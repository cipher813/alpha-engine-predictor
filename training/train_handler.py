"""
training/train_handler.py — Weekly GBM retraining pipeline for Lambda.

Called by inference/handler.py when event["action"] == "train".

Pipeline:
  1. Download per-ticker OHLCV Parquet files + sector_map.json from S3.
  1b. Refresh any stale parquets with recent yfinance data and upload back to S3.
  2. Build regression arrays (29 features) and apply 70/15/15 time-based split
     with purge gaps of FORWARD_DAYS between train/val and val/test.
  3. Train GBMScorer with Optuna-tuned params from config.GBM_TUNED_PARAMS
     (n_estimators=2000, early_stopping=50).
  4. Evaluate on test set: IC, IC IR, positive-period rate.
  5. Upload dated backup unconditionally; promote to gbm_latest.txt only if IC gate passes.
  5b. Write slim 2-year price cache to predictor/price_cache_slim/ so the daily
      inference Lambda can skip the 2y yfinance fetch and read from S3 instead.
  6. Send training summary email with results.

S3 layout:
  predictor/price_cache/*.parquet       — adjusted OHLCV price history per ticker (10y,
                                          auto_adjust=True; rewritten weekly by this handler)
  predictor/price_cache/sector_map.json — ticker → sector ETF symbol
  predictor/weights/gbm_latest.txt      — (output) active inference weights
  predictor/weights/gbm_{date}.txt      — (output) dated backup

  predictor/daily_closes/{date}.parquet — Backward-split-adjusted OHLCV snapshot per
                                          trading day (auto_adjust=False; yfinance has no
                                          truly-raw mode — all modes apply retroactive split
                                          adjustment).  Written by save_daily_closes() each
                                          morning.  Used as the Mon–Fri delta source by
                                          load_price_data_from_cache(), reducing daily yfinance
                                          fetches to only the ~1-2 split tickers per week.
  predictor/price_cache_slim/*.parquet  — 2-year slice of each ticker (written by
                                          write_slim_cache() after each Sunday training run).
                                          The inference Lambda downloads this (~9 MB) instead
                                          of re-fetching 2y from yfinance each morning.

Lambda environment variables (same as predictor email):
  EMAIL_SENDER / EMAIL_RECIPIENTS / GMAIL_APP_PASSWORD / AWS_REGION
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import tempfile
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── S3 data download ───────────────────────────────────────────────────────────

def download_price_cache(bucket: str, prefix: str, local_dir: Path) -> int:
    """
    Download all objects under S3 prefix (Parquets + sector_map.json) to local_dir.
    Returns number of files downloaded.
    """
    import boto3
    s3 = boto3.client("s3")
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]   # strip prefix, keep filename
            if not filename or filename.endswith("/"):
                continue
            local_path = local_dir / filename
            s3.download_file(bucket, key, str(local_path))
            n += 1

    log.info("Downloaded %d files from s3://%s/%s → %s", n, bucket, prefix, local_dir)
    return n


# ── Incremental price cache refresh ────────────────────────────────────────────

# yfinance requires a leading caret for these index / rate tickers.
_CARET_SYMBOLS = {"VIX", "TNX", "IRX"}

# Batch size for yfinance multi-ticker downloads.
_REFRESH_BATCH_SIZE = 50


def refresh_price_cache(
    bucket: str,
    prefix: str,
    local_dir: Path,
    fetch_period: str = "10y",
    staleness_threshold_days: int = 2,
) -> int:
    """
    Fully rewrite stale .parquet files in local_dir with fresh yfinance data and
    upload the new files back to S3.

    For each <ticker>.parquet in local_dir, the last date in the file is checked.
    If that date is more than ``staleness_threshold_days`` business days before today,
    the ticker is added to the refresh list.  A full ``fetch_period`` history is
    downloaded from yfinance and **replaces** (not appends to) the existing parquet.

    **Why full replace, not append:**
    yfinance's ``auto_adjust=True`` retroactively adjusts the *entire* price history
    whenever a stock split or large dividend occurs.  Any shorter incremental window
    creates a price-level discontinuity at the splice point for splits that occurred
    between the bootstrap and the window start — breaking momentum, MA, and every
    other feature that crosses that date boundary.  A full rewrite guarantees the
    stored history is always internally consistent with the current adjustment factors.
    If a download fails for any ticker, the existing parquet is left untouched.

    Parameters
    ----------
    bucket                   : S3 bucket name.
    prefix                   : S3 key prefix, e.g. "predictor/price_cache/".
    local_dir                : Local directory containing the downloaded parquets.
    fetch_period             : yfinance period string (default: "10y" — matches
                               the original bootstrap and covers the full training
                               window).
    staleness_threshold_days : Minimum business-day lag before a file is refreshed.

    Returns
    -------
    Number of tickers successfully refreshed.
    """
    import pandas as pd
    import yfinance as yf
    import boto3
    from data.dataset import _parquet_engine

    engine = _parquet_engine()
    today  = pd.Timestamp.now().normalize()  # timezone-naive, matches bootstrap index

    # ── Identify stale tickers ────────────────────────────────────────────────
    stale: list[tuple[str, Path]] = []  # [(ticker_name, parquet_path), ...]
    for parquet_path in sorted(local_dir.glob("*.parquet")):
        stem = parquet_path.stem
        if stem in ("sector_map",):
            continue
        try:
            df = pd.read_parquet(parquet_path, engine=engine)
            if df.empty:
                continue
            last_ts = pd.Timestamp(df.index.max())
            # Strip timezone for comparison against timezone-naive today
            if last_ts.tzinfo is not None:
                last_ts = last_ts.tz_convert("UTC").tz_localize(None)
            bdays_lag = len(pd.bdate_range(last_ts, today)) - 1
            if bdays_lag >= staleness_threshold_days:
                stale.append((stem, parquet_path))
        except Exception as exc:
            log.warning("Could not check staleness for %s: %s", stem, exc)

    if not stale:
        log.info("Price cache is current — no refresh needed")
        return 0

    log.info(
        "Refreshing %d stale tickers (fetch_period=%s, engine=%s) ...",
        len(stale), fetch_period, engine,
    )

    s3        = boto3.client("s3")
    refreshed = 0

    # ── Batch-fetch and append ────────────────────────────────────────────────
    for batch_start in range(0, len(stale), _REFRESH_BATCH_SIZE):
        batch        = stale[batch_start : batch_start + _REFRESH_BATCH_SIZE]
        ticker_names = [t for t, _ in batch]
        yf_symbols   = [
            f"^{t}" if t in _CARET_SYMBOLS else t
            for t in ticker_names
        ]

        try:
            tickers_arg = yf_symbols[0] if len(yf_symbols) == 1 else yf_symbols
            raw = yf.download(
                tickers=tickers_arg,
                period=fetch_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            is_multi = isinstance(raw.columns, pd.MultiIndex)
        except Exception as exc:
            log.warning(
                "yfinance batch download failed for %s...: %s",
                ticker_names[:3], exc,
            )
            continue

        for ticker, parquet_path in batch:
            yf_sym = f"^{ticker}" if ticker in _CARET_SYMBOLS else ticker
            try:
                new_df = (raw[yf_sym] if is_multi else raw).copy()
                if "Close" not in new_df.columns or new_df.empty:
                    log.debug("No new data returned for %s — skipping", ticker)
                    continue
                new_df = new_df.dropna(subset=["Close"])
                if new_df.empty:
                    continue

                # Normalize index to timezone-naive (matches bootstrap format and
                # avoids TypeError in compute_features' reindex() calls).
                idx = pd.to_datetime(new_df.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                new_df.index = idx
                new_df = new_df.sort_index()

                # Full replace — no append, no concat.  The fresh download from
                # yfinance already has all splits and dividends correctly reflected
                # throughout the entire history.  If this write fails for any reason,
                # the except clause leaves the old parquet intact.
                new_df.to_parquet(parquet_path, engine=engine, compression="snappy")

                # Upload back to S3
                s3_key = f"{prefix}{ticker}.parquet"
                s3.upload_file(str(parquet_path), bucket, s3_key)
                refreshed += 1

            except Exception as exc:
                log.warning("Refresh failed for %s: %s", ticker, exc)

        pct = 100 * min(batch_start + _REFRESH_BATCH_SIZE, len(stale)) / len(stale)
        log.info(
            "Refresh batch %d/%d done — %.0f%% complete",
            batch_start // _REFRESH_BATCH_SIZE + 1,
            -(-len(stale) // _REFRESH_BATCH_SIZE),  # ceiling div
            pct,
        )

    log.info(
        "Price cache refresh complete: %d / %d tickers updated",
        refreshed, len(stale),
    )
    return refreshed


# ── Slim cache writer ──────────────────────────────────────────────────────────

def write_slim_cache(
    bucket: str,
    full_cache_dir: Path,
    slim_prefix: str = "predictor/price_cache_slim/",
    lookback_days: int = 730,
) -> int:
    """
    After weekly training, write a 2-year slice of each ticker parquet to S3 at
    the slim cache prefix.  The inference Lambda downloads this slim cache at
    6:15 AM instead of fetching 2 years from yfinance — reducing daily yfinance
    calls from ~450 000 rows to at most a few hundred (the Mon–Fri delta rows
    sourced from predictor/daily_closes/).

    Strategy
    --------
    For each <ticker>.parquet in full_cache_dir (already refreshed to current):
        1. Load and slice rows where index >= today - lookback_days.
        2. Write slim parquet to a temp file, upload to S3, delete temp.

    The slim prefix is completely overwritten on each training run to stay
    consistent with the freshly refreshed full cache.

    Parameters
    ----------
    bucket          : S3 bucket name.
    full_cache_dir  : Local directory containing the full 10y parquets (after
                      refresh and before training completes).
    slim_prefix     : S3 key prefix for slim parquets.
    lookback_days   : Calendar days of history to keep (default 730 ≈ 2 years).

    Returns
    -------
    Number of ticker slim parquets successfully uploaded.
    """
    import pandas as pd
    import boto3
    from data.dataset import _parquet_engine

    engine  = _parquet_engine()
    s3      = boto3.client("s3")
    cutoff  = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    written = 0

    parquet_files = sorted(full_cache_dir.glob("*.parquet"))
    log.info(
        "Writing slim cache: %d parquets → s3://%s/%s (cutoff %s) …",
        len(parquet_files), bucket, slim_prefix, cutoff.date(),
    )

    for parquet_path in parquet_files:
        try:
            df = pd.read_parquet(parquet_path, engine=engine)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)

            slim_df = df[df.index >= cutoff]
            if slim_df.empty:
                continue

            # Write to temp sibling file, upload, then delete
            slim_path = parquet_path.with_name("_slim_" + parquet_path.name)
            slim_df.to_parquet(slim_path, engine=engine, compression="snappy")

            s3_key = f"{slim_prefix}{parquet_path.name}"
            s3.upload_file(str(slim_path), bucket, s3_key)
            slim_path.unlink()
            written += 1

        except Exception as exc:
            log.warning("Slim cache write failed for %s: %s", parquet_path.stem, exc)

    log.info(
        "Slim cache written: %d / %d tickers uploaded to s3://%s/%s",
        written, len(parquet_files), bucket, slim_prefix,
    )
    return written


# ── Training pipeline ──────────────────────────────────────────────────────────

def run_gbm_training(
    data_dir: str,
    bucket: str,
    date_str: str,
    dry_run: bool = False,
) -> dict:
    """
    Build dataset, train GBMScorer, evaluate, and upload to S3.

    Parameters
    ----------
    data_dir  : Local directory containing *.parquet + sector_map.json.
    bucket    : S3 bucket for model upload.
    date_str  : Training date label (YYYY-MM-DD) for the dated backup key.
    dry_run   : Skip S3 upload if True.

    Returns
    -------
    Result dict with val_ic, test_ic, ic_ir, promoted flag, feature_importance, etc.
    """
    import config as cfg
    from data.dataset import build_regression_arrays
    from model.gbm_scorer import GBMScorer

    start_ts = datetime.now(timezone.utc)

    # ── Build arrays (no PyTorch dependency) ──────────────────────────────────
    log.info("Building regression arrays from %s ...", data_dir)
    X_all, fwd_all, all_dates = build_regression_arrays(
        data_dir=data_dir,
        config_module=cfg,
    )

    N = len(fwd_all)

    # ── Time-based split with purge gaps (mirrors dataset.py logic) ───────────
    purge_days = cfg.FORWARD_DAYS
    n_train = int(N * cfg.TRAIN_FRAC)
    n_val_raw = int(N * cfg.VAL_FRAC)

    # Purge gap 1: train → val
    train_end_date = all_dates[n_train - 1]
    unique_post_train = sorted(set(d for d in all_dates[n_train:] if d > train_end_date))
    if len(unique_post_train) >= purge_days:
        purge_cutoff_1 = unique_post_train[purge_days - 1]
        val_start = next(i for i in range(n_train, N) if all_dates[i] > purge_cutoff_1)
    else:
        val_start = n_train

    # Val slice
    val_end = min(val_start + n_val_raw, N)

    # Purge gap 2: val → test
    val_end_date = all_dates[min(val_end - 1, N - 1)]
    unique_post_val = sorted(set(d for d in all_dates[val_end:] if d > val_end_date))
    if len(unique_post_val) >= purge_days:
        purge_cutoff_2 = unique_post_val[purge_days - 1]
        test_start = next(i for i in range(val_end, N) if all_dates[i] > purge_cutoff_2)
    else:
        test_start = val_end

    X_train = X_all[:n_train]
    y_train = fwd_all[:n_train]
    X_val   = X_all[val_start:val_end]
    y_val   = fwd_all[val_start:val_end]
    X_test  = X_all[test_start:]
    y_test  = fwd_all[test_start:]

    log.info(
        "Dataset ready: train=%d  val=%d  test=%d  features=%d  "
        "purged=%d samples",
        len(y_train), len(y_val), len(y_test), X_train.shape[1],
        (val_start - n_train) + (test_start - val_end),
    )

    # ── Train with Optuna-tuned hyperparameters ───────────────────────────────
    tuned_params = getattr(cfg, "GBM_TUNED_PARAMS", None)
    log.info(
        "Training GBMScorer (n_estimators=2000, early_stopping=50, %s) ...",
        "Optuna-tuned params" if tuned_params else "default params",
    )
    scorer = GBMScorer(
        params=tuned_params,
        n_estimators=2000,
        early_stopping_rounds=50,
    )
    scorer.fit(X_train, y_train, X_val, y_val, feature_names=cfg.FEATURES)

    # ── Evaluate (numpy-only, no scipy dependency) ──────────────────────────
    test_preds = scorer.predict(X_test)
    test_ic = float(np.corrcoef(test_preds, y_test)[0, 1])

    n_chunks   = 20
    chunk_size = len(test_preds) // n_chunks
    chunk_ics  = np.array([
        float(np.corrcoef(
            test_preds[i*chunk_size:(i+1)*chunk_size],
            y_test[i*chunk_size:(i+1)*chunk_size],
        )[0, 1])
        for i in range(n_chunks)
    ])
    ic_ir = float(chunk_ics.mean() / (chunk_ics.std() + 1e-8))

    importance = scorer.feature_importance(importance_type="gain")
    top10      = sorted(importance.items(), key=lambda x: -x[1])[:10]

    passes_ic    = float(test_ic) >= cfg.MIN_IC
    passes_ic_ir = ic_ir >= 0.3
    elapsed_s    = (datetime.now(timezone.utc) - start_ts).total_seconds()
    model_version = f"GBM-v{scorer._best_iteration}"

    log.info(
        "Training complete: val_IC=%.4f  test_IC=%.4f  IC_IR=%.3f  "
        "passes_ic=%s  elapsed=%.0fs",
        scorer._val_ic, test_ic, ic_ir, passes_ic, elapsed_s,
    )

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    promoted = False
    if not dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            booster_path = Path(tmp) / "gbm_model.txt"
            scorer.save(booster_path)

            import boto3
            s3 = boto3.client("s3")

            # Always save a dated backup
            dated_key = f"predictor/weights/gbm_{date_str}.txt"
            s3.upload_file(str(booster_path), bucket, dated_key)
            log.info("Uploaded dated backup: s3://%s/%s", bucket, dated_key)

            if passes_ic:
                # Promote to active weights
                s3.upload_file(str(booster_path), bucket, cfg.GBM_WEIGHTS_KEY)
                meta = {
                    "model_version":  model_version,
                    "val_ic":         round(float(scorer._val_ic), 6),
                    "test_ic":        round(float(test_ic), 6),
                    "ic_ir":          round(ic_ir, 4),
                    "trained_date":   date_str,
                    "best_iteration": scorer._best_iteration,
                    "n_train":        int(len(y_train)),
                }
                s3.put_object(
                    Bucket=bucket,
                    Key=cfg.GBM_WEIGHTS_META_KEY,
                    Body=json.dumps(meta, indent=2).encode(),
                    ContentType="application/json",
                )
                log.info("Promoted to active weights: s3://%s/%s", bucket, cfg.GBM_WEIGHTS_KEY)
                promoted = True
            else:
                log.warning(
                    "IC gate FAILED (%.4f < %.4f) — backup saved, NOT promoted",
                    test_ic, cfg.MIN_IC,
                )

    return {
        "model_version":         model_version,
        "best_iteration":        scorer._best_iteration,
        "val_ic":                round(float(scorer._val_ic), 6),
        "test_ic":               round(float(test_ic), 6),
        "test_ic_p":             0.0,  # p-value omitted (numpy-only path)
        "ic_ir":                 round(ic_ir, 4),
        "ic_positive_20":        int((chunk_ics > 0).sum()),
        "passes_ic_gate":        passes_ic,
        "passes_ic_ir_gate":     passes_ic_ir,
        "promoted":              promoted,
        "n_train":               int(len(y_train)),
        "n_val":                 int(len(y_val)),
        "n_test":                int(len(y_test)),
        "elapsed_s":             round(elapsed_s, 1),
        "feature_importance_top10": [
            {"feature": f, "gain": round(s, 2)} for f, s in top10
        ],
    }


# ── Training email ─────────────────────────────────────────────────────────────

def send_training_email(result: dict, date_str: str) -> bool:
    """
    Send GBM training summary email via Gmail SMTP (primary) or SES (fallback).
    Returns True on success. Never raises.
    """
    import config as cfg

    sender     = cfg.EMAIL_SENDER
    recipients = cfg.EMAIL_RECIPIENTS

    if not sender or not recipients:
        log.info("Training email skipped — EMAIL_SENDER/EMAIL_RECIPIENTS not set")
        return False

    promoted     = result.get("promoted", False)
    passes_ic    = result.get("passes_ic_gate", False)
    val_ic       = result.get("val_ic", 0.0)
    test_ic      = result.get("test_ic", 0.0)
    ic_ir        = result.get("ic_ir", 0.0)
    version      = result.get("model_version", "unknown")
    elapsed_s    = result.get("elapsed_s", 0)
    n_train      = result.get("n_train", 0)
    ic_pos       = result.get("ic_positive_20", 0)
    top10        = result.get("feature_importance_top10", [])

    ic_color    = "#2e7d32" if passes_ic else "#c62828"
    ic_label    = "PASS ✓" if passes_ic else "FAIL ✗"
    promo_color = "#2e7d32" if promoted else "#c62828"
    promo_label = "Promoted → gbm_latest ✓" if promoted else "NOT promoted (IC gate failed ✗)"
    status_str  = "PASS" if passes_ic else "FAIL"

    subject = (
        f"GBM Trainer | {date_str} | IC {test_ic:.4f} {status_str} | "
        f"{'Promoted' if promoted else 'Not promoted'}"
    )

    # Feature importance bar chart (top 5)
    max_gain  = max((r["gain"] for r in top10), default=1)
    feat_rows = "".join(
        f'<tr>'
        f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{r["feature"]}</td>'
        f'<td style="padding:2px 8px; width:130px;">'
        f'<div style="background:#1976d2; height:10px; width:{int(r["gain"]/max_gain*100)}%;"></div>'
        f'</td>'
        f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{r["gain"]:.1f}</td>'
        f'</tr>'
        for r in top10[:5]
    )

    html_body = (
        f'<html><body style="font-family:sans-serif; font-size:13px; color:#222; max-width:600px;">'
        f'<h2 style="margin-bottom:4px;">GBM Weekly Retrain — {date_str}</h2>'
        f'<p style="color:#555; font-size:12px; margin-top:0;">'
        f'Model: <b>{version}</b> &nbsp;|&nbsp;'
        f'Training samples: <b>{n_train:,}</b> &nbsp;|&nbsp;'
        f'Elapsed: <b>{elapsed_s:.0f}s ({elapsed_s/60:.1f} min)</b></p>'

        f'<table style="border-collapse:collapse; width:100%; margin-bottom:12px;">'
        f'<tr style="background:#f9f9f9;">'
        f'  <td style="padding:5px 10px; color:#555; width:160px;">Val IC</td>'
        f'  <td style="padding:5px 10px; font-family:monospace; font-weight:bold;">{val_ic:.4f}</td>'
        f'</tr>'
        f'<tr>'
        f'  <td style="padding:5px 10px; color:#555;">Test IC</td>'
        f'  <td style="padding:5px 10px; font-family:monospace; font-weight:bold; color:{ic_color};">'
        f'  {test_ic:.4f} — {ic_label}</td>'
        f'</tr>'
        f'<tr style="background:#f9f9f9;">'
        f'  <td style="padding:5px 10px; color:#555;">IC IR</td>'
        f'  <td style="padding:5px 10px; font-family:monospace;">'
        f'  {ic_ir:.3f} {"✓" if ic_ir >= 0.3 else "✗"}</td>'
        f'</tr>'
        f'<tr>'
        f'  <td style="padding:5px 10px; color:#555;">IC positive periods</td>'
        f'  <td style="padding:5px 10px; font-family:monospace;">{ic_pos}/20</td>'
        f'</tr>'
        f'<tr style="background:#f9f9f9;">'
        f'  <td style="padding:5px 10px; color:#555; font-weight:bold;">Promotion</td>'
        f'  <td style="padding:5px 10px; font-weight:bold; color:{promo_color};">{promo_label}</td>'
        f'</tr>'
        f'</table>'

        f'<h3 style="margin-bottom:4px;">Top 5 Features by Gain</h3>'
        f'<table>{feat_rows}</table>'

        f'<p style="font-size:11px; color:#aaa; margin-top:20px;">'
        f'IC gate: ≥{cfg.MIN_IC:.2f} to promote &nbsp;|&nbsp; IC IR gate: ≥0.30</p>'
        f'</body></html>'
    )

    plain_body = (
        f"GBM Weekly Retrain — {date_str}\n"
        f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
        f"\nVal IC:    {val_ic:.4f}"
        f"\nTest IC:   {test_ic:.4f} — {ic_label}"
        f"\nIC IR:     {ic_ir:.3f}"
        f"\nIC pos:    {ic_pos}/20"
        f"\nPromotion: {promo_label}\n"
        f"\nTop features: " + ", ".join(r["feature"] for r in top10[:5]) + "\n"
    )

    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()

    if app_password:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body,  "html",  "utf-8"))
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender, app_password.replace(" ", ""))
                server.sendmail(sender, recipients, msg.as_string())
            log.info("Training email sent via Gmail SMTP: '%s'", subject)
            return True
        except Exception as exc:
            log.warning("Gmail SMTP failed (%s) — trying SES fallback", exc)

    # SES fallback
    try:
        import boto3
        ses = boto3.client("ses", region_name=cfg.AWS_REGION)
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body,  "Charset": "UTF-8"},
                },
            },
        )
        log.info("Training email sent via SES: '%s'", subject)
        return True
    except Exception as exc:
        log.warning("SES failed: %s — training email not delivered", exc)
        return False


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main(
    bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Entry point called from inference/handler.py for the "train" action.

    1.  Download price cache from S3 to /tmp.
    1b. Refresh any stale parquets with recent yfinance data → upload back to S3.
    2.  Run GBM training pipeline on the now-current local cache.
    2b. Write a 2-year slim cache to S3 (predictor/price_cache_slim/).
        The inference Lambda uses this slim cache + daily_closes delta instead
        of fetching 2 years from yfinance every morning, reducing daily yfinance
        API calls by ~125×.
    3.  Send training summary email.

    Returns the result dict from run_gbm_training().
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    log.info("GBM training run: date=%s  bucket=%s  dry_run=%s", date_str, bucket, dry_run)

    # Step 1: Download price cache (Parquets + sector_map.json)
    tmp_cache = Path(tempfile.mkdtemp()) / "cache"
    n_files = download_price_cache(
        bucket=bucket,
        prefix="predictor/price_cache/",
        local_dir=tmp_cache,
    )

    if n_files == 0:
        raise RuntimeError(
            f"No files found at s3://{bucket}/predictor/price_cache/ — "
            "run bootstrap_fetcher.py first to populate the price cache."
        )

    # Step 1b: Refresh any stale parquets with recent data → upload back to S3
    n_refreshed = refresh_price_cache(
        bucket=bucket,
        prefix="predictor/price_cache/",
        local_dir=tmp_cache,
    )
    log.info("Price cache refresh: %d tickers updated", n_refreshed)

    # Step 2: Train + upload
    result = run_gbm_training(
        data_dir=str(tmp_cache),
        bucket=bucket,
        date_str=date_str,
        dry_run=dry_run,
    )

    # Step 2b: Write slim cache for inference (2-year slice of each ticker)
    # The slim cache lets the daily inference Lambda skip the 2y yfinance fetch
    # and instead download ~9 MB of parquets + a few daily_closes delta rows.
    # Skipped on dry_run since no model upload happened either.
    if not dry_run:
        n_slim = write_slim_cache(
            bucket=bucket,
            full_cache_dir=tmp_cache,
        )
        log.info("Slim cache written: %d tickers", n_slim)
        result["slim_cache_tickers"] = n_slim
    else:
        log.info("[dry-run] Skipping slim cache write")

    # Step 3: Email
    if not dry_run:
        send_training_email(result, date_str)

    return result
