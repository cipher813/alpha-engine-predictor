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
                                          write_slim_cache() after each weekly training run).
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
import time
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

# Batch size for yfinance multi-ticker downloads (from config).
from config import REFRESH_BATCH_SIZE as _REFRESH_BATCH_SIZE


def refresh_price_cache(
    bucket: str,
    prefix: str,
    local_dir: Path,
    fetch_period: str | None = None,
    staleness_threshold_days: int | None = None,
    fd=None,
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
    from config import BOOTSTRAP_PERIOD, STALENESS_THRESHOLD_DAYS as _DEFAULT_STALE_DAYS

    if fetch_period is None:
        fetch_period = BOOTSTRAP_PERIOD
    if staleness_threshold_days is None:
        staleness_threshold_days = _DEFAULT_STALE_DAYS

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
    lookback_days: int | None = None,
    fd=None,
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
    from config import SLIM_CACHE_LOOKBACK_DAYS

    if lookback_days is None:
        lookback_days = SLIM_CACHE_LOOKBACK_DAYS

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


# ── Walk-forward validation ───────────────────────────────────────────────────

def _find_date_boundary(all_dates: list, target_idx: int, purge_days: int, N: int) -> int:
    """Find the first sample index after purging `purge_days` unique dates past target_idx."""
    if target_idx >= N:
        return N
    boundary_date = all_dates[min(target_idx - 1, N - 1)]
    unique_post = sorted(set(d for d in all_dates[target_idx:] if d > boundary_date))
    if len(unique_post) >= purge_days:
        purge_cutoff = unique_post[purge_days - 1]
        return next(i for i in range(target_idx, N) if all_dates[i] > purge_cutoff)
    return target_idx


def run_walk_forward(
    X_all: np.ndarray,
    fwd_all: np.ndarray,
    all_dates: list,
    cfg,
) -> dict:
    """
    Expanding-window walk-forward validation.

    Splits data into ~15 folds of WF_TEST_WINDOW_DAYS each, with expanding
    training windows (all data before the fold boundary). Each fold trains a
    fresh GBM and computes IC on the out-of-sample test window.

    Returns dict with fold_ics, median_ic, pct_positive, passes_wf flags,
    and per-fold detail for email reporting.
    """
    from model.gbm_scorer import GBMScorer

    N = len(fwd_all)
    unique_dates = sorted(set(all_dates))
    n_unique = len(unique_dates)
    test_window = cfg.WF_TEST_WINDOW_DAYS
    min_train = cfg.WF_MIN_TRAIN_DAYS
    purge_days = cfg.WF_PURGE_DAYS

    # Build date → index mapping for efficient boundary lookup
    date_to_first_idx: dict = {}
    for i, d in enumerate(all_dates):
        if d not in date_to_first_idx:
            date_to_first_idx[d] = i

    # Generate fold boundaries: expanding train, fixed-size test windows.
    # After the last full fold, include a partial final fold if at least
    # half a test window of data remains — ensures recent data is validated.
    min_partial_days = test_window // 2  # minimum 63 days for partial fold
    folds: list[dict] = []
    fold_start_date_idx = min_train  # index into unique_dates
    while fold_start_date_idx < n_unique:
        remaining = n_unique - fold_start_date_idx
        # Skip if less than half a test window remains
        if remaining < min_partial_days:
            break

        test_start_date = unique_dates[fold_start_date_idx]
        test_end_date_idx = min(fold_start_date_idx + test_window - 1, n_unique - 1)
        test_end_date = unique_dates[test_end_date_idx]

        # Train end: purge_days before test start
        train_end_date_idx = fold_start_date_idx - purge_days
        if train_end_date_idx < min_train // 2:
            fold_start_date_idx += test_window
            continue
        train_end_date = unique_dates[train_end_date_idx]

        # Convert date boundaries to sample indices
        train_mask = np.array([d <= train_end_date for d in all_dates])
        test_mask = np.array([test_start_date <= d <= test_end_date for d in all_dates])

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        if len(train_indices) < 1000 or len(test_indices) < 100:
            fold_start_date_idx += test_window
            continue

        is_partial = remaining < test_window
        folds.append({
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_end_date": str(train_end_date),
            "test_start_date": str(test_start_date),
            "test_end_date": str(test_end_date),
            "partial": is_partial,
        })
        if is_partial:
            log.info("  Partial final fold: %d/%d test days", remaining, test_window)

        fold_start_date_idx += test_window

    log.info("Walk-forward: %d folds generated from %d unique dates", len(folds), n_unique)

    if len(folds) < 3:
        log.warning("Walk-forward: too few folds (%d) — falling back to single split", len(folds))
        return {"folds": [], "median_ic": 0.0, "pct_positive": 0.0, "passes_wf": False,
                "fallback": True}

    tuned_params = getattr(cfg, "GBM_TUNED_PARAMS", None)
    # Walk-forward fast mode: use lighter estimators/early-stopping for fold training
    wf_n_est = getattr(cfg, "WF_N_ESTIMATORS", None) or cfg.GBM_N_ESTIMATORS
    wf_early_stop = getattr(cfg, "WF_EARLY_STOPPING", None) or cfg.GBM_EARLY_STOPPING_ROUNDS
    log.info("Walk-forward training params: n_estimators=%d, early_stopping=%d", wf_n_est, wf_early_stop)
    fold_results: list[dict] = []

    for i, fold in enumerate(folds):
        fold_start = time.time()
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]
        X_train_fold = X_all[train_idx]
        y_train_fold = fwd_all[train_idx]
        X_test_fold = X_all[test_idx]
        y_test_fold = fwd_all[test_idx]

        # Split train into sub-train (85%) + sub-val (15%) for early stopping
        n_sub_train = int(len(y_train_fold) * 0.85)
        X_sub_train = X_train_fold[:n_sub_train]
        y_sub_train = y_train_fold[:n_sub_train]
        X_sub_val = X_train_fold[n_sub_train:]
        y_sub_val = y_train_fold[n_sub_train:]

        scorer = GBMScorer(
            params=tuned_params,
            n_estimators=wf_n_est,
            early_stopping_rounds=wf_early_stop,
        )
        scorer.fit(X_sub_train, y_sub_train, X_sub_val, y_sub_val,
                    feature_names=cfg.GBM_FEATURES)

        test_preds = scorer.predict(X_test_fold)

        # IC for this fold
        if len(test_preds) > 1 and np.std(test_preds) > 1e-10 and np.std(y_test_fold) > 1e-10:
            fold_ic = float(np.corrcoef(test_preds, y_test_fold)[0, 1])
        else:
            fold_ic = 0.0

        fold_elapsed = time.time() - fold_start
        fold_result = {
            "fold": i + 1,
            "train_end": fold["train_end_date"],
            "test_start": fold["test_start_date"],
            "test_end": fold["test_end_date"],
            "n_train": len(y_train_fold),
            "n_test": len(y_test_fold),
            "ic": round(fold_ic, 6),
            "best_iteration": scorer._best_iteration,
            "elapsed_s": round(fold_elapsed, 1),
        }
        fold_results.append(fold_result)
        log.info(
            "  Fold %d/%d: train=%d test=%d  [%s → %s]  IC=%.4f  (%.1fs)",
            i + 1, len(folds), len(y_train_fold), len(y_test_fold),
            fold["test_start_date"], fold["test_end_date"], fold_ic, fold_elapsed,
        )

    fold_ics = np.array([f["ic"] for f in fold_results])
    median_ic = float(np.median(fold_ics))
    pct_positive = float((fold_ics > 0).mean())

    passes_median = median_ic >= cfg.WF_MEDIAN_IC_GATE
    passes_pct = pct_positive >= cfg.WF_MIN_FOLDS_POSITIVE
    passes_wf = passes_median and passes_pct

    log.info(
        "Walk-forward summary: median_IC=%.4f (%s %.4f)  "
        "pct_positive=%.1f%% (%s %.0f%%)  passes=%s",
        median_ic, ">=" if passes_median else "<", cfg.WF_MEDIAN_IC_GATE,
        pct_positive * 100, ">=" if passes_pct else "<", cfg.WF_MIN_FOLDS_POSITIVE * 100,
        passes_wf,
    )

    return {
        "folds": fold_results,
        "median_ic": round(median_ic, 6),
        "pct_positive": round(pct_positive, 4),
        "passes_median_ic": passes_median,
        "passes_pct_positive": passes_pct,
        "passes_wf": passes_wf,
        "fallback": False,
    }


# ── Training pipeline ──────────────────────────────────────────────────────────

def run_gbm_training(
    data_dir: str,
    bucket: str,
    date_str: str,
    dry_run: bool = False,
) -> dict:
    """
    Build dataset, train GBMScorer, evaluate, and upload to S3.

    If walk-forward validation is enabled (config.WF_ENABLED), runs expanding-
    window cross-validation across multiple regime periods before training the
    final production model. The walk-forward IC summary is included in the
    result dict and email.

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
        feature_list=cfg.GBM_FEATURES,
    )

    N = len(fwd_all)

    # ── Walk-forward validation (if enabled) ──────────────────────────────────
    wf_result: Optional[dict] = None
    if cfg.WF_ENABLED:
        log.info("Walk-forward validation enabled — running expanding-window CV ...")
        wf_result = run_walk_forward(X_all, fwd_all, all_dates, cfg)
        if wf_result.get("fallback"):
            log.warning("Walk-forward fell back — proceeding with single-split evaluation")
        elif not wf_result["passes_wf"]:
            log.warning(
                "Walk-forward FAILED gates (median_IC=%.4f, pct_positive=%.1f%%) "
                "— training production model anyway but will NOT promote",
                wf_result["median_ic"], wf_result["pct_positive"] * 100,
            )

    # ── Train final production model on all data ──────────────────────────────
    # Use 70/15/15 split for single-split IC metrics (reporting) and early stopping.
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
    ensemble_enabled = getattr(cfg, "GBM_ENSEMBLE_LAMBDARANK", True)
    train_dates_slice = all_dates[:n_train]
    val_dates_slice = all_dates[val_start:val_end]

    log.info(
        "Training GBMScorer (n_estimators=%d, early_stopping=%d, %s%s) ...",
        cfg.GBM_N_ESTIMATORS, cfg.GBM_EARLY_STOPPING_ROUNDS,
        "Optuna-tuned params" if tuned_params else "default params",
        " + lambdarank ensemble" if ensemble_enabled else "",
    )

    # ── MSE model (always trained) ────────────────────────────────────────────
    scorer = GBMScorer(
        params=tuned_params,
        n_estimators=cfg.GBM_N_ESTIMATORS,
        early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
        ranking_objective=False,
    )
    scorer.fit(X_train, y_train, X_val, y_val, feature_names=cfg.GBM_FEATURES)

    # ── Lambdarank model (when ensemble enabled) ──────────────────────────────
    rank_scorer = None
    if ensemble_enabled:
        try:
            rank_scorer = GBMScorer(
                params=tuned_params,
                n_estimators=cfg.GBM_N_ESTIMATORS,
                early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
                ranking_objective=True,
            )
            rank_scorer.fit(
                X_train, y_train, X_val, y_val,
                feature_names=cfg.GBM_FEATURES,
                train_dates=train_dates_slice,
                val_dates=val_dates_slice,
            )
            log.info("Lambdarank model trained: val_IC=%.4f", rank_scorer._val_ic)
        except Exception as e:
            log.warning("Lambdarank training failed — falling back to MSE-only: %s", e)
            rank_scorer = None

    # ── Evaluate ─────────────────────────────────────────────────────────────
    try:
        from scipy.stats import rankdata
    except ImportError:
        # Numpy-only fallback for rankdata
        def rankdata(a):
            arr = np.asarray(a)
            order = arr.argsort()
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
            return ranks

    test_preds_mse = scorer.predict(X_test)
    mse_ic = float(np.corrcoef(test_preds_mse, y_test)[0, 1])

    # Lambdarank IC + ensemble IC
    rank_ic = 0.0
    ensemble_ic = mse_ic  # default to MSE if no rank model
    if rank_scorer is not None:
        test_preds_rank = rank_scorer.predict(X_test)
        rank_ic = float(np.corrcoef(test_preds_rank, y_test)[0, 1])
        # Ensemble: rank-normalize each model's predictions, then average
        mse_ranked = rankdata(test_preds_mse).astype(np.float32)
        rank_ranked = rankdata(test_preds_rank).astype(np.float32)
        ensemble_preds = 0.5 * mse_ranked + 0.5 * rank_ranked
        ensemble_ic = float(np.corrcoef(ensemble_preds, y_test)[0, 1])
        log.info(
            "Ensemble ICs: MSE=%.4f  Lambdarank=%.4f  Ensemble=%.4f",
            mse_ic, rank_ic, ensemble_ic,
        )

    # Pick the best IC among all candidates for the gate
    if rank_scorer is not None:
        candidates = {"mse": mse_ic, "ensemble": ensemble_ic}
        candidates["rank"] = rank_ic
    else:
        candidates = {"mse": mse_ic}
    best_mode = max(candidates, key=candidates.get)
    best_ic = candidates[best_mode]
    test_ic = best_ic
    test_preds = test_preds_mse  # for chunk IC and backwards compat
    log.info(
        "Best IC selection: %s (%.4f) from candidates %s",
        best_mode, best_ic,
        {k: round(v, 4) for k, v in candidates.items()},
    )

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

    # SHAP feature importance (more reliable than gain-based) — computed on MSE model
    shap_importance = None
    try:
        import shap
        explainer = shap.TreeExplainer(scorer._booster)
        shap_values = explainer.shap_values(X_test[:500])  # cap at 500 rows for speed
        shap_importance = dict(zip(
            cfg.GBM_FEATURES,
            [round(float(v), 4) for v in np.abs(shap_values).mean(axis=0)]
        ))
        shap_top10 = sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
        log.info("SHAP top 5: %s", shap_top10[:5])
    except Exception as e:
        log.warning("SHAP computation failed (non-blocking): %s", e)

    # ── Per-feature IC tracking ────────────────────────────────────────────────
    feature_ics = {}
    for i, fname in enumerate(cfg.GBM_FEATURES):
        feat_vals = X_test[:, i]
        if np.std(feat_vals) > 1e-10:
            fic = float(np.corrcoef(feat_vals, y_test)[0, 1])
        else:
            fic = 0.0
        feature_ics[fname] = round(fic, 6)

    sorted_by_abs_ic = sorted(feature_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    log.info("Per-feature IC — top 5: %s", sorted_by_abs_ic[:5])
    log.info("Per-feature IC — bottom 5: %s", sorted_by_abs_ic[-5:])

    # ── SHAP-based noise feature detection ─────────────────────────────────────
    noise_candidates = []
    if shap_importance:
        max_shap = max(shap_importance.values()) if shap_importance else 0.0
        shap_noise_thresh = max_shap * (cfg.SHAP_NOISE_THRESHOLD_PCT / 100.0)
        ic_noise_thresh = cfg.IC_NOISE_THRESHOLD
        for fname in cfg.GBM_FEATURES:
            shap_val = shap_importance.get(fname, 0.0)
            ic_val = abs(feature_ics.get(fname, 0.0))
            if shap_val < shap_noise_thresh and ic_val < ic_noise_thresh:
                noise_candidates.append(fname)
        if noise_candidates:
            log.info(
                "Noise feature candidates (%d): %s (SHAP < %.4f AND |IC| < %.4f)",
                len(noise_candidates), noise_candidates, shap_noise_thresh, ic_noise_thresh,
            )
        else:
            log.info("No noise feature candidates detected")

    # ── Promotion decision ────────────────────────────────────────────────────
    # If walk-forward enabled and passed: use WF gate for promotion.
    # If walk-forward enabled but failed: do NOT promote regardless of single-split IC.
    # If walk-forward disabled: fall back to single-split IC gate.
    passes_single_ic = float(test_ic) >= cfg.MIN_IC
    passes_ic_ir = ic_ir >= cfg.GBM_IC_IR_GATE

    if wf_result and not wf_result.get("fallback"):
        passes_ic = wf_result["passes_wf"]
    else:
        # Single-split fallback: also require chunk-level consistency
        pct_chunks_positive = float((chunk_ics > 0).mean())
        if passes_single_ic and pct_chunks_positive < 0.50:
            log.warning(
                "Single-split IC passes (%.4f) but chunk consistency fails "
                "(%.0f%% positive < 50%%) — NOT promoting",
                float(test_ic), pct_chunks_positive * 100,
            )
        passes_ic = passes_single_ic and (pct_chunks_positive >= 0.50)

    elapsed_s = (datetime.now(timezone.utc) - start_ts).total_seconds()
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

            # Always save dated backups
            dated_key = f"predictor/weights/gbm_{date_str}.txt"
            s3.upload_file(str(booster_path), bucket, dated_key)
            log.info("Uploaded dated backup (MSE): s3://%s/%s", bucket, dated_key)

            # Also save as gbm_mse_{date}.txt
            dated_mse_key = f"predictor/weights/gbm_mse_{date_str}.txt"
            s3.upload_file(str(booster_path), bucket, dated_mse_key)

            # Save lambdarank model if trained
            rank_booster_path = None
            if rank_scorer is not None:
                rank_booster_path = Path(tmp) / "gbm_rank_model.txt"
                rank_scorer.save(rank_booster_path)
                dated_rank_key = f"predictor/weights/gbm_rank_{date_str}.txt"
                s3.upload_file(str(rank_booster_path), bucket, dated_rank_key)
                log.info("Uploaded dated backup (lambdarank): s3://%s/%s", bucket, dated_rank_key)

            if passes_ic:
                # Always upload BOTH models — inference runs both side by side.
                # MSE → predicted_alpha (calibrated returns)
                # Lambdarank → model_rank (cross-sectional ranking)
                s3.upload_file(str(booster_path), bucket, cfg.GBM_MSE_WEIGHTS_KEY)
                log.info("Uploaded MSE model: s3://%s/%s", bucket, cfg.GBM_MSE_WEIGHTS_KEY)
                if rank_booster_path is not None:
                    s3.upload_file(str(rank_booster_path), bucket, cfg.GBM_RANK_WEIGHTS_KEY)
                    log.info("Uploaded lambdarank model: s3://%s/%s", bucket, cfg.GBM_RANK_WEIGHTS_KEY)

                # Promote best_mode to gbm_latest (for model_version tracking)
                if best_mode == "mse":
                    s3.upload_file(str(booster_path), bucket, cfg.GBM_WEIGHTS_KEY)
                elif best_mode == "rank" and rank_booster_path is not None:
                    s3.upload_file(str(rank_booster_path), bucket, cfg.GBM_WEIGHTS_KEY)
                elif best_mode == "ensemble":
                    s3.upload_file(str(booster_path), bucket, cfg.GBM_WEIGHTS_KEY)
                log.info("Promoted %s to active weights: s3://%s/%s", best_mode, bucket, cfg.GBM_WEIGHTS_KEY)

                # Write gbm_mode.json to S3
                mode_payload = json.dumps({"mode": best_mode}, indent=2).encode()
                s3.put_object(
                    Bucket=bucket,
                    Key=cfg.GBM_MODE_KEY,
                    Body=mode_payload,
                    ContentType="application/json",
                )
                log.info("Wrote gbm_mode.json: mode=%s", best_mode)

                # ── Append to mode_history.json on S3 ────────────────────
                try:
                    MODE_HISTORY_KEY = "predictor/metrics/mode_history.json"
                    try:
                        hist_obj = s3.get_object(Bucket=bucket, Key=MODE_HISTORY_KEY)
                        mode_history = json.loads(hist_obj["Body"].read())
                        if not isinstance(mode_history, list):
                            mode_history = []
                    except Exception:
                        mode_history = []

                    mode_history_entry = {
                        "date": date_str,
                        "mse_ic": round(float(mse_ic), 6),
                        "rank_ic": round(float(rank_ic), 6) if rank_scorer is not None else None,
                        "ensemble_ic": round(float(ensemble_ic), 6) if rank_scorer is not None else None,
                        "best_mode": best_mode,
                        "promoted": True,
                        "ic_delta_rank_vs_mse": round(float(rank_ic - mse_ic), 6) if rank_scorer is not None else None,
                        "ic_delta_ens_vs_mse": round(float(ensemble_ic - mse_ic), 6) if rank_scorer is not None else None,
                        "n_train": int(len(y_train)),
                        "ic_ir": round(ic_ir, 4),
                    }

                    # Deduplicate by date (replace if same date already exists)
                    mode_history = [e for e in mode_history if e.get("date") != date_str]
                    mode_history.append(mode_history_entry)
                    # Cap at 52 entries (1 year of weekly runs)
                    mode_history = mode_history[-52:]

                    s3.put_object(
                        Bucket=bucket,
                        Key=MODE_HISTORY_KEY,
                        Body=json.dumps(mode_history, indent=2).encode(),
                        ContentType="application/json",
                    )
                    log.info("Appended mode history entry for %s (mode=%s)", date_str, best_mode)
                except Exception as e:
                    log.warning("Failed to update mode_history.json: %s", e)

                meta = {
                    "model_version":  model_version,
                    "val_ic":         round(float(scorer._val_ic), 6),
                    "test_ic":        round(float(test_ic), 6),
                    "mse_ic":         round(float(mse_ic), 6),
                    "ic_ir":          round(ic_ir, 4),
                    "trained_date":   date_str,
                    "best_iteration": scorer._best_iteration,
                    "n_train":        int(len(y_train)),
                    "rank_normalized": True,
                    "feature_list":   cfg.GBM_FEATURES,
                    "n_features":     len(cfg.GBM_FEATURES),
                    "gain_importance": dict(importance),
                    "shap_importance": shap_importance,
                    "feature_ics": feature_ics,
                    "noise_candidates": noise_candidates,
                    "ensemble_enabled": rank_scorer is not None,
                    "promoted_mode":  best_mode,
                }
                if rank_scorer is not None:
                    meta["rank_ic"] = round(float(rank_ic), 6)
                    meta["ensemble_ic"] = round(float(ensemble_ic), 6)
                if wf_result and not wf_result.get("fallback"):
                    meta["walk_forward"] = {
                        "median_ic": wf_result["median_ic"],
                        "pct_positive": wf_result["pct_positive"],
                        "n_folds": len(wf_result["folds"]),
                    }
                s3.put_object(
                    Bucket=bucket,
                    Key=cfg.GBM_WEIGHTS_META_KEY,
                    Body=json.dumps(meta, indent=2).encode(),
                    ContentType="application/json",
                )

                promoted = True
            else:
                reason = "walk-forward" if (wf_result and not wf_result.get("fallback")) else "single-split IC"
                log.warning(
                    "IC gate FAILED (%s) — backup saved, NOT promoted",
                    reason,
                )

    result = {
        "model_version":         model_version,
        "best_iteration":        scorer._best_iteration,
        "val_ic":                round(float(scorer._val_ic), 6),
        "test_ic":               round(float(test_ic), 6),
        "mse_ic":                round(float(mse_ic), 6),
        "rank_ic":               round(float(rank_ic), 6) if rank_scorer is not None else None,
        "ensemble_ic":           round(float(ensemble_ic), 6) if rank_scorer is not None else None,
        "ensemble_enabled":      rank_scorer is not None,
        "promoted_mode":         best_mode if promoted else None,
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
        "feature_importance_shap_top10": [
            {"feature": f, "shap": round(s, 4)} for f, s in
            sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
        ] if shap_importance else [],
        "feature_ics": feature_ics,
        "noise_candidates": noise_candidates,
    }

    # Attach walk-forward detail for email reporting
    if wf_result and not wf_result.get("fallback"):
        result["walk_forward"] = wf_result

    # ── SHAP history: store dated SHAP and compute drift ──────────────────
    if shap_importance and not dry_run:
        try:
            import boto3
            s3_shap = boto3.client("s3")

            # Write current SHAP to dated S3 key
            shap_key = f"predictor/metrics/shap_{date_str}.json"
            s3_shap.put_object(
                Bucket=bucket,
                Key=shap_key,
                Body=json.dumps(shap_importance, indent=2).encode(),
                ContentType="application/json",
            )
            log.info("SHAP importance saved to s3://%s/%s", bucket, shap_key)

            # Load previous week's SHAP for drift detection
            from datetime import timedelta
            prev_date = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=7)
            prev_key = f"predictor/metrics/shap_{prev_date.strftime('%Y-%m-%d')}.json"
            try:
                prev_obj = s3_shap.get_object(Bucket=bucket, Key=prev_key)
                prev_shap = json.loads(prev_obj["Body"].read().decode())

                # Compute Spearman rank correlation between current and previous SHAP
                common_features = [f for f in cfg.GBM_FEATURES
                                   if f in shap_importance and f in prev_shap]
                if len(common_features) >= 5:
                    from scipy.stats import spearmanr
                    curr_vals = [shap_importance[f] for f in common_features]
                    prev_vals = [prev_shap[f] for f in common_features]
                    rho, _ = spearmanr(curr_vals, prev_vals)
                    rho = round(float(rho), 4)
                    result["shap_rank_stability"] = rho
                    log.info("SHAP rank stability (Spearman): %.4f", rho)
                    if rho < 0.80:
                        log.warning(
                            "SHAP feature drift detected: rank correlation %.4f < 0.80",
                            rho,
                        )
                else:
                    log.info("Not enough common features for SHAP drift (%d)", len(common_features))
            except s3_shap.exceptions.NoSuchKey:
                log.info("No previous SHAP found at %s — skipping drift check", prev_key)
            except Exception as e:
                log.warning("SHAP drift check failed (non-blocking): %s", e)
        except Exception as e:
            log.warning("SHAP history storage failed (non-blocking): %s", e)

    return result


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
    promoted_mode = result.get("promoted_mode")
    passes_ic    = result.get("passes_ic_gate", False)
    val_ic       = result.get("val_ic", 0.0)
    test_ic      = result.get("test_ic", 0.0)
    mse_ic       = result.get("mse_ic", test_ic)
    rank_ic      = result.get("rank_ic")
    ensemble_ic  = result.get("ensemble_ic")
    ensemble_on  = result.get("ensemble_enabled", False)
    ic_ir        = result.get("ic_ir", 0.0)
    version      = result.get("model_version", "unknown")
    elapsed_s    = result.get("elapsed_s", 0)
    n_train      = result.get("n_train", 0)
    ic_pos       = result.get("ic_positive_20", 0)
    top10        = result.get("feature_importance_top10", [])

    ic_color    = "#2e7d32" if passes_ic else "#c62828"
    ic_label    = "PASS ✓" if passes_ic else "FAIL ✗"
    promo_color = "#2e7d32" if promoted else "#c62828"
    promo_label = (
        f"Promoted → gbm_latest ({promoted_mode}) ✓"
        if promoted else "NOT promoted (IC gate failed ✗)"
    )
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

    # Walk-forward section for email
    wf_data = result.get("walk_forward")
    wf_html = ""
    wf_plain = ""
    if wf_data and wf_data.get("folds"):
        wf_median = wf_data["median_ic"]
        wf_pct = wf_data["pct_positive"]
        wf_pass = wf_data.get("passes_wf", False)
        wf_color = "#2e7d32" if wf_pass else "#c62828"
        wf_label = "PASS ✓" if wf_pass else "FAIL ✗"
        fold_rows = "".join(
            f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
            f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["fold"]}</td>'
            f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_start"]}</td>'
            f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_end"]}</td>'
            f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["n_train"]:,}</td>'
            f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
            f'color:{"#2e7d32" if f["ic"] > 0 else "#c62828"};">{f["ic"]:.4f}</td>'
            f'</tr>'
            for i, f in enumerate(wf_data["folds"])
        )
        wf_html = (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Walk-Forward Validation '
            f'({len(wf_data["folds"])} folds)</h3>'
            f'<p style="font-size:12px; margin:2px 0;">Median IC: <b style="color:{wf_color};">'
            f'{wf_median:.4f}</b> — {wf_label} &nbsp;|&nbsp; '
            f'Positive folds: <b>{wf_pct*100:.0f}%</b></p>'
            f'<table style="border-collapse:collapse; width:100%; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 6px;">Fold</th>'
            f'<th style="padding:3px 6px;">Test Start</th>'
            f'<th style="padding:3px 6px;">Test End</th>'
            f'<th style="padding:3px 6px;">Train N</th>'
            f'<th style="padding:3px 6px;">IC</th></tr>'
            f'{fold_rows}</table>'
        )
        wf_plain = (
            f"\n--- Walk-Forward ({len(wf_data['folds'])} folds) ---"
            f"\nMedian IC: {wf_median:.4f} — {wf_label}"
            f"\nPositive folds: {wf_pct*100:.0f}%\n"
            + "\n".join(
                f"  Fold {f['fold']}: [{f['test_start']} → {f['test_end']}] "
                f"train={f['n_train']:,}  IC={f['ic']:.4f}"
                for f in wf_data["folds"]
            )
            + "\n"
        )

    # ── SHAP vs Gain comparison section ──────────────────────────────────────
    shap_top10    = result.get("feature_importance_shap_top10", [])
    shap_stability = result.get("shap_rank_stability")

    shap_html = ""
    shap_plain = ""
    if shap_top10 and top10:
        # Build rank lookup: feature → rank (1-based)
        gain_rank = {r["feature"]: i + 1 for i, r in enumerate(top10)}
        shap_rank = {r["feature"]: i + 1 for i, r in enumerate(shap_top10)}
        all_features = list(dict.fromkeys(
            [r["feature"] for r in top10] + [r["feature"] for r in shap_top10]
        ))

        comparison_rows = ""
        comparison_plain_lines = []
        for feat in all_features[:10]:
            g_rank = gain_rank.get(feat, "-")
            s_rank = shap_rank.get(feat, "-")
            divergence = ""
            if isinstance(g_rank, int) and isinstance(s_rank, int):
                diff = abs(g_rank - s_rank)
                if diff > 3:
                    divergence = f' style="color:#c62828; font-weight:bold;"'
            comparison_rows += (
                f'<tr>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{feat}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px; text-align:center;">{g_rank}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px; text-align:center;"'
                f'{divergence}>{s_rank}</td>'
                f'</tr>'
            )
            flag = " ***" if isinstance(g_rank, int) and isinstance(s_rank, int) and abs(g_rank - s_rank) > 3 else ""
            comparison_plain_lines.append(f"  {feat:<22} Gain:{g_rank}  SHAP:{s_rank}{flag}")

        stability_note = ""
        stability_plain = ""
        if shap_stability is not None:
            stab_color = "#2e7d32" if shap_stability >= 0.80 else "#c62828"
            stab_label = "stable" if shap_stability >= 0.80 else "DRIFT WARNING"
            stability_note = (
                f'<p style="font-size:12px; margin:4px 0;">SHAP rank stability (vs last week): '
                f'<b style="color:{stab_color};">rho={shap_stability:.4f} — {stab_label}</b></p>'
            )
            stability_plain = f"\nSHAP rank stability: rho={shap_stability:.4f} — {stab_label}"

        shap_html = (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Feature Importance: Gain vs SHAP</h3>'
            f'{stability_note}'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Feature</th>'
            f'<th style="padding:3px 8px;">Gain Rank</th>'
            f'<th style="padding:3px 8px;">SHAP Rank</th></tr>'
            f'{comparison_rows}</table>'
            f'<p style="font-size:10px; color:#888;">Features with rank divergence &gt;3 highlighted in red.</p>'
        )
        shap_plain = (
            "\n--- Gain vs SHAP Rank ---"
            + stability_plain
            + "\n" + "\n".join(comparison_plain_lines)
            + "\n  (*** = rank divergence > 3)\n"
        )

    # ── Feature Health section (per-feature IC + noise detection) ──────────
    feat_ics = result.get("feature_ics", {})
    noise_cands = result.get("noise_candidates", [])
    feat_health_html = ""
    feat_health_plain = ""
    if feat_ics:
        sorted_ics = sorted(feat_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        ic_rows = "".join(
            f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
            f'<td style="padding:2px 8px; font-family:monospace; font-size:11px;">{fname}</td>'
            f'<td style="padding:2px 8px; font-family:monospace; font-size:11px; '
            f'color:{"#2e7d32" if fic > 0 else "#c62828"};">{fic:.4f}</td>'
            f'</tr>'
            for i, (fname, fic) in enumerate(sorted_ics[:10])
        )
        noise_note = ""
        if noise_cands:
            noise_note = (
                f'<p style="font-size:11px; color:#c62828; margin:4px 0;">'
                f'Noise candidates ({len(noise_cands)}): {", ".join(noise_cands)}</p>'
            )
        feat_health_html = (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Feature Health</h3>'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Feature</th>'
            f'<th style="padding:3px 8px;">IC vs Forward</th></tr>'
            f'{ic_rows}</table>'
            f'{noise_note}'
        )
        feat_health_plain = (
            "\n--- Feature Health (top 10 by |IC|) ---\n"
            + "\n".join(f"  {fname:<22} IC={fic:.4f}" for fname, fic in sorted_ics[:10])
            + (f"\nNoise candidates: {', '.join(noise_cands)}" if noise_cands else "")
            + "\n"
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
        f'  <td style="padding:5px 10px; color:#555;">MSE Model IC</td>'
        f'  <td style="padding:5px 10px; font-family:monospace; font-weight:bold;'
        f'{f" color:{ic_color};" if promoted_mode == "mse" else ""}">'
        f'{mse_ic:.4f}{f" — {ic_label}" if promoted_mode == "mse" else ""}'
        f'{"  ✓" if promoted_mode == "mse" else ""}</td>'
        f'</tr>'
        + (
            f'<tr style="background:#f9f9f9;">'
            f'  <td style="padding:5px 10px; color:#555;">Lambdarank Model IC</td>'
            f'  <td style="padding:5px 10px; font-family:monospace; font-weight:bold;'
            f'{f" color:{ic_color};" if promoted_mode == "rank" else ""}">'
            f'{rank_ic:.4f}{f" — {ic_label}" if promoted_mode == "rank" else ""}'
            f'{"  ✓" if promoted_mode == "rank" else ""}</td>'
            f'</tr>'
            f'<tr>'
            f'  <td style="padding:5px 10px; color:#555;">Ensemble IC</td>'
            f'  <td style="padding:5px 10px; font-family:monospace; font-weight:bold;'
            f'{f" color:{ic_color};" if promoted_mode == "ensemble" else ""}">'
            f'{ensemble_ic:.4f}{f" — {ic_label}" if promoted_mode == "ensemble" else ""}'
            f'{"  ✓" if promoted_mode == "ensemble" else ""}</td>'
            f'</tr>'
            if ensemble_on and rank_ic is not None and ensemble_ic is not None else
            f'<tr>'
            f'  <td style="padding:5px 10px; color:#555;">Test IC</td>'
            f'  <td style="padding:5px 10px; font-family:monospace; font-weight:bold; color:{ic_color};">'
            f'  {test_ic:.4f} — {ic_label}</td>'
            f'</tr>'
        ) +
        f'<tr style="background:#f9f9f9;">'
        f'  <td style="padding:5px 10px; color:#555;">IC IR</td>'
        f'  <td style="padding:5px 10px; font-family:monospace; color:#555;">'
        f'  {ic_ir:.3f} ({ic_pos}/20 positive)</td>'
        f'</tr>'
        f'<tr style="background:#f9f9f9;">'
        f'  <td style="padding:5px 10px; color:#555; font-weight:bold;">Promotion</td>'
        f'  <td style="padding:5px 10px; font-weight:bold; color:{promo_color};">{promo_label}</td>'
        f'</tr>'
        f'</table>'

        f'{wf_html}'

        f'<h3 style="margin-bottom:4px;">Top 5 Features by Gain</h3>'
        f'<table>{feat_rows}</table>'

        f'{shap_html}'

        f'{feat_health_html}'

        f'<p style="font-size:11px; color:#aaa; margin-top:20px;">'
        f'IC gate: ≥{cfg.MIN_IC:.2f} to promote &nbsp;|&nbsp; Walk-forward: median IC ≥{cfg.WF_MEDIAN_IC_GATE:.2f}, {cfg.WF_MIN_FOLDS_POSITIVE*100:.0f}%+ positive folds</p>'
        f'</body></html>'
    )

    _mse_mark  = " ✓" if promoted_mode == "mse" else ""
    _rank_mark = " ✓" if promoted_mode == "rank" else ""
    _ens_mark  = " ✓" if promoted_mode == "ensemble" else ""
    plain_body = (
        f"GBM Weekly Retrain — {date_str}\n"
        f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
        f"\nVal IC:             {val_ic:.4f}"
        f"\nMSE Model IC:       {mse_ic:.4f}{' — ' + ic_label if promoted_mode == 'mse' else ''}{_mse_mark}"
        + (
            f"\nLambdarank Model IC: {rank_ic:.4f}{' — ' + ic_label if promoted_mode == 'rank' else ''}{_rank_mark}"
            f"\nEnsemble IC:        {ensemble_ic:.4f}{' — ' + ic_label if promoted_mode == 'ensemble' else ''}{_ens_mark}"
            if ensemble_on and rank_ic is not None and ensemble_ic is not None else
            f"\nTest IC:            {test_ic:.4f} — {ic_label}"
        ) +
        f"\nPromoted:           {promoted_mode if promoted else 'none'}"
        f"\nIC IR:              {ic_ir:.3f} ({ic_pos}/20 positive)"
        f"\nPromotion:          {promo_label}\n"
        f"{wf_plain}"
        f"\nTop features: " + ", ".join(r["feature"] for r in top10[:5])
        + f"\n{shap_plain}"
        + f"{feat_health_plain}\n"
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

    fd = None

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
        fd=fd,
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
            fd=fd,
        )
        log.info("Slim cache written: %d tickers", n_slim)
        result["slim_cache_tickers"] = n_slim
    else:
        log.info("[dry-run] Skipping slim cache write")

    # Step 3: Email
    if not dry_run:
        send_training_email(result, date_str)

    # Step 4: Health status
    if not dry_run:
        try:
            from health_status import write_health
            write_health(
                bucket=bucket,
                module_name="predictor_training",
                status="ok",
                run_date=date_str,
                duration_seconds=result.get("train_time_seconds", 0),
                summary={
                    "promoted": result.get("promoted", False),
                    "ic_30d": result.get("ic_30d"),
                    "n_train": result.get("n_train"),
                    "slim_cache_tickers": result.get("slim_cache_tickers"),
                },
            )
        except Exception as _he:
            log.warning("Health status write failed: %s", _he)

    return result
