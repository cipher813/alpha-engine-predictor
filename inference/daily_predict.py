"""
inference/daily_predict.py — Daily prediction run.

Called by the Lambda handler (inference/handler.py) and optionally from the
CLI for local testing. Orchestrates:
  1. Load model weights from S3 (or local path with --local flag).
  2. Determine the active ticker universe.
  3. Fetch today's 1-year OHLCV data via yfinance.
  4. Compute features and run inference for each ticker.
  5. Write predictions JSON to S3 at both dated path and latest.json.
  6. Write metrics/latest.json.

Usage:
    # Full universe (~900 tickers, uses S3 signals.json):
    python inference/daily_predict.py [--date DATE] [--dry-run] [--local]

    # Focused watchlist mode — predict only the research module's tracked +
    # buy-candidate tickers (typically ~10–30 names):
    python inference/daily_predict.py --watchlist auto \
        --model-type gbm [--dry-run]               # pulls signals.json from S3

    python inference/daily_predict.py \
        --watchlist /path/to/signals.json \        # local signals.json for offline use
        --local --model-type gbm --dry-run

Flags:
    --date DATE          Override prediction date (YYYY-MM-DD). Default: today.
    --dry-run            Run inference but skip S3 writes. Print output to stdout.
    --local              Load model weights from local checkpoints/ instead of S3.
    --model-type mlp|gbm Which model to use. Default: mlp.
    --watchlist auto|PATH
                         Restrict predictions to research-module tickers only.
                         "auto"  → reads today's signals/{date}/signals.json from S3.
                         PATH    → reads a local signals.json from alpha-engine-research.
                         Omit    → full ~900-ticker universe (default behaviour).
                         Each prediction result includes watchlist_source:
                         "tracked" | "buy_candidate" | "both".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure the project root (parent of inference/) is on sys.path so that
# root-level modules (config, model/, data/) are importable when this script
# is invoked directly as `python inference/daily_predict.py` from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as cfg

log = logging.getLogger(__name__)

# Fallback universe if signals.json is unavailable
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B",
    "LLY", "JPM", "V", "UNH", "XOM", "COST", "TSLA", "HD",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    s3_bucket: str,
    weights_key: str,
    device: str = "cpu",
) -> tuple:
    """
    Download model weights from S3 to /tmp and load the checkpoint.

    Parameters
    ----------
    s3_bucket :   S3 bucket name.
    weights_key : S3 key for the weights file (e.g. predictor/weights/latest.pt).
    device :      Torch device string.

    Returns
    -------
    (model, checkpoint_dict)
    """
    from model.predictor import load_checkpoint

    try:
        import boto3
        s3 = boto3.client("s3")
        local_path = Path(tempfile.mkdtemp()) / "model_weights.pt"
        log.info("Downloading model weights from s3://%s/%s", s3_bucket, weights_key)
        s3.download_file(s3_bucket, weights_key, str(local_path))
        log.info("Downloaded to %s", local_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model weights from s3://{s3_bucket}/{weights_key}: {exc}"
        ) from exc

    model, checkpoint = load_checkpoint(str(local_path), device=device)
    log.info(
        "Model loaded: version=%s  epoch=%d  val_loss=%.4f",
        checkpoint.get("model_version", "unknown"),
        checkpoint.get("epoch", -1),
        checkpoint.get("val_loss", float("nan")),
    )
    return model, checkpoint


def load_model_local(
    path: str = "checkpoints/best.pt",
    device: str = "cpu",
) -> tuple:
    """Load model weights from a local file path."""
    from model.predictor import load_checkpoint
    model, checkpoint = load_checkpoint(path, device=device)
    log.info(
        "Model loaded (local): version=%s  epoch=%d",
        checkpoint.get("model_version", "unknown"),
        checkpoint.get("epoch", -1),
    )
    return model, checkpoint


def load_gbm_local(path: str = "checkpoints/gbm_best.txt"):
    """Load GBMScorer from a local file path."""
    from model.gbm_scorer import GBMScorer
    scorer = GBMScorer.load(path)
    log.info(
        "GBMScorer loaded (local): val_IC=%.4f  best_iter=%d",
        scorer._val_ic, scorer._best_iteration,
    )
    return scorer


def load_gbm_s3(s3_bucket: str, weights_key: str):
    """
    Download GBM booster + meta from S3 to /tmp and load.

    Parameters
    ----------
    s3_bucket :   S3 bucket name.
    weights_key : S3 key for the booster text file (e.g. predictor/weights/gbm_latest.txt).

    Returns
    -------
    GBMScorer instance with booster loaded.
    """
    from model.gbm_scorer import GBMScorer
    try:
        import boto3
        s3 = boto3.client("s3")
        tmp_dir = Path(tempfile.mkdtemp())
        local_path = tmp_dir / "gbm_model.txt"
        meta_path  = tmp_dir / "gbm_model.txt.meta.json"
        log.info("Downloading GBM booster from s3://%s/%s", s3_bucket, weights_key)
        s3.download_file(s3_bucket, weights_key, str(local_path))
        # Meta file is best-effort — GBMScorer.load() handles its absence gracefully
        try:
            s3.download_file(s3_bucket, weights_key + ".meta.json", str(meta_path))
        except Exception:
            pass
        scorer = GBMScorer.load(str(local_path))
        log.info(
            "GBMScorer loaded from S3: val_IC=%.4f  best_iter=%d",
            scorer._val_ic, scorer._best_iteration,
        )
        return scorer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download GBM booster from s3://{s3_bucket}/{weights_key}: {exc}"
        ) from exc


# ── Universe ──────────────────────────────────────────────────────────────────

def get_universe_tickers(s3_bucket: str, date_str: Optional[str] = None) -> list[str]:
    """
    Read the active ticker universe from today's signals.json in S3.
    Falls back to _FALLBACK_TICKERS if signals.json is not available.

    Parameters
    ----------
    s3_bucket : S3 bucket name.
    date_str :  Date string YYYY-MM-DD. Uses today if None.

    Returns
    -------
    list of ticker symbols.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    signals_key = f"signals/signals_{date_str.replace('-', '')}.json"

    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key=signals_key)
        signals_data = json.loads(obj["Body"].read().decode("utf-8"))

        # signals.json has a top-level "signals" list with per-ticker records
        signals_list = signals_data.get("signals", [])
        tickers = [s["ticker"] for s in signals_list if "ticker" in s]

        if tickers:
            log.info("Universe: %d tickers from %s", len(tickers), signals_key)
            return tickers
        else:
            log.warning("signals.json found but no tickers extracted — using fallback")
    except Exception as exc:
        log.info("Could not read signals.json (%s) — using fallback universe", exc)

    log.info("Using fallback universe: %d tickers", len(_FALLBACK_TICKERS))
    return _FALLBACK_TICKERS


# ── Watchlist ─────────────────────────────────────────────────────────────────

def load_watchlist(
    path: str,
    s3_bucket: Optional[str] = None,
    date_str: Optional[str] = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Build a focused prediction universe from the research module's signals.json.

    signals.json (written by alpha-engine-research) contains two relevant keys:

        "universe"       — tickers the research pipeline actively monitors
                           → watchlist_source = "tracked"
        "buy_candidates" — scanner-identified entry candidates for the run
                           → watchlist_source = "buy_candidate"

    Parameters
    ----------
    path      : "auto" → fetch today's signals.json from S3
                         at signals/{date}/signals.json.
                Any other string → local file path to a signals.json produced
                by alpha-engine-research (for offline / dry-run use).
    s3_bucket : S3 bucket name. Required when path="auto".
    date_str  : YYYY-MM-DD override. Defaults to today. Used for the S3 key
                when path="auto".

    Returns
    -------
    tickers : Deduplicated, sorted list of tickers from universe + buy_candidates.
    sources : Dict mapping ticker → "tracked" | "buy_candidate" | "both".
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # ── Load raw signals payload ──────────────────────────────────────────────
    if path == "auto":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when --watchlist auto is used")
        signals_key = f"signals/{date_str}/signals.json"
        try:
            import boto3
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=s3_bucket, Key=signals_key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            log.info(
                "Watchlist: loaded signals from s3://%s/%s",
                s3_bucket, signals_key,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load signals.json from s3://{s3_bucket}/{signals_key}: {exc}\n"
                "Ensure the research pipeline has run for today before the predictor, "
                "or provide a local path: --watchlist /path/to/signals.json"
            ) from exc
    else:
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(
                f"signals.json not found: {path}\n"
                "Use --watchlist auto to pull from S3, or provide the path to a "
                "local signals.json from alpha-engine-research."
            )
        data = json.loads(local_path.read_text())
        log.info("Watchlist: loaded signals from %s", path)

    # ── Extract and annotate tickers ─────────────────────────────────────────
    universe_tickers = {
        e["ticker"].upper() for e in data.get("universe", []) if "ticker" in e
    }
    buy_cand_tickers = {
        e["ticker"].upper() for e in data.get("buy_candidates", []) if "ticker" in e
    }

    sources: dict[str, str] = {}
    for t in universe_tickers:
        sources[t] = "both" if t in buy_cand_tickers else "tracked"
    for t in buy_cand_tickers:
        if t not in sources:
            sources[t] = "buy_candidate"

    tickers = sorted(sources.keys())
    n_overlap = len(universe_tickers & buy_cand_tickers)
    log.info(
        "Watchlist: %d universe (tracked) + %d buy_candidates "
        "= %d unique tickers (%d overlap)",
        len(universe_tickers), len(buy_cand_tickers), len(tickers), n_overlap,
    )
    return tickers, sources


# ── Price fetch ───────────────────────────────────────────────────────────────

def fetch_today_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch 1-year OHLCV history for each ticker via yfinance.
    Returns a dict of ticker → DataFrame. Empty DataFrames on failure.
    One year provides sufficient history for MA200 (200 rows) with some buffer.
    """
    import yfinance as yf

    # 2y lookback: compute_features needs 252 rows of warmup (52w rolling windows).
    # With only 1y of data (~252 rows) the dropna step in compute_features leaves
    # 0–1 rows, causing empty featured_df and skipped predictions.
    log.info("Fetching 2y price data for %d tickers...", len(tickers))
    result: dict[str, pd.DataFrame] = {}

    # Batch download for efficiency
    batch_size = 100
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in batches:
        try:
            if len(batch) == 1:
                raw = yf.download(
                    batch[0],
                    period="2y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                raw.index = pd.to_datetime(raw.index)
                raw = raw.dropna(subset=["Close"])
                result[batch[0]] = raw
            else:
                raw = yf.download(
                    tickers=batch,
                    period="2y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                for ticker in batch:
                    try:
                        df = raw[ticker].copy()
                        df.index = pd.to_datetime(df.index)
                        df = df.dropna(subset=["Close"])
                        result[ticker] = df
                    except (KeyError, AttributeError):
                        result[ticker] = pd.DataFrame()
        except Exception as exc:
            log.warning("Batch price fetch failed: %s", exc)
            for ticker in batch:
                result[ticker] = pd.DataFrame()

    n_success = sum(1 for df in result.values() if not df.empty)
    log.info("Price fetch complete: %d / %d succeeded", n_success, len(tickers))
    return result


# ── Macro series fetch ────────────────────────────────────────────────────────

def fetch_macro_series(
    extra_tickers: list[str] | None = None,
    period: str = "2y",
) -> dict[str, pd.Series]:
    """
    Fetch macro indicator Close-price series needed for feature computation.

    Always fetches:
        SPY   — used for return_vs_spy_5d and as SPY leg of sector_vs_spy_5d
        VIX   — (^VIX) vix_level feature
        TNX   — (^TNX) 10Y Treasury yield → yield_10y + yield_curve_slope
        IRX   — (^IRX) 3M T-bill yield → yield_curve_slope
        GLD   — gold_mom_5d
        USO   — oil_mom_5d

    Parameters
    ----------
    extra_tickers : Additional ticker symbols to fetch (e.g. sector ETFs like
                    XLK, XLF). Returned in the dict keyed by their uppercase
                    symbol.
    period :        yfinance period string. Matches the price-fetch period so
                    all series cover the same date range.

    Returns
    -------
    dict mapping key → pd.Series of Close prices with a DatetimeIndex.
    Keys: SPY, VIX, TNX, IRX, GLD, USO  plus any extra_tickers symbols.
    Missing series are omitted (the caller's compute_features falls back to
    neutral constants when a series is None).
    """
    import yfinance as yf

    # Core macro symbols: internal key → yfinance ticker
    _MACRO_MAP = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "TNX": "^TNX",
        "IRX": "^IRX",
        "GLD": "GLD",
        "USO": "USO",
    }

    all_yf_tickers = list(_MACRO_MAP.values())
    extra = [t.upper() for t in (extra_tickers or [])]
    all_yf_tickers += extra

    log.info("Fetching macro/sector series (%d symbols, period=%s)…", len(all_yf_tickers), period)

    result: dict[str, pd.Series] = {}
    try:
        if len(all_yf_tickers) == 1:
            raw = yf.download(
                all_yf_tickers[0],
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            series = raw["Close"].dropna() if "Close" in raw.columns else pd.Series(dtype=float)
            series.index = pd.to_datetime(series.index)
            if not series.empty:
                result[all_yf_tickers[0]] = series
        else:
            raw = yf.download(
                tickers=all_yf_tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            for yf_ticker in all_yf_tickers:
                try:
                    s = raw[yf_ticker]["Close"].dropna()
                    s.index = pd.to_datetime(s.index)
                    if not s.empty:
                        result[yf_ticker] = s
                except (KeyError, AttributeError):
                    log.debug("Macro series unavailable: %s", yf_ticker)
    except Exception as exc:
        log.warning("Macro series fetch error: %s — features will use neutral defaults", exc)
        return result

    # Re-key from yfinance ticker to internal key
    keyed: dict[str, pd.Series] = {}
    for key, yf_ticker in _MACRO_MAP.items():
        if yf_ticker in result:
            keyed[key] = result[yf_ticker]
        else:
            log.warning("Macro series missing: %s (%s) — feature will use neutral default", key, yf_ticker)
    # Sector ETFs and other extras are keyed by their uppercase symbol
    for yf_ticker in extra:
        if yf_ticker in result:
            keyed[yf_ticker] = result[yf_ticker]

    log.info("Macro series loaded: %s", sorted(keyed.keys()))
    return keyed


# ── Per-ticker prediction ─────────────────────────────────────────────────────

def predict_ticker(
    ticker: str,
    df: pd.DataFrame,
    model,  # torch.nn.Module — imported lazily below (not needed for GBM path)
    norm_stats: dict,
    macro: dict[str, pd.Series] | None = None,
    sector_etf_series: pd.Series | None = None,
) -> Optional[dict]:
    """
    Compute features for one ticker and run inference.

    Parameters
    ----------
    ticker :           Ticker symbol.
    df :               2-year OHLCV DataFrame (2y needed for 52w rolling windows).
    model :            Loaded DirectionPredictor in eval mode.
    norm_stats :       Dict with 'mean' and 'std' lists for z-score normalization.
    macro :            Dict of macro Close-price Series from fetch_macro_series().
                       Keys: SPY, VIX, TNX, IRX, GLD, USO.  None → neutral defaults.
    sector_etf_series: Sector ETF Close prices for this ticker's sector.  None → 0.0.

    Returns
    -------
    Prediction dict or None if insufficient data.

    Output dict schema:
        {
            "ticker": "AAPL",
            "predicted_direction": "UP",
            "prediction_confidence": 0.74,
            "p_up": 0.74,
            "p_flat": 0.18,
            "p_down": 0.08
        }
    """
    from data.feature_engineer import compute_features

    if df.empty or len(df) < 265:
        # Need 252 rows for 52w rolling windows + buffer; 265 is a safe minimum
        log.debug("Skipping %s: insufficient data (%d rows)", ticker, len(df))
        return None

    try:
        featured_df = compute_features(
            df,
            spy_series=macro.get("SPY") if macro else None,
            vix_series=macro.get("VIX") if macro else None,
            sector_etf_series=sector_etf_series,
            tnx_series=macro.get("TNX") if macro else None,
            irx_series=macro.get("IRX") if macro else None,
            gld_series=macro.get("GLD") if macro else None,
            uso_series=macro.get("USO") if macro else None,
        )
    except Exception as exc:
        log.warning("Feature computation failed for %s: %s", ticker, exc)
        return None

    if featured_df.empty:
        log.debug("No rows after feature computation for %s", ticker)
        return None

    # Use the most recent row (today's feature vector)
    latest = featured_df.iloc[-1]

    feature_cols = cfg.FEATURES
    x_raw = latest[feature_cols].to_numpy(dtype=np.float32)

    # Z-score normalize using stored training statistics
    try:
        mean = np.array(norm_stats["mean"], dtype=np.float32)
        std = np.array(norm_stats["std"], dtype=np.float32)
        std = np.where(std == 0, 1.0, std)
        x_norm = (x_raw - mean) / std
    except Exception as exc:
        log.warning("Normalization failed for %s: %s", ticker, exc)
        return None

    # Inference — lazy-import torch so the module loads without PyTorch when
    # model_type="gbm" (LightGBM only) is used in the Lambda environment.
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    x_tensor = torch.FloatTensor(x_norm).unsqueeze(0)  # shape (1, 8)
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    # probs indices: 0=DOWN, 1=FLAT, 2=UP (matches CLASS_LABELS in config)
    p_down = float(probs[0])
    p_flat = float(probs[1])
    p_up = float(probs[2])

    # Predicted class = argmax
    class_idx = int(np.argmax(probs))
    predicted_direction = cfg.CLASS_LABELS[class_idx]
    confidence = float(probs[class_idx])

    return {
        "ticker": ticker,
        "predicted_direction": predicted_direction,
        "prediction_confidence": round(confidence, 4),
        "p_up": round(p_up, 4),
        "p_flat": round(p_flat, 4),
        "p_down": round(p_down, 4),
    }


def predict_ticker_gbm(
    ticker: str,
    df: pd.DataFrame,
    scorer,
    macro: dict[str, pd.Series] | None = None,
    sector_etf_series: pd.Series | None = None,
) -> Optional[dict]:
    """
    Compute features for one ticker and run GBMScorer inference.
    GBM is scale-invariant — no z-score normalization needed.

    Parameters
    ----------
    ticker :           Ticker symbol.
    df :               2-year OHLCV DataFrame (2y needed for 52w rolling windows).
    scorer :           Loaded GBMScorer instance.
    macro :            Dict of macro Close-price Series from fetch_macro_series().
                       Keys: SPY, VIX, TNX, IRX, GLD, USO.  None → neutral defaults.
    sector_etf_series: Sector ETF Close prices for this ticker's sector.  None → 0.0.

    Returns
    -------
    Prediction dict or None if insufficient data.

    Output dict schema matches predict_ticker() with one extra field:
        "predicted_alpha": float  — raw continuous alpha score (predicted 5d return vs benchmark)
    """
    from data.feature_engineer import compute_features

    if df.empty or len(df) < 265:
        # Need 252 rows for 52w rolling windows + buffer; 265 is a safe minimum
        log.debug("Skipping %s: insufficient data (%d rows)", ticker, len(df))
        return None

    try:
        featured_df = compute_features(
            df,
            spy_series=macro.get("SPY") if macro else None,
            vix_series=macro.get("VIX") if macro else None,
            sector_etf_series=sector_etf_series,
            tnx_series=macro.get("TNX") if macro else None,
            irx_series=macro.get("IRX") if macro else None,
            gld_series=macro.get("GLD") if macro else None,
            uso_series=macro.get("USO") if macro else None,
        )
    except Exception as exc:
        log.warning("Feature computation failed for %s: %s", ticker, exc)
        return None

    if featured_df.empty:
        log.debug("No rows after feature computation for %s", ticker)
        return None

    latest = featured_df.iloc[-1]
    x_raw = latest[cfg.FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)

    try:
        s = float(scorer.predict(x_raw)[0])
    except Exception as exc:
        log.warning("GBM inference failed for %s: %s", ticker, exc)
        return None

    # Convert continuous alpha scalar → pseudo-probabilities for output compatibility.
    # Linear map over [-LABEL_CLIP, +LABEL_CLIP] → p_up ∈ [0, 1]; clamp outside range.
    max_r = getattr(cfg, "LABEL_CLIP", 0.15)
    p_up   = float(np.clip(0.5 + s / (2.0 * max_r), 0.0, 1.0))
    p_down = float(np.clip(0.5 - s / (2.0 * max_r), 0.0, 1.0))
    p_flat = float(max(0.0, 1.0 - p_up - p_down))

    if s > cfg.UP_THRESHOLD:
        predicted_direction = "UP"
        confidence = p_up
    elif s < cfg.DOWN_THRESHOLD:
        predicted_direction = "DOWN"
        confidence = p_down
    else:
        predicted_direction = "FLAT"
        confidence = 1.0 - abs(p_up - p_down)

    return {
        "ticker":                ticker,
        "predicted_direction":   predicted_direction,
        "prediction_confidence": round(confidence, 4),
        "predicted_alpha":       round(s, 6),
        "p_up":                  round(p_up, 4),
        "p_flat":                round(p_flat, 4),
        "p_down":                round(p_down, 4),
    }


# ── Output writing ────────────────────────────────────────────────────────────

def write_predictions(
    predictions: list[dict],
    date_str: str,
    s3_bucket: str,
    metrics: dict,
    dry_run: bool = False,
) -> None:
    """
    Write predictions JSON to S3 at both the dated key and latest.json.
    Also writes metrics/latest.json. All S3 operations are best-effort.

    Parameters
    ----------
    predictions : List of per-ticker prediction dicts.
    date_str :    Date string YYYY-MM-DD.
    s3_bucket :   S3 bucket name.
    metrics :     Metrics dict to write to predictor/metrics/latest.json.
    dry_run :     If True, print to stdout instead of writing to S3.
    """
    # Build the predictions envelope
    n_high_confidence = sum(
        1 for p in predictions
        if p.get("prediction_confidence", 0) >= cfg.MIN_CONFIDENCE
    )

    output = {
        "date": date_str,
        "model_version": metrics.get("model_version", "unknown"),
        "model_hit_rate_30d": metrics.get("hit_rate_30d_rolling", None),
        "n_predictions": len(predictions),
        "n_high_confidence": n_high_confidence,
        "predictions": predictions,
    }

    metrics_out = {
        **metrics,
        "n_predictions_today": len(predictions),
        "n_high_confidence": n_high_confidence,
        "last_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
    }

    predictions_json = json.dumps(output, indent=2)
    metrics_json = json.dumps(metrics_out, indent=2)

    if dry_run:
        print("=== PREDICTIONS (dry-run) ===")
        print(predictions_json)
        print("\n=== METRICS (dry-run) ===")
        print(metrics_json)
        return

    dated_key = cfg.PREDICTIONS_KEY.format(date=date_str)
    latest_key = cfg.PREDICTIONS_LATEST_KEY
    metrics_key = cfg.METRICS_KEY

    try:
        import boto3
        s3 = boto3.client("s3")

        s3.put_object(
            Bucket=s3_bucket,
            Key=dated_key,
            Body=predictions_json.encode("utf-8"),
            ContentType="application/json",
        )
        log.info("Written s3://%s/%s", s3_bucket, dated_key)

        s3.put_object(
            Bucket=s3_bucket,
            Key=latest_key,
            Body=predictions_json.encode("utf-8"),
            ContentType="application/json",
        )
        log.info("Written s3://%s/%s", s3_bucket, latest_key)

        s3.put_object(
            Bucket=s3_bucket,
            Key=metrics_key,
            Body=metrics_json.encode("utf-8"),
            ContentType="application/json",
        )
        log.info("Written s3://%s/%s", s3_bucket, metrics_key)

    except Exception as exc:
        log.error("S3 write failed: %s", exc)
        log.error("Predictions not written to S3. Check IAM permissions for s3://%s", s3_bucket)


# ── Orchestration ─────────────────────────────────────────────────────────────

def main(
    date_str: Optional[str] = None,
    dry_run: bool = False,
    local: bool = False,
    s3_bucket: Optional[str] = None,
    model_type: str = "mlp",
    watchlist_path: Optional[str] = None,
) -> None:
    """
    Run the full daily prediction pipeline.

    Parameters
    ----------
    date_str :       Override prediction date YYYY-MM-DD. Default: today.
    dry_run :        Skip S3 writes; print output to stdout.
    local :          Load model from local checkpoints/ instead of S3.
    s3_bucket :      Override S3 bucket. Falls back to S3_BUCKET env var or config default.
    model_type :     Which model to run: 'mlp' (default) or 'gbm'.
    watchlist_path : Path to watchlist.json. When provided, predictions are
                     restricted to the tickers in 'tracked' + 'buy_candidates'.
                     Each result gets a 'watchlist_source' annotation.
                     When None, the full signals.json universe is used.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    bucket = s3_bucket or os.environ.get("S3_BUCKET", cfg.S3_BUCKET)

    log.info(
        "Daily predictor run: date=%s  bucket=%s  dry_run=%s  local=%s  "
        "model_type=%s  watchlist=%s",
        date_str, bucket, dry_run, local, model_type,
        watchlist_path or "full universe",
    )

    # ── Step 1: Load model ────────────────────────────────────────────────────
    scorer = None      # GBM path
    model = None       # MLP path
    checkpoint = {}    # MLP path

    if model_type == "gbm":
        if local:
            scorer = load_gbm_local("checkpoints/gbm_best.txt")
        else:
            scorer = load_gbm_s3(bucket, cfg.GBM_WEIGHTS_KEY)
        model_version = f"GBM-v{scorer._best_iteration}"
        val_loss = scorer._val_ic   # IC is the GBM analogue of val_loss
    else:
        # mlp (default)
        if local:
            model, checkpoint = load_model_local("checkpoints/best.pt")
        else:
            model, checkpoint = load_model(bucket, cfg.MODEL_WEIGHTS_KEY)
        norm_stats = checkpoint.get("norm_stats", {})
        if not norm_stats:
            log.warning("No norm_stats in checkpoint — features may not normalize correctly")
        model_version = checkpoint.get("model_version", "unknown")
        val_loss = checkpoint.get("val_loss", float("nan"))

    # ── Step 2: Get universe ──────────────────────────────────────────────────
    ticker_sources: dict[str, str] = {}   # ticker → watchlist_source annotation
    if watchlist_path:
        tickers, ticker_sources = load_watchlist(
            path=watchlist_path,
            s3_bucket=bucket,
            date_str=date_str,
        )
    else:
        tickers = get_universe_tickers(bucket, date_str)

    # ── Step 3: Fetch prices ──────────────────────────────────────────────────
    price_data = fetch_today_prices(tickers)

    # ── Step 3b: Load sector map + fetch macro/sector ETF series ─────────────
    # sector_map: ticker → sector ETF symbol (e.g. "AAPL" → "XLK")
    # Built by bootstrap_fetcher.py and stored in data/cache/sector_map.json.
    sector_map: dict[str, str] = {}
    sector_map_path = Path("data/cache/sector_map.json")
    if sector_map_path.exists():
        try:
            sector_map = json.loads(sector_map_path.read_text())
            log.info("Sector map loaded: %d mappings", len(sector_map))
        except Exception as exc:
            log.warning("Could not load sector_map.json: %s — sector_vs_spy_5d will be 0", exc)
    else:
        log.warning(
            "data/cache/sector_map.json not found — sector_vs_spy_5d will be 0. "
            "Run bootstrap_fetcher.py to generate it."
        )

    # Collect the unique sector ETFs needed for the active ticker universe
    sector_etfs_needed = sorted({
        sector_map[t] for t in tickers if t in sector_map
    })

    # Fetch macro series + any required sector ETFs in one batch
    macro = fetch_macro_series(extra_tickers=sector_etfs_needed)

    # ── Step 4: Run inference ─────────────────────────────────────────────────
    predictions: list[dict] = []
    n_skipped = 0

    for ticker in tickers:
        df = price_data.get(ticker, pd.DataFrame())
        # Look up this ticker's sector ETF series (None if not in sector map)
        sector_etf_sym = sector_map.get(ticker)
        sector_etf_series = macro.get(sector_etf_sym) if sector_etf_sym else None

        if model_type == "gbm":
            result = predict_ticker_gbm(
                ticker, df, scorer,
                macro=macro,
                sector_etf_series=sector_etf_series,
            )
        else:
            result = predict_ticker(
                ticker, df, model, norm_stats,
                macro=macro,
                sector_etf_series=sector_etf_series,
            )
        if result is not None:
            # Annotate with watchlist source when running in watchlist mode
            if ticker_sources:
                result["watchlist_source"] = ticker_sources.get(ticker, "unknown")
            predictions.append(result)
        else:
            n_skipped += 1

    log.info(
        "Inference complete: %d predictions  %d skipped",
        len(predictions),
        n_skipped,
    )

    # Sort by descending (p_up - p_down) for readability
    predictions.sort(key=lambda p: p["p_up"] - p["p_down"], reverse=True)

    # ── Step 5: Build metrics ─────────────────────────────────────────────────
    if model_type == "gbm":
        last_trained = scorer._best_iteration
    else:
        last_trained = checkpoint.get("epoch", "unknown")

    metrics = {
        "model_version": model_version,
        "model_type": model_type,
        "last_trained": last_trained,
        "val_loss": round(float(val_loss), 6) if isinstance(val_loss, (int, float)) else None,
        # Rolling hit rate requires outcome tracking (populated by backtester)
        "hit_rate_30d_rolling": None,
        "ic_30d": None,
        "ic_ir_30d": None,
    }

    # ── Step 6: Write output ──────────────────────────────────────────────────
    write_predictions(predictions, date_str, bucket, metrics, dry_run=dry_run)

    log.info("Predictor run complete for %s", date_str)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run daily direction predictions and write to S3."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override prediction date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip S3 writes; print predictions JSON to stdout.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Load model from local checkpoints/best.pt instead of S3.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help=f"Override S3 bucket. Default: {cfg.S3_BUCKET}",
    )
    parser.add_argument(
        "--model-type",
        default="mlp",
        choices=["mlp", "gbm"],
        help=(
            "Which model to load for inference: "
            "'mlp' loads checkpoints/best.pt (or S3 predictor/weights/latest.pt), "
            "'gbm' loads checkpoints/gbm_best.txt (or S3 predictor/weights/gbm_latest.txt). "
            "Default: mlp"
        ),
    )
    parser.add_argument(
        "--watchlist",
        default=None,
        metavar="auto|PATH",
        help=(
            "Restrict predictions to the research module's tracked + buy-candidate "
            "tickers instead of the full ~900-stock universe. "
            "'auto' reads today's signals/{date}/signals.json from S3. "
            "Any other value is treated as a local file path to a signals.json "
            "produced by alpha-engine-research (useful for offline / dry-run testing). "
            "Each prediction result gets a 'watchlist_source' annotation: "
            "'tracked' (universe tickers), 'buy_candidate' (scanner picks), or 'both'. "
            "Omit this flag to predict the full universe. "
            "Example: --watchlist auto   or   --watchlist /tmp/signals.json"
        ),
    )
    args = parser.parse_args()

    main(
        date_str=args.date,
        dry_run=args.dry_run,
        local=args.local,
        s3_bucket=args.s3_bucket,
        model_type=args.model_type,
        watchlist_path=args.watchlist,
    )
