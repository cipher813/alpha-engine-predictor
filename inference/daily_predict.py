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
    python inference/daily_predict.py [--date DATE] [--dry-run] [--local]

Flags:
    --date DATE     Override prediction date (YYYY-MM-DD). Default: today.
    --dry-run       Run inference but skip S3 writes. Print output to stdout.
    --local         Load model from local checkpoints/best.pt instead of S3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


# ── Price fetch ───────────────────────────────────────────────────────────────

def fetch_today_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch 1-year OHLCV history for each ticker via yfinance.
    Returns a dict of ticker → DataFrame. Empty DataFrames on failure.
    One year provides sufficient history for MA200 (200 rows) with some buffer.
    """
    import yfinance as yf

    log.info("Fetching 1y price data for %d tickers...", len(tickers))
    result: dict[str, pd.DataFrame] = {}

    # Batch download for efficiency
    batch_size = 100
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in batches:
        try:
            if len(batch) == 1:
                raw = yf.download(
                    batch[0],
                    period="1y",
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
                    period="1y",
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


# ── Per-ticker prediction ─────────────────────────────────────────────────────

def predict_ticker(
    ticker: str,
    df: pd.DataFrame,
    model: torch.nn.Module,
    norm_stats: dict,
) -> Optional[dict]:
    """
    Compute features for one ticker and run inference.

    Parameters
    ----------
    ticker :     Ticker symbol.
    df :         1-year OHLCV DataFrame.
    model :      Loaded DirectionPredictor in eval mode.
    norm_stats : Dict with 'mean' and 'std' lists for z-score normalization.

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

    if df.empty or len(df) < 205:
        # Need 200 rows for MA200 + a few extra for robustness
        log.debug("Skipping %s: insufficient data (%d rows)", ticker, len(df))
        return None

    try:
        featured_df = compute_features(df)
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

    # Inference
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
) -> None:
    """
    Run the full daily prediction pipeline.

    Parameters
    ----------
    date_str :  Override prediction date YYYY-MM-DD. Default: today.
    dry_run :   Skip S3 writes; print output to stdout.
    local :     Load model from local checkpoints/best.pt.
    s3_bucket : Override S3 bucket. Falls back to S3_BUCKET env var or config default.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    bucket = s3_bucket or os.environ.get("S3_BUCKET", cfg.S3_BUCKET)

    log.info("Daily predictor run: date=%s  bucket=%s  dry_run=%s  local=%s",
             date_str, bucket, dry_run, local)

    # ── Step 1: Load model ────────────────────────────────────────────────────
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
    tickers = get_universe_tickers(bucket, date_str)

    # ── Step 3: Fetch prices ──────────────────────────────────────────────────
    price_data = fetch_today_prices(tickers)

    # ── Step 4: Run inference ─────────────────────────────────────────────────
    predictions: list[dict] = []
    n_skipped = 0

    for ticker in tickers:
        df = price_data.get(ticker, pd.DataFrame())
        result = predict_ticker(ticker, df, model, norm_stats)
        if result is not None:
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
    metrics = {
        "model_version": model_version,
        "last_trained": checkpoint.get("epoch", "unknown"),
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
    args = parser.parse_args()

    main(
        date_str=args.date,
        dry_run=args.dry_run,
        local=args.local,
        s3_bucket=args.s3_bucket,
    )
