"""
inference/daily_predict.py — Daily prediction run.

Called by the Lambda handler (inference/handler.py) and optionally from the
CLI for local testing. Orchestrates:
  1. Load model weights from S3 (or local path with --local flag).
  2. Determine the active ticker universe.
  3. Fetch today's 2-year adjusted OHLCV data via yfinance (for features).
  3b. Load sector map + fetch macro/sector ETF series.
  3c. Save raw (unadjusted) daily closes to S3 at
      predictor/daily_closes/{date}.parquet — building the independent
      price archive over time (see save_daily_closes() for rationale).
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


# ── S3-delivered predictor params (veto threshold) ────────────────────────────

_predictor_params_cache: dict | None = None
_predictor_params_loaded: bool = False
# Local cache persists last known optimal across Lambda cold-starts (via /tmp)
# and EC2 restarts (via project dir).
_PREDICTOR_PARAMS_CACHE_PATH = Path(
    os.environ.get("PREDICTOR_PARAMS_CACHE", "/tmp/predictor_params_cache.json")
)


def _load_predictor_params_from_s3(s3_bucket: str) -> dict | None:
    """Read config/predictor_params.json from S3. Cache per cold-start.

    Fallback chain: S3 → local cache file → None (hardcoded defaults).
    On successful S3 read, writes a local cache so the last known optimal
    params survive transient S3 failures.
    """
    global _predictor_params_cache, _predictor_params_loaded
    if _predictor_params_loaded:
        return _predictor_params_cache
    _predictor_params_loaded = True

    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key="config/predictor_params.json")
        data = json.loads(obj["Body"].read())
        if "veto_confidence" in data:
            _predictor_params_cache = data
            log.info("Loaded predictor params from S3: veto_confidence=%.2f", data["veto_confidence"])
            # Persist to local cache for fault tolerance
            try:
                _PREDICTOR_PARAMS_CACHE_PATH.write_text(json.dumps(data, indent=2))
            except Exception:
                pass  # best-effort
        return _predictor_params_cache
    except Exception as e:
        log.warning("Could not read predictor params from S3: %s", e)

    # Fallback: last known optimal from local cache
    try:
        if _PREDICTOR_PARAMS_CACHE_PATH.exists():
            data = json.loads(_PREDICTOR_PARAMS_CACHE_PATH.read_text())
            if "veto_confidence" in data:
                _predictor_params_cache = data
                log.info(
                    "Loaded predictor params from local cache (last known optimal): veto_confidence=%.2f",
                    data["veto_confidence"],
                )
                return _predictor_params_cache
    except Exception as e2:
        log.debug("Could not read local predictor params cache: %s", e2)

    return None


def get_veto_threshold(s3_bucket: str, market_regime: str = "") -> float:
    """
    Return the active veto confidence threshold, adjusted by market regime.

    In bear/caution regimes, the threshold is lowered (more aggressive vetoing)
    to protect capital. In bull regimes, the threshold is raised (more permissive)
    to avoid missing opportunities.

    Regime adjustments (applied to the base threshold from S3 or config):
      bear:    -0.10  (e.g., 0.65 → 0.55 — veto more aggressively)
      caution: -0.05  (e.g., 0.65 → 0.60)
      neutral:  0.00  (no adjustment)
      bullish: +0.05  (e.g., 0.65 → 0.70 — allow more entries)
    """
    params = _load_predictor_params_from_s3(s3_bucket)
    if params and "veto_confidence" in params:
        base = float(params["veto_confidence"])
    else:
        base = cfg.MIN_CONFIDENCE

    # Regime-adaptive adjustment
    regime = market_regime.lower().strip() if market_regime else ""
    regime_adjustments = {
        "bear": -0.10,
        "bearish": -0.10,
        "caution": -0.05,
        "neutral": 0.0,
        "bull": 0.05,
        "bullish": 0.05,
    }
    adjustment = regime_adjustments.get(regime, 0.0)
    adjusted = max(0.40, min(0.90, base + adjustment))

    if adjustment != 0.0:
        log.info(
            "Veto threshold regime-adjusted: base=%.2f %+.2f (%s) → %.2f",
            base, adjustment, regime, adjusted,
        )

    return adjusted


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
    Build a focused prediction universe from Research's population or signals.

    Priority order for ``path="auto"``:
      1. population/latest.json  — new population-based architecture
      2. signals/{date}/signals.json  — legacy fallback

    Parameters
    ----------
    path      : "auto" → fetch from S3 (population first, then signals).
                Any other string → local file path to signals.json or
                population.json for offline / dry-run use.
    s3_bucket : S3 bucket name. Required when path="auto".
    date_str  : YYYY-MM-DD override. Defaults to today.

    Returns
    -------
    tickers : Deduplicated, sorted list of tickers.
    sources : Dict mapping ticker → "population" | "tracked" | "buy_candidate" | "both".
    data    : Raw JSON payload (population or signals).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    data = None

    # ── Load raw payload ──────────────────────────────────────────────────────
    if path == "auto":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when --watchlist auto is used")

        import boto3
        s3 = boto3.client("s3")

        # Try population/latest.json first (new architecture)
        try:
            obj = s3.get_object(Bucket=s3_bucket, Key="population/latest.json")
            data = json.loads(obj["Body"].read().decode("utf-8"))
            pop_tickers = [p["ticker"] for p in data.get("population", []) if "ticker" in p]
            if pop_tickers:
                sources = {t.upper(): "population" for t in pop_tickers}
                tickers = sorted(sources.keys())
                log.info(
                    "Watchlist: loaded %d tickers from population/latest.json",
                    len(tickers),
                )
                return tickers, sources, data
        except Exception as exc:
            log.info("population/latest.json not available (%s), falling back to signals.json", exc)

        # Fallback: signals/{date}/signals.json with date lookback
        # Walk back up to 5 calendar days (skipping weekends) to find the
        # most recent signals — mirrors executor's read_signals_with_fallback.
        from datetime import date as _date, timedelta as _td
        from botocore.exceptions import ClientError

        start = _date.fromisoformat(date_str)
        max_lookback = 5
        tried: list[str] = []

        for days_back in range(max_lookback + 1):
            candidate = start - _td(days=days_back)
            if candidate.weekday() >= 5:  # skip Saturday/Sunday
                continue
            signals_key = f"signals/{candidate}/signals.json"
            try:
                obj = s3.get_object(Bucket=s3_bucket, Key=signals_key)
                data = json.loads(obj["Body"].read().decode("utf-8"))
                if days_back > 0:
                    log.warning(
                        "Watchlist: no signals for %s — using %s (%d day(s) old). Tried: %s",
                        start, candidate, days_back, tried,
                    )
                else:
                    log.info("Watchlist: loaded signals from s3://%s/%s", s3_bucket, signals_key)
                break
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("NoSuchKey", "AccessDenied", "403"):
                    log.info("No signals for %s (%s), looking further back...", candidate, code)
                    tried.append(str(candidate))
                    continue
                raise
            except Exception as exc:
                log.info("Error reading signals for %s: %s", candidate, exc)
                tried.append(str(candidate))
                continue
        else:
            raise RuntimeError(
                f"No signals found within {max_lookback} days of {start}. "
                f"Dates tried: {tried}. Ensure research pipeline ran recently, "
                "or provide a local path: --watchlist /path/to/signals.json"
            )
    else:
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Use --watchlist auto to pull from S3, or provide a local path."
            )
        data = json.loads(local_path.read_text())
        log.info("Watchlist: loaded from %s", path)

        # Check if this is a population file
        if "population" in data and isinstance(data["population"], list):
            pop_tickers = [p["ticker"] for p in data["population"] if "ticker" in p]
            if pop_tickers:
                sources = {t.upper(): "population" for t in pop_tickers}
                tickers = sorted(sources.keys())
                log.info("Watchlist: %d tickers from population file", len(tickers))
                return tickers, sources, data

    # ── Extract and annotate tickers from signals.json ─────────────────────
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
    return tickers, sources, data


# ── Price fetch ───────────────────────────────────────────────────────────────

def fetch_today_prices(tickers: list[str], fd=None) -> dict[str, pd.DataFrame]:
    """
    Fetch 1-year OHLCV history for each ticker via yfinance.
    Returns a dict of ticker → DataFrame. Empty DataFrames on failure.
    One year provides sufficient history for MA200 (200 rows) with some buffer.
    """
    import yfinance as yf

    # 2y lookback: compute_features needs 252 rows of warmup (52w rolling windows).
    # With only 1y of data (~252 rows) the dropna step in compute_features leaves
    # 0–1 rows, causing empty featured_df and skipped predictions.
    log.info("Fetching %s price data for %d tickers...", cfg.INFERENCE_PERIOD, len(tickers))
    result: dict[str, pd.DataFrame] = {}

    # Batch download for efficiency
    batch_size = cfg.INFERENCE_BATCH_SIZE
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in batches:
        try:
            if len(batch) == 1:
                raw = yf.download(
                    batch[0],
                    period=cfg.INFERENCE_PERIOD,
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
                    period=cfg.INFERENCE_PERIOD,
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
            if fd:
                fd.report(exc, severity="error", context={
                    "site": "batch_price_fetch",
                    "batch_size": len(batch),
                })
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


# ── Daily raw-price archive ────────────────────────────────────────────────────

def save_daily_closes(
    tickers: list[str],
    date_str: str,
    s3_bucket: str,
    dry_run: bool = False,
) -> int:
    """
    Fetch and archive today's OHLCV for all tickers using ``auto_adjust=False``.

    Writes one parquet file per trading day:
        predictor/daily_closes/{date_str}.parquet
    Schema: index=ticker (str), columns=[date, open, high, low, close, adj_close, volume]

    **What yfinance actually stores — no truly raw mode exists:**
    yfinance provides NO mode that yields truly raw (unadjusted) prices via batch
    download.  ``auto_adjust=False``, ``back_adjust=False``, and ``actions=False``
    all return the same *backward-split-adjusted* series — every historical price
    is retroactively divided by the cumulative split factor so the current price is
    the reference point.  Example: NVDA's true raw close on Jan 2, 2024 was ~$481;
    yfinance returns ~$48 because it has already divided by the 10× June 2024 split.
    ``adj_close / close`` captures only the dividend adjustment factor (~0.999 on
    dividend dates, 1.0 otherwise).  True raw prices would require a separate data
    provider (e.g., Polygon.io) that stores prices and corporate actions independently.

    What we actually save:
    - ``close``     = backward-split-adjusted close (same basis as the training
                      cache / slim cache which use auto_adjust=True).  The two modes
                      differ only by the dividend factor, which is negligible for
                      momentum/RSI/MACD features (~0.1% per dividend event).
    - ``adj_close`` = fully adjusted (splits + dividends) = auto_adjust=True close.

    **Split-boundary staleness and how it is handled:**
    When a stock splits after a daily_closes entry has been saved, yfinance
    retroactively revises that ticker's historical prices to the new split basis.
    The saved entry retains its original value, creating a price discontinuity
    in the combined slim-cache + daily-closes series.  ``load_price_data_from_cache``
    detects any |single-day return| > 45% as a split event and re-fetches that
    ticker from yfinance.  Splits affect ~1-2 tickers per week; the remaining ~898
    tickers every day require no yfinance call at all.  The slim cache is rebuilt
    every Sunday, clearing all stale split-boundary entries.

    Parameters
    ----------
    tickers :   Tickers to capture.
    date_str :  Trading date label YYYY-MM-DD (file key and 'date' column value).
    s3_bucket : S3 bucket name.
    dry_run :   Log what would be written but skip the S3 put.

    Returns
    -------
    Number of tickers successfully captured (0 on failure).
    """
    import io

    import boto3
    import yfinance as yf

    log.info(
        "Capturing daily closes for %d tickers, date=%s …",
        len(tickers), date_str,
    )

    records: list[dict] = []
    batch_size = cfg.INFERENCE_BATCH_SIZE
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in batches:
        try:
            tickers_arg = batch[0] if len(batch) == 1 else batch
            raw = yf.download(
                tickers=tickers_arg,
                period=cfg.DAILY_CLOSES_PERIOD,
                interval="1d",
                auto_adjust=False,  # keeps both Close (split-adj) and Adj Close (full-adj)
                progress=False,
                group_by="ticker",
                threads=True,
            )
            is_multi = isinstance(raw.columns, pd.MultiIndex)

            for ticker in batch:
                try:
                    df = (raw[ticker] if is_multi else raw).copy()
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert("UTC").tz_localize(None)
                    df = df.dropna(subset=["Close"])
                    if df.empty:
                        continue

                    last = df.iloc[-1]
                    raw_close = float(last["Close"])
                    adj_close = float(last["Adj Close"]) if "Adj Close" in df.columns else raw_close
                    records.append({
                        "ticker":    ticker,
                        "date":      date_str,
                        "open":      round(float(last["Open"]),  4),
                        "high":      round(float(last["High"]),  4),
                        "low":       round(float(last["Low"]),   4),
                        "close":     round(raw_close,            4),
                        "adj_close": round(adj_close,            4),
                        "volume":    int(last["Volume"]) if pd.notna(last.get("Volume")) else 0,
                    })
                except Exception as exc:
                    log.debug("Close extract failed for %s: %s", ticker, exc)

        except Exception as exc:
            log.warning("Raw price batch failed: %s", exc)

    if not records:
        log.warning("No raw closes captured for %s — skipping S3 write", date_str)
        return 0

    closes_df = pd.DataFrame(records).set_index("ticker")
    log.info(
        "Raw closes captured: %d / %d tickers for %s",
        len(closes_df), len(tickers), date_str,
    )

    if dry_run:
        log.info(
            "[dry-run] Would write %d closes to s3://%s/predictor/daily_closes/%s.parquet",
            len(closes_df), s3_bucket, date_str,
        )
        return len(closes_df)

    try:
        s3  = boto3.client("s3")
        buf = io.BytesIO()
        closes_df.to_parquet(buf, engine="pyarrow", compression="snappy", index=True)
        buf.seek(0)
        key = f"predictor/daily_closes/{date_str}.parquet"
        s3.put_object(
            Bucket=s3_bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        log.info(
            "Daily closes written to s3://%s/%s  (%d tickers)",
            s3_bucket, key, len(closes_df),
        )
        return len(closes_df)
    except Exception as exc:
        log.error("Failed to write daily closes to S3: %s", exc)
        return 0


# ── Slim-cache + daily-closes price loader ────────────────────────────────────

_SLIM_PREFIX   = "predictor/price_cache_slim/"
_CLOSES_PREFIX = "predictor/daily_closes/"

# Single-day return magnitude above which we assume a split occurred.
# A legitimate limit-up/down is ±20%; a split creates ±50%+ moves.
from config import SPLIT_RETURN_THRESHOLD as _SPLIT_RETURN_THRESHOLD


def _download_slim_cache(s3_bucket: str, local_dir: Path) -> int:
    """
    Download all parquets from predictor/price_cache_slim/ to local_dir in parallel.
    Returns number of files downloaded, or 0 if the prefix has no objects.
    """
    import boto3
    from concurrent.futures import ThreadPoolExecutor

    s3 = boto3.client("s3")
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=_SLIM_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)

    if not keys:
        return 0

    def _get(key: str) -> None:
        fname = key[len(_SLIM_PREFIX):]
        s3.download_file(s3_bucket, key, str(local_dir / fname))

    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(_get, keys))

    return len(keys)


def _load_delta_from_daily_closes(
    s3_bucket: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, list[dict]]:
    """
    Load daily_closes parquets for every trading day in (start_date, end_date).
    Returns dict: ticker → list of {date, open, high, low, close, adj_close} dicts,
    one entry per trading day found in S3.

    start_date is exclusive (we already have it in the slim cache).
    end_date   is inclusive (we want data up to and including this date).
    """
    import io
    import boto3

    s3 = boto3.client("s3")

    # Build business-day range strictly after start_date and up to end_date
    delta_dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.bdate_range(
            start_date + pd.Timedelta(days=1), end_date
        )
    ]

    if not delta_dates:
        return {}

    log.info(
        "Loading daily_closes delta: %d trading days (%s → %s)",
        len(delta_dates), delta_dates[0], delta_dates[-1],
    )

    ticker_rows: dict[str, list[dict]] = {}

    for d in delta_dates:
        key = f"{_CLOSES_PREFIX}{d}.parquet"
        try:
            resp = s3.get_object(Bucket=s3_bucket, Key=key)
            buf  = io.BytesIO(resp["Body"].read())
            day_df = pd.read_parquet(buf, engine="pyarrow")
            # index is ticker string
            for ticker, row in day_df.iterrows():
                if ticker not in ticker_rows:
                    ticker_rows[ticker] = []
                ticker_rows[ticker].append({
                    "date":      pd.Timestamp(d),
                    "open":      float(row.get("open",      row.get("Open",      np.nan))),
                    "high":      float(row.get("high",      row.get("High",      np.nan))),
                    "low":       float(row.get("low",       row.get("Low",       np.nan))),
                    "close":     float(row.get("close",     row.get("Close",     np.nan))),
                    "adj_close": float(row.get("adj_close", row.get("close",     np.nan))),
                    "volume":    int(row.get("volume",      row.get("Volume",    0))),
                })
        except s3.exceptions.NoSuchKey:
            log.debug("daily_closes/%s.parquet not found in S3 (non-trading day?)", d)
        except Exception as exc:
            log.warning("Could not load daily_closes/%s: %s", d, exc)

    n_tickers = len(ticker_rows)
    n_rows    = sum(len(v) for v in ticker_rows.values())
    log.info("Delta loaded: %d rows across %d tickers", n_rows, n_tickers)
    return ticker_rows


def load_price_data_from_cache(
    tickers: list[str],
    date_str: str,
    s3_bucket: str,
) -> "tuple[dict[str, pd.DataFrame] | None, dict[str, pd.Series] | None]":
    """
    Load price history for inference from the slim cache + daily_closes delta.

    This replaces the 2-year-per-day yfinance full download with:
      1. S3 download of the slim cache (~9 MB, created weekly by train_handler).
      2. S3 reads of daily_closes/{date}.parquet for each trading day between the
         slim cache's last date and today (typically 1–4 files, each ~100 KB).
      3. For tickers where the combined series shows a single-day price move
         > ±45% (a stock split), re-fetch only those tickers from yfinance.

    Also builds a macro dict (SPY, VIX, TNX, IRX, GLD, USO, sector ETFs) from
    the slim cache, eliminating fetch_macro_series() entirely.

    Returns
    -------
    (price_data, macro) in the same formats as fetch_today_prices() and
    fetch_macro_series().  Returns (None, None) if the slim cache does not yet
    exist — callers should fall back to fetch_today_prices() + fetch_macro_series().
    """
    import tempfile
    from data.dataset import _load_ticker_parquet

    today = pd.Timestamp(date_str).normalize()

    # ── Step 1: Download slim cache ───────────────────────────────────────────
    tmp_dir  = Path(tempfile.mkdtemp())
    slim_dir = tmp_dir / "slim_cache"

    log.info("Attempting slim-cache load from s3://%s/%s …", s3_bucket, _SLIM_PREFIX)
    n_slim = _download_slim_cache(s3_bucket, slim_dir)

    if n_slim == 0:
        log.info("Slim cache not found — will fall back to yfinance")
        return None, None

    log.info("Slim cache downloaded: %d parquets", n_slim)

    # ── Step 2: Load slim cache into memory ───────────────────────────────────
    slim_data: dict[str, pd.DataFrame] = {}
    for pq in slim_dir.glob("*.parquet"):
        df = _load_ticker_parquet(pq)
        if not df.empty:
            slim_data[pq.stem] = df

    if not slim_data:
        log.warning("Slim cache parquets empty — falling back to yfinance")
        return None, None

    slim_last_date = max(df.index.max() for df in slim_data.values())
    slim_last_date = pd.Timestamp(slim_last_date).normalize()
    log.info(
        "Slim cache loaded: %d tickers, last date: %s",
        len(slim_data), slim_last_date.date(),
    )

    # ── Step 3: Load daily_closes delta ──────────────────────────────────────
    ticker_rows = _load_delta_from_daily_closes(
        s3_bucket, slim_last_date, today - pd.Timedelta(days=1),
    )

    # ── Step 4: Build combined price_data per ticker ──────────────────────────
    price_data:    dict[str, pd.DataFrame] = {}
    split_tickers: set[str]                = set()

    # Start with all tickers found in the slim cache
    for ticker, slim_df in slim_data.items():
        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        # Only keep columns that exist
        base = slim_df[[c for c in base_cols if c in slim_df.columns]].copy()

        delta = ticker_rows.get(ticker, [])
        if not delta:
            price_data[ticker] = base
            continue

        # Build delta DataFrame (capitalize to match slim cache schema)
        delta_df = pd.DataFrame(
            [{
                "Open":   r["open"],
                "High":   r["high"],
                "Low":    r["low"],
                "Close":  r["close"],
                "Volume": r["volume"],
            } for r in delta],
            index=pd.DatetimeIndex([r["date"] for r in delta]),
        )

        combined = pd.concat([base, delta_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        # Split detection: any single-day Close return beyond threshold
        returns = combined["Close"].pct_change().dropna()
        if (returns.abs() > _SPLIT_RETURN_THRESHOLD).any():
            log.info(
                "Split/large-move detected in %s — will re-fetch from yfinance", ticker,
            )
            split_tickers.add(ticker)
        else:
            price_data[ticker] = combined

    # Tickers in requested list but not in slim cache get re-fetched too
    missing = set(tickers) - set(slim_data.keys())
    if missing:
        log.info("%d tickers not in slim cache — adding to re-fetch list", len(missing))
        split_tickers.update(missing)

    # ── Step 5: Re-fetch split/missing tickers from yfinance ─────────────────
    if split_tickers:
        log.info(
            "Re-fetching %d tickers from yfinance (splits/missing): %s …",
            len(split_tickers),
            sorted(split_tickers)[:10],
        )
        fresh = fetch_today_prices(sorted(split_tickers))
        price_data.update(fresh)

    n_success = sum(1 for df in price_data.values() if not df.empty)
    log.info(
        "Cache-based price load complete: %d/%d tickers  "
        "(%d from slim+delta, %d yfinance re-fetches)",
        n_success, len(tickers),
        n_success - len(split_tickers) + sum(1 for t in split_tickers if not price_data.get(t, pd.DataFrame()).empty),
        len(split_tickers),
    )

    # ── Step 6: Build macro dict from slim cache ──────────────────────────────
    # These symbols are stored without caret in the training cache parquets.
    _MACRO_SLIM_KEYS = {
        "SPY": "SPY",
        "VIX": "VIX",   # stored as VIX, yfinance ticker is ^VIX
        "TNX": "TNX",   # stored as TNX, yfinance ticker is ^TNX
        "IRX": "IRX",
        "GLD": "GLD",
        "USO": "USO",
    }
    macro: dict[str, pd.Series] = {}
    for key, stem in _MACRO_SLIM_KEYS.items():
        # Prefer combined price_data (includes delta) over raw slim_data
        source = price_data.get(stem) or slim_data.get(stem)
        if source is not None and "Close" in source.columns:
            macro[key] = source["Close"].dropna()
        else:
            log.debug("Macro series %s not in slim cache", key)

    # Sector ETFs: any XL* symbols in the slim cache
    for stem, df in slim_data.items():
        if stem.startswith("XL") and "Close" in df.columns:
            source = price_data.get(stem) or df
            macro[stem] = source["Close"].dropna()

    return price_data, macro


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

    if df.empty or len(df) < cfg.MIN_ROWS_FOR_FEATURES:
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

    if df.empty or len(df) < cfg.MIN_ROWS_FOR_FEATURES:
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
    x_raw = latest[cfg.GBM_FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)

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
    veto_threshold: float | None = None,
    fd=None,
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
    veto_threshold : Confidence threshold for veto gate. Defaults to cfg.MIN_CONFIDENCE.
    """
    threshold = veto_threshold if veto_threshold is not None else cfg.MIN_CONFIDENCE
    # Build the predictions envelope
    n_high_confidence = sum(
        1 for p in predictions
        if p.get("prediction_confidence", 0) >= threshold
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
        if fd:
            fd.report(exc, severity="critical", context={
                "site": "s3_predictions_write",
                "bucket": s3_bucket,
                "date": date_str,
            })


# ── Predictor email ────────────────────────────────────────────────────────────

def _build_predictor_email(
    predictions: list[dict],
    metrics: dict,
    date_str: str,
    signals_data: dict | None = None,
    veto_threshold: float | None = None,
) -> tuple[str, str, str]:
    """
    Build subject, HTML body, and plain-text body for the combined morning briefing.

    When signals_data is supplied (the raw signals.json payload from the research
    pipeline), a research section is prepended containing market regime, buy
    candidates, and sector ratings. The GBM predictions follow as the second half.

    Returns
    -------
    (subject, html_body, plain_body)
    """
    import datetime as _dt

    _vt = veto_threshold if veto_threshold is not None else cfg.MIN_CONFIDENCE
    model_version = metrics.get("model_version", "unknown")
    val_ic        = metrics.get("ic_30d")        # 30-day information coefficient
    n_total       = len(predictions)

    # Group by direction (predictions are pre-sorted descending p_up - p_down)
    ups   = [p for p in predictions if p.get("predicted_direction") == "UP"]
    flats = [p for p in predictions if p.get("predicted_direction") == "FLAT"]
    downs = [p for p in predictions if p.get("predicted_direction") == "DOWN"]

    # Option A vetoes: high-confidence DOWN signals that will trigger HOLD overrides
    vetoes   = [p for p in downs if p.get("prediction_confidence", 0) >= _vt]
    n_vetoed = len(vetoes)

    # ── Research data extraction ───────────────────────────────────────────────
    sd = signals_data or {}
    market_regime    = sd.get("market_regime", "")
    buy_candidates   = sd.get("buy_candidates", [])
    sector_ratings   = sd.get("sector_ratings", {})
    sorted_sectors: list = []

    # ── Subject ───────────────────────────────────────────────────────────────
    veto_str    = f" | {n_vetoed} veto{'es' if n_vetoed != 1 else ''}" if n_vetoed else ""
    regime_str  = f" | {market_regime.upper()}" if market_regime else ""
    cand_str    = f" | {len(buy_candidates)} candidates" if buy_candidates else ""
    subject = (
        f"Alpha Engine Brief | {date_str}{regime_str}{cand_str} | "
        f"{len(ups)} UP / {len(flats)} FLAT / {len(downs)} DOWN"
        f"{veto_str}"
    )

    # ── Helpers ───────────────────────────────────────────────────────────────
    run_time = _dt.datetime.now().strftime("%-I:%M %p PT")
    ic_str   = f"{val_ic:.4f}" if isinstance(val_ic, (int, float)) else "—"

    def _source_tag(p: dict) -> str:
        return " ★" if p.get("watchlist_source") in ("buy_candidate", "both") else ""

    def _alpha_str(p: dict) -> str:
        a = p.get("predicted_alpha")
        if a is None:
            return "—"
        return f"{'+' if a >= 0 else ''}{a * 100:.2f}%"

    def _conf_pct(p: dict) -> str:
        return f"{p.get('prediction_confidence', 0) * 100:.0f}%"

    # ── HTML ──────────────────────────────────────────────────────────────────
    TH = 'style="background:#f0f0f0; padding:4px 8px; text-align:left; border:1px solid #ccc;"'
    TD = 'style="padding:4px 8px; border:1px solid #ddd;"'
    TDR = 'style="padding:4px 8px; border:1px solid #ddd; text-align:right;"'
    TABLE = 'style="border-collapse:collapse; width:100%; font-family:monospace; font-size:12px;"'

    def _html_rows(group: list[dict]) -> str:
        if not group:
            return '<tr><td colspan="4" style="padding:4px 8px; color:#888; font-style:italic;">none</td></tr>'
        rows = []
        for p in group:
            is_veto = (
                p.get("predicted_direction") == "DOWN"
                and p.get("prediction_confidence", 0) >= _vt
            )
            veto_badge = ' <span style="color:#c62828; font-weight:bold;">⚠ VETO</span>' if is_veto else ""
            rows.append(
                f'<tr>'
                f'<td {TD}><b>{p["ticker"]}{_source_tag(p)}</b>{veto_badge}</td>'
                f'<td {TDR}>{_alpha_str(p)}</td>'
                f'<td {TDR}>{_conf_pct(p)}</td>'
                f'<td {TD}>{p.get("watchlist_source", "—")}</td>'
                f'</tr>'
            )
        return "\n".join(rows)

    def _html_section(title: str, color: str, group: list[dict]) -> str:
        return (
            f'<h3 style="color:{color}; margin-bottom:4px;">{title} ({len(group)})</h3>'
            f'<table {TABLE}>'
            f'<tr><th {TH}>Ticker</th><th {TH}>α score</th><th {TH}>Conf</th><th {TH}>Source</th></tr>'
            f'{_html_rows(group)}'
            f'</table>'
        )

    veto_section_html = ""
    if vetoes:
        veto_tickers = ", ".join(p["ticker"] for p in vetoes)
        pct = int(_vt * 100)
        veto_section_html = (
            f'<hr style="border:1px solid #eee; margin:16px 0;">'
            f'<h3 style="color:#c62828;">⚠ Option A Vetoes ({n_vetoed})</h3>'
            f'<p style="font-size:12px; margin:4px 0;">'
            f'DOWN predictions with confidence ≥{pct}% — executor will override ENTER → HOLD:</p>'
            f'<p style="font-family:monospace; font-size:13px;"><b>{veto_tickers}</b></p>'
        )

    # ── Research section HTML ─────────────────────────────────────────────────
    research_html = ""
    if sd:
        # Market regime pill
        regime_color = {"bullish": "#2e7d32", "bearish": "#c62828"}.get(
            market_regime.lower(), "#555"
        )
        regime_pill = (
            f'<span style="display:inline-block; background:{regime_color}; color:#fff; '
            f'font-size:11px; padding:2px 8px; border-radius:3px; font-weight:bold;">'
            f'{market_regime.upper() if market_regime else "NEUTRAL"}</span>'
        )

        # Buy candidates table
        cand_rows = ""
        for c in buy_candidates:
            score      = c.get("score") or "—"
            conviction = c.get("conviction", "—")
            signal     = c.get("signal", "—")
            sector     = c.get("sector", "—")
            gbm_veto   = c.get("gbm_veto", False)
            veto_badge = ' <span style="color:#c62828; font-weight:bold; font-size:10px;">GBM⚠</span>' if gbm_veto else ""
            score_str  = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            cand_rows += (
                f'<tr>'
                f'<td {TD}><b>{c.get("ticker","?")}</b>{veto_badge}</td>'
                f'<td {TDR}>{score_str}</td>'
                f'<td {TD}>{conviction}</td>'
                f'<td {TD}>{signal}</td>'
                f'<td {TD}>{sector}</td>'
                f'</tr>'
            )
        if not cand_rows:
            cand_rows = f'<tr><td colspan="5" style="padding:4px 8px; color:#888; font-style:italic;">none</td></tr>'
        cand_table = (
            f'<table {TABLE}>'
            f'<tr><th {TH}>Ticker</th><th {TH}>Score</th><th {TH}>Conviction</th>'
            f'<th {TH}>Signal</th><th {TH}>Sector</th></tr>'
            f'{cand_rows}'
            f'</table>'
        )

        # Sector ratings (top sectors sorted by rating desc, skip empty)
        sector_rows = ""
        sorted_sectors = sorted(
            [(s, v) for s, v in sector_ratings.items() if isinstance(v, dict)],
            key=lambda x: x[1].get("rating", 0),
            reverse=True,
        )
        for sector, v in sorted_sectors[:8]:
            rating   = v.get("rating", "—")
            modifier = v.get("modifier", "—")
            rating_str   = f"{rating:.0f}" if isinstance(rating, (int, float)) else str(rating)
            modifier_str = f"{modifier:.2f}x" if isinstance(modifier, (int, float)) else str(modifier)
            sector_rows += f'<tr><td {TD}>{sector}</td><td {TDR}>{rating_str}</td><td {TDR}>{modifier_str}</td></tr>'
        sector_table = ""
        if sector_rows:
            sector_table = (
                f'<table {TABLE}>'
                f'<tr><th {TH}>Sector</th><th {TH}>Rating</th><th {TH}>Modifier</th></tr>'
                f'{sector_rows}'
                f'</table>'
            )

        research_html = (
            f'<div style="background:#f8f9fa; border-left:3px solid #555; padding:12px 16px; margin-bottom:16px;">'
            f'<h3 style="margin:0 0 8px 0; font-size:14px; color:#333;">Research Brief</h3>'
            f'<p style="margin:0 0 8px 0;">Market Regime: {regime_pill}</p>'
            f'<h4 style="margin:8px 0 4px 0; font-size:12px; color:#555;">Buy Candidates ({len(buy_candidates)})</h4>'
            f'{cand_table}'
            f'{"<h4 style=margin:8px 0 4px 0; font-size:12px; color:#555;>Sector Ratings</h4>" + sector_table if sector_table else ""}'
            f'</div>'
        )

    html_body = (
        f'<html><body style="font-family:sans-serif; font-size:13px; color:#222; max-width:700px;">'
        f'<h2 style="margin-bottom:4px;">Alpha Engine Brief — {date_str}</h2>'
        f'<p style="color:#555; font-size:12px; margin-top:0;">'
        f'Model: <b>{model_version}</b> &nbsp;|&nbsp;'
        f'IC (val): <b>{ic_str}</b> &nbsp;|&nbsp;'
        f'Universe: <b>{n_total}</b> tickers &nbsp;|&nbsp;'
        f'Run at <b>{run_time}</b></p>'
        f'{research_html}'
        f'<h3 style="font-size:13px; color:#333; margin-bottom:4px;">GBM Predictions</h3>'
        f'{_html_section("↑ BULLISH", "#2e7d32", ups)}'
        f'{_html_section("→ NEUTRAL", "#888888", flats)}'
        f'{_html_section("↓ BEARISH", "#c62828", downs)}'
        f'{veto_section_html}'
        f'<p style="font-size:11px; color:#aaa; margin-top:24px;">'
        f'★ = buy_candidate from research signals.json &nbsp;|&nbsp;'
        f'⚠ VETO = Option A gate trigger (conf ≥{int(_vt * 100)}%)</p>'
        f'</body></html>'
    )

    # ── Plain text ────────────────────────────────────────────────────────────
    def _plain_rows(group: list[dict]) -> str:
        if not group:
            return "  (none)\n"
        lines = []
        for p in group:
            veto = " [VETO]" if (
                p.get("predicted_direction") == "DOWN"
                and p.get("prediction_confidence", 0) >= _vt
            ) else ""
            lines.append(
                f"  {p['ticker']:<6}  α={_alpha_str(p):>7}  conf={_conf_pct(p)}"
                f"  {p.get('watchlist_source', '—')}{_source_tag(p)}{veto}"
            )
        return "\n".join(lines) + "\n"

    # Research plain section
    research_plain = ""
    if sd:
        research_plain = (
            f"\n{'='*60}\n"
            f"RESEARCH BRIEF\n"
            f"{'='*60}\n"
            f"Market Regime: {market_regime.upper() if market_regime else 'NEUTRAL'}\n"
        )
        if buy_candidates:
            research_plain += f"\nBuy Candidates ({len(buy_candidates)}):\n"
            for c in buy_candidates:
                score = c.get("score")
                score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "—"
                veto_str_c = " [GBM VETO]" if c.get("gbm_veto") else ""
                research_plain += (
                    f"  {c.get('ticker','?'):<6}  score={score_str:>5}  "
                    f"{c.get('conviction','—'):<10}  {c.get('signal','—'):<8}  "
                    f"{c.get('sector','—')}{veto_str_c}\n"
                )
        if sector_ratings:
            research_plain += "\nSector Ratings:\n"
            for sector, v in sorted_sectors[:8]:
                rating = v.get("rating", "—")
                modifier = v.get("modifier", "—")
                rating_str   = f"{rating:.0f}" if isinstance(rating, (int, float)) else str(rating)
                modifier_str = f"{modifier:.2f}x" if isinstance(modifier, (int, float)) else str(modifier)
                research_plain += f"  {sector:<20}  rating={rating_str:>3}  modifier={modifier_str}\n"

    plain_body = (
        f"Alpha Engine Brief — {date_str}\n"
        f"Model: {model_version}  IC(val): {ic_str}  Universe: {n_total}  Run: {run_time}\n"
        f"{research_plain}"
        f"\n{'='*60}\n"
        f"GBM PREDICTIONS\n"
        f"{'='*60}\n"
        f"\nBULLISH ({len(ups)})\n{_plain_rows(ups)}"
        f"\nNEUTRAL ({len(flats)})\n{_plain_rows(flats)}"
        f"\nBEARISH ({len(downs)})\n{_plain_rows(downs)}"
    )
    if vetoes:
        veto_tickers = ", ".join(p["ticker"] for p in vetoes)
        plain_body += (
            f"\nOPTION A VETOES ({n_vetoed}): {veto_tickers}\n"
            f"(DOWN + conf >= {int(_vt * 100)}% → executor HOLD override)\n"
        )

    return subject, html_body, plain_body


def send_predictor_email(
    predictions: list[dict],
    metrics: dict,
    date_str: str,
    signals_data: dict | None = None,
    veto_threshold: float | None = None,
) -> bool:
    """
    Send combined morning briefing email via Gmail SMTP (primary) or SES (fallback).

    When signals_data is provided (research pipeline's signals.json payload),
    the email includes a research section (market regime, buy candidates, sector
    ratings) followed by the GBM predictions — one complete morning briefing.

    Reads from environment / config:
        EMAIL_SENDER        — from-address
        EMAIL_RECIPIENTS    — list of recipient addresses
        GMAIL_APP_PASSWORD  — enables Gmail SMTP path (recommended)
        AWS_REGION          — SES region fallback

    Returns True on success, False on any failure. Never raises.
    """
    sender     = cfg.EMAIL_SENDER
    recipients = cfg.EMAIL_RECIPIENTS

    if not sender or not recipients:
        log.info(
            "Predictor email skipped — set EMAIL_SENDER and EMAIL_RECIPIENTS "
            "env vars in the Lambda to enable"
        )
        return False

    try:
        subject, html_body, plain_body = _build_predictor_email(
            predictions, metrics, date_str, signals_data=signals_data,
            veto_threshold=veto_threshold,
        )
    except Exception as exc:
        log.warning("Failed to build predictor email body: %s", exc)
        return False

    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()

    if app_password:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

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
            log.info("Predictor email sent via Gmail SMTP: '%s'", subject)
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
        log.info("Predictor email sent via SES: '%s'", subject)
        return True
    except Exception as exc:
        log.warning("SES send failed: %s — predictor email not delivered", exc)
        return False


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

    fd = None
    try:
        import flow_doctor
        fd = flow_doctor.init(config_path=os.path.join(
            str(Path(__file__).resolve().parent.parent), "flow-doctor.yaml"))
    except Exception:
        pass

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
    signals_data: dict = {}               # raw signals.json payload for email
    if watchlist_path:
        tickers, ticker_sources, signals_data = load_watchlist(
            path=watchlist_path,
            s3_bucket=bucket,
            date_str=date_str,
        )
    else:
        tickers = get_universe_tickers(bucket, date_str)

    # ── Step 3: Load sector map ───────────────────────────────────────────────
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

    # ── Step 3b: Prices + macro — slim cache preferred, yfinance fallback ─────
    # Primary path (after first Sunday training run):
    #   slim cache (2y, weekly) + daily_closes delta (Mon–Fri, 1–4 rows/ticker)
    #   → eliminates the 2y × 900 ticker yfinance fetch entirely.
    # Fallback (slim cache not yet created or download failure):
    #   fetch_today_prices() + fetch_macro_series() from yfinance as before.
    price_data: dict[str, pd.DataFrame] = {}
    macro:      dict[str, pd.Series]    = {}

    cached_prices, cached_macro = load_price_data_from_cache(
        tickers, date_str, bucket,
    )

    if cached_prices is not None:
        price_data = cached_prices
        macro      = cached_macro or {}
        log.info("Using slim-cache + daily_closes for prices and macro")
    else:
        log.info("Slim cache unavailable — fetching from yfinance (full 2y)")
        price_data = fetch_today_prices(tickers, fd=fd)
        sector_etfs_needed = sorted({sector_map[t] for t in tickers if t in sector_map})
        macro = fetch_macro_series(extra_tickers=sector_etfs_needed)

    # Ensure all sector ETFs needed are present in macro (may be missing from
    # slim cache if a ticker's sector changed since the last bootstrap)
    sector_etfs_needed = sorted({sector_map[t] for t in tickers if t in sector_map})
    missing_etfs = [e for e in sector_etfs_needed if e not in macro]
    if missing_etfs:
        log.info("Fetching %d missing sector ETFs from yfinance: %s", len(missing_etfs), missing_etfs)
        extra = fetch_macro_series(extra_tickers=missing_etfs)
        macro.update({k: v for k, v in extra.items() if k not in macro})

    # ── Step 3c: Persist daily closes to S3 (independent price archive) ───────
    # Saves split-adjusted (auto_adjust=False) OHLCV + adj_close to:
    #   predictor/daily_closes/{date_str}.parquet
    # These files are the delta source for the slim-cache inference path.
    # adj_close = full (split + dividend) adjusted close; the ratio adj_close/close
    # captures the dividend factor and helps detect splits via sudden price jumps.
    save_daily_closes(tickers, date_str, bucket, dry_run=dry_run)

    # ── Step 4: Run inference ─────────────────────────────────────────────────
    predictions: list[dict] = []
    n_skipped = 0

    if model_type == "gbm":
        # Batch GBM path: compute features for all tickers first, then
        # cross-sectional rank-normalize, then run inference.
        # This ensures rank normalization has the full cross-section.
        from data.feature_engineer import compute_features as _compute_features

        gbm_feature_cols = cfg.GBM_FEATURES
        raw_vectors: dict[str, np.ndarray] = {}   # ticker → raw feature vector
        for ticker in tickers:
            df = price_data.get(ticker, pd.DataFrame())
            if df.empty or len(df) < cfg.MIN_ROWS_FOR_FEATURES:
                n_skipped += 1
                continue
            sector_etf_sym = sector_map.get(ticker)
            sector_etf_series = macro.get(sector_etf_sym) if sector_etf_sym else None
            try:
                featured_df = _compute_features(
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
                n_skipped += 1
                continue
            if featured_df.empty:
                n_skipped += 1
                continue
            latest = featured_df.iloc[-1]
            try:
                raw_vectors[ticker] = latest[gbm_feature_cols].to_numpy(dtype=np.float32)
            except KeyError:
                n_skipped += 1
                continue

        # Cross-sectional rank normalization across all tickers
        if raw_vectors:
            ordered_tickers = list(raw_vectors.keys())
            X_batch = np.stack([raw_vectors[t] for t in ordered_tickers])  # (N_tickers, N_features)
            n_tickers = X_batch.shape[0]
            if n_tickers > 1:
                for f in range(X_batch.shape[1]):
                    vals = X_batch[:, f]
                    order = vals.argsort()
                    ranks = np.empty_like(order, dtype=np.float32)
                    ranks[order] = np.arange(n_tickers, dtype=np.float32)
                    # Average ranks for ties
                    unique_vals, inverse = np.unique(vals, return_inverse=True)
                    if len(unique_vals) < n_tickers:
                        for uv_idx in range(len(unique_vals)):
                            mask = inverse == uv_idx
                            if mask.sum() > 1:
                                ranks[mask] = ranks[mask].mean()
                    X_batch[:, f] = ranks / max(n_tickers - 1, 1)
                log.info("Rank-normalized GBM features across %d tickers", n_tickers)
            else:
                X_batch[:, :] = 0.5  # single ticker defaults to median percentile

            # Run batch inference and build prediction dicts
            try:
                scores = scorer.predict(X_batch)  # (N_tickers,)
            except Exception as exc:
                log.error("Batch GBM inference failed: %s", exc)
                if fd:
                    fd.report(exc, severity="critical", context={
                        "site": "batch_gbm_inference",
                        "n_tickers": len(ordered_tickers),
                    })
                scores = np.full(n_tickers, np.nan)

            max_r = getattr(cfg, "LABEL_CLIP", 0.15)
            for i, ticker in enumerate(ordered_tickers):
                s = float(scores[i])
                if np.isnan(s):
                    n_skipped += 1
                    continue
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
                result = {
                    "ticker":                ticker,
                    "predicted_direction":   predicted_direction,
                    "prediction_confidence": round(confidence, 4),
                    "predicted_alpha":       round(s, 6),
                    "p_up":                  round(p_up, 4),
                    "p_flat":                round(p_flat, 4),
                    "p_down":                round(p_down, 4),
                }
                if ticker_sources:
                    result["watchlist_source"] = ticker_sources.get(ticker, "unknown")
                predictions.append(result)
    else:
        # NN path — per-ticker inference (unchanged)
        for ticker in tickers:
            df = price_data.get(ticker, pd.DataFrame())
            sector_etf_sym = sector_map.get(ticker)
            sector_etf_series = macro.get(sector_etf_sym) if sector_etf_sym else None
            result = predict_ticker(
                ticker, df, model, norm_stats,
                macro=macro,
                sector_etf_series=sector_etf_series,
            )
            if result is not None:
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
    # For GBM: read training-time meta from S3 to populate training_samples,
    # last_trained (date string), ic_30d, ic_ir_30d. Falls back gracefully if
    # meta is absent (e.g. model was never promoted through train_handler.py).
    gbm_meta: dict = {}
    if model_type == "gbm" and not local:
        try:
            import boto3 as _boto3
            _s3 = _boto3.client("s3")
            _resp = _s3.get_object(Bucket=bucket, Key=cfg.GBM_WEIGHTS_META_KEY)
            gbm_meta = json.loads(_resp["Body"].read())
            log.info("GBM weights meta loaded: trained_date=%s  n_train=%s",
                     gbm_meta.get("trained_date"), gbm_meta.get("n_train"))
        except Exception as _exc:
            log.debug("GBM weights meta not found or unreadable: %s", _exc)

    if model_type == "gbm":
        last_trained = gbm_meta.get("trained_date", scorer._best_iteration)
    else:
        last_trained = checkpoint.get("epoch", "unknown")

    metrics = {
        "model_version": model_version,
        "model_type": model_type,
        "last_trained": last_trained,
        "training_samples": gbm_meta.get("n_train") if model_type == "gbm" else None,
        "val_loss": round(float(val_loss), 6) if isinstance(val_loss, (int, float)) else None,
        # ic_30d/ic_ir_30d seeded from training-time values; overwritten by backtester
        # once 30 days of live outcome data accumulates.
        "ic_30d": gbm_meta.get("test_ic") if model_type == "gbm" else None,
        "ic_ir_30d": gbm_meta.get("ic_ir") if model_type == "gbm" else None,
        # Rolling hit rate requires 30 days of resolved outcomes (populated by backtester)
        "hit_rate_30d_rolling": None,
    }

    # ── Step 6: Resolve veto threshold (S3 override, regime-adjusted) ───────
    market_regime = signals_data.get("market_regime", "") if signals_data else ""
    veto_thresh = get_veto_threshold(bucket, market_regime=market_regime)

    # ── Step 7: Write output ──────────────────────────────────────────────────
    write_predictions(predictions, date_str, bucket, metrics, dry_run=dry_run,
                      veto_threshold=veto_thresh, fd=fd)

    # ── Step 8: Send combined morning briefing email ──────────────────────────
    if not dry_run:
        send_predictor_email(predictions, metrics, date_str, signals_data=signals_data,
                             veto_threshold=veto_thresh)

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
