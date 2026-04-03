"""
feature_store/compute.py — Standalone feature computation for the full universe.

Decouples feature computation from model inference. Loads price + macro data
from S3 (slim cache + daily_closes delta), computes all 54 features for every
ticker in the universe, and writes dated Parquet snapshots to S3.

Usage:
    python -m feature_store.compute                          # today's date
    python -m feature_store.compute --date 2026-04-02        # specific date
    python -m feature_store.compute --dry-run                # compute but skip S3 write
    python -m feature_store.compute --source full             # use full 10y cache (Saturday)

Data sources:
    Prices:       predictor/price_cache_slim/*.parquet + predictor/daily_closes/{date}.parquet
    Macro:        SPY, VIX, TNX, IRX, GLD, USO, VIX3M (from slim cache)
    Sector map:   predictor/price_cache/sector_map.json
    Fundamentals: archive/fundamentals/{date}.json (cached by prior inference)
    Alt data:     market_data/weekly/{latest}/alternative/{TICKER}.json (from DataPhase2)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from data.feature_engineer import compute_features
from feature_store.registry import upload_registry
from feature_store.writer import write_feature_snapshot

log = logging.getLogger(__name__)

# Tickers that are macro/index series, not stocks
_SKIP_TICKERS = {
    "SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO",
    "^VIX", "^VIX3M", "^TNX", "^IRX",
}
# Sector ETFs to skip (not individual stocks)
_SECTOR_ETF_PREFIXES = {"XL"}


def _is_sector_etf(ticker: str) -> bool:
    return len(ticker) == 3 and ticker[:2] in _SECTOR_ETF_PREFIXES


def _load_sector_map(s3, bucket: str) -> dict[str, str]:
    """Load ticker → sector ETF mapping from S3."""
    try:
        obj = s3.get_object(Bucket=bucket, Key="predictor/price_cache/sector_map.json")
        return json.loads(obj["Body"].read())
    except Exception as exc:
        log.warning("Failed to load sector_map.json: %s", exc)
        return {}


def _load_cached_fundamentals(s3, bucket: str, date_str: str) -> dict[str, dict]:
    """Load cached fundamental data from S3 (written by prior inference)."""
    # Try exact date, then scan for most recent
    for key in [
        f"archive/fundamentals/{date_str}.json",
    ]:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            log.info("Loaded cached fundamentals from s3://%s/%s (%d tickers)", bucket, key, len(data))
            return data
        except Exception:
            pass

    # Scan for most recent fundamentals file
    try:
        resp = s3.list_objects_v2(
            Bucket=bucket, Prefix="archive/fundamentals/", MaxKeys=100,
        )
        keys = sorted(
            [c["Key"] for c in resp.get("Contents", []) if c["Key"].endswith(".json")],
            reverse=True,
        )
        if keys:
            obj = s3.get_object(Bucket=bucket, Key=keys[0])
            data = json.loads(obj["Body"].read())
            log.info("Loaded cached fundamentals from s3://%s/%s (%d tickers)", bucket, keys[0], len(data))
            return data
    except Exception as exc:
        log.warning("Failed to scan for cached fundamentals: %s", exc)

    log.info("No cached fundamentals found — fundamental features will use defaults")
    return {}


def _load_cached_alternative(s3, bucket: str) -> dict[str, dict]:
    """Load cached alternative data from the most recent DataPhase2 output."""
    try:
        # Find latest weekly date
        obj = s3.get_object(Bucket=bucket, Key="market_data/latest_weekly.json")
        latest = json.loads(obj["Body"].read())
        latest_date = latest.get("date", "")
        prefix = f"market_data/weekly/{latest_date}/alternative/"

        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=200)
        contents = resp.get("Contents", [])

        alt_data: dict[str, dict] = {}
        for item in contents:
            key = item["Key"]
            if key.endswith("manifest.json") or not key.endswith(".json"):
                continue
            ticker = key.split("/")[-1].replace(".json", "")
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                ticker_data = json.loads(obj["Body"].read())
                # Extract into the format compute_features expects
                alt_data[ticker] = {
                    "earnings": {
                        "surprise_pct": ticker_data.get("eps_revision", {}).get("surprise_pct",
                                        ticker_data.get("analyst_consensus", {}).get("surprise_pct", 0.0)),
                        "days_since_earnings": ticker_data.get("eps_revision", {}).get("days_since_earnings", 0.0),
                    },
                    "revisions": {
                        "eps_revision_4w": ticker_data.get("eps_revision", {}).get("revision_4w", 0.0),
                        "revision_streak": ticker_data.get("eps_revision", {}).get("streak", 0),
                    },
                    "options": {
                        "put_call_ratio": ticker_data.get("options_flow", {}).get("put_call_ratio"),
                        "iv_rank": ticker_data.get("options_flow", {}).get("iv_rank"),
                        "atm_iv": ticker_data.get("options_flow", {}).get("expected_move_pct"),
                    },
                }
            except Exception:
                pass

        if alt_data:
            log.info("Loaded cached alternative data for %d tickers from %s", len(alt_data), latest_date)
        return alt_data

    except Exception as exc:
        log.debug("No cached alternative data available: %s", exc)
        return {}


def _load_prices_and_macro(
    s3_bucket: str,
    date_str: str,
    source: str = "slim",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """
    Load price data and macro series from S3 cache.

    Parameters
    ----------
    s3_bucket : S3 bucket name
    date_str : Target date (YYYY-MM-DD)
    source : "slim" (2y slim cache, default) or "full" (10y full cache)

    Returns
    -------
    (price_data, macro) — dict of ticker→DataFrame, dict of key→Series
    """
    from inference.stages.load_prices import (
        load_price_data_from_cache,
        fetch_macro_series,
    )

    # Try S3 cache first (slim cache + daily_closes delta)
    log.info("Loading price data from S3 cache (source=%s)...", source)

    if source == "slim":
        # Get all tickers from slim cache — pass empty list, function loads all
        result = load_price_data_from_cache([], date_str, s3_bucket)
        if result[0] is not None:
            price_data, macro = result
            log.info("Loaded %d tickers from slim cache + delta", len(price_data))
            return price_data, macro
        log.warning("Slim cache not available, falling back to yfinance")

    elif source == "full":
        # Load full 10y cache from a local download
        from data.dataset import _load_ticker_parquet

        tmp_dir = Path(tempfile.mkdtemp()) / "full_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        import boto3
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=s3_bucket, Prefix="predictor/price_cache/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])

        log.info("Downloading %d full-cache parquets...", len(keys))
        from concurrent.futures import ThreadPoolExecutor

        def _get(key):
            fname = key.split("/")[-1]
            s3.download_file(s3_bucket, key, str(tmp_dir / fname))

        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(_get, keys))

        price_data = {}
        for pq in tmp_dir.glob("*.parquet"):
            df = _load_ticker_parquet(pq)
            if not df.empty:
                price_data[pq.stem] = df

        log.info("Loaded %d tickers from full cache", len(price_data))

        # Build macro from the price data (SPY, VIX, etc. are in the cache)
        macro = {}
        _macro_keys = {"SPY": "SPY", "VIX": "VIX", "VIX3M": "VIX3M",
                       "TNX": "TNX", "IRX": "IRX", "GLD": "GLD", "USO": "USO"}
        for key, ticker in _macro_keys.items():
            if ticker in price_data and "Close" in price_data[ticker].columns:
                macro[key] = price_data[ticker]["Close"]
        # Sector ETFs
        for ticker, df in price_data.items():
            if _is_sector_etf(ticker) and "Close" in df.columns:
                macro[ticker] = df["Close"]

        return price_data, macro

    # Fallback: fetch from yfinance (slow, last resort)
    log.warning("Falling back to yfinance for price + macro data")
    from inference.stages.load_prices import fetch_today_prices, fetch_macro_series

    # We need a ticker list — load constituents from S3
    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key="market_data/latest_weekly.json")
        latest = json.loads(obj["Body"].read())
        latest_date = latest.get("date", "")
        obj2 = s3.get_object(
            Bucket=s3_bucket,
            Key=f"market_data/weekly/{latest_date}/constituents.json",
        )
        const = json.loads(obj2["Body"].read())
        tickers = const.get("tickers", [])
    except Exception:
        tickers = []
        log.error("Cannot determine ticker universe — no constituents in S3")

    if tickers:
        price_data = fetch_today_prices(tickers)
        macro = fetch_macro_series()
        return price_data, macro

    return {}, {}


def compute_and_write(
    date_str: str,
    bucket: str = cfg.S3_BUCKET,
    source: str = "slim",
    dry_run: bool = False,
) -> dict:
    """
    Compute all 54 features for the full universe and write to S3.

    Returns summary dict with counts and timing.
    """
    import boto3

    s3 = boto3.client("s3")
    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    price_data, macro = _load_prices_and_macro(bucket, date_str, source)
    if not price_data:
        log.error("No price data loaded — cannot compute features")
        return {"status": "error", "error": "no_price_data"}

    sector_map = _load_sector_map(s3, bucket)
    fundamentals = _load_cached_fundamentals(s3, bucket, date_str)
    alt_data = _load_cached_alternative(s3, bucket)

    t_load = time.time() - t0
    log.info(
        "Data loaded in %.1fs: %d tickers, %d macro series, %d sector mappings, "
        "%d fundamentals, %d alt data",
        t_load, len(price_data), len(macro), len(sector_map),
        len(fundamentals), len(alt_data),
    )

    # ── 2. Compute features for each ticker ──────────────────────────────────
    store_rows: list[dict] = []
    n_ok = 0
    n_skip = 0
    n_err = 0

    # Filter to stock tickers only
    universe_tickers = [
        t for t in price_data
        if t not in _SKIP_TICKERS
        and not _is_sector_etf(t)
        and price_data[t] is not None
        and len(price_data[t]) >= cfg.MIN_ROWS_FOR_FEATURES
    ]

    log.info("Computing features for %d tickers...", len(universe_tickers))

    # Extract macro series once
    spy_series = macro.get("SPY")
    vix_series = macro.get("VIX")
    tnx_series = macro.get("TNX")
    irx_series = macro.get("IRX")
    gld_series = macro.get("GLD")
    uso_series = macro.get("USO")
    vix3m_series = macro.get("VIX3M")

    for ticker in universe_tickers:
        try:
            df = price_data[ticker]
            sector_etf_sym = sector_map.get(ticker)
            sector_etf_series = macro.get(sector_etf_sym) if sector_etf_sym else None

            # Get alt data for this ticker (if available)
            ticker_alt = alt_data.get(ticker, {})
            earnings_data = ticker_alt.get("earnings")
            revision_data = ticker_alt.get("revisions")
            options_data = ticker_alt.get("options")
            fundamental_data = fundamentals.get(ticker)

            featured_df = compute_features(
                df,
                spy_series=spy_series,
                vix_series=vix_series,
                sector_etf_series=sector_etf_series,
                tnx_series=tnx_series,
                irx_series=irx_series,
                gld_series=gld_series,
                uso_series=uso_series,
                vix3m_series=vix3m_series,
                earnings_data=earnings_data,
                revision_data=revision_data,
                options_data=options_data,
                fundamental_data=fundamental_data,
            )

            if featured_df.empty:
                n_skip += 1
                continue

            latest = featured_df.iloc[-1]
            row = {"ticker": ticker}
            for f in cfg.FEATURES:
                val = latest[f] if f in latest.index else 0.0
                row[f] = float(val) if pd.notna(val) else 0.0
            store_rows.append(row)
            n_ok += 1

        except Exception as exc:
            log.debug("Feature computation failed for %s: %s", ticker, exc)
            n_err += 1

    t_compute = time.time() - t0 - t_load
    log.info(
        "Feature computation complete in %.1fs: %d OK, %d skipped, %d errors",
        t_compute, n_ok, n_skip, n_err,
    )

    if not store_rows:
        log.error("No features computed — nothing to write")
        return {"status": "error", "error": "no_features_computed"}

    # ── 3. Write to S3 ───────────────────────────────────────────────────────
    features_df = pd.DataFrame(store_rows)

    if dry_run:
        log.info(
            "[dry-run] Would write feature snapshot: %d tickers, %d features, date=%s",
            len(features_df), len(cfg.FEATURES), date_str,
        )
        summary = {"groups": {g: len(features_df) for g in ["technical", "macro", "interaction", "alternative", "fundamental"]}}
    else:
        summary = write_feature_snapshot(
            date_str, features_df, bucket,
            prefix=cfg.FEATURE_STORE_PREFIX,
        )
        upload_registry(bucket, prefix=cfg.FEATURE_STORE_PREFIX)
        log.info("Feature snapshot + registry written to s3://%s/%s%s/", bucket, cfg.FEATURE_STORE_PREFIX, date_str)

    t_total = time.time() - t0

    result = {
        "status": "ok",
        "date": date_str,
        "tickers_computed": n_ok,
        "tickers_skipped": n_skip,
        "tickers_errored": n_err,
        "groups_written": summary,
        "load_seconds": round(t_load, 1),
        "compute_seconds": round(t_compute, 1),
        "total_seconds": round(t_total, 1),
        "dry_run": dry_run,
    }

    log.info("Feature store compute complete: %s", json.dumps(result, default=str))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute and write feature store snapshots to S3",
    )
    parser.add_argument(
        "--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Target date (YYYY-MM-DD, default: today UTC)",
    )
    parser.add_argument(
        "--source", choices=["slim", "full"], default="slim",
        help="Price data source: slim (2y cache, default) or full (10y cache)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute features but skip S3 write",
    )
    parser.add_argument(
        "--bucket", default=cfg.S3_BUCKET,
        help=f"S3 bucket (default: {cfg.S3_BUCKET})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    result = compute_and_write(
        date_str=args.date,
        bucket=args.bucket,
        source=args.source,
        dry_run=args.dry_run,
    )

    if result["status"] != "ok":
        log.error("Feature compute failed: %s", result.get("error"))
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
