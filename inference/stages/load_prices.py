"""Stage: load_prices — Load price data and macro series."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from config import SPLIT_RETURN_THRESHOLD as _SPLIT_RETURN_THRESHOLD
from inference.pipeline import PipelineContext, PipelineAbort
from inference.s3_io import _s3_put_bytes
from retry import retry

log = logging.getLogger(__name__)

_SLIM_PREFIX   = "predictor/price_cache_slim/"
_CLOSES_PREFIX = "predictor/daily_closes/"


# ── Price functions (migrated from daily_predict.py) ─────────────────────────

def _safe_last_date(idx: "pd.Index") -> "pd.Timestamp | None":
    """Return the normalized last date from a DatetimeIndex, or None if empty/NaT.

    Centralizes the NaT guard that caused the 2026-04-01 crash.  Every call
    site that previously did ``pd.Timestamp(df.index.max()).normalize()``
    should use this instead.
    """
    if idx is None or idx.empty:
        return None
    last = idx.max()
    if pd.isna(last):
        return None
    return pd.Timestamp(last).normalize()


@retry(max_attempts=2, retryable=(Exception,), label="yfinance")
def _yf_download_batch(tickers, **kwargs):
    """Download a batch of tickers from yfinance with retry."""
    import yfinance as yf
    return yf.download(tickers=tickers, **kwargs)


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
                raw = _yf_download_batch(
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
                raw = _yf_download_batch(
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
            for ticker in batch:
                result[ticker] = pd.DataFrame()

    n_success = sum(1 for df in result.values() if not df.empty)
    log.info("Price fetch complete: %d / %d succeeded", n_success, len(tickers))
    return result


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
        "VIX3M": "^VIX3M",
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


def save_daily_closes(
    tickers: list[str],
    date_str: str,
    s3_bucket: str,
    dry_run: bool = False,
) -> int:
    """
    Fetch and archive today's OHLCV for all tickers.

    **Primary writer is alpha-engine-data's collectors/daily_closes.py**, which
    runs as DataPhase1 on the EC2 micro instance before this function is
    invoked by the inference Lambda. This function exists only as a fallback
    for the case where DataPhase1 failed or has not run yet — it skips the
    write if the file already exists so it does not clobber the data module's
    richer output (full universe coverage + polygon true VWAP).

    Historical context: before alpha-engine-data was extracted as a separate
    module, this function was the sole writer. The extraction copied the
    logic into alpha-engine-data but both sides kept writing to the same S3
    path, with the predictor's lowercase-no-VWAP schema silently overwriting
    the data module's richer version every day. Caused the 2026-04-10 "no
    daily_closes with VWAP" incident.

    Writes to:
        predictor/daily_closes/{date_str}.parquet

    Schema (when this function writes): index=ticker, columns=[date, Open,
    High, Low, Close, Adj_Close, Volume, VWAP]. Matches the data module's
    uppercase schema. VWAP uses polygon's true intraday VWAP when available,
    typical-price proxy (H+L+C)/3 when falling back to yfinance.

    Parameters
    ----------
    tickers :   Tickers to capture.
    date_str :  Trading date label YYYY-MM-DD.
    s3_bucket : S3 bucket name.
    dry_run :   Log what would be written but skip the S3 put.

    Returns
    -------
    Number of tickers successfully captured (0 if skipped or no data).
    """
    import io

    import boto3

    # Skip write if data module already produced this file. Check first so
    # we don't do any polygon / yfinance work unnecessarily.
    s3 = boto3.client("s3")
    key = f"{_CLOSES_PREFIX}{date_str}.parquet"
    if not dry_run:
        try:
            s3.head_object(Bucket=s3_bucket, Key=key)
            log.info(
                "daily_closes/%s.parquet already exists — skipping "
                "(data module DataPhase1 is the primary writer)",
                date_str,
            )
            return 0
        except Exception:
            log.info(
                "daily_closes/%s.parquet not found — predictor will write "
                "as fallback (data module DataPhase1 may not have run)",
                date_str,
            )

    log.info(
        "Capturing daily closes for %d tickers, date=%s (fallback path)",
        len(tickers), date_str,
    )

    records: list[dict] = []

    # Try polygon grouped-daily first (all US stocks in 1 API call)
    try:
        from polygon_client import polygon_client
        grouped = polygon_client().get_grouped_daily(date_str)
        if grouped:
            ticker_set = set(tickers) | {t.lstrip("^") for t in tickers}
            for ticker in tickers:
                store_ticker = ticker.lstrip("^")
                g = grouped.get(store_ticker)
                if g:
                    records.append({
                        "ticker":    store_ticker,
                        "date":      date_str,
                        "Open":      round(g["open"],   4),
                        "High":      round(g["high"],   4),
                        "Low":       round(g["low"],    4),
                        "Close":     round(g["close"],  4),
                        "Adj_Close": round(g["close"],  4),  # split-adjusted only; negligible diff
                        "Volume":    int(g["volume"]),
                        "VWAP":      round(g["vwap"], 4) if g.get("vwap") else None,
                    })
            log.info("Polygon grouped-daily: %d/%d tickers captured", len(records), len(tickers))
    except Exception as exc:
        log.warning("Polygon grouped-daily failed, falling back to yfinance: %s", exc)

    # Fallback to yfinance for any tickers polygon missed
    captured_tickers = {r["ticker"] for r in records}
    missing_tickers = [t for t in tickers if t.lstrip("^") not in captured_tickers]
    if missing_tickers:
        import yfinance as yf
        log.info("Fetching %d remaining tickers from yfinance…", len(missing_tickers))
        batch_size = cfg.INFERENCE_BATCH_SIZE
        batches = [missing_tickers[i : i + batch_size] for i in range(0, len(missing_tickers), batch_size)]

        for batch in batches:
            try:
                tickers_arg = batch[0] if len(batch) == 1 else batch
                raw = yf.download(
                    tickers=tickers_arg,
                    period=cfg.DAILY_CLOSES_PERIOD,
                    interval="1d",
                    auto_adjust=False,
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
                        raw_high = float(last["High"])
                        raw_low = float(last["Low"])
                        adj_close = float(last["Adj Close"]) if "Adj Close" in df.columns else raw_close
                        # Typical-price proxy for VWAP since yfinance does not
                        # expose true intraday VWAP. Matches the proxy used in
                        # alpha-engine-data/collectors/daily_closes.py fallback.
                        vwap_proxy = round((raw_high + raw_low + raw_close) / 3.0, 4)
                        store_ticker = ticker.lstrip("^")
                        records.append({
                            "ticker":    store_ticker,
                            "date":      date_str,
                            "Open":      round(float(last["Open"]),  4),
                            "High":      round(raw_high,             4),
                            "Low":       round(raw_low,              4),
                            "Close":     round(raw_close,            4),
                            "Adj_Close": round(adj_close,            4),
                            "Volume":    int(last["Volume"]) if pd.notna(last.get("Volume")) else 0,
                            "VWAP":      vwap_proxy,
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
        _s3_put_bytes(s3, s3_bucket, key, buf.getvalue())
        log.info(
            "Daily closes written to s3://%s/%s  (%d tickers)",
            s3_bucket, key, len(closes_df),
        )
        return len(closes_df)
    except Exception as exc:
        log.error("Failed to write daily closes to S3: %s", exc)
        return 0


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

    _candidate_dates = [_safe_last_date(df.index) for df in slim_data.values()]
    _valid_dates = [d for d in _candidate_dates if d is not None]
    if not _valid_dates:
        log.warning("Slim cache loaded but all indices are empty/NaT — falling back to yfinance")
        return None, None
    slim_last_date = max(_valid_dates)
    log.info(
        "Slim cache loaded: %d tickers, last date: %s",
        len(slim_data), slim_last_date.date(),
    )

    # ── Step 3: Load daily_closes delta ──────────────────────────────────────
    ticker_rows = _load_delta_from_daily_closes(
        s3_bucket, slim_last_date, today,
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
        # Map bare macro tickers to yfinance caret format before re-fetch
        _CARET_MAP = {"VIX": "^VIX", "TNX": "^TNX", "IRX": "^IRX"}
        yf_tickers = [_CARET_MAP.get(t, t) for t in sorted(split_tickers)]
        fresh = fetch_today_prices(yf_tickers)
        # Map caret tickers back to bare keys for price_data
        for yf_sym, df in fresh.items():
            cache_key = yf_sym.lstrip("^")
            price_data[cache_key] = df

    n_success = sum(1 for df in price_data.values() if not df.empty)
    log.info(
        "Cache-based price load complete: %d/%d tickers  "
        "(%d from slim+delta, %d yfinance re-fetches)",
        n_success, len(tickers),
        n_success - len(split_tickers) + sum(1 for t in split_tickers if not price_data.get(t, pd.DataFrame()).empty),
        len(split_tickers),
    )

    # ── Step 6: Build macro dict from slim cache + delta ─────────────────────
    # These symbols are stored without caret in the training cache parquets.
    _MACRO_SLIM_KEYS = {
        "SPY": "SPY",
        "VIX": "VIX",   # stored as VIX, yfinance ticker is ^VIX
        "VIX3M": "VIX3M",  # stored as VIX3M, yfinance ticker is ^VIX3M
        "TNX": "TNX",   # stored as TNX, yfinance ticker is ^TNX
        "IRX": "IRX",
        "GLD": "GLD",
        "USO": "USO",
    }
    macro: dict[str, pd.Series] = {}
    _stale_macros: list[str] = []  # yfinance tickers to re-fetch
    for key, stem in _MACRO_SLIM_KEYS.items():
        source = price_data.get(stem) if stem in price_data else slim_data.get(stem)
        if source is not None and "Close" in source.columns:
            last_macro_date = _safe_last_date(source.index)
            if last_macro_date is None:
                log.warning("Macro series %s has empty/NaT index — skipping", key)
                continue
            if (today - last_macro_date).days > 1:
                # Macro series is stale — needs fresh data from yfinance
                _yf_sym = f"^{stem}" if stem in ("VIX", "VIX3M", "TNX", "IRX") else stem
                _stale_macros.append(_yf_sym)
            macro[key] = source["Close"].dropna()
        else:
            log.debug("Macro series %s not in slim cache", key)

    # Sector ETFs: any XL* symbols in the slim cache
    for stem, df in slim_data.items():
        if stem.startswith("XL") and "Close" in df.columns:
            source = price_data.get(stem) if stem in price_data else df
            last_etf_date = _safe_last_date(source.index)
            if last_etf_date is None:
                log.warning("Sector ETF %s has empty/NaT index — skipping", stem)
                continue
            if (today - last_etf_date).days > 1:
                _stale_macros.append(stem)
            macro[stem] = source["Close"].dropna()

    # Re-fetch stale macro series from yfinance so feature dropna doesn't
    # truncate featured_df to the slim cache's last date
    if _stale_macros:
        log.info("Re-fetching %d stale macro series from yfinance: %s", len(_stale_macros), _stale_macros[:10])
        _fresh = fetch_today_prices(_stale_macros)
        for yf_sym, fresh_df in _fresh.items():
            if fresh_df.empty:
                continue
            # Map yfinance ticker back to slim cache key
            cache_key = yf_sym.lstrip("^")
            macro_key = cache_key  # SPY→SPY, VIX→VIX, XLK→XLK
            if cache_key in _MACRO_SLIM_KEYS.values():
                # Find the macro dict key (SPY→SPY, VIX→VIX, etc.)
                macro_key = next(k for k, v in _MACRO_SLIM_KEYS.items() if v == cache_key)
            if "Close" in fresh_df.columns:
                macro[macro_key] = fresh_df["Close"].dropna()
                # Also update price_data so the delta merge benefits other tickers
                price_data[cache_key] = fresh_df
        log.info("Macro refresh complete: %d series updated", len(_fresh))

    return price_data, macro


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Load prices from slim cache or yfinance fallback, plus macro series."""
    from inference.stages.write_output import write_predictions

    # ── Try slim cache first, yfinance fallback ──────────────────────────────
    cached_prices, cached_macro = load_price_data_from_cache(
        ctx.tickers, ctx.date_str, ctx.bucket,
    )

    if cached_prices is not None:
        ctx.price_data = cached_prices
        ctx.macro = cached_macro or {}
        log.info("Using slim-cache + daily_closes for prices and macro")
    else:
        log.info("Slim cache unavailable — fetching from yfinance (full 2y)")
        ctx.price_data = fetch_today_prices(ctx.tickers, fd=ctx.fd)
        _n_ok = sum(1 for df in ctx.price_data.values() if not df.empty)
        if _n_ok == 0:
            log.error("yfinance returned zero usable price data — writing empty predictions")
            write_predictions([], ctx.date_str, ctx.bucket,
                              {"model_version": "no_price_data"},
                              dry_run=ctx.dry_run, fd=ctx.fd)
            raise PipelineAbort("zero price data from yfinance")
        sector_etfs_needed = sorted({ctx.sector_map[t] for t in ctx.tickers if t in ctx.sector_map})
        ctx.macro = fetch_macro_series(extra_tickers=sector_etfs_needed)

    # ── Compute per-ticker price data age ────────────────────────────────────
    if ctx.price_data:
        _today_ts = pd.Timestamp(ctx.date_str).normalize()
        for _tk, _df in ctx.price_data.items():
            if _df is not None and not _df.empty:
                _last_date = _safe_last_date(_df.index)
                if _last_date is not None:
                    ctx.ticker_data_age[_tk] = (_today_ts - _last_date).days
    if ctx.ticker_data_age:
        log.info(
            "Price data age: max=%d days, n_stale(>1d)=%d / %d tickers",
            max(ctx.ticker_data_age.values()),
            sum(1 for d in ctx.ticker_data_age.values() if d > 1),
            len(ctx.ticker_data_age),
        )

    # ── Timeout gate ─────────────────────────────────────────────────────────
    if ctx.near_timeout():
        log.warning("Soft timeout before sector ETF fetch — writing partial predictions")
        write_predictions(ctx.predictions, ctx.date_str, ctx.bucket,
                          {"model_version": "timeout", "timed_out": True},
                          dry_run=ctx.dry_run, fd=ctx.fd)
        raise PipelineAbort("soft timeout after price load")

    # ── Fill missing sector ETFs from yfinance ───────────────────────────────
    sector_etfs_needed = sorted({ctx.sector_map[t] for t in ctx.tickers if t in ctx.sector_map})
    missing_etfs = [e for e in sector_etfs_needed if e not in ctx.macro]
    if missing_etfs:
        log.info("Fetching %d missing sector ETFs from yfinance: %s", len(missing_etfs), missing_etfs)
        extra = fetch_macro_series(extra_tickers=missing_etfs)
        ctx.macro.update({k: v for k, v in extra.items() if k not in ctx.macro})

    # ── Persist daily closes to S3 ───────────────────────────────────────────
    _MACRO_YFINANCE = {"SPY": "SPY", "^VIX": "VIX", "^TNX": "TNX", "^IRX": "IRX", "GLD": "GLD", "USO": "USO"}
    _slim_source = ctx.price_data if ctx.price_data is not None else {}
    _sector_etfs = [s for s in _slim_source if s.startswith("XL")]
    _macro_yf_tickers = list(_MACRO_YFINANCE.keys()) + _sector_etfs
    _all_daily_tickers = sorted(set(ctx.tickers + _macro_yf_tickers))
    save_daily_closes(_all_daily_tickers, ctx.date_str, ctx.bucket, dry_run=ctx.dry_run)

    # ── Timeout gate ─────────────────────────────────────────────────────────
    if ctx.near_timeout():
        log.warning("Soft timeout before alternative data fetch — writing partial predictions")
        write_predictions(ctx.predictions, ctx.date_str, ctx.bucket,
                          {"model_version": "timeout", "timed_out": True},
                          dry_run=ctx.dry_run, fd=ctx.fd)
        raise PipelineAbort("soft timeout after daily closes")
