"""Stage: load_prices — Load price data and macro series."""

from __future__ import annotations

import logging

import pandas as pd

import config as cfg
from inference.pipeline import PipelineContext, PipelineAbort

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> None:
    """Load prices from slim cache or yfinance fallback, plus macro series."""
    from inference.daily_predict import (
        load_price_data_from_cache, fetch_today_prices, fetch_macro_series,
        save_daily_closes, write_predictions, _safe_last_date,
    )

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
