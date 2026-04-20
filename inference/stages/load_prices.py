"""Stage: load_prices — Load OHLCV + macro series from ArcticDB (the same
source used by training).

The prior implementation chained S3 slim-cache → daily_closes delta → yfinance
fallbacks with a default-True silent fall-through. That chain was the root of
the 2026-04-14 silent-ImportError incident and the 2026-04-15 duplicate-row
blockage, and it held training and inference on two different canonical price
sources. Phase 7a deletes the chain in favor of a single ArcticDB read with
hard-fail semantics — matching the pattern run_inference.py already uses for
feature loading.
"""

from __future__ import annotations

import logging
import os

import arcticdb as adb  # Hard dep: matches run_inference.py pattern. No try/except
                        # ImportError — if the Lambda image lacks arcticdb, fail
                        # loud at cold start rather than silently degrading.
import pandas as pd

from alpha_engine_lib.trading_calendar import prior_trading_day

from inference.pipeline import PipelineContext, PipelineAbort

log = logging.getLogger(__name__)

# OHLCV columns in ArcticDB's universe library (title-case; matches the schema
# that alpha-engine-data's builders/backfill.py + daily_append.py write).
_ARCTIC_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Macro symbols expected in ArcticDB. Historical backfill writes these to the
# universe library with full OHLCV; daily_append writes them to the macro
# library with Close only. The reader tries universe first, falls back to
# macro, so both writers are supported.
_ARCTIC_MACRO_STEMS = ["SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO"]
_ARCTIC_SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
                       "XLP", "XLRE", "XLU", "XLV", "XLY"]


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


def _verify_arctic_fresh(macro_lib, date_str: str) -> None:
    """Assert ArcticDB's SPY close series has a row for the most recent
    CLOSED trading day relative to ``date_str``.

    The morning SF runs at 6:05 AM PT — before the market closes — so
    today's close does not yet exist. What we DO need is the most recent
    completed trading day's close (Friday's close on a Monday run). That
    is what ``prior_trading_day`` returns.

    Stricter-than-equal checks here caused the 2026-04-20 Monday abort:
    the old ``expected = pd.Timestamp(date_str).normalize()`` compared
    SPY's last_date against today's calendar date, so a Monday run with
    SPY@2026-04-17 (Friday) was flagged stale even though Friday is the
    latest close available to any consumer on Monday pre-open.

    Raises PipelineAbort on stale/missing SPY.
    """
    try:
        df = macro_lib.read("SPY", columns=["Close"]).data
    except Exception as exc:
        raise PipelineAbort(
            f"ArcticDB macro SPY unreadable: {exc} — DataPhase1 did not run "
            f"or the macro library is broken."
        ) from exc

    last_date = _safe_last_date(df.index)
    if last_date is None:
        raise PipelineAbort(
            "ArcticDB macro SPY has no rows — DataPhase1 has never written."
        )

    inference_date = pd.Timestamp(date_str).normalize()
    expected_min = pd.Timestamp(prior_trading_day(inference_date.date())).normalize()
    if last_date < expected_min:
        raise PipelineAbort(
            f"ArcticDB macro SPY last_date={last_date.date()} is stale for "
            f"inference date={inference_date.date()} (expected last_date >= "
            f"{expected_min.date()}, the most recent closed trading day). "
            f"The post-close daily-data job likely failed to update macro — "
            f"refusing to run inference on pre-last-close prices."
        )


def _connect_arctic(bucket: str) -> "tuple[object, object]":
    """Open the ArcticDB universe + macro libraries. Hard-fail on unreachable."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = f"s3s://s3.{region}.amazonaws.com:{bucket}?path_prefix=arcticdb&aws_auth=true"
    try:
        arctic = adb.Arctic(uri)
        return arctic.get_library("universe"), arctic.get_library("macro")
    except Exception as exc:
        raise PipelineAbort(
            f"ArcticDB unreachable at {uri}: {exc}"
        ) from exc


def load_price_data_from_arctic(
    tickers: list[str],
    date_str: str,
    bucket: str,
    *,
    lookback_days: int = 730,
) -> "tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]":
    """
    Read OHLCV prices + macro series from ArcticDB — the same source training uses.

    Parameters
    ----------
    tickers : list[str]
        Stocks to read from the ``universe`` library. Macro + sector ETFs are
        pulled separately from ``_ARCTIC_MACRO_STEMS`` and ``_ARCTIC_SECTOR_ETFS``;
        callers don't need to include them.
    date_str : str
        End date (inclusive) of the read window in ``YYYY-MM-DD`` form.
    bucket : str
        S3 bucket that hosts the ArcticDB path prefix.
    lookback_days : int
        Calendar-day window ending at ``date_str`` (default 730 ≈ 2y).

    Returns
    -------
    (price_data, macro):
      * ``price_data[ticker]`` — ``DataFrame(Open, High, Low, Close, Volume)``
        with ``DatetimeIndex``
      * ``macro[key]`` — ``Series`` of Close prices (``SPY``, ``VIX``, sector ETFs)

    Failure semantics
    -----------------
    * ArcticDB unreachable → ``PipelineAbort`` (hard fail). No yfinance fallback —
      upstream is canonical, and the prior chained fallbacks masked data bugs for
      days at a time (2026-04-14 silent ImportError, 2026-04-15 duplicate rows).
    * Per-ticker read error rate > 5% → ``PipelineAbort``.
    * Individual ticker missing/empty → logged WARNING and dropped from output.

    Defensive dedup
    ---------------
    The 2026-04-15 write-path fix (builders/daily_append.py → ``update()`` instead
    of ``append()``) prevents new duplicate-date rows, but historical rows may
    still carry duplicates. Each frame is deduped on read with ``keep="last"``.
    Remove after 1-2 clean Saturday cycles confirm upstream is clean.
    """
    end_ts   = pd.Timestamp(date_str).normalize()
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    universe_lib, macro_lib = _connect_arctic(bucket)

    def _read_ohlcv(lib, sym: str) -> pd.DataFrame:
        res = lib.read(sym, date_range=(start_ts, end_ts), columns=_ARCTIC_OHLCV_COLS)
        df = res.data
        if df.empty:
            return df
        return df[~df.index.duplicated(keep="last")].sort_index()

    def _read_close(lib, sym: str) -> pd.DataFrame:
        res = lib.read(sym, date_range=(start_ts, end_ts), columns=["Close"])
        df = res.data
        if df.empty:
            return df
        return df[~df.index.duplicated(keep="last")].sort_index()

    # ── Stocks: universe library only ────────────────────────────────────────
    price_data: dict[str, pd.DataFrame] = {}
    n_err = 0
    for ticker in tickers:
        try:
            df = _read_ohlcv(universe_lib, ticker)
        except Exception as exc:
            log.warning("ArcticDB universe read failed for %s: %s", ticker, exc)
            n_err += 1
            continue
        if df.empty:
            log.warning("ArcticDB universe returned empty frame for %s", ticker)
            n_err += 1
            continue
        price_data[ticker] = df

    err_rate = n_err / max(len(tickers), 1)
    if err_rate > 0.05:
        raise PipelineAbort(
            f"ArcticDB per-ticker error rate {err_rate:.1%} exceeds 5% threshold "
            f"({n_err} failed of {len(tickers)}) — treating as pipeline failure"
        )

    # ── Macros + sector ETFs: try universe first (full OHLCV from backfill),
    #     fall back to macro library (Close only from daily_append) ───────────
    all_macro_syms = _ARCTIC_MACRO_STEMS + _ARCTIC_SECTOR_ETFS
    for sym in all_macro_syms:
        df: pd.DataFrame | None = None
        try:
            df = _read_ohlcv(universe_lib, sym)
        except Exception:
            df = None  # sym not in universe library — expected for post-backfill-only macros

        if df is None or df.empty:
            try:
                df = _read_close(macro_lib, sym)
            except Exception as exc:
                log.warning("ArcticDB macro read failed for %s: %s", sym, exc)
                continue

        if df is None or df.empty:
            log.warning("ArcticDB: %s not found in universe or macro libraries", sym)
            continue

        price_data[sym] = df

    # ── Build macro dict (key → Close Series) from what we loaded ────────────
    macro: dict[str, pd.Series] = {}
    for sym in all_macro_syms:
        if sym in price_data and "Close" in price_data[sym].columns:
            s = price_data[sym]["Close"].dropna()
            if not s.empty:
                macro[sym] = s

    n_stocks = sum(1 for t in price_data.keys() if t not in all_macro_syms)
    log.info(
        "[data_source=arcticdb] Loaded %d/%d stocks, %d macro series, window %s → %s",
        n_stocks, len(tickers), len(macro),
        start_ts.date(), end_ts.date(),
    )
    return price_data, macro


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Load OHLCV + macro series from ArcticDB.

    Hard-fails on ArcticDB unreachability, stale SPY, or per-ticker read error
    rate > 5%. No yfinance fallback — upstream ArcticDB is canonical.
    """
    from inference.stages.write_output import write_predictions

    # ── ArcticDB read ────────────────────────────────────────────────────────
    ctx.price_data, ctx.macro = load_price_data_from_arctic(
        ctx.tickers, ctx.date_str, ctx.bucket,
    )

    # ── Freshness gate: SPY must have today's close in ArcticDB ──────────────
    if not ctx.dry_run:
        _, macro_lib = _connect_arctic(ctx.bucket)
        _verify_arctic_fresh(macro_lib, ctx.date_str)

    # ── Per-ticker price data age (downstream telemetry) ─────────────────────
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
        log.warning("Soft timeout after price load — writing partial predictions")
        write_predictions(ctx.predictions, ctx.date_str, ctx.bucket,
                          {"model_version": "timeout", "timed_out": True},
                          dry_run=ctx.dry_run, fd=ctx.fd)
        raise PipelineAbort("soft timeout after price load")
