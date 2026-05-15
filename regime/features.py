"""
regime/features.py — Macro feature fetcher for the regime substrate.

Reads the FRED-historical parquets that alpha-engine-data writes to
``s3://alpha-engine-research/predictor/price_cache/`` and assembles them
into the weekly feature DataFrame the HMM + composite + BOCPD consume.

Source-of-truth boundary: features (raw + derived series) live in
alpha-engine-data; models live here in alpha-engine-predictor. This
fetcher is a model-side consumer that reads data-side artifacts — no
fetcher logic duplicates collection / refresh / repair work that the
data side already owns (FRED ingestion, validation, repair scripts).

Feature inventory
-----------------
Six macros land in the regime substrate. Five are sourced here in PR 2;
``market_breadth`` requires per-ticker computation outside the FRED
parquets and is folded in by a follow-up PR when the historical breadth
time series lands.

- ``spy_20d_return``    derived from ``SPY.parquet`` Close column
- ``vix_level``         from ``VIX.parquet`` Close (FRED VIXCLS via
                        ``daily_closes._FRED_INDEX_MAP`` since the
                        Stage 2.5 cutover)
- ``vix_term_slope``    ``VIX3M.parquet`` Close − ``VIX.parquet`` Close
                        (term-structure spread)
- ``hy_oas_bps``        ``HYOAS.parquet`` Close × 100 (FRED publishes
                        decimal percent; convert to basis points)
- ``yield_curve_slope`` ``TNX.parquet`` Close − ``TWO.parquet`` Close
                        (10y minus 2y in percent)

All inputs already align to weekly cadence via the data side's
weekly-collector aggregation; here we re-confirm by resampling to
Friday-anchored weeks and forward-filling stale-from-holiday values.
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Mapping

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_S3_BUCKET: str = "alpha-engine-research"
DEFAULT_PRICE_CACHE_PREFIX: str = "predictor/price_cache/"


# Source tickers expected as ``{prefix}{ticker}.parquet`` keys. Mirror
# data/bootstrap_fetcher.py's _CARET_SYMBOLS convention — stored without
# the yfinance leading caret. FRED-sourced symbols (HYOAS, TWO) come
# from collectors/fred_history.py per the Stage 2.5b cutover.
SOURCE_TICKERS: tuple[str, ...] = ("SPY", "VIX", "VIX3M", "TNX", "TWO", "HYOAS")


def _read_parquet_close(
    ticker: str,
    *,
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> pd.Series:
    """Read ``{prefix}{ticker}.parquet`` from S3 and return its Close
    column as a Series indexed by date. Returns an empty Series when
    the key is missing — callers decide whether absence is fatal."""
    key = f"{prefix}{ticker}.parquet"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        logger.warning(
            "[regime_features] missing %s at s3://%s/%s (%s) — returning empty series",
            ticker, bucket, key, type(e).__name__,
        )
        return pd.Series(name=ticker, dtype="float64")

    df = pd.read_parquet(BytesIO(obj["Body"].read()), engine="pyarrow")
    if "Close" not in df.columns:
        logger.warning("[regime_features] %s parquet missing Close column", ticker)
        return pd.Series(name=ticker, dtype="float64")
    s = df["Close"].copy()
    s.name = ticker
    # Normalize DatetimeIndex — the data side writes either a DatetimeIndex
    # or a 'date' column; handle both defensively.
    if not isinstance(s.index, pd.DatetimeIndex):
        if "date" in df.columns:
            s.index = pd.to_datetime(df["date"])
        else:
            s.index = pd.to_datetime(df.index)
    return s.sort_index()


def fetch_macro_feature_history(
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_PRICE_CACHE_PREFIX,
    lookback_weeks: int = 520,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the weekly macro feature DataFrame from price-cache parquets.

    Parameters
    ----------
    s3_client
        boto3 S3 client (or test stub with ``get_object``).
    bucket, prefix
        S3 location of the per-ticker parquets.
    lookback_weeks
        Trim to the most recent N weeks. Default 520 (~10y).
    as_of
        Truncate the right edge of the series to this timestamp. Used
        by the backfill script (Stage A PR 4) to reproduce historical
        feature snapshots; ``None`` = use all available data.

    Returns
    -------
    DataFrame indexed by weekly Friday timestamp with columns matching
    ``regime/substrate.py:REGIME_FEATURES`` (excluding ``market_breadth``
    which is filed by a follow-up PR). Forward-filled across any
    single-week gaps so the HMM never sees NaN rows.
    """
    close_series: dict[str, pd.Series] = {}
    for ticker in SOURCE_TICKERS:
        s = _read_parquet_close(ticker, s3_client=s3_client, bucket=bucket, prefix=prefix)
        if not s.empty and as_of is not None:
            s = s.loc[:as_of]
        close_series[ticker] = s

    # Align to a common daily date index then resample to weekly Friday.
    # Only concat non-empty series — an empty Series has no DatetimeIndex
    # and would poison the combined index. Missing tickers are
    # re-attached as all-NaN columns post-concat so downstream code can
    # still reference the column name.
    non_empty = {t: s for t, s in close_series.items() if not s.empty}
    if not non_empty:
        return pd.DataFrame(columns=list(_REGIME_FEATURE_LIST))
    daily = pd.concat(non_empty, axis=1).sort_index()
    for ticker in SOURCE_TICKERS:
        if ticker not in daily.columns:
            daily[ticker] = np.nan
    if daily.empty:
        return pd.DataFrame(columns=list(_REGIME_FEATURE_LIST))
    # Forward-fill across NYSE holidays (FRED weekend NaNs etc.) before
    # resampling so the weekly snapshot has the last-observed value.
    daily = daily.ffill()
    weekly = daily.resample("W-FRI").last()

    # Derive features
    features = pd.DataFrame(index=weekly.index)
    spy = weekly["SPY"]
    features["spy_20d_return"] = spy.pct_change(periods=4, fill_method=None)  # ~4 weeks ≈ 20 trading days
    features["vix_level"] = weekly["VIX"]
    features["vix_term_slope"] = weekly["VIX3M"] - weekly["VIX"]
    features["hy_oas_bps"] = weekly["HYOAS"] * 100.0  # FRED publishes decimal percent
    features["yield_curve_slope"] = weekly["TNX"] - weekly["TWO"]

    # market_breadth not yet sourced — fill with NaN. Composite z-score
    # gracefully handles missing features; HMM is configured at fit-time
    # with the subset of features actually populated.
    features["market_breadth"] = np.nan

    # Drop rows where the SPY-derived 20d return is undefined (the first
    # 4 weekly observations).
    features = features.dropna(subset=["spy_20d_return"])

    if lookback_weeks is not None and len(features) > lookback_weeks:
        features = features.tail(lookback_weeks).copy()

    logger.info(
        "[regime_features] assembled weekly history: %d rows %s → %s",
        len(features),
        features.index.min().date() if len(features) else None,
        features.index.max().date() if len(features) else None,
    )
    return features


def fetch_macro_feature_history_daily(
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_PRICE_CACHE_PREFIX,
    lookback_days: int | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Daily-cadence sibling of ``fetch_macro_feature_history``.

    The weekly fetcher already builds a daily frame internally and then
    downsamples (``resample("W-FRI")``). The daily fast signal (Stage F1
    of regime-fast-signal-260515.md) needs the *pre-downsample* series
    to recalibrate the daily BOCPD hazard on a 10y backfill, so this
    function returns the same five features on the daily grid:

    - ``spy_20d_return`` uses a **20-trading-day** window
      (``pct_change(20)``) — the daily analog of the weekly fetcher's
      ``pct_change(periods=4)`` (~4 weeks).
    - the FRED-sourced levels/spreads are taken on the daily index
      directly (no resample).

    The live weekly substrate path is untouched — this is an additive,
    backfill/eval-only reader. ``market_breadth`` is filled NaN
    identically to the weekly fetcher (composite handles missing
    features gracefully).
    """
    close_series: dict[str, pd.Series] = {}
    for ticker in SOURCE_TICKERS:
        s = _read_parquet_close(ticker, s3_client=s3_client, bucket=bucket, prefix=prefix)
        if not s.empty and as_of is not None:
            s = s.loc[:as_of]
        close_series[ticker] = s

    non_empty = {t: s for t, s in close_series.items() if not s.empty}
    if not non_empty:
        return pd.DataFrame(columns=list(_REGIME_FEATURE_LIST))
    daily = pd.concat(non_empty, axis=1).sort_index()
    for ticker in SOURCE_TICKERS:
        if ticker not in daily.columns:
            daily[ticker] = np.nan
    if daily.empty:
        return pd.DataFrame(columns=list(_REGIME_FEATURE_LIST))
    daily = daily.ffill()

    features = pd.DataFrame(index=daily.index)
    spy = daily["SPY"]
    features["spy_20d_return"] = spy.pct_change(periods=20, fill_method=None)
    features["vix_level"] = daily["VIX"]
    features["vix_term_slope"] = daily["VIX3M"] - daily["VIX"]
    features["hy_oas_bps"] = daily["HYOAS"] * 100.0
    features["yield_curve_slope"] = daily["TNX"] - daily["TWO"]
    features["market_breadth"] = np.nan

    features = features.dropna(subset=["spy_20d_return"])
    if lookback_days is not None and len(features) > lookback_days:
        features = features.tail(lookback_days).copy()

    logger.info(
        "[regime_features] assembled DAILY history: %d rows %s → %s",
        len(features),
        features.index.min().date() if len(features) else None,
        features.index.max().date() if len(features) else None,
    )
    return features


def current_features_from_history(
    history: pd.DataFrame,
    extra: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Extract the most recent observation's feature row from history.

    Optional ``extra`` mapping is merged on top — used by the handler
    to inject ``spy_30d_return`` (read separately from the SPY close
    series) into the dict that the guardrail-flag computation consumes,
    without polluting the HMM training matrix.
    """
    if history.empty:
        raise ValueError("Empty feature history — cannot extract current row.")
    last = history.iloc[-1].to_dict()
    # Drop NaN-valued keys so consumers can rely on np.isfinite() checks
    cleaned = {k: float(v) for k, v in last.items() if v is not None and np.isfinite(v)}
    if extra:
        for k, v in extra.items():
            if v is not None and np.isfinite(float(v)):
                cleaned[k] = float(v)
    return cleaned


# Local mirror of ``regime/substrate.py:REGIME_FEATURES`` — kept here as
# a tuple constant so this module's tests don't have to import substrate
# (avoiding circular import + keeping the contract explicit at the
# feature-fetcher boundary). The substrate module's import is the
# authoritative one.
_REGIME_FEATURE_LIST: tuple[str, ...] = (
    "spy_20d_return",
    "vix_level",
    "vix_term_slope",
    "hy_oas_bps",
    "yield_curve_slope",
    "market_breadth",
)
