"""
data/feature_engineer.py — Rolling technical feature computation.

Mirrors compute_technical_indicators() from alpha-engine-research exactly,
but operates on a full OHLCV DataFrame (rolling window per row) rather than
returning a single snapshot dict. Every row in the output has a complete
21-feature vector. Rows lacking sufficient price history for any indicator
are dropped after all features are computed.

Feature list (must stay in sync with config.FEATURES):
    rsi_14              RSI(14), range 0–100
    macd_cross          +1 bullish cross / -1 bearish cross / 0 no cross (last 3 days)
    macd_above_zero     1.0 if MACD line > 0, else 0.0
    macd_line_last      raw MACD line value
    price_vs_ma50       (close - ma50) / ma50
    price_vs_ma200      (close - ma200) / ma200
    momentum_20d        (close / close.shift(20)) - 1
    avg_volume_20d      20-day average volume normalized by full-series mean volume
    dist_from_52w_high  (close - 252d rolling max) / 252d rolling max      [v1.1]
    momentum_5d         (close / close.shift(5)) - 1                        [v1.1]
    rel_volume_ratio    volume / 20d rolling mean volume                     [v1.1]
    return_vs_spy_5d    momentum_5d(stock) - momentum_5d(SPY)               [v1.1]
    vix_level           VIX Close / 20.0 (1.0 = neutral, >1 = elevated fear)[v1.2]
    dist_from_52w_low   (close - 252d rolling min) / 252d rolling min       [v1.2]
    vol_ratio_10_60     10d realized vol / 60d realized vol                  [v1.2]
    bollinger_pct       (close - lower_bb20) / (upper_bb20 - lower_bb20)    [v1.2]
    sector_vs_spy_5d    sector ETF 5d return - SPY 5d return                [v1.2]
    yield_10y           10Y Treasury yield / 10.0 (rate level)              [v1.3]
    yield_curve_slope   (10Y yield - 3M yield) / 10.0 (>0 normal, <0 inv.) [v1.3]
    gold_mom_5d         5d momentum of GLD (risk-off indicator)             [v1.3]
    oil_mom_5d          5d momentum of USO (commodity cycle / inflation)    [v1.3]

RSI uses EWM with com=13 (equiv. to Wilder's 14-period smoothing), identical
to the research pipeline.

Optional series parameters (all default to neutral values when None):
    spy_series        SPY Close prices (DatetimeIndex) — return_vs_spy_5d + sector_vs_spy_5d
    vix_series        VIX Close prices (DatetimeIndex) — vix_level
    sector_etf_series Sector ETF Close prices (DatetimeIndex) — sector_vs_spy_5d
    tnx_series        10Y Treasury yield (DatetimeIndex, percent) — yield_10y, yield_curve_slope
    irx_series        3M T-bill yield (DatetimeIndex, percent) — yield_curve_slope
    gld_series        GLD Close prices (DatetimeIndex) — gold_mom_5d
    uso_series        USO Close prices (DatetimeIndex) — oil_mom_5d
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI using EWM (com = period - 1).
    Matches research's compute_technical_indicators() exactly.
    Returns a Series of RSI values, same index as close.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """
    Standard MACD. Returns (macd_line, signal_line) as Series.
    All EWM calls use adjust=False to match research.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _macd_cross_series(macd_line: pd.Series, signal_line: pd.Series, window: int = 3) -> pd.Series:
    """
    For each row, determine whether a MACD cross occurred in the last `window` days.
    Returns +1 (bullish), -1 (bearish), or 0 (no cross) as a float Series.

    A bullish cross at row t means: macd was below signal at t-1 and at or above at t.
    A bearish cross at row t means: macd was above signal at t-1 and at or below at t.
    We look back `window` rows and report the most recent cross signal.
    """
    diff = macd_line - signal_line
    cross = pd.Series(0.0, index=macd_line.index)

    prev_diff = diff.shift(1)
    # Bullish: previously negative, now non-negative
    bullish = (prev_diff < 0) & (diff >= 0)
    # Bearish: previously positive, now non-positive
    bearish = (prev_diff > 0) & (diff <= 0)

    # Rolling look-back: if any of the last `window` rows had a cross, report it
    # Most recent cross in the window takes precedence
    result = pd.Series(0.0, index=macd_line.index)
    for offset in range(window - 1, -1, -1):
        result[bullish.shift(offset, fill_value=False)] = 1.0
        result[bearish.shift(offset, fill_value=False)] = -1.0

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    spy_series: pd.Series | None = None,
    vix_series: pd.Series | None = None,
    sector_etf_series: pd.Series | None = None,
    tnx_series: pd.Series | None = None,
    irx_series: pd.Series | None = None,
    gld_series: pd.Series | None = None,
    uso_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute all 21 technical and macro features for a full OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, Volume (Open/High/Low optional).
        Index should be a DatetimeIndex sorted ascending.
    spy_series : pd.Series or None
        SPY Close prices with a DatetimeIndex. Used for return_vs_spy_5d
        and as the SPY leg of sector_vs_spy_5d. Defaults to 0.0 when None.
    vix_series : pd.Series or None
        VIX Close prices with a DatetimeIndex. Used for vix_level feature.
        Defaults to 1.0 (neutral VIX=20) when None.
    sector_etf_series : pd.Series or None
        Sector ETF Close prices (e.g. XLK for tech stocks) with a
        DatetimeIndex. Used for sector_vs_spy_5d. Defaults to 0.0 when None.
    tnx_series : pd.Series or None
        10-Year Treasury yield in percent (e.g. 4.5 for 4.5%) with a
        DatetimeIndex. Used for yield_10y and yield_curve_slope.
        Defaults to 0.4 (neutral ~4%) when None.
    irx_series : pd.Series or None
        3-Month T-bill yield in percent with a DatetimeIndex. Used as the
        short leg of yield_curve_slope. Defaults to 0.3 when None.
    gld_series : pd.Series or None
        GLD (Gold ETF) Close prices with a DatetimeIndex. Used for
        gold_mom_5d. Defaults to 0.0 when None.
    uso_series : pd.Series or None
        USO (Oil ETF) Close prices with a DatetimeIndex. Used for
        oil_mom_5d. Defaults to 0.0 when None.

    Returns
    -------
    pd.DataFrame
        Original columns plus the 21 feature columns. Rows without
        sufficient history for any feature are dropped.

    Notes
    -----
    Effective warmup is 252 rows (dist_from_52w_high / dist_from_52w_low).
    Tickers with fewer than ~265 rows should be skipped by the caller.
    avg_volume_20d is normalized by the mean volume of the ENTIRE series
    (same scalar applied to all rows) to keep the feature stable.
    Macro series (tnx, irx, gld, uso) are forward-filled on alignment so
    that weekends/holidays in the source series do not create NaNs.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # Ensure index is sorted
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    # ── RSI 14 ────────────────────────────────────────────────────────────────
    df["rsi_14"] = _compute_rsi(close, period=14)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line, signal_line = _compute_macd(close, fast=12, slow=26, signal=9)
    df["macd_line_last"] = macd_line
    df["macd_above_zero"] = (macd_line > 0).astype(float)

    # macd_cross: +1/−1/0 for a cross in the last 3 days
    diff = macd_line - signal_line
    prev_diff = diff.shift(1)
    bullish = (prev_diff < 0) & (diff >= 0)
    bearish = (prev_diff > 0) & (diff <= 0)

    # Check last 3 days: most recent cross wins
    macd_cross = pd.Series(0.0, index=df.index)
    for lag in [2, 1, 0]:  # older to newer so most recent overwrites
        macd_cross[bullish.shift(lag, fill_value=False)] = 1.0
        macd_cross[bearish.shift(lag, fill_value=False)] = -1.0
    df["macd_cross"] = macd_cross

    # ── Moving averages ───────────────────────────────────────────────────────
    ma50 = close.rolling(window=50, min_periods=50).mean()
    ma200 = close.rolling(window=200, min_periods=200).mean()

    # price_vs_ma50 / price_vs_ma200 expressed as ratio (not ×100) for ML input
    # Research uses ×100 for human-readable output; we keep raw ratio here
    # so that z-score normalization works cleanly.
    df["price_vs_ma50"] = (close - ma50) / ma50
    df["price_vs_ma200"] = (close - ma200) / ma200

    # ── 20-day momentum ───────────────────────────────────────────────────────
    df["momentum_20d"] = (close / close.shift(20)) - 1.0

    # ── Average volume (normalized) ───────────────────────────────────────────
    # 20-day rolling mean volume, then normalized by the global mean of the
    # full series. This keeps the feature on a stable scale across all rows.
    volume_global_mean = volume.mean()
    if volume_global_mean > 0:
        avg_vol_20d = volume.rolling(window=20, min_periods=20).mean()
        df["avg_volume_20d"] = avg_vol_20d / volume_global_mean
    else:
        df["avg_volume_20d"] = 1.0  # degenerate case

    # ── Distance from 52-week high ─────────────────────────────────────────────
    # Documented momentum factor: stocks near their 52w high tend to continue up.
    # Expressed as a negative ratio: 0.0 = at the high, -0.20 = 20% below.
    rolling_max_252 = close.rolling(window=252, min_periods=252).max()
    df["dist_from_52w_high"] = (close - rolling_max_252) / rolling_max_252

    # ── 5-day momentum ─────────────────────────────────────────────────────────
    # Short-term reversal / continuation signal.
    df["momentum_5d"] = (close / close.shift(5)) - 1.0

    # ── Relative volume ratio ──────────────────────────────────────────────────
    # Current 20d avg volume vs. the rolling mean — flags institutional activity.
    # Differs from avg_volume_20d: this is a relative ratio, not an absolute level.
    rolling_mean_vol_20 = volume.rolling(window=20, min_periods=20).mean()
    df["rel_volume_ratio"] = volume / rolling_mean_vol_20.replace(0, float("nan"))
    df["rel_volume_ratio"] = df["rel_volume_ratio"].fillna(1.0)

    # ── Return vs SPY (5-day relative strength) ────────────────────────────────
    # Stock's 5d momentum minus SPY's 5d momentum on the same date.
    # Captures stock-specific alpha rather than market-wide moves.
    spy_mom_5d: pd.Series | None = None
    if spy_series is not None:
        spy_aligned = spy_series.reindex(df.index)
        spy_mom_5d = (spy_aligned / spy_aligned.shift(5)) - 1.0
        df["return_vs_spy_5d"] = df["momentum_5d"] - spy_mom_5d
    else:
        df["return_vs_spy_5d"] = 0.0

    # ── v1.2 features ──────────────────────────────────────────────────────────

    # VIX level: normalised around the long-run average (~20). Values > 1 = fear.
    # Forward-filled so that non-trading days in the VIX series don't create NaNs.
    if vix_series is not None:
        vix_aligned = vix_series.reindex(df.index, method="ffill")
        df["vix_level"] = vix_aligned / 20.0
    else:
        df["vix_level"] = 1.0  # neutral (VIX ≈ 20)

    # Distance from 52-week low — symmetric counterpart to dist_from_52w_high.
    # Positive ratio: 0.0 = exactly at the low, larger = further above.
    rolling_min_252 = close.rolling(window=252, min_periods=252).min()
    df["dist_from_52w_low"] = (close - rolling_min_252) / rolling_min_252

    # Historical volatility ratio (10d / 60d). > 1 = vol expansion (breakout),
    # < 1 = vol compression (consolidating). Capped to avoid extreme outliers.
    log_ret = np.log(close / close.shift(1))
    vol_10d = log_ret.rolling(window=10, min_periods=10).std() * np.sqrt(252)
    vol_60d = log_ret.rolling(window=60, min_periods=60).std() * np.sqrt(252)
    df["vol_ratio_10_60"] = (vol_10d / vol_60d.replace(0, float("nan"))).fillna(1.0)

    # Bollinger band position: 0 = at lower band, 0.5 = at mid, 1 = at upper band.
    # Values outside [0, 1] are possible (close outside bands).
    ma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    upper_bb = ma20 + 2.0 * std20
    lower_bb = ma20 - 2.0 * std20
    band_width = (upper_bb - lower_bb).replace(0, float("nan"))
    df["bollinger_pct"] = ((close - lower_bb) / band_width).fillna(0.5)

    # Sector ETF vs SPY (5-day): captures sector rotation relative to the market.
    # Positive = sector outperforming SPY; negative = sector underperforming.
    if sector_etf_series is not None and spy_mom_5d is not None:
        sec_aligned = sector_etf_series.reindex(df.index)
        sec_mom_5d = (sec_aligned / sec_aligned.shift(5)) - 1.0
        df["sector_vs_spy_5d"] = sec_mom_5d - spy_mom_5d
    elif sector_etf_series is not None:
        # SPY not available — compute sector momentum only (no relative adjustment)
        sec_aligned = sector_etf_series.reindex(df.index)
        df["sector_vs_spy_5d"] = (sec_aligned / sec_aligned.shift(5)) - 1.0
    else:
        df["sector_vs_spy_5d"] = 0.0  # no sector info: neutral signal

    # ── v1.3 features — macro regime ──────────────────────────────────────────

    # yield_10y: 10-year Treasury yield normalized to a 0–1 scale.
    # TNX from yfinance is in percent (e.g. 4.5 for 4.5%); divide by 10
    # so the typical 0–10% range maps to 0.0–1.0.
    # Forward-fill so bond market holidays don't create equity-day NaNs.
    if tnx_series is not None:
        tnx_aligned = tnx_series.reindex(df.index, method="ffill")
        df["yield_10y"] = tnx_aligned / 10.0
    else:
        df["yield_10y"] = 0.4  # neutral: ~4% 10Y yield

    # yield_curve_slope: spread between 10Y and 3M yields, normalized.
    # Positive = normal/steep curve; negative = inverted (recession signal).
    # IRX from yfinance is the 13-week T-bill annualized yield in percent.
    if tnx_series is not None and irx_series is not None:
        irx_aligned = irx_series.reindex(df.index, method="ffill")
        df["yield_curve_slope"] = (tnx_aligned - irx_aligned) / 10.0
    elif tnx_series is not None:
        # No short rate — use yield level as a rough slope proxy (vs zero)
        df["yield_curve_slope"] = tnx_aligned / 10.0
    else:
        df["yield_curve_slope"] = 0.0  # neutral: assume normal curve

    # gold_mom_5d: 5-day momentum of GLD. Rising gold = risk-off / inflation
    # fears; falling = risk-on or deflationary. Market-wide signal applied to
    # all tickers; z-score normalization lets the model weight it appropriately.
    if gld_series is not None:
        gld_aligned = gld_series.reindex(df.index, method="ffill")
        df["gold_mom_5d"] = (gld_aligned / gld_aligned.shift(5)) - 1.0
        df["gold_mom_5d"] = df["gold_mom_5d"].fillna(0.0)
    else:
        df["gold_mom_5d"] = 0.0

    # oil_mom_5d: 5-day momentum of USO (WTI crude proxy).
    # Rising oil = inflationary pressure / energy sector tailwind;
    # falling = deflationary / energy headwind. Particularly signal-rich
    # for energy, industrials, and consumer discretionary sectors.
    if uso_series is not None:
        uso_aligned = uso_series.reindex(df.index, method="ffill")
        df["oil_mom_5d"] = (uso_aligned / uso_aligned.shift(5)) - 1.0
        df["oil_mom_5d"] = df["oil_mom_5d"].fillna(0.0)
    else:
        df["oil_mom_5d"] = 0.0

    # ── Drop rows with any NaN in the feature columns ─────────────────────────
    feature_cols = [
        "rsi_14",
        "macd_cross",
        "macd_above_zero",
        "macd_line_last",
        "price_vs_ma50",
        "price_vs_ma200",
        "momentum_20d",
        "avg_volume_20d",
        "dist_from_52w_high",
        "momentum_5d",
        "rel_volume_ratio",
        "return_vs_spy_5d",
        "vix_level",
        "dist_from_52w_low",
        "vol_ratio_10_60",
        "bollinger_pct",
        "sector_vs_spy_5d",
        "yield_10y",
        "yield_curve_slope",
        "gold_mom_5d",
        "oil_mom_5d",
    ]
    df = df.dropna(subset=feature_cols)

    return df


def features_to_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the 17 feature columns from a featured DataFrame as a float32 array.
    Shape: (N, 17).
    """
    from config import FEATURES
    return df[FEATURES].to_numpy(dtype=np.float32)
