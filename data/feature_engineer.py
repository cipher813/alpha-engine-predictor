"""
data/feature_engineer.py — Rolling technical feature computation.

Mirrors compute_technical_indicators() from alpha-engine-research exactly,
but operates on a full OHLCV DataFrame (rolling window per row) rather than
returning a single snapshot dict. Every row in the output has a complete
8-feature vector. Rows lacking sufficient price history for a given indicator
are dropped after all features are computed.

Feature list (must stay in sync with config.FEATURES):
    rsi_14         RSI(14), range 0–100
    macd_cross     +1 bullish cross / -1 bearish cross / 0 no cross (last 3 days)
    macd_above_zero  1.0 if MACD line > 0, else 0.0
    macd_line_last   raw MACD line value
    price_vs_ma50    (close - ma50) / ma50
    price_vs_ma200   (close - ma200) / ma200
    momentum_20d     (close / close.shift(20)) - 1
    avg_volume_20d   20-day average volume normalized by full-series mean volume

RSI uses EWM with com=13 (equiv. to Wilder's 14-period smoothing), identical
to the research pipeline.
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

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 8 technical features for a full OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, Volume (Open/High/Low optional).
        Index should be a DatetimeIndex sorted ascending.

    Returns
    -------
    pd.DataFrame
        Original columns plus the 8 feature columns. Rows without
        sufficient history for any feature are dropped.

    Notes
    -----
    MA200 requires 200 rows of history, so effective warmup is 200 rows.
    avg_volume_20d is normalized by the mean volume of the ENTIRE series
    (same scalar applied to all rows), not a rolling mean — this keeps the
    feature stable across the training set.
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
    # (close / close.shift(20)) - 1
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
    ]
    df = df.dropna(subset=feature_cols)

    return df


def features_to_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the 8 feature columns from a featured DataFrame as a float32 array.
    Shape: (N, 8).
    """
    from config import FEATURES
    return df[FEATURES].to_numpy(dtype=np.float32)
