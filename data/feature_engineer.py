"""
data/feature_engineer.py — Rolling technical feature computation.

Mirrors compute_technical_indicators() from alpha-engine-research exactly,
but operates on a full OHLCV DataFrame (rolling window per row) rather than
returning a single snapshot dict. Every row in the output has a complete
feature vector. Rows lacking sufficient price history for any indicator
are dropped after all features are computed.

Feature groups:
  v1.0-v1.5: 34 price/volume/macro features (29 core + 5 regime interactions)
  v2.0 (O10-O12): 7 alternative data features (earnings, revisions, options)

Feature list (must stay in sync with config.FEATURES):
    rsi_14              RSI(14), range 0–100
    macd_cross          +1 bullish / -1 bearish / 0 no cross (last 3 days)
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
    price_accel         momentum_5d - momentum_20d (acceleration)           [v1.4]
    ema_cross_8_21      EMA(8) / EMA(21) - 1 (short vs medium trend)       [v1.4]
    atr_14_pct          ATR(14) / close (normalized volatility)             [v1.4]
    realized_vol_20d    20d realized vol annualized (√252 scaled)           [v1.4]
    volume_trend        SMA(vol,5) / SMA(vol,20) (short-term vol surge)    [v1.4]
    obv_slope_10d       (OBV_fast - OBV_slow) / SMA(vol,20) (accumulation) [v1.4]
    rsi_slope_5d        (RSI - RSI.shift(5)) / 5 (RSI momentum)            [v1.4]
    volume_price_div    sign(volume_trend-1) * sign(momentum_5d)            [v1.4]

  v2.0 — Alternative data features (O10-O12):
    earnings_surprise_pct  most recent quarterly EPS surprise %               [O10]
    days_since_earnings    days since last earnings (0-1, capped 90d)         [O10]
    eps_revision_4w        4-week cumulative EPS revision %                   [O11]
    revision_streak        consecutive weeks same-direction revisions         [O11]
    put_call_ratio         log-transformed put/call OI ratio                  [O12]
    iv_rank                IV percentile rank (0-1)                           [O12]
    iv_vs_rv               implied vol / realized vol ratio                   [O12]

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

from config import FEATURE_CFG as _FC


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int | None = None) -> pd.Series:
    """
    Wilder's RSI using EWM (com = period - 1).
    Matches research's compute_technical_indicators() exactly.
    Returns a Series of RSI values, same index as close.
    """
    if period is None:
        period = _FC["rsi_period"]
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
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Standard MACD. Returns (macd_line, signal_line) as Series.
    All EWM calls use adjust=False to match research.
    """
    if fast is None:
        fast = _FC["macd_fast"]
    if slow is None:
        slow = _FC["macd_slow"]
    if signal is None:
        signal = _FC["macd_signal"]
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
    earnings_data: dict | None = None,
    revision_data: dict | None = None,
    options_data: dict | None = None,
    fundamental_data: dict | None = None,
) -> pd.DataFrame:
    """
    Compute all technical, macro, and alternative data features for a full OHLCV DataFrame.

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
    earnings_data : dict or None
        Per-ticker earnings data with keys: surprise_pct, days_since_earnings.
        Used for O10 PEAD features. Defaults to neutral when None.
    revision_data : dict or None
        Per-ticker revision data with keys: eps_revision_4w, revision_streak.
        Used for O11 revision features. Defaults to neutral when None.
    options_data : dict or None
        Per-ticker options data with keys: put_call_ratio, iv_rank, atm_iv.
        Used for O12 options features. Defaults to neutral when None.
    fundamental_data : dict or None
        Per-ticker fundamental ratios with keys: pe_ratio, pb_ratio,
        debt_to_equity, revenue_growth_yoy, fcf_yield, gross_margin, roe,
        current_ratio. Pre-normalized by fundamental_fetcher. Defaults to
        neutral (0.0) when None.

    Returns
    -------
    pd.DataFrame
        Original columns plus feature columns. Rows without sufficient
        history for any feature are dropped.

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

    # ── RSI ─────────────────────────────────────────────────────────────────
    df["rsi_14"] = _compute_rsi(close)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line, signal_line = _compute_macd(close)
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
    _ma_short = _FC["ma_short"]
    _ma_long = _FC["ma_long"]
    ma50 = close.rolling(window=_ma_short, min_periods=_ma_short).mean()
    ma200 = close.rolling(window=_ma_long, min_periods=_ma_long).mean()

    # price_vs_ma50 / price_vs_ma200 expressed as ratio (not ×100) for ML input
    # Research uses ×100 for human-readable output; we keep raw ratio here
    # so that z-score normalization works cleanly.
    df["price_vs_ma50"] = (close - ma50) / ma50
    df["price_vs_ma200"] = (close - ma200) / ma200

    # ── 20-day momentum ───────────────────────────────────────────────────────
    _mom_long = _FC["momentum_long"]
    df["momentum_20d"] = (close / close.shift(_mom_long)) - 1.0

    # ── Average volume (normalized) ───────────────────────────────────────────
    # 20-day rolling mean volume, then normalized by the global mean of the
    # full series. This keeps the feature on a stable scale across all rows.
    _vol_slow = _FC["volume_slow"]
    volume_global_mean = volume.mean()
    if volume_global_mean > 0:
        avg_vol_20d = volume.rolling(window=_vol_slow, min_periods=_vol_slow).mean()
        df["avg_volume_20d"] = avg_vol_20d / volume_global_mean
    else:
        df["avg_volume_20d"] = 1.0  # degenerate case

    # ── Distance from 52-week high ─────────────────────────────────────────────
    # Documented momentum factor: stocks near their 52w high tend to continue up.
    # Expressed as a negative ratio: 0.0 = at the high, -0.20 = 20% below.
    _52w = _FC["weeks_52_days"]
    rolling_max_252 = close.rolling(window=_52w, min_periods=_52w).max()
    df["dist_from_52w_high"] = (close - rolling_max_252) / rolling_max_252

    # ── 5-day momentum ─────────────────────────────────────────────────────────
    # Short-term reversal / continuation signal.
    _mom_short = _FC["momentum_short"]
    df["momentum_5d"] = (close / close.shift(_mom_short)) - 1.0

    # ── Relative volume ratio ──────────────────────────────────────────────────
    # Current 20d avg volume vs. the rolling mean — flags institutional activity.
    # Differs from avg_volume_20d: this is a relative ratio, not an absolute level.
    rolling_mean_vol_20 = volume.rolling(window=_vol_slow, min_periods=_vol_slow).mean()
    df["rel_volume_ratio"] = volume / rolling_mean_vol_20.replace(0, float("nan"))
    df["rel_volume_ratio"] = df["rel_volume_ratio"].fillna(1.0)

    # ── Return vs SPY (5-day relative strength) ────────────────────────────────
    # Stock's 5d momentum minus SPY's 5d momentum on the same date.
    # Captures stock-specific alpha rather than market-wide moves.
    spy_mom_5d: pd.Series | None = None
    if spy_series is not None:
        spy_aligned = spy_series.reindex(df.index)
        spy_mom_5d = (spy_aligned / spy_aligned.shift(_mom_short)) - 1.0
        df["return_vs_spy_5d"] = df["momentum_5d"] - spy_mom_5d
    else:
        df["return_vs_spy_5d"] = 0.0

    # ── v1.2 features ──────────────────────────────────────────────────────────

    # VIX level: normalised around the long-run average (~20). Values > 1 = fear.
    # Forward-filled so that non-trading days in the VIX series don't create NaNs.
    if vix_series is not None:
        vix_aligned = vix_series.reindex(df.index, method="ffill")
        df["vix_level"] = vix_aligned / _FC["vix_baseline"]
    else:
        df["vix_level"] = 1.0  # neutral (VIX ≈ 20)

    # Distance from 52-week low — symmetric counterpart to dist_from_52w_high.
    # Positive ratio: 0.0 = exactly at the low, larger = further above.
    rolling_min_252 = close.rolling(window=_52w, min_periods=_52w).min()
    df["dist_from_52w_low"] = (close - rolling_min_252) / rolling_min_252

    # Historical volatility ratio (10d / 60d). > 1 = vol expansion (breakout),
    # < 1 = vol compression (consolidating). Capped to avoid extreme outliers.
    _vol_short = _FC["vol_short_window"]
    _vol_long = _FC["vol_long_window"]
    log_ret = np.log(close / close.shift(1))
    vol_10d = log_ret.rolling(window=_vol_short, min_periods=_vol_short).std() * np.sqrt(_52w)
    vol_60d = log_ret.rolling(window=_vol_long, min_periods=_vol_long).std() * np.sqrt(_52w)
    df["vol_ratio_10_60"] = (vol_10d / vol_60d.replace(0, float("nan"))).fillna(1.0)

    # Bollinger band position: 0 = at lower band, 0.5 = at mid, 1 = at upper band.
    # Values outside [0, 1] are possible (close outside bands).
    _bb_win = _FC["bollinger_window"]
    _bb_std = _FC["bollinger_std"]
    ma20 = close.rolling(window=_bb_win, min_periods=_bb_win).mean()
    std20 = close.rolling(window=_bb_win, min_periods=_bb_win).std()
    upper_bb = ma20 + _bb_std * std20
    lower_bb = ma20 - _bb_std * std20
    band_width = (upper_bb - lower_bb).replace(0, float("nan"))
    df["bollinger_pct"] = ((close - lower_bb) / band_width).fillna(0.5)

    # Sector ETF vs SPY (5-day): captures sector rotation relative to the market.
    # Positive = sector outperforming SPY; negative = sector underperforming.
    if sector_etf_series is not None and spy_mom_5d is not None:
        sec_aligned = sector_etf_series.reindex(df.index)
        sec_mom_5d = (sec_aligned / sec_aligned.shift(_mom_short)) - 1.0
        df["sector_vs_spy_5d"] = sec_mom_5d - spy_mom_5d
    elif sector_etf_series is not None:
        # SPY not available — compute sector momentum only (no relative adjustment)
        sec_aligned = sector_etf_series.reindex(df.index)
        df["sector_vs_spy_5d"] = (sec_aligned / sec_aligned.shift(_mom_short)) - 1.0
    else:
        df["sector_vs_spy_5d"] = 0.0  # no sector info: neutral signal

    # ── v1.3 features — macro regime ──────────────────────────────────────────

    # yield_10y: 10-year Treasury yield normalized to a 0–1 scale.
    # TNX from yfinance is in percent (e.g. 4.5 for 4.5%); divide by 10
    # so the typical 0–10% range maps to 0.0–1.0.
    # Forward-fill so bond market holidays don't create equity-day NaNs.
    if tnx_series is not None:
        _tnx_norm = _FC["tnx_normalizer"]
        tnx_aligned = tnx_series.reindex(df.index, method="ffill")
        df["yield_10y"] = tnx_aligned / _tnx_norm
    else:
        df["yield_10y"] = 0.4  # neutral: ~4% 10Y yield

    # yield_curve_slope: spread between 10Y and 3M yields, normalized.
    # Positive = normal/steep curve; negative = inverted (recession signal).
    # IRX from yfinance is the 13-week T-bill annualized yield in percent.
    if tnx_series is not None and irx_series is not None:
        irx_aligned = irx_series.reindex(df.index, method="ffill")
        df["yield_curve_slope"] = (tnx_aligned - irx_aligned) / _tnx_norm
    elif tnx_series is not None:
        # No short rate — use yield level as a rough slope proxy (vs zero)
        df["yield_curve_slope"] = tnx_aligned / _tnx_norm
    else:
        df["yield_curve_slope"] = 0.0  # neutral: assume normal curve

    # gold_mom_5d: 5-day momentum of GLD. Rising gold = risk-off / inflation
    # fears; falling = risk-on or deflationary. Market-wide signal applied to
    # all tickers; z-score normalization lets the model weight it appropriately.
    if gld_series is not None:
        gld_aligned = gld_series.reindex(df.index, method="ffill")
        df["gold_mom_5d"] = (gld_aligned / gld_aligned.shift(_mom_short)) - 1.0
        df["gold_mom_5d"] = df["gold_mom_5d"].fillna(0.0)
    else:
        df["gold_mom_5d"] = 0.0

    # oil_mom_5d: 5-day momentum of USO (WTI crude proxy).
    # Rising oil = inflationary pressure / energy sector tailwind;
    # falling = deflationary / energy headwind. Particularly signal-rich
    # for energy, industrials, and consumer discretionary sectors.
    if uso_series is not None:
        uso_aligned = uso_series.reindex(df.index, method="ffill")
        df["oil_mom_5d"] = (uso_aligned / uso_aligned.shift(_mom_short)) - 1.0
        df["oil_mom_5d"] = df["oil_mom_5d"].fillna(0.0)
    else:
        df["oil_mom_5d"] = 0.0

    # ── v1.4 features — design doc Appendix A completions ────────────────────

    # price_accel: momentum acceleration. Positive = momentum is increasing
    # (price accelerating upward); negative = momentum is decelerating.
    df["price_accel"] = df["momentum_5d"] - df["momentum_20d"]

    # ema_cross_8_21: short vs medium-term trend alignment.
    # Positive = short-term EMA above medium-term (bullish alignment).
    ema_8 = close.ewm(span=_FC["ema_fast_span"], adjust=False).mean()
    ema_21 = close.ewm(span=_FC["ema_slow_span"], adjust=False).mean()
    df["ema_cross_8_21"] = ema_8 / ema_21 - 1.0

    # atr_14_pct: normalized average true range. Measures intraday volatility
    # independent of price level. Requires High/Low; falls back to close if
    # OHLCV data only has Close (e.g. some data providers).
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=_FC["atr_period"], adjust=False).mean()
    df["atr_14_pct"] = atr / close

    # realized_vol_20d: 20-day annualized realized volatility of daily returns.
    # Distinct from vol_ratio_10_60 which is a *ratio* of two vol windows;
    # this is the absolute volatility level itself.
    daily_returns = close.pct_change()
    df["realized_vol_20d"] = daily_returns.rolling(_FC["realized_vol_window"]).std() * np.sqrt(_52w)

    # volume_trend: ratio of short-term to medium-term average volume.
    # > 1 = volume is expanding (institutional activity); < 1 = contracting.
    _vf = _FC["volume_fast"]
    vol_5 = volume.rolling(_vf).mean()
    vol_20 = volume.rolling(_vol_slow).mean()
    df["volume_trend"] = (vol_5 / vol_20.replace(0, float("nan"))).fillna(1.0)

    # obv_slope_10d: On-Balance Volume trend, normalized by average volume.
    # OBV accumulates volume on up-days and subtracts on down-days.
    # The fast/slow EMA crossover of OBV flags accumulation (positive) or
    # distribution (negative) patterns.
    obv_direction = np.sign(close.diff()).fillna(0)
    obv = (obv_direction * volume).cumsum()
    obv_fast = obv.rolling(_FC["obv_fast"]).mean()
    obv_slow = obv.rolling(_FC["obv_slow"]).mean()
    df["obv_slope_10d"] = ((obv_fast - obv_slow) / vol_20.replace(0, float("nan"))).fillna(0.0)

    # rsi_slope_5d: rate of change of RSI over 5 days. Rising RSI = strengthening
    # momentum; falling RSI = weakening. Captures RSI trend, not just level.
    rsi = df["rsi_14"]
    _rsi_slope_w = _FC["rsi_slope_window"]
    df["rsi_slope_5d"] = (rsi - rsi.shift(_rsi_slope_w)) / float(_rsi_slope_w)

    # volume_price_div: sign divergence between volume and price momentum.
    # +1 = volume and price moving in same direction (confirmation)
    # -1 = divergence (volume up / price down, or vice versa — reversal signal)
    #  0 = neutral (volume or price flat)
    df["volume_price_div"] = np.sign(df["volume_trend"] - 1.0) * np.sign(df["momentum_5d"])

    # ── v1.5 features — regime interaction terms ───────────────────────────────
    # Pure macro features (VIX, yields) are identical across all tickers on a
    # given day and cannot predict cross-sectional alpha.  But *interactions*
    # between macro regime indicators and ticker-specific signals ARE cross-
    # sectional: the product varies by ticker, capturing how technical signals
    # behave differently under different regimes (e.g., momentum works in bull
    # markets but mean-reverts in high-fear environments).

    # VIX regime indicator: >1 when VIX above baseline (elevated fear).
    # Centered at 0 so the interaction term is neutral at the baseline.
    vix_regime = df["vix_level"] - 1.0  # 0 = baseline, +0.5 = VIX at 30

    # momentum × VIX regime: captures regime-dependent momentum behavior.
    # In high-fear regimes, short-term momentum tends to be mean-reverting
    # (oversold bounces); in low-fear regimes, momentum tends to persist.
    df["mom5d_x_vix"] = df["momentum_5d"] * vix_regime

    # RSI × VIX regime: overbought/oversold signals are more predictive
    # in high-volatility regimes (mean-reversion works better).
    # Center RSI at 50 so neutral → 0 interaction.
    rsi_centered = (df["rsi_14"] - 50.0) / 50.0  # range [-1, +1]
    df["rsi_x_vix"] = rsi_centered * vix_regime

    # sector relative strength × market trend: captures whether sector rotation
    # signals are more/less predictive in trending vs range-bound markets.
    # SPY 20d momentum as market trend proxy (already available from momentum_20d).
    if spy_series is not None:
        spy_aligned = spy_series.reindex(df.index, method="ffill")
        spy_trend = (spy_aligned / spy_aligned.shift(20)) - 1.0
        spy_trend = spy_trend.fillna(0.0)
    else:
        spy_trend = pd.Series(0.0, index=df.index)
    df["sector_x_trend"] = df["sector_vs_spy_5d"] * spy_trend

    # volatility regime interaction: high atr_14_pct in high-VIX environments
    # signals panic; high atr_14_pct in low-VIX signals stock-specific event.
    df["atr_x_vix"] = df["atr_14_pct"] * vix_regime

    # volume surge × VIX: distinguishes institutional accumulation in calm
    # markets from panic liquidation in fearful markets.
    df["vol_trend_x_vix"] = (df["volume_trend"] - 1.0) * vix_regime

    # ── v2.0 features — alternative data signals (O10-O12) ────────────────────
    # These features come from external data sources (FMP, yfinance options)
    # rather than price/volume data. They are passed in as scalar values per
    # ticker and broadcast across all rows (constant within a ticker's series).

    # O10: PEAD — earnings surprise magnitude and recency
    if earnings_data:
        df["earnings_surprise_pct"] = float(earnings_data.get("surprise_pct", 0.0))
        days_since = float(earnings_data.get("days_since_earnings", 90))
        df["days_since_earnings"] = days_since / 90.0  # normalize to [0, 1]
    else:
        df["earnings_surprise_pct"] = 0.0
        df["days_since_earnings"] = 1.0  # 90/90 = no recent earnings

    # O11: EPS revision momentum
    if revision_data:
        df["eps_revision_4w"] = float(revision_data.get("eps_revision_4w", 0.0))
        df["revision_streak"] = float(revision_data.get("revision_streak", 0))
    else:
        df["eps_revision_4w"] = 0.0
        df["revision_streak"] = 0.0

    # O12: Options-derived signals
    if options_data:
        df["put_call_ratio"] = float(options_data.get("put_call_ratio", 0.0))
        df["iv_rank"] = float(options_data.get("iv_rank", 0.5))
        # IV vs realized vol ratio
        atm_iv = float(options_data.get("atm_iv", 0.0))
        realized_vol = df["realized_vol_20d"].iloc[-1] if "realized_vol_20d" in df.columns else 0.0
        if realized_vol > 0 and atm_iv > 0:
            df["iv_vs_rv"] = atm_iv / realized_vol
        else:
            df["iv_vs_rv"] = 1.0  # neutral
    else:
        df["put_call_ratio"] = 0.0   # log(1.0) = 0
        df["iv_rank"] = 0.5          # 50th percentile
        df["iv_vs_rv"] = 1.0         # neutral

    # ── v3.0 features — fundamental ratios (quarterly, from FMP) ─────────────
    # Pre-normalized by fundamental_fetcher.py (P/E / 30, P/B / 5, etc.).
    # Broadcast as constants across all rows (same pattern as earnings_data).
    if fundamental_data:
        df["pe_ratio"] = float(fundamental_data.get("pe_ratio", 0.0))
        df["pb_ratio"] = float(fundamental_data.get("pb_ratio", 0.0))
        df["debt_to_equity"] = float(fundamental_data.get("debt_to_equity", 0.0))
        df["revenue_growth_yoy"] = float(fundamental_data.get("revenue_growth_yoy", 0.0))
        df["fcf_yield"] = float(fundamental_data.get("fcf_yield", 0.0))
        df["gross_margin"] = float(fundamental_data.get("gross_margin", 0.0))
        df["roe"] = float(fundamental_data.get("roe", 0.0))
        df["current_ratio"] = float(fundamental_data.get("current_ratio", 0.0))
    else:
        df["pe_ratio"] = 0.0
        df["pb_ratio"] = 0.0
        df["debt_to_equity"] = 0.0
        df["revenue_growth_yoy"] = 0.0
        df["fcf_yield"] = 0.0
        df["gross_margin"] = 0.0
        df["roe"] = 0.0
        df["current_ratio"] = 0.0

    # ── Drop rows with any NaN in the feature columns ─────────────────────────
    from config import FEATURES as feature_cols
    df = df.dropna(subset=feature_cols)

    return df


def features_to_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract all feature columns from a featured DataFrame as a float32 array.
    Shape: (N, N_FEATURES).
    """
    from config import FEATURES
    return df[FEATURES].to_numpy(dtype=np.float32)
