"""
feature_store/registry.py — Feature catalog and S3 registry.

Defines all features with metadata (group, description, source, refresh frequency).
Generates registry.json for cross-repo consumers that read features from S3.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureEntry:
    name: str
    group: str          # technical | macro | interaction | alternative | fundamental
    description: str
    dtype: str = "float32"
    source: str = ""    # yfinance | fmp | computed
    refresh: str = ""   # daily | weekly | quarterly
    per_ticker: bool = True  # False for macro features (one row per date)


# ── Feature Catalog ──────────────────────────────────────────────────────────
# Authoritative list of all features. Must stay in sync with
# config.FEATURES and data/feature_engineer.py::compute_features().

CATALOG: list[FeatureEntry] = [
    # ── Technical (29) ────────────────────────────────────────────────────────
    FeatureEntry("rsi_14", "technical", "RSI(14), range 0-100", source="yfinance", refresh="daily"),
    FeatureEntry("macd_cross", "technical", "+1 bullish / -1 bearish / 0 no cross (last 3 days)", source="yfinance", refresh="daily"),
    FeatureEntry("macd_above_zero", "technical", "1 if MACD line > 0, else 0", source="yfinance", refresh="daily"),
    FeatureEntry("macd_line_last", "technical", "MACD line value (fast EMA - slow EMA)", source="yfinance", refresh="daily"),
    FeatureEntry("price_vs_ma50", "technical", "Close / SMA(50) ratio", source="yfinance", refresh="daily"),
    FeatureEntry("price_vs_ma200", "technical", "Close / SMA(200) ratio", source="yfinance", refresh="daily"),
    FeatureEntry("momentum_20d", "technical", "20-day price return", source="yfinance", refresh="daily"),
    FeatureEntry("avg_volume_20d", "technical", "20-day avg volume / global mean volume", source="yfinance", refresh="daily"),
    FeatureEntry("dist_from_52w_high", "technical", "(Close - 52w high) / 52w high", source="yfinance", refresh="daily"),
    FeatureEntry("momentum_5d", "technical", "5-day price return", source="yfinance", refresh="daily"),
    FeatureEntry("rel_volume_ratio", "technical", "Today volume / 20-day avg volume", source="yfinance", refresh="daily"),
    FeatureEntry("return_vs_spy_5d", "technical", "5-day stock return minus SPY return", source="yfinance", refresh="daily"),
    FeatureEntry("dist_from_52w_low", "technical", "(Close - 52w low) / 52w low", source="yfinance", refresh="daily"),
    FeatureEntry("vol_ratio_10_60", "technical", "10-day vol / 60-day vol", source="yfinance", refresh="daily"),
    FeatureEntry("bollinger_pct", "technical", "Position within Bollinger Bands (0-1)", source="yfinance", refresh="daily"),
    FeatureEntry("sector_vs_spy_5d", "technical", "5-day sector ETF return minus SPY return", source="yfinance", refresh="daily"),
    FeatureEntry("price_accel", "technical", "Momentum acceleration (5d mom - 20d mom)", source="yfinance", refresh="daily"),
    FeatureEntry("ema_cross_8_21", "technical", "EMA(8) / EMA(21) ratio", source="yfinance", refresh="daily"),
    FeatureEntry("atr_14_pct", "technical", "ATR(14) / Close, normalized volatility", source="yfinance", refresh="daily"),
    FeatureEntry("realized_vol_20d", "technical", "20-day annualized return std dev", source="yfinance", refresh="daily"),
    FeatureEntry("volume_trend", "technical", "5-day avg volume / 20-day avg volume", source="yfinance", refresh="daily"),
    FeatureEntry("obv_slope_10d", "technical", "OBV linear regression slope over 10 days", source="yfinance", refresh="daily"),
    FeatureEntry("rsi_slope_5d", "technical", "5-day RSI slope", source="yfinance", refresh="daily"),
    FeatureEntry("volume_price_div", "technical", "sign(volume_trend-1) * sign(momentum_5d)", source="yfinance", refresh="daily"),

    # ── Macro (5) — identical across all tickers on a given day ───────────────
    FeatureEntry("vix_level", "macro", "VIX / 20 (normalized around long-run avg)", source="yfinance", refresh="daily", per_ticker=False),
    FeatureEntry("yield_10y", "macro", "10Y Treasury yield normalized to 0-1", source="yfinance", refresh="daily", per_ticker=False),
    FeatureEntry("yield_curve_slope", "macro", "10Y - 2Y spread, normalized", source="yfinance", refresh="daily", per_ticker=False),
    FeatureEntry("gold_mom_5d", "macro", "5-day gold (GLD) momentum", source="yfinance", refresh="daily", per_ticker=False),
    FeatureEntry("oil_mom_5d", "macro", "5-day oil (USO) momentum", source="yfinance", refresh="daily", per_ticker=False),

    # ── Regime interactions (5) — macro × ticker-specific signals ─────────────
    FeatureEntry("mom5d_x_vix", "interaction", "momentum_5d * VIX regime indicator", source="computed", refresh="daily"),
    FeatureEntry("rsi_x_vix", "interaction", "RSI deviation from 50 * VIX regime", source="computed", refresh="daily"),
    FeatureEntry("sector_x_trend", "interaction", "Sector-relative return * price trend", source="computed", refresh="daily"),
    FeatureEntry("atr_x_vix", "interaction", "ATR * VIX regime indicator", source="computed", refresh="daily"),
    FeatureEntry("vol_trend_x_vix", "interaction", "Volume trend * VIX regime indicator", source="computed", refresh="daily"),

    # ── Alternative data (7) — O10-O12 signals ───────────────────────────────
    FeatureEntry("earnings_surprise_pct", "alternative", "Most recent quarterly EPS surprise %", source="fmp", refresh="weekly"),
    FeatureEntry("days_since_earnings", "alternative", "Days since last earnings (0-1, capped 90d)", source="fmp", refresh="weekly"),
    FeatureEntry("eps_revision_4w", "alternative", "4-week cumulative EPS revision %", source="fmp", refresh="weekly"),
    FeatureEntry("revision_streak", "alternative", "Consecutive weeks of same-direction revisions", source="fmp", refresh="weekly"),
    FeatureEntry("put_call_ratio", "alternative", "Log-transformed put/call OI ratio", source="yfinance", refresh="weekly"),
    FeatureEntry("iv_rank", "alternative", "IV percentile rank (0-1)", source="yfinance", refresh="weekly"),
    FeatureEntry("iv_vs_rv", "alternative", "Implied vol / realized vol ratio", source="yfinance", refresh="weekly"),

    # ── Fundamental (8) — quarterly financials ────────────────────────────────
    FeatureEntry("pe_ratio", "fundamental", "Trailing P/E ratio, normalized (PE / 30)", source="fmp", refresh="quarterly"),
    FeatureEntry("pb_ratio", "fundamental", "Price-to-book ratio, normalized (PB / 5)", source="fmp", refresh="quarterly"),
    FeatureEntry("debt_to_equity", "fundamental", "Total debt / total equity, normalized (D/E / 2)", source="fmp", refresh="quarterly"),
    FeatureEntry("revenue_growth_yoy", "fundamental", "Year-over-year revenue growth (decimal)", source="fmp", refresh="quarterly"),
    FeatureEntry("fcf_yield", "fundamental", "Free cash flow / market cap (decimal)", source="fmp", refresh="quarterly"),
    FeatureEntry("gross_margin", "fundamental", "Gross profit / revenue (0-1)", source="fmp", refresh="quarterly"),
    FeatureEntry("roe", "fundamental", "Return on equity (decimal)", source="fmp", refresh="quarterly"),
    FeatureEntry("current_ratio", "fundamental", "Current assets / current liabilities, normalized (CR / 3)", source="fmp", refresh="quarterly"),
]

# Quick lookup by name
_CATALOG_BY_NAME: dict[str, FeatureEntry] = {f.name: f for f in CATALOG}

# Group membership
GROUPS: dict[str, list[str]] = {}
for _f in CATALOG:
    GROUPS.setdefault(_f.group, []).append(_f.name)


def get_feature(name: str) -> FeatureEntry:
    """Look up a feature entry by name."""
    return _CATALOG_BY_NAME[name]


def get_group_features(group: str) -> list[str]:
    """Return feature names for a group."""
    return GROUPS.get(group, [])


def generate_registry_json() -> str:
    """Serialize the full catalog to JSON for S3 consumers."""
    return json.dumps(
        {"features": [asdict(f) for f in CATALOG]},
        indent=2,
    )


def upload_registry(bucket: str, prefix: str = "features/") -> None:
    """Write registry.json to S3."""
    import boto3

    s3 = boto3.client("s3")
    body = generate_registry_json()
    key = f"{prefix}registry.json"
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode(), ContentType="application/json")
    logger.info("Feature registry uploaded to s3://%s/%s (%d features)", bucket, key, len(CATALOG))
