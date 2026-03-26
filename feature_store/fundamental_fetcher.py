"""
feature_store/fundamental_fetcher.py — FMP fundamental data fetcher.

Provides quarterly financial ratios for GBM features:
P/E, P/B, D/E, revenue growth YoY, FCF yield, gross margin, ROE, current ratio.

FMP free tier: 250 req/day. Each ticker uses up to 3 calls (key-metrics,
income-statement, balance-sheet). Budget accordingly.

Rate limiting: 4 req/sec to stay under 5/sec FMP limit.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_TIMEOUT = 10
_RATE_LIMIT_DELAY = 0.25  # 4 req/sec


def _fmp_get(endpoint: str, params: Optional[dict] = None) -> dict | list:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        raise RuntimeError("FMP_API_KEY environment variable not set.")
    url = f"{_FMP_BASE}/{endpoint}"
    p = {"apikey": api_key}
    if params:
        p.update(params)
    resp = requests.get(url, params=p, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# Neutral values for graceful degradation
_NEUTRAL = {
    "pe_ratio": 0.0,
    "pb_ratio": 0.0,
    "debt_to_equity": 0.0,
    "revenue_growth_yoy": 0.0,
    "fcf_yield": 0.0,
    "gross_margin": 0.0,
    "roe": 0.0,
    "current_ratio": 0.0,
}


def fetch_fundamental_data(
    tickers: list[str],
) -> dict[str, dict]:
    """
    Fetch fundamental ratios from FMP for all tickers.

    Returns per ticker:
        pe_ratio: float         — trailing P/E, normalized (PE / 30)
        pb_ratio: float         — price-to-book, normalized (PB / 5)
        debt_to_equity: float   — total debt / total equity, normalized (D/E / 2)
        revenue_growth_yoy: float — year-over-year revenue growth (decimal)
        fcf_yield: float        — free cash flow / market cap (decimal)
        gross_margin: float     — gross profit / revenue (0-1)
        roe: float              — return on equity (decimal)
        current_ratio: float    — current assets / current liabilities, normalized (CR / 3)

    Graceful degradation: returns neutral values on API failure.
    """
    results: dict[str, dict] = {}

    for ticker in tickers:
        try:
            data = _fetch_single_ticker(ticker)
            results[ticker] = data
        except Exception as e:
            log.warning("Fundamental fetch failed for %s: %s", ticker, e)
            results[ticker] = _NEUTRAL.copy()

    fetched = sum(1 for v in results.values() if v != _NEUTRAL)
    log.info("Fetched fundamentals for %d/%d tickers", fetched, len(tickers))
    return results


def _fetch_single_ticker(ticker: str) -> dict:
    """Fetch and normalize fundamental data for a single ticker."""
    # key-metrics gives us P/E, P/B, FCF yield, ROE, current ratio
    metrics = _fmp_get(f"key-metrics-ttm/{ticker}")
    time.sleep(_RATE_LIMIT_DELAY)

    if not isinstance(metrics, list) or not metrics:
        log.debug("No key-metrics for %s", ticker)
        return _NEUTRAL.copy()

    m = metrics[0]

    pe_raw = _safe_float(m.get("peRatioTTM"))
    pb_raw = _safe_float(m.get("pbRatioTTM"))
    roe_raw = _safe_float(m.get("roeTTM"))
    fcf_yield_raw = _safe_float(m.get("freeCashFlowYieldTTM"))
    current_ratio_raw = _safe_float(m.get("currentRatioTTM"))
    de_raw = _safe_float(m.get("debtToEquityTTM"))

    # income-statement for revenue growth and gross margin
    income = _fmp_get(f"income-statement/{ticker}", params={"period": "quarter", "limit": 5})
    time.sleep(_RATE_LIMIT_DELAY)

    revenue_growth = 0.0
    gross_margin = 0.0

    if isinstance(income, list) and len(income) >= 5:
        # YoY: compare most recent quarter to same quarter last year
        recent_rev = _safe_float(income[0].get("revenue"))
        year_ago_rev = _safe_float(income[4].get("revenue"))
        if year_ago_rev > 0:
            revenue_growth = (recent_rev - year_ago_rev) / year_ago_rev

        gross_profit = _safe_float(income[0].get("grossProfit"))
        if recent_rev > 0:
            gross_margin = gross_profit / recent_rev
    elif isinstance(income, list) and income:
        # Fewer than 5 quarters — use what we have for gross margin
        recent_rev = _safe_float(income[0].get("revenue"))
        gross_profit = _safe_float(income[0].get("grossProfit"))
        if recent_rev > 0:
            gross_margin = gross_profit / recent_rev

    # Normalize to reasonable scales for ML consumption
    return {
        "pe_ratio": _clip(pe_raw / 30.0, -3.0, 3.0),       # P/E=30 → 1.0
        "pb_ratio": _clip(pb_raw / 5.0, -3.0, 3.0),        # P/B=5 → 1.0
        "debt_to_equity": _clip(de_raw / 2.0, -3.0, 3.0),  # D/E=2 → 1.0
        "revenue_growth_yoy": _clip(revenue_growth, -1.0, 2.0),
        "fcf_yield": _clip(fcf_yield_raw, -0.5, 0.5),
        "gross_margin": _clip(gross_margin, 0.0, 1.0),
        "roe": _clip(roe_raw, -1.0, 1.0),
        "current_ratio": _clip(current_ratio_raw / 3.0, 0.0, 3.0),  # CR=3 → 1.0
    }


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ── S3 caching ───────────────────────────────────────────────────────────────

def cache_fundamentals_to_s3(
    data: dict[str, dict],
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> None:
    """Cache fundamental data to S3 for historical lookback."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/fundamentals/{date_str}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json",
        )
        log.info("Cached fundamentals to s3://%s/%s", bucket, key)
    except Exception as e:
        log.warning("Failed to cache fundamentals to S3: %s", e)


def load_fundamentals_from_s3(
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> Optional[dict[str, dict]]:
    """Load cached fundamental data from S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/fundamentals/{date_str}.json"
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None
