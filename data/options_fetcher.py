"""
data/options_fetcher.py — Options data fetcher for predictor features (O12).

Fetches options-derived signals (put/call ratio, IV rank) for GBM features.
Optimized for batch operation at inference time. For training, loads
historical snapshots from S3 archive/options/ directory.

Graceful degradation: fills neutral values when data unavailable.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_FETCH_DELAY = 0.5


def fetch_options_features(
    tickers: list[str],
    reference_date: Optional[str] = None,
) -> dict[str, dict]:
    """
    Fetch options-derived features for predictor inference.

    Returns per ticker:
        put_call_ratio: float — log-transformed P/C OI ratio
        iv_rank: float — IV percentile rank normalized to [0, 1]
        atm_iv: float — raw ATM implied volatility
    """
    try:
        import yfinance
    except ImportError:
        log.warning("yfinance not available for options features")
        return {t: _neutral_features() for t in tickers}

    results: dict[str, dict] = {}

    for ticker in tickers:
        try:
            t = yfinance.Ticker(ticker)
            expiries = t.options
            if not expiries:
                results[ticker] = _neutral_features()
                continue

            # Select expiry ~30 DTE
            expiry = _select_expiry(expiries, reference_date)
            if not expiry:
                results[ticker] = _neutral_features()
                continue

            chain = t.option_chain(expiry)
            calls, puts = chain.calls, chain.puts

            # Put/call ratio
            put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            raw_pc = put_oi / max(call_oi, 1)
            pc_ratio_log = float(np.log(max(raw_pc, 0.01)))  # log-transform

            # ATM IV
            info = t.info if hasattr(t, "info") else {}
            price = info.get("regularMarketPrice") or info.get("previousClose", 0)
            if not price:
                hist = t.history(period="1d")
                price = float(hist["Close"].iloc[-1]) if not hist.empty else 0

            atm_iv = _get_atm_iv(calls, puts, price)

            # IV rank (normalized to [0, 1])
            iv_rank = _compute_iv_rank(t, atm_iv) / 100.0

            results[ticker] = {
                "put_call_ratio": round(pc_ratio_log, 4),
                "iv_rank": round(iv_rank, 4),
                "atm_iv": round(atm_iv, 4),
            }

            time.sleep(_FETCH_DELAY)

        except Exception as e:
            log.debug("Options features failed for %s: %s", ticker, e)
            results[ticker] = _neutral_features()

    log.info("Fetched options features for %d/%d tickers", len(results), len(tickers))
    return results


def load_historical_options(
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> Optional[dict[str, dict]]:
    """Load historical options data from S3 for training features."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/options/{date_str}.json"
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = json.loads(obj["Body"].read())
        # Convert research-format to predictor features
        result = {}
        for ticker, data in raw.items():
            pc = data.get("put_call_ratio", 1.0)
            result[ticker] = {
                "put_call_ratio": float(np.log(max(pc, 0.01))),
                "iv_rank": data.get("iv_rank", 50.0) / 100.0,
                "atm_iv": data.get("atm_iv", 0.0),
            }
        return result
    except Exception:
        return None


def _neutral_features() -> dict:
    """Neutral values when options data unavailable."""
    return {
        "put_call_ratio": 0.0,   # log(1.0) = 0
        "iv_rank": 0.5,          # 50th percentile
        "atm_iv": 0.0,
    }


def _select_expiry(
    expiries: tuple[str, ...],
    reference_date: Optional[str] = None,
) -> Optional[str]:
    """Select expiry closest to 30 DTE."""
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
    best, best_diff = None, float("inf")
    for exp_str in expiries:
        try:
            dte = (datetime.strptime(exp_str, "%Y-%m-%d") - today).days
            if 7 < dte < 60 and abs(dte - 30) < best_diff:
                best, best_diff = exp_str, abs(dte - 30)
        except ValueError:
            continue
    return best


def _get_atm_iv(calls, puts, price: float) -> float:
    """Get ATM implied volatility."""
    if price <= 0 or "strike" not in calls.columns or "impliedVolatility" not in calls.columns:
        return 0.0
    try:
        strikes = calls["strike"].values
        if len(strikes) == 0:
            return 0.0
        idx = np.abs(strikes - price).argmin()
        call_iv = float(calls.iloc[idx]["impliedVolatility"])
        if "strike" in puts.columns and "impliedVolatility" in puts.columns:
            put_strikes = puts["strike"].values
            if len(put_strikes) > 0:
                put_idx = np.abs(put_strikes - price).argmin()
                put_iv = float(puts.iloc[put_idx]["impliedVolatility"])
                return (call_iv + put_iv) / 2
        return call_iv
    except Exception:
        return 0.0


def _compute_iv_rank(ticker_obj, current_iv: float) -> float:
    """Approximate IV rank using historical realized vol percentile."""
    if current_iv <= 0:
        return 50.0
    try:
        hist = ticker_obj.history(period="1y")
        if hist.empty or len(hist) < 30:
            return 50.0
        returns = hist["Close"].pct_change().dropna()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) < 10:
            return 50.0
        rank = float((rolling_vol < current_iv).sum() / len(rolling_vol) * 100)
        return min(100.0, max(0.0, rank))
    except Exception:
        return 50.0
