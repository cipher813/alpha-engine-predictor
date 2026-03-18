"""
data/earnings_fetcher.py — FMP earnings surprise + calendar fetcher.

Provides earnings surprise magnitude and recency data for GBM features (O10)
and revision tracking (O11).

FMP free tier: 250 req/day. Each ticker uses 1 call for surprises +
1 call for calendar = 2 calls per ticker. Budget accordingly.

Rate limiting: 5 req/sec for FMP free tier.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_FMP_STABLE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 10
_RATE_LIMIT_DELAY = 0.25  # 4 req/sec to stay under 5/sec limit


def _fmp_get(endpoint: str, params: Optional[dict] = None, base: str = _FMP_BASE) -> dict | list:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        raise RuntimeError("FMP_API_KEY environment variable not set.")
    url = f"{base}/{endpoint}"
    p = {"apikey": api_key}
    if params:
        p.update(params)
    resp = requests.get(url, params=p, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse YYYY-MM-DD date string, return None on failure."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def fetch_earnings_data(
    tickers: list[str],
    reference_date: Optional[str] = None,
) -> dict[str, dict]:
    """
    Fetch earnings surprise data for a list of tickers from FMP.

    Returns per ticker:
        surprise_pct: float — most recent quarterly surprise percentage
        days_since_earnings: int — days since last earnings report (capped at 90)
        next_earnings_date: str or None — next earnings date if available
        next_earnings_days: int or None — days until next earnings

    Graceful degradation: returns neutral values on API failure.
    """
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
    results: dict[str, dict] = {}

    for ticker in tickers:
        try:
            # Fetch historical earnings surprises
            surprises = _fmp_get(f"earning_surprises/{ticker}")
            time.sleep(_RATE_LIMIT_DELAY)

            surprise_pct = 0.0
            days_since = 90  # default: no recent earnings

            if isinstance(surprises, list) and surprises:
                # FMP returns most recent first
                latest = surprises[0]
                report_date = _parse_date(latest.get("date", ""))
                if report_date:
                    days_since = min((today - report_date).days, 90)

                actual = latest.get("actualEarningResult")
                estimated = latest.get("estimatedEarning")
                if actual is not None and estimated is not None and estimated != 0:
                    surprise_pct = round((actual - estimated) / abs(estimated) * 100, 2)

            # Fetch upcoming earnings calendar
            next_earnings_date = None
            next_earnings_days = None
            try:
                date_from = today.strftime("%Y-%m-%d")
                date_to = (today + timedelta(days=90)).strftime("%Y-%m-%d")
                calendar = _fmp_get(
                    "earning_calendar",
                    params={"symbol": ticker, "from": date_from, "to": date_to},
                )
                time.sleep(_RATE_LIMIT_DELAY)
                if isinstance(calendar, list) and calendar:
                    # Find the nearest future date
                    for entry in calendar:
                        ed = _parse_date(entry.get("date", ""))
                        if ed and ed >= today:
                            next_earnings_date = entry["date"]
                            next_earnings_days = (ed - today).days
                            break
            except Exception as e:
                log.debug("Earnings calendar fetch failed for %s: %s", ticker, e)

            results[ticker] = {
                "surprise_pct": surprise_pct,
                "days_since_earnings": days_since,
                "next_earnings_date": next_earnings_date,
                "next_earnings_days": next_earnings_days,
            }

        except Exception as e:
            log.warning("Earnings data fetch failed for %s: %s", ticker, e)
            results[ticker] = {
                "surprise_pct": 0.0,
                "days_since_earnings": 90,
                "next_earnings_date": None,
                "next_earnings_days": None,
            }

    log.info("Fetched earnings data for %d/%d tickers", len(results), len(tickers))
    return results


def cache_earnings_to_s3(
    data: dict[str, dict],
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> None:
    """Cache earnings data to S3 for historical lookback."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"predictor/earnings_cache/{date_str}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json",
        )
        log.info("Cached earnings data to s3://%s/%s", bucket, key)
    except Exception as e:
        log.warning("Failed to cache earnings data to S3: %s", e)


def load_earnings_from_s3(
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> Optional[dict[str, dict]]:
    """Load cached earnings data from S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"predictor/earnings_cache/{date_str}.json"
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


# ── O11: Revision tracking ──────────────────────────────────────────────────

def fetch_eps_estimates(
    tickers: list[str],
) -> dict[str, dict]:
    """
    Fetch current EPS consensus estimates from FMP for revision tracking.

    Returns per ticker:
        eps_current: float — current consensus EPS estimate
        eps_next_year: float — next year estimate (if available)
    """
    results: dict[str, dict] = {}

    for ticker in tickers:
        try:
            data = _fmp_get(f"analyst-estimates/{ticker}", params={"limit": 1})
            time.sleep(_RATE_LIMIT_DELAY)

            if isinstance(data, list) and data:
                entry = data[0]
                results[ticker] = {
                    "eps_current": entry.get("estimatedEpsAvg", 0.0),
                    "revenue_current": entry.get("estimatedRevenueAvg", 0.0),
                }
            else:
                results[ticker] = {"eps_current": 0.0, "revenue_current": 0.0}

        except Exception as e:
            log.debug("EPS estimate fetch failed for %s: %s", ticker, e)
            results[ticker] = {"eps_current": 0.0, "revenue_current": 0.0}

    return results


def fetch_revision_history(
    tickers: list[str],
    bucket: str = "alpha-engine-research",
    lookback_weeks: int = 8,
    reference_date: Optional[str] = None,
) -> dict[str, dict]:
    """
    Load historical EPS snapshots from S3 to construct revision time series.

    Returns per ticker:
        eps_revision_4w: float — 4-week cumulative EPS revision percentage
        revision_streak: int — consecutive weeks of same-direction revisions
            (positive = upward, negative = downward)
    """
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
    results: dict[str, dict] = {}

    # Load weekly snapshots going back lookback_weeks
    snapshots: list[tuple[str, dict]] = []
    for weeks_ago in range(lookback_weeks + 1):
        target_date = today - timedelta(weeks=weeks_ago)
        # Try each day in the week (Monday preference)
        for day_offset in range(7):
            check_date = target_date - timedelta(days=day_offset)
            date_str = check_date.strftime("%Y-%m-%d")
            cached = _load_revision_snapshot(date_str, bucket)
            if cached is not None:
                snapshots.append((date_str, cached))
                break

    if len(snapshots) < 2:
        # Not enough history for revision computation
        for ticker in tickers:
            results[ticker] = {"eps_revision_4w": 0.0, "revision_streak": 0}
        return results

    # Sort oldest first
    snapshots.sort(key=lambda x: x[0])

    for ticker in tickers:
        try:
            # Get weekly EPS values for this ticker
            weekly_eps: list[tuple[str, float]] = []
            for date_str, snapshot in snapshots:
                ticker_data = snapshot.get(ticker, {})
                eps = ticker_data.get("eps_current", 0.0)
                if eps != 0.0:
                    weekly_eps.append((date_str, eps))

            if len(weekly_eps) < 2:
                results[ticker] = {"eps_revision_4w": 0.0, "revision_streak": 0}
                continue

            # 4-week cumulative revision
            current_eps = weekly_eps[-1][1]
            four_weeks_ago_idx = max(0, len(weekly_eps) - 5)
            base_eps = weekly_eps[four_weeks_ago_idx][1]
            if abs(base_eps) > 0.001:
                eps_revision_4w = round((current_eps - base_eps) / abs(base_eps) * 100, 2)
            else:
                eps_revision_4w = 0.0

            # Revision streak: consecutive weeks of same-direction changes
            streak = 0
            for i in range(len(weekly_eps) - 1, 0, -1):
                delta = weekly_eps[i][1] - weekly_eps[i - 1][1]
                if delta > 0.001:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                elif delta < -0.001:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
                else:
                    break

            results[ticker] = {
                "eps_revision_4w": eps_revision_4w,
                "revision_streak": streak,
            }
        except Exception:
            results[ticker] = {"eps_revision_4w": 0.0, "revision_streak": 0}

    return results


def save_revision_snapshot(
    data: dict[str, dict],
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> None:
    """Save current EPS estimates snapshot to S3 for revision tracking."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/revisions/{date_str}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json",
        )
        log.info("Saved revision snapshot to s3://%s/%s", bucket, key)
    except Exception as e:
        log.warning("Failed to save revision snapshot to S3: %s", e)


def _load_revision_snapshot(
    date_str: str,
    bucket: str = "alpha-engine-research",
) -> Optional[dict]:
    """Load a single revision snapshot from S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/revisions/{date_str}.json"
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None
