"""
Polygon.io (Massive) market data client with rate limiting and dividend adjustment.

Replaces yfinance as primary price data source. Free tier: 5 API calls/min,
~2 years historical depth, EOD data only. Index tickers (VIX/TNX/IRX) are
not available on free tier — use FRED or yfinance for those.

Usage:
    from polygon_client import PolygonClient, polygon_client

    # Singleton (reads POLYGON_API_KEY from env):
    client = polygon_client()
    bars = client.get_daily_bars("AAPL", "2025-01-01", "2026-03-28")

    # Dividend-adjusted (matches yfinance auto_adjust=True):
    bars = client.get_daily_bars_dividend_adjusted("XOM", "2025-01-01", "2026-03-28")

    # All US stocks for a single date:
    prices = client.get_grouped_daily("2026-03-28")
    # -> {"AAPL": {"open": 253.9, "high": 255.5, ...}, ...}
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import date, datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_MAX_BARS_PER_REQUEST = 50_000  # polygon limit param max


class PolygonRateLimitError(Exception):
    """Raised when rate limit is exhausted and caller should backoff."""


class PolygonClient:
    """Rate-limited polygon.io REST client with dividend adjustment."""

    def __init__(self, api_key: str | None = None, calls_per_min: int = 5):
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError("POLYGON_API_KEY not set")
        self._calls_per_min = calls_per_min
        self._call_times: deque[float] = deque()
        self._session = requests.Session()
        self._session.params = {"apiKey": self._api_key}  # type: ignore[assignment]

    # ── Rate limiter ──────────────────────────────────────────────────────

    def _wait_for_slot(self) -> None:
        """Block until a rate limit slot is available."""
        now = time.monotonic()
        window = 60.0  # 1 minute window
        # Purge old timestamps
        while self._call_times and now - self._call_times[0] > window:
            self._call_times.popleft()
        if len(self._call_times) >= self._calls_per_min:
            wait = window - (now - self._call_times[0]) + 0.5
            logger.debug("Rate limit: waiting %.1fs", wait)
            time.sleep(wait)
            # Purge again after sleep
            now = time.monotonic()
            while self._call_times and now - self._call_times[0] > window:
                self._call_times.popleft()
        self._call_times.append(time.monotonic())

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a rate-limited GET request. Handles 429 with retry."""
        self._wait_for_slot()
        url = f"{_BASE_URL}{path}"
        for attempt in range(3):
            resp = self._session.get(url, params=params or {}, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 15))
                logger.warning("Rate limited (429), waiting %ds", retry_after)
                time.sleep(retry_after)
                self._call_times.clear()  # Reset window after forced wait
                continue
            if resp.status_code == 403:
                data = resp.json()
                msg = data.get("message", "Not authorized")
                logger.warning("Polygon 403: %s (path=%s)", msg, path)
                return {"results": [], "resultsCount": 0, "status": "FORBIDDEN"}
            resp.raise_for_status()
            return resp.json()
        raise PolygonRateLimitError("Rate limited after 3 retries")

    # ── Core endpoints ────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a single ticker.

        Returns DataFrame with DatetimeIndex and columns:
        [Open, High, Low, Close, Volume]

        Prices are split-adjusted (adjusted=True) but NOT dividend-adjusted.
        Use get_daily_bars_dividend_adjusted() for fully-adjusted prices.
        """
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": _MAX_BARS_PER_REQUEST,
        }
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            params=params,
        )
        results = data.get("results", [])
        if not results:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
        df = df.sort_index()
        return df

    def get_grouped_daily(self, date_str: str) -> dict[str, dict]:
        """Fetch OHLCV for ALL US stocks on a single date.

        Returns {ticker: {"open": float, "high": float, "low": float,
                          "close": float, "volume": float}}
        """
        data = self._get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}",
            params={"adjusted": "true"},
        )
        results = data.get("results", [])
        return {
            r["T"]: {
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r["v"],
            }
            for r in results
            if "T" in r
        }

    def get_dividends(
        self,
        ticker: str,
        start: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch dividend history for a ticker.

        Returns list of dicts with keys:
        ex_dividend_date, cash_amount, frequency, declaration_date, pay_date, etc.
        """
        params: dict = {"ticker": ticker, "limit": limit, "sort": "ex_dividend_date"}
        if start:
            params["ex_dividend_date.gte"] = start
        all_dividends: list[dict] = []
        next_url: str | None = None

        # First page
        data = self._get("/v3/reference/dividends", params=params)
        all_dividends.extend(data.get("results", []))
        next_url = data.get("next_url")

        # Paginate
        while next_url:
            resp = self._get_raw_url(next_url)
            all_dividends.extend(resp.get("results", []))
            next_url = resp.get("next_url")

        return all_dividends

    def _get_raw_url(self, url: str) -> dict:
        """GET a full URL (for pagination next_url)."""
        self._wait_for_slot()
        # next_url already includes apiKey
        if "apiKey" not in url:
            url += f"&apiKey={self._api_key}" if "?" in url else f"?apiKey={self._api_key}"
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Dividend adjustment ───────────────────────────────────────────────

    def get_daily_bars_dividend_adjusted(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch daily bars with full adjustment (splits + dividends).

        Produces prices equivalent to yfinance auto_adjust=True.
        """
        bars = self.get_daily_bars(ticker, start, end, adjusted=True)
        if bars.empty:
            return bars

        divs = self.get_dividends(ticker, start=start)
        if not divs:
            return bars  # No dividends → split-adjusted is sufficient

        return _apply_dividend_adjustment(bars, divs)

    # ── Batch helpers ─────────────────────────────────────────────────────

    def fetch_batch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        dividend_adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple tickers with rate limiting.

        Returns dict[ticker, DataFrame].
        """
        results: dict[str, pd.DataFrame] = {}
        fetch_fn = (
            self.get_daily_bars_dividend_adjusted
            if dividend_adjusted
            else self.get_daily_bars
        )
        for i, ticker in enumerate(tickers):
            try:
                df = fetch_fn(ticker, start, end)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", ticker, e)
            if (i + 1) % 50 == 0:
                logger.info("Batch progress: %d/%d tickers", i + 1, len(tickers))
        return results

    def get_single_close(self, ticker: str, date_str: str) -> float | None:
        """Get closing price for a single ticker on a single date.

        Tries grouped daily first (if we happen to have it cached),
        falls back to per-ticker bars.
        """
        bars = self.get_daily_bars(ticker, date_str, date_str, adjusted=True)
        if not bars.empty:
            return float(bars["Close"].iloc[-1])
        return None


# ── Dividend adjustment logic ─────────────────────────────────────────────

def _apply_dividend_adjustment(
    bars: pd.DataFrame,
    dividends: list[dict],
) -> pd.DataFrame:
    """Apply backward dividend adjustment to split-adjusted OHLCV bars.

    For each bar date, computes:
        factor = product(1 - div_amount / close_before_ex)
        for all dividends with ex_date > bar_date

    Then: adjusted_price = split_adjusted_price * factor
    """
    df = bars.copy()
    price_cols = ["Open", "High", "Low", "Close"]

    # Parse and sort dividends by ex-date ascending
    last_bar_date = df.index[-1]
    div_records = []
    for d in dividends:
        ex_date = d.get("ex_dividend_date")
        amount = d.get("cash_amount")
        if ex_date and amount and float(amount) > 0:
            ex_ts = pd.Timestamp(ex_date)
            # Skip future dividends not yet ex within the data range
            if ex_ts > last_bar_date:
                continue
            div_records.append({
                "ex_date": ex_ts,
                "amount": float(amount),
            })
    if not div_records:
        return df

    div_records.sort(key=lambda x: x["ex_date"])

    # For each dividend, find the close price on the trading day before ex-date
    # to compute the adjustment ratio
    adjustment_factors = []
    for div in div_records:
        ex_date = div["ex_date"]
        # Find closest trading day before ex-date
        prior_bars = df[df.index < ex_date]
        if prior_bars.empty:
            # Dividend ex-date is before our data range — skip
            continue
        close_before = prior_bars["Close"].iloc[-1]
        if close_before <= 0:
            continue
        ratio = 1.0 - div["amount"] / close_before
        if ratio <= 0 or ratio > 1:
            logger.warning(
                "Skipping suspicious dividend ratio %.4f (amount=%.2f, close=%.2f)",
                ratio, div["amount"], close_before,
            )
            continue
        adjustment_factors.append({"ex_date": ex_date, "ratio": ratio})

    if not adjustment_factors:
        return df

    # Apply cumulative backward adjustment:
    # Bars before the earliest ex-date get ALL factors applied
    # Bars between ex-dates get progressively fewer factors
    # Bars on/after the latest ex-date get no adjustment
    for col in price_cols:
        adjusted = df[col].copy()
        for af in adjustment_factors:
            mask = df.index < af["ex_date"]
            adjusted[mask] *= af["ratio"]
        df[col] = adjusted

    return df


# ── Singleton ─────────────────────────────────────────────────────────────

_singleton: PolygonClient | None = None


def polygon_client(api_key: str | None = None) -> PolygonClient:
    """Get or create a singleton PolygonClient."""
    global _singleton
    if _singleton is None:
        _singleton = PolygonClient(api_key=api_key)
    return _singleton
