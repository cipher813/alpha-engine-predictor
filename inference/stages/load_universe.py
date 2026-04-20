"""Stage: load_universe — Determine ticker universe and load sector map."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)

# Fallback universe if signals.json is unavailable
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B",
    "LLY", "JPM", "V", "UNH", "XOM", "COST", "TSLA", "HD",
]


# ── Universe functions (migrated from daily_predict.py) ──────────────────────

def get_universe_tickers(
    s3_bucket: str, date_str: Optional[str] = None,
) -> tuple[list[str], dict]:
    """
    Read the active ticker universe from today's signals.json in S3.
    Falls back to _FALLBACK_TICKERS if signals.json is not available.

    Parameters
    ----------
    s3_bucket : S3 bucket name.
    date_str :  Date string YYYY-MM-DD. Uses today if None.

    Returns
    -------
    (tickers, signals_data) — ticker list and full signals payload for email.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    signals_key = f"signals/{date_str}/signals.json"

    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key=signals_key)
        signals_data = json.loads(obj["Body"].read().decode("utf-8"))

        # signals.json has a top-level "signals" list with per-ticker records
        signals_list = signals_data.get("signals", [])
        tickers = [s["ticker"] for s in signals_list if "ticker" in s]

        if tickers:
            log.info("Universe: %d tickers from %s", len(tickers), signals_key)
            return tickers, signals_data
        else:
            log.warning("signals.json found but no tickers extracted — using fallback")
    except Exception as exc:
        log.info("Could not read signals.json (%s) — using fallback universe", exc)

    log.info("Using fallback universe: %d tickers", len(_FALLBACK_TICKERS))
    return _FALLBACK_TICKERS, {}


def load_watchlist(
    path: str,
    s3_bucket: Optional[str] = None,
    date_str: Optional[str] = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Build a focused prediction universe from Research's population or signals.

    Priority order for ``path="auto"``:
      1. population/latest.json  — new population-based architecture
      2. signals/{date}/signals.json  — legacy fallback

    Parameters
    ----------
    path      : "auto" → fetch from S3 (population first, then signals).
                Any other string → local file path to signals.json or
                population.json for offline / dry-run use.
    s3_bucket : S3 bucket name. Required when path="auto".
    date_str  : YYYY-MM-DD override. Defaults to today.

    Returns
    -------
    tickers : Deduplicated, sorted list of tickers.
    sources : Dict mapping ticker → "population" | "tracked" | "buy_candidate" | "both".
    data    : Raw JSON payload (population or signals).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    data = None

    # ── Load raw payload ──────────────────────────────────────────────────────
    if path == "auto":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when --watchlist auto is used")

        import boto3
        s3 = boto3.client("s3")

        # Try population/latest.json first (new architecture)
        try:
            obj = s3.get_object(Bucket=s3_bucket, Key="population/latest.json")
            data = json.loads(obj["Body"].read().decode("utf-8"))
            pop_tickers = [p["ticker"] for p in data.get("population", []) if "ticker" in p]
            if pop_tickers:
                sources = {t.upper(): "population" for t in pop_tickers}
                tickers = sorted(sources.keys())
                log.info(
                    "Watchlist: loaded %d tickers from population/latest.json",
                    len(tickers),
                )
                return tickers, sources, data
        except Exception as exc:
            log.info("population/latest.json not available (%s), falling back to signals.json", exc)

        # Fallback: signals/{date}/signals.json with date lookback
        # Walk back up to 5 calendar days (skipping weekends) to find the
        # most recent signals — mirrors executor's read_signals_with_fallback.
        from datetime import date as _date, timedelta as _td
        from botocore.exceptions import ClientError

        start = _date.fromisoformat(date_str)
        max_lookback = 5
        tried: list[str] = []

        for days_back in range(max_lookback + 1):
            candidate = start - _td(days=days_back)
            if candidate.weekday() >= 5:  # skip Saturday/Sunday
                continue
            signals_key = f"signals/{candidate}/signals.json"
            try:
                obj = s3.get_object(Bucket=s3_bucket, Key=signals_key)
                data = json.loads(obj["Body"].read().decode("utf-8"))
                if days_back > 0:
                    log.warning(
                        "Watchlist: no signals for %s — using %s (%d day(s) old). Tried: %s",
                        start, candidate, days_back, tried,
                    )
                else:
                    log.info("Watchlist: loaded signals from s3://%s/%s", s3_bucket, signals_key)
                break
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("NoSuchKey", "AccessDenied", "403"):
                    log.info("No signals for %s (%s), looking further back...", candidate, code)
                    tried.append(str(candidate))
                    continue
                raise
            except Exception as exc:
                log.info("Error reading signals for %s: %s", candidate, exc)
                tried.append(str(candidate))
                continue
        else:
            raise RuntimeError(
                f"No signals found within {max_lookback} days of {start}. "
                f"Dates tried: {tried}. Ensure research pipeline ran recently, "
                "or provide a local path: --watchlist /path/to/signals.json"
            )
    else:
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Use --watchlist auto to pull from S3, or provide a local path."
            )
        data = json.loads(local_path.read_text())
        log.info("Watchlist: loaded from %s", path)

        # Check if this is a population file
        if "population" in data and isinstance(data["population"], list):
            pop_tickers = [p["ticker"] for p in data["population"] if "ticker" in p]
            if pop_tickers:
                sources = {t.upper(): "population" for t in pop_tickers}
                tickers = sorted(sources.keys())
                log.info("Watchlist: %d tickers from population file", len(tickers))
                return tickers, sources, data

    # ── Extract tickers from signals.json ───────────────────────────────────
    universe_tickers = {
        e["ticker"].upper() for e in data.get("universe", []) if "ticker" in e
    }

    sources: dict[str, str] = {}
    for t in universe_tickers:
        sources[t] = "tracked"

    tickers = sorted(sources.keys())
    log.info(
        "Watchlist: %d universe tickers",
        len(tickers),
    )
    return tickers, sources, data


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Load ticker universe from watchlist or signals, plus sector map.

    Supplemental-scoring mode: if ``ctx.explicit_tickers`` is non-empty, the
    universe is forced to that list (regardless of signals.json content).
    signals_data is still loaded from S3 when available so the email renderer
    and ticker-source annotations have the research context; if signals.json
    is unavailable in this mode we proceed with an empty dict — the Step
    Function coverage-gap path has already validated the tickers exist in
    signals.json before invoking, so we don't re-enforce that here.
    """

    # ── Explicit-tickers (supplemental-scoring) short-circuit ────────────────
    if ctx.explicit_tickers:
        ctx.tickers = list(ctx.explicit_tickers)
        try:
            _, ctx.signals_data = get_universe_tickers(ctx.bucket, ctx.date_str)
        except Exception as exc:
            log.warning(
                "Supplemental mode: signals.json load failed (%s) — "
                "proceeding without research context", exc,
            )
            ctx.signals_data = {}
        # Annotate sources from signals_data when possible
        buy_set = {
            e.get("ticker", "").upper()
            for e in (ctx.signals_data.get("buy_candidates") or [])
            if isinstance(e, dict)
        }
        universe_set = {
            e.get("ticker", "").upper()
            for e in (ctx.signals_data.get("universe") or [])
            if isinstance(e, dict)
        }
        ctx.ticker_sources = {
            t: (
                "buy_candidate" if t in buy_set
                else "tracked" if t in universe_set
                else "supplemental"
            )
            for t in ctx.tickers
        }
        log.info(
            "Supplemental-scoring mode: %d explicit tickers (%s)",
            len(ctx.tickers), ", ".join(ctx.tickers[:10]) + ("…" if len(ctx.tickers) > 10 else ""),
        )
    # ── Ticker universe ──────────────────────────────────────────────────────
    elif ctx.watchlist_path:
        ctx.tickers, ctx.ticker_sources, ctx.signals_data = load_watchlist(
            path=ctx.watchlist_path,
            s3_bucket=ctx.bucket,
            date_str=ctx.date_str,
        )
    else:
        ctx.tickers, ctx.signals_data = get_universe_tickers(ctx.bucket, ctx.date_str)

    # ── Sector map ───────────────────────────────────────────────────────────
    sector_map_path = Path("data/cache/sector_map.json")
    if sector_map_path.exists():
        try:
            ctx.sector_map = json.loads(sector_map_path.read_text())
            log.info("Sector map loaded: %d mappings", len(ctx.sector_map))
        except Exception as exc:
            log.warning("Could not load sector_map.json: %s — sector_vs_spy_5d will be 0", exc)
    else:
        log.warning(
            "data/cache/sector_map.json not found — sector_vs_spy_5d will be 0. "
            "Run bootstrap_fetcher.py to generate it."
        )
