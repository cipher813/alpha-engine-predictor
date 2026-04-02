"""Stage: load_universe — Determine ticker universe and load sector map."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> None:
    """Load ticker universe from watchlist or signals, plus sector map."""
    from inference.daily_predict import load_watchlist, get_universe_tickers

    # ── Ticker universe ──────────────────────────────────────────────────────
    if ctx.watchlist_path:
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
