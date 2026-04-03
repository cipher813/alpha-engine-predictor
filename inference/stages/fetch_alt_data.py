"""Stage: fetch_alt_data — Fetch alternative data (earnings, revisions, options, fundamentals)."""

from __future__ import annotations

import logging

from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> None:
    """Fetch alternative data sources. Each source fails independently."""

    # O10: Earnings
    try:
        from data.earnings_fetcher import fetch_earnings_data, cache_earnings_to_s3
        ctx.earnings_all = fetch_earnings_data(ctx.tickers, reference_date=ctx.date_str)
        cache_earnings_to_s3(ctx.earnings_all, ctx.date_str, ctx.bucket)
        log.info("O10: Fetched earnings data for %d tickers", len(ctx.earnings_all))
    except Exception as exc:
        log.warning("O10: Earnings data fetch failed (features will use defaults): %s", exc)

    # O11: Revisions
    try:
        from data.earnings_fetcher import fetch_revision_history
        ctx.revision_all = fetch_revision_history(
            ctx.tickers, bucket=ctx.bucket, reference_date=ctx.date_str,
        )
        log.info("O11: Loaded revision data for %d tickers", len(ctx.revision_all))
    except Exception as exc:
        log.warning("O11: Revision data fetch failed (features will use defaults): %s", exc)

    # O12: Options
    try:
        from data.options_fetcher import load_historical_options, fetch_options_features
        ctx.options_all = load_historical_options(ctx.date_str, ctx.bucket) or {}
        if ctx.options_all:
            log.info("O12: Loaded cached options for %d tickers from S3", len(ctx.options_all))
        else:
            ctx.options_all = fetch_options_features(ctx.tickers, reference_date=ctx.date_str)
            log.info("O12: Fetched options features for %d tickers via yfinance", len(ctx.options_all))
    except Exception as exc:
        log.warning("O12: Options features fetch failed (features will use defaults): %s", exc)

    # Fundamentals — read from S3 cache (written weekly by DataPhase1)
    try:
        import boto3, json
        _s3_fund = boto3.client("s3")
        # Try today's date, then scan for most recent
        _fund_data = None
        for _fund_key in [f"archive/fundamentals/{ctx.date_str}.json"]:
            try:
                _obj = _s3_fund.get_object(Bucket=ctx.bucket, Key=_fund_key)
                _fund_data = json.loads(_obj["Body"].read())
                break
            except Exception:
                pass
        if not _fund_data:
            # Scan for most recent
            _resp = _s3_fund.list_objects_v2(Bucket=ctx.bucket, Prefix="archive/fundamentals/", MaxKeys=100)
            _keys = sorted([c["Key"] for c in _resp.get("Contents", []) if c["Key"].endswith(".json")], reverse=True)
            if _keys:
                _obj = _s3_fund.get_object(Bucket=ctx.bucket, Key=_keys[0])
                _fund_data = json.loads(_obj["Body"].read())
                log.info("Fundamentals: Using cached data from %s", _keys[0])
        if _fund_data:
            ctx.fundamental_all = _fund_data
            log.info("Fundamentals: Loaded cached data for %d tickers from S3", len(_fund_data))
        else:
            log.info("Fundamentals: No cached data found — features will use defaults")
    except Exception as exc:
        log.warning("Fundamentals: S3 load failed (features will use defaults): %s", exc)

    # Aggregate alert
    _alt_data_sources = {
        "O10_earnings": ctx.earnings_all,
        "O11_revisions": ctx.revision_all,
        "O12_options": ctx.options_all,
        "fundamentals": ctx.fundamental_all,
    }
    _alt_data_populated = {k: len(v) for k, v in _alt_data_sources.items() if v}
    if not _alt_data_populated:
        log.error(
            "ALL alternative data sources failed — model running on technical "
            "features only. Predictions may be degraded. Sources attempted: %s",
            list(_alt_data_sources.keys()),
        )
    else:
        log.info("Alternative data summary: %s", _alt_data_populated)
