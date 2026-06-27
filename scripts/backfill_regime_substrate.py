"""scripts/backfill_regime_substrate.py — historical regime substrate
backfill.

Generates one ``regime/{YYMMDDHHMM}.json`` artifact per historical
Saturday going back N years, reproducing what the live Saturday SF
``RegimeSubstrate`` Lambda would have written had it been running at
the time. Populates the dashboard's history window + enables backtester
sweeps with v3-conditioned signals.

Each historical Saturday is processed independently:

1. ``as_of`` truncates the price-cache history to the date (no
   look-ahead into the future).
2. A fresh HMM is fit on the truncated window (matching production's
   always-refit semantics).
3. Composite z-score + BOCPD warm-up + guardrail flags computed on
   the same truncated state.
4. Artifact written to ``regime/{run_id}.json`` where the run_id is
   the synthetic ``YYMMDDHHMM`` encoding the Saturday's calendar date
   at 02:00 UTC (matches the Saturday SF cron).

CRITICAL: ``write_latest=False`` for every backfill write. The
``regime/latest.json`` sidecar reflects the live current-week substrate
and must NEVER be clobbered by a backfilled run.

Idempotency:
  Default behavior skips dates whose ``regime/{run_id}.json`` artifact
  already exists. Pass ``--force`` to overwrite.

Runtime: ~30 min for a full 10y backfill (~520 weeks × ~3s per fit).
Memory: ~500 MB peak (HMM fit on 520 weekly observations is small).

Usage:

  # Full 10y backfill, skip-existing
  python -m scripts.backfill_regime_substrate

  # Specific date range
  python -m scripts.backfill_regime_substrate --start 2024-01-01 --end 2024-12-31

  # Dry run (list dates, don't write)
  python -m scripts.backfill_regime_substrate --dry-run

  # Force overwrite existing artifacts
  python -m scripts.backfill_regime_substrate --force
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date as Date
from datetime import timedelta

# Project root on path so ``regime/`` is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3  # noqa: E402
import pandas as pd  # noqa: E402

from regime.features import DEFAULT_PRICE_CACHE_PREFIX  # noqa: E402
from regime.handler import (  # noqa: E402
    DEFAULT_HMM_FEATURE_COLUMNS,
    produce_regime_substrate,
)
from regime.substrate import DEFAULT_S3_BUCKET, DEFAULT_S3_PREFIX  # noqa: E402


logger = logging.getLogger("backfill_regime_substrate")


def _saturday_run_id(d: Date, hour: int = 2, minute: int = 0) -> str:
    """Encode a historical Saturday's date into a synthetic YYMMDDHHMM
    run_id. Hour 02:00 UTC mirrors the live Saturday SF cron, so
    backfilled artifacts sort chronologically alongside live ones."""
    return f"{d:%y%m%d}{hour:02d}{minute:02d}"


def _generate_saturdays(start: Date, end: Date) -> list[Date]:
    """Return every Saturday in [start, end] inclusive."""
    if start > end:
        return []
    # Advance start to the next Saturday (weekday() == 5)
    offset = (5 - start.weekday()) % 7
    first_sat = start + timedelta(days=offset)
    out: list[Date] = []
    cur = first_sat
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=7)
    return out


def _artifact_exists(s3_client, bucket: str, prefix: str, run_id: str) -> bool:
    """HeadObject probe — cheap idempotency check."""
    from nousergon_lib.eval_artifacts import eval_artifact_key

    key = eval_artifact_key(prefix, run_id)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def backfill_one(
    *,
    saturday: Date,
    s3_client,
    bucket: str,
    price_cache_prefix: str,
    regime_prefix: str,
    hmm_feature_columns: tuple[str, ...],
    force: bool,
    dry_run: bool,
) -> dict:
    """Process a single Saturday. Returns a status dict."""
    run_id = _saturday_run_id(saturday)
    if not force and _artifact_exists(s3_client, bucket, regime_prefix, run_id):
        return {"saturday": saturday.isoformat(), "run_id": run_id, "status": "skipped_existing"}

    if dry_run:
        return {"saturday": saturday.isoformat(), "run_id": run_id, "status": "would_write"}

    # Saturday is the calendar_date; trading_day is the last NYSE session
    # which is Friday (Saturday is not a trading day).
    trading_day = saturday - timedelta(days=1)  # naive Friday — close enough for backfill metadata

    t0 = time.time()
    try:
        result = produce_regime_substrate(
            s3_client=s3_client,
            bucket=bucket,
            price_cache_prefix=price_cache_prefix,
            regime_prefix=regime_prefix,
            hmm_feature_columns=hmm_feature_columns,
            write=True,
            as_of=pd.Timestamp(saturday),
            run_id=run_id,
            calendar_date=saturday.isoformat(),
            trading_day=trading_day.isoformat(),
            write_latest=False,  # CRITICAL: never clobber live latest.json
        )
    except Exception as e:
        elapsed = time.time() - t0
        logger.warning(
            "[backfill] %s (run_id=%s) FAILED in %.1fs: %s: %s",
            saturday, run_id, elapsed, type(e).__name__, e,
        )
        return {
            "saturday": saturday.isoformat(),
            "run_id": run_id,
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
        }

    elapsed = time.time() - t0
    hmm_argmax = result["payload"]["hmm"]["argmax"]
    intensity = result["payload"]["composite"]["intensity_z"]
    logger.info(
        "[backfill] %s (run_id=%s) OK in %.1fs — argmax=%s intensity_z=%+.2f",
        saturday, run_id, elapsed, hmm_argmax, intensity,
    )
    return {
        "saturday": saturday.isoformat(),
        "run_id": run_id,
        "status": "wrote",
        "artifact_key": result["artifact_key"],
        "hmm_argmax": hmm_argmax,
        "intensity_z": intensity,
        "elapsed_sec": elapsed,
    }


def run_backfill(
    *,
    start: Date,
    end: Date,
    s3_client=None,
    bucket: str = DEFAULT_S3_BUCKET,
    price_cache_prefix: str = DEFAULT_PRICE_CACHE_PREFIX,
    regime_prefix: str = DEFAULT_S3_PREFIX,
    hmm_feature_columns: tuple[str, ...] = DEFAULT_HMM_FEATURE_COLUMNS,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Drive the backfill across a date range. Returns per-Saturday
    status dicts. Errors are captured per-date rather than aborting —
    a single failed week shouldn't kill a 10y run."""
    if s3_client is None:
        s3_client = boto3.client("s3")

    saturdays = _generate_saturdays(start, end)
    logger.info(
        "[backfill] processing %d Saturdays %s → %s (force=%s dry_run=%s)",
        len(saturdays), start, end, force, dry_run,
    )

    results: list[dict] = []
    for sat in saturdays:
        results.append(backfill_one(
            saturday=sat,
            s3_client=s3_client,
            bucket=bucket,
            price_cache_prefix=price_cache_prefix,
            regime_prefix=regime_prefix,
            hmm_feature_columns=hmm_feature_columns,
            force=force,
            dry_run=dry_run,
        ))

    # Summary
    n_wrote = sum(1 for r in results if r["status"] == "wrote")
    n_skipped = sum(1 for r in results if r["status"] == "skipped_existing")
    n_would = sum(1 for r in results if r["status"] == "would_write")
    n_error = sum(1 for r in results if r["status"] == "error")
    logger.info(
        "[backfill] summary — wrote=%d skipped_existing=%d would_write=%d error=%d",
        n_wrote, n_skipped, n_would, n_error,
    )
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", type=_parse_date, default=None,
                   help="Start date YYYY-MM-DD. Default: 10y ago.")
    p.add_argument("--end", type=_parse_date, default=None,
                   help="End date YYYY-MM-DD. Default: yesterday.")
    p.add_argument("--bucket", default=DEFAULT_S3_BUCKET)
    p.add_argument("--price-cache-prefix", default=DEFAULT_PRICE_CACHE_PREFIX)
    p.add_argument("--regime-prefix", default=DEFAULT_S3_PREFIX)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing artifacts (default: skip).")
    p.add_argument("--dry-run", action="store_true",
                   help="List dates that would be processed; do not write.")
    return p.parse_args()


def _parse_date(s: str) -> Date:
    return Date.fromisoformat(s)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    today = Date.today()
    end = args.end or (today - timedelta(days=1))
    start = args.start or (today - timedelta(days=365 * 10))

    results = run_backfill(
        start=start, end=end,
        bucket=args.bucket,
        price_cache_prefix=args.price_cache_prefix,
        regime_prefix=args.regime_prefix,
        force=args.force,
        dry_run=args.dry_run,
    )

    n_error = sum(1 for r in results if r["status"] == "error")
    # Don't fail-hard on partial errors — backfill is best-effort over
    # a long range. Exit-1 only on systemic failure (no Saturdays found).
    if not results:
        return 1
    if n_error == len(results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
