"""
regime/handler.py — Lambda entry point for the weekly RegimeSubstrate run.

Orchestrates the Saturday SF ``RegimeSubstrate`` state (inserted between
``RAGIngestion`` and ``Research`` per regime-v3-260514.md §5.4):

1. Fetch the historical macro feature DataFrame from data-side parquets
   (``regime/features.py``).
2. Fit a fresh HMM weekly on the 10y rolling history (always-refit
   design — no weights persistence in this PR; refitting is cheap
   relative to Lambda cold-start and keeps the artifact reproducible
   from the data side alone).
3. Assemble the substrate payload via ``regime/substrate.py``.
4. Publish the canonical-shape artifact to S3
   (``s3://alpha-engine-research/regime/{YYMMDDHHMM}.json`` +
   ``latest.json`` sidecar).

The macro economist agent (Stage C) reads ``latest.json`` and consumes
the substrate as a strong prior. The agent remains the final authority;
this Lambda is informational.

Why always-refit
----------------
HMM fit on ~520 weekly observations is seconds; Lambda cold-start +
boto3 init dominates. Persisting weights and partial-refitting adds
complexity without saving runtime, and obscures forensic replay (the
artifact is reproducible from the price-cache parquets alone). Weight
persistence may be revisited if HMM fit time grows materially or if
inter-week posterior continuity becomes a downstream requirement.

Invocation modes
----------------
- ``action == "produce"`` (default) — fit HMM, build substrate, write to S3.
- ``action == "dry_run"`` — fit HMM, build substrate, return payload
  without writing (for ad-hoc inspection + Stage A backfill validation).

Event payload (all optional with sensible defaults):

    {
      "action": "produce" | "dry_run",
      "bucket": "alpha-engine-research",
      "price_cache_prefix": "predictor/price_cache/",
      "regime_prefix": "regime",
      "lookback_weeks": 520,
      "hmm_feature_columns": ["spy_20d_return", "vix_level", "hy_oas_bps"]
    }

Lambda configuration (set at deploy time — separate alpha-engine-config PR):
- Runtime: container image (predictor's existing ECR image, separate CMD)
- Memory: 1024 MB (HMM fit is the bottleneck; ~520 × 3 features is tiny)
- Timeout: 300 seconds (fit ~10s + S3 IO ~5s + headroom)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

# Ensure project root on sys.path for sibling imports (regime/, etc.).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from regime.features import (
    DEFAULT_PRICE_CACHE_PREFIX,
    current_features_from_history,
    fetch_macro_feature_history,
)
from nousergon_lib.logging import monitor_handler, setup_logging
from regime.hmm import HMMRegimeClassifier
from regime.substrate import (
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    REGIME_FEATURES,
    build_regime_substrate,
    write_regime_substrate,
)


# Structured logging + flow-doctor. setup_logging attaches a FlowDoctorHandler
# at ERROR (off under pytest) so every log.error() in this regime Lambda routes
# through flow-doctor's capture -> dedupe -> alert dispatch. flow-doctor-regime.yaml
# ships at the Lambda task root (Dockerfile COPY); ${VAR} secrets come from the
# Lambda --environment block. Mirrors inference/handler.py.
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor-regime.yaml",
)
setup_logging(
    "predictor-regime",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)


# HMM feature subset — must include PIN_FEATURE (``spy_20d_return``) and
# at minimum two more for K=3 to discover meaningful states. The default
# matches what's available in the FRED price-cache today; can be
# overridden via event payload as more features land.
DEFAULT_HMM_FEATURE_COLUMNS: tuple[str, ...] = (
    "spy_20d_return",
    "vix_level",
    "hy_oas_bps",
)


def produce_regime_substrate(
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    price_cache_prefix: str = DEFAULT_PRICE_CACHE_PREFIX,
    regime_prefix: str = DEFAULT_S3_PREFIX,
    lookback_weeks: int = 520,
    hmm_feature_columns: tuple[str, ...] = DEFAULT_HMM_FEATURE_COLUMNS,
    write: bool = True,
    as_of: "pd.Timestamp | None" = None,
    run_id: str | None = None,
    calendar_date: str | None = None,
    trading_day: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Run the substrate pipeline end-to-end.

    Returns the substrate payload dict. When ``write=True``, also
    publishes to S3 and includes ``artifact_key`` + ``latest_key`` in
    the return value alongside the payload.

    Backfill parameters (use defaults for live Saturday SF invocation):

    - ``as_of``         truncate the feature history to this timestamp,
                        reproducing the substrate as it would have been
                        written on that date. Used by the historical
                        backfill script (scripts/backfill_regime_substrate.py)
                        to populate the dashboard's history window with
                        ~10y of weekly artifacts.
    - ``run_id``        override the canonical YYMMDDHHMM run_id with a
                        caller-supplied string. Backfill encodes the
                        historical Saturday's date into the run_id so
                        artifacts sort chronologically alongside live
                        weekly runs.
    - ``calendar_date`` / ``trading_day``  override the dual-tracked
                        dates so the artifact payload reflects the
                        historical run rather than today's clock.
    - ``write_latest``  when False, skip the latest.json sidecar update.
                        Backfill must NEVER touch latest.json — that
                        pointer reflects live state.
    """
    history = fetch_macro_feature_history(
        s3_client=s3_client,
        bucket=bucket,
        prefix=price_cache_prefix,
        lookback_weeks=lookback_weeks,
        as_of=as_of,
    )
    if history.empty:
        raise RuntimeError(
            f"No macro feature history available at s3://{bucket}/{price_cache_prefix} — "
            f"upstream FRED ingestion likely failed."
        )

    # Drop any HMM-feature columns that came back all-NaN (e.g., breadth
    # before its historical series lands) before fitting.
    available_hmm_cols = tuple(
        c for c in hmm_feature_columns
        if c in history.columns and not history[c].dropna().empty
    )
    if "spy_20d_return" not in available_hmm_cols:
        raise RuntimeError(
            "spy_20d_return is required for HMM state pinning and was not "
            "available in fetched history."
        )

    hmm_input = history[list(available_hmm_cols)].dropna()
    hmm = HMMRegimeClassifier(n_states=3, random_state=42).fit(hmm_input, available_hmm_cols)

    current = current_features_from_history(history)

    # ── Drawdown regime leg — weekly forensic block (additive) ───────────
    # regime-drawdown-hysteresis-260518.md §4.1. Reproducible from the
    # price cache (replayed through the hysteresis), mirroring the HMM
    # refit-from-history discipline. Best-effort: a failure here writes
    # the substrate WITHOUT the drawdown key (S3-contract-safe, ADD-only)
    # — the brief/macro-agent fall back to the HMM-only path. The excess
    # leg is live-only (point-in-time eod_pnl is not reproducible for an
    # ``as_of`` historical backfill).
    drawdown_block = None
    try:
        from regime.drawdown import block_from_history, read_eod_pnl_nav
        from regime.features import _read_parquet_close

        spy_close = _read_parquet_close(
            "SPY", s3_client=s3_client, bucket=bucket,
            prefix=price_cache_prefix,
        )
        if as_of is not None and len(spy_close):
            spy_close = spy_close[spy_close.index <= as_of]
        nav_hist = (
            read_eod_pnl_nav(s3_client, bucket=bucket)
            if as_of is None else None
        )
        drawdown_block = block_from_history(spy_close, nav_history=nav_hist)
    except Exception as _dd_err:  # noqa: BLE001 — additive, never block
        logger.warning(
            "[regime_substrate] drawdown block assembly failed (%s) — "
            "substrate written without it (additive, S3-contract-safe).",
            _dd_err,
        )
        drawdown_block = None

    payload = build_regime_substrate(
        feature_history=hmm_input,  # only the cols HMM saw at fit time
        current_features=current,
        hmm=hmm,
        run_id=run_id,
        calendar_date=calendar_date,
        trading_day=trading_day,
        fit_window_start=str(hmm_input.index.min().date()),
        fit_window_end=str(hmm_input.index.max().date()),
        drawdown_block=drawdown_block,
    )

    if not write:
        return {"payload": payload, "wrote": False}

    keys = write_regime_substrate(
        payload, s3_client=s3_client, bucket=bucket, prefix=regime_prefix,
        write_latest=write_latest,
    )
    return {"payload": payload, "wrote": True, **keys}


@monitor_handler
def lambda_handler(event: dict | None, context: Any) -> dict[str, Any]:
    """AWS Lambda entry point.

    Routes ``event["action"]`` to either ``produce`` (default, writes
    to S3) or ``dry_run`` (returns payload without writing).
    """
    event = event or {}
    action = event.get("action", "produce")

    import boto3
    s3_client = boto3.client("s3")

    kwargs = {
        "s3_client": s3_client,
        "bucket": event.get("bucket", DEFAULT_S3_BUCKET),
        "price_cache_prefix": event.get("price_cache_prefix", DEFAULT_PRICE_CACHE_PREFIX),
        "regime_prefix": event.get("regime_prefix", DEFAULT_S3_PREFIX),
        "lookback_weeks": int(event.get("lookback_weeks", 520)),
        "hmm_feature_columns": tuple(event.get("hmm_feature_columns", DEFAULT_HMM_FEATURE_COLUMNS)),
    }

    if action == "produce":
        result = produce_regime_substrate(**kwargs, write=True)
        return {
            "statusCode": 200,
            "action": action,
            "run_id": result["payload"]["run_id"],
            "artifact_key": result["artifact_key"],
            "latest_key": result["latest_key"],
            "hmm_argmax": result["payload"]["hmm"]["argmax"],
            "regime_change_signal": result["payload"]["bocpd"]["change_signal"],
        }
    elif action == "dry_run":
        result = produce_regime_substrate(**kwargs, write=False)
        return {
            "statusCode": 200,
            "action": action,
            "payload": result["payload"],
        }
    else:
        return {
            "statusCode": 400,
            "error": f"unknown action {action!r}; expected 'produce' or 'dry_run'",
        }
