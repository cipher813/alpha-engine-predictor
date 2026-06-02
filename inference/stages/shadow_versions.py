"""Stage: shadow_versions — champion/challenger Phase 1 shadow runner (L4469).

After the live (champion) inference has written predictions/{date}.json, re-score
the SAME prices/universe/alt-data with each registered CHALLENGER's weights and
write ``predictor/predictions_shadow/{version_id}/{date}.json``. Challengers
**trade on none** — only the live predictions feed the executor; these shadow
files exist so Phase 2 can score each version on realized outcomes under a fixed
policy (plan: ``alpha-engine-docs/private/model-shadow-comparison-260601.md``).

Cost-aware design: the expensive fetches (load_universe / load_prices /
fetch_alt_data) are NOT repeated — the shadow context reuses the live context's
already-loaded universe + prices + alt-data and only swaps the model weights (via
``ctx.weights_prefix_override`` → the challenger's registry bundle) and re-runs
load_model + run_inference.

Safety: this stage is non-critical and time-guarded. It runs AFTER write_output
(the live predictions are already on S3 and untouched), skips remaining
challengers near the Lambda soft-timeout, and lets any single-challenger failure
log-and-continue. It can never delay, corrupt, or block the live path. No-op
until challengers exist in the registry (they register from each retrain via the
L4469 capture fix).
"""
from __future__ import annotations

import json
import logging

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)

_SHADOW_PREFIX = "predictor/predictions_shadow"
_REGISTRY_PREFIX = "predictor/registry/"

# EXPENSIVE-to-recompute fields, identical across model versions → shared
# (shallow-copied so per-shadow key mutations don't leak between challengers;
# the underlying price frames are read-only during feature compute).
_SHARED_FIELDS = (
    "tickers", "ticker_sources", "signals_data", "sector_map",
    "price_data", "macro", "ticker_data_age",
    "earnings_all", "revision_all", "options_all", "fundamental_all",
)
_IDENTITY_FIELDS = (
    "date_str", "bucket", "dry_run", "local", "model_type",
    "start_ts", "soft_timeout_s",
)


def _clone_for_shadow(ctx: PipelineContext, *, weights_prefix: str) -> PipelineContext:
    """Fresh context that reuses the live run's loaded data but resets the
    model/result state and points the loader at a challenger's bundle."""
    shadow = PipelineContext()
    for f in _IDENTITY_FIELDS:
        setattr(shadow, f, getattr(ctx, f))
    for f in _SHARED_FIELDS:
        val = getattr(ctx, f)
        setattr(
            shadow, f,
            dict(val) if isinstance(val, dict)
            else list(val) if isinstance(val, list) else val,
        )
    shadow.weights_prefix_override = weights_prefix
    return shadow


def _write_shadow(ctx: PipelineContext, version_id: str) -> None:
    import boto3

    payload = {
        "date": ctx.date_str,
        "version_id": version_id,
        "model_version": getattr(ctx, "model_version", "unknown"),
        "n_predictions": len(ctx.predictions),
        "shadow": True,  # marker — NEVER consumed by the executor
        "predictions": ctx.predictions,
    }
    key = f"{_SHADOW_PREFIX}/{version_id}/{ctx.date_str}.json"
    boto3.client("s3").put_object(
        Bucket=ctx.bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode(),
        ContentType="application/json",
    )
    log.info(
        "Shadow predictions written: s3://%s/%s (%d preds)",
        ctx.bucket, key, len(ctx.predictions),
    )


def run(ctx: PipelineContext) -> None:
    if not getattr(cfg, "SHADOW_VERSIONS_ENABLED", True):
        log.info("Shadow runner disabled (SHADOW_VERSIONS_ENABLED=False) — skip.")
        return
    if ctx.dry_run:
        log.info("Shadow runner skipped in dry-run.")
        return
    if getattr(ctx, "explicit_tickers", None):
        # Coverage-gap re-invocation scores only a subset — shadowing a partial
        # batch would write an incomparable shadow. Skip.
        log.info("Shadow runner skipped on supplemental (explicit_tickers) run.")
        return

    try:
        import boto3

        from model.registry import list_versions

        challengers = list_versions(
            boto3.client("s3"), ctx.bucket, stage="challenger",
        )
    except Exception:
        log.warning("Shadow runner: failed to list challengers — skip.", exc_info=True)
        return

    max_n = int(getattr(cfg, "SHADOW_VERSIONS_MAX_N", 3))
    challengers = challengers[:max_n]
    if not challengers:
        log.info("Shadow runner: no registered challengers — no-op.")
        return

    import importlib

    load_model = importlib.import_module("inference.stages.load_model")
    run_inference = importlib.import_module("inference.stages.run_inference")

    log.info(
        "Shadow runner: shadowing %d challenger(s) (max_n=%d), trade-on-none.",
        len(challengers), max_n,
    )
    for lineage in challengers:
        vid = lineage.get("version_id")
        if not vid:
            continue
        if ctx.near_timeout():
            log.warning(
                "Shadow runner: near Lambda soft-timeout — skipping remaining "
                "challengers (stopped before %s).", vid,
            )
            break
        try:
            shadow_ctx = _clone_for_shadow(
                ctx, weights_prefix=f"{_REGISTRY_PREFIX}{vid}/",
            )
            load_model.run(shadow_ctx)
            run_inference.run(shadow_ctx)
            _write_shadow(shadow_ctx, vid)
        except Exception:
            log.warning(
                "Shadow runner: challenger %s failed — continuing (live path "
                "unaffected).", vid, exc_info=True,
            )
            continue
