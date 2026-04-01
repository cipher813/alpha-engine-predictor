"""Stage: write_output — Build metrics, apply veto, write predictions, send email, write health."""

from __future__ import annotations

import json
import logging
import time as _time

import config as cfg
from inference.pipeline import PipelineContext

log = logging.getLogger(__name__)


def run(ctx: PipelineContext) -> None:
    """Write predictions, metrics, email, and health status."""
    from inference.daily_predict import (
        get_veto_threshold, write_predictions, send_predictor_email,
    )

    # ── Build metrics ────────────────────────────────────────────────────────
    gbm_meta = _load_gbm_meta(ctx)

    if ctx.model_type == "gbm":
        last_trained = gbm_meta.get("trained_date", ctx.scorer._best_iteration)
    else:
        last_trained = ctx.checkpoint.get("epoch", "unknown")

    metrics = {
        "model_version": ctx.model_version,
        "model_type": ctx.model_type,
        "inference_mode": ctx.inference_mode if ctx.model_type == "gbm" else "mlp",
        "last_trained": last_trained,
        "training_samples": gbm_meta.get("n_train") if ctx.model_type == "gbm" else None,
        "val_loss": round(float(ctx.val_loss), 6) if isinstance(ctx.val_loss, (int, float)) else None,
        "ic_30d": gbm_meta.get("test_ic") if ctx.model_type == "gbm" else None,
        "ic_ir_30d": gbm_meta.get("ic_ir") if ctx.model_type == "gbm" else None,
        "hit_rate_30d_rolling": None,
        "price_freshness": {
            "max_age_days": max(ctx.ticker_data_age.values()) if ctx.ticker_data_age else -1,
            "n_stale": sum(1 for d in ctx.ticker_data_age.values() if d > 1),
        },
    }

    # ── Veto logic ───────────────────────────────────────────────────────────
    market_regime = ctx.signals_data.get("market_regime", "") if ctx.signals_data else ""
    veto_thresh = get_veto_threshold(ctx.bucket, market_regime=market_regime)

    n_preds = len(ctx.predictions)
    for p in ctx.predictions:
        cr = p.get("combined_rank")
        alpha = p.get("predicted_alpha", 0) or 0
        p["gbm_veto"] = (alpha < 0 and cr is not None and cr > n_preds / 2)

    # ── Write predictions ────────────────────────────────────────────────────
    write_predictions(ctx.predictions, ctx.date_str, ctx.bucket, metrics,
                      dry_run=ctx.dry_run, veto_threshold=veto_thresh, fd=ctx.fd)

    # ── Send email ───────────────────────────────────────────────────────────
    if not ctx.dry_run:
        email_sent = send_predictor_email(
            ctx.predictions, metrics, ctx.date_str,
            signals_data=ctx.signals_data, veto_threshold=veto_thresh,
        )
        if not email_sent:
            log.warning("Predictor email failed to send (Gmail + SES both failed)")

    # ── Health status ────────────────────────────────────────────────────────
    try:
        from health_status import write_health
        n_up = sum(1 for p in ctx.predictions if p.get("predicted_direction") == "UP")
        n_down = sum(1 for p in ctx.predictions if p.get("predicted_direction") == "DOWN")
        write_health(
            bucket=ctx.bucket,
            module_name="predictor_inference",
            status="ok",
            run_date=ctx.date_str,
            duration_seconds=ctx.elapsed_seconds(),
            summary={
                "n_predictions": len(ctx.predictions),
                "n_up": n_up,
                "n_down": n_down,
            },
        )
    except Exception as _he:
        log.warning("Health status write failed: %s", _he)

    # ── Data manifest ────────────────────────────────────────────────────────
    try:
        from health_status import write_data_manifest
        write_data_manifest(
            bucket=ctx.bucket,
            module_name="predictor_inference",
            run_date=ctx.date_str,
            manifest={
                "n_predictions": len(ctx.predictions),
                "n_up": sum(1 for p in ctx.predictions if p.get("predicted_direction") == "UP"),
                "n_down": sum(1 for p in ctx.predictions if p.get("predicted_direction") == "DOWN"),
                "n_tickers_failed": len(getattr(cfg, 'FAILED_TICKERS', [])),
                "model_version": getattr(cfg, 'GBM_VERSION', 'unknown'),
            },
        )
    except Exception as _me:
        log.warning("Data manifest write failed: %s", _me)

    log.info("Predictor run complete for %s", ctx.date_str)


def _load_gbm_meta(ctx: PipelineContext) -> dict:
    """Load GBM training metadata from S3 (best-effort)."""
    if ctx.model_type != "gbm" or ctx.local:
        return {}
    try:
        import boto3 as _boto3
        _s3 = _boto3.client("s3")
        _resp = _s3.get_object(Bucket=ctx.bucket, Key=cfg.GBM_WEIGHTS_META_KEY)
        meta = json.loads(_resp["Body"].read())
        log.info("GBM weights meta loaded: trained_date=%s  n_train=%s",
                 meta.get("trained_date"), meta.get("n_train"))
        return meta
    except Exception as _exc:
        log.debug("GBM weights meta not found or unreadable: %s", _exc)
        return {}
