"""
inference/handler.py — AWS Lambda handler (inference only).

Lambda runs daily GBM + CatBoost inference. Training has moved to EC2 spot
instance (infrastructure/spot_train.sh) due to CatBoost + multi-horizon
training exceeding Lambda's 15-minute timeout.

  action == "predict" (default, or omitted):
    Triggered by EventBridge Mon–Fri at 6:15am PT. Loads LightGBM + CatBoost
    models from S3, runs blended inference on the research watchlist, writes
    predictions to S3. Sends predictor email.

  action == "check_coverage":
    Compute buy_candidates - predictions delta and return the missing tickers.
    Used by the weekday Step Function's coverage-gap Choice state to decide
    whether to re-invoke `action=predict` with a `tickers` payload.

  action == "check_deploy_drift":
    Compare the deployed Step Function definition + CloudFormation stack
    SHAs against alpha-engine-data @main HEAD on GitHub. Used by the
    weekday Step Function's first state (DeployDriftCheck) to halt the
    pipeline when infrastructure code has been merged but not deployed.

  action == "train":
    DEPRECATED — returns error directing to spot_train.sh.

Lambda configuration:
  - Runtime: container image (public.ecr.aws/lambda/python:3.12)
  - Memory: 3072 MB  (inference with LightGBM + CatBoost + multi-horizon)
  - Timeout: 900 seconds  (inference takes ~3–4 min)
  - Environment variables:
      S3_BUCKET          — override default bucket (optional)
      EMAIL_SENDER       — from-address for notification emails
      EMAIL_RECIPIENTS   — comma-separated recipient list
      GMAIL_APP_PASSWORD — Gmail App Password (enables SMTP path)
      AWS_REGION         — SES fallback region (default: us-east-1)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Load secrets from SSM Parameter Store (must run before any os.environ.get)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ssm_secrets import load_secrets
load_secrets()

log = logging.getLogger(__name__)


def handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    event may contain:
        action    (str)        : "predict" (default) | "train"
        date      (str)        : Override date YYYY-MM-DD.
        dry_run   (bool)       : If True, skip S3 writes and email (for testing).
        tickers   (list[str])  : Supplemental-scoring mode. When non-empty,
                                 score ONLY these tickers and merge into the
                                 existing predictions/{date}.json. Used by the
                                 weekday Step Function's coverage-gap re-invoke.
    """
    os.environ.setdefault("S3_BUCKET", "alpha-engine-research")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    # Structured logging + flow-doctor via alpha-engine-lib (shared with
    # alpha-engine-data). Replaces the duplicated local log_config.py.
    from alpha_engine_lib.logging import setup_logging
    setup_logging(
        "predictor",
        flow_doctor_yaml=str(Path(__file__).parent.parent / "flow-doctor.yaml"),
    )
    logging.getLogger().setLevel(logging.INFO)

    # Preflight — fail fast on env / connectivity / ArcticDB freshness
    # before loading models or touching inference. See PR #5 and
    # inference/preflight.py.
    #
    # Action-aware dispatch: drift-check is a Step Function gate that only
    # needs env + image-SHA validation. Running the full preflight here
    # (200s universe scan + 5 macro reads + model-weights check) caused
    # the 2026-05-01 SF DeployDriftCheck timeout cascade. Other actions
    # need the full preflight before doing real work.
    from inference.preflight import PredictorPreflight
    _bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")
    _action = event.get("action", "predict")
    _pf = PredictorPreflight(bucket=_bucket)
    if _action == "check_deploy_drift":
        _pf.run_for_drift_gate()
    else:
        _pf.run()

    fd = None

    action  = event.get("action", "predict")
    date_str = event.get("date", None)
    dry_run  = bool(event.get("dry_run", False))
    raw_tickers = event.get("tickers") or []
    if isinstance(raw_tickers, str):
        raw_tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
    explicit_tickers = [t.upper() for t in raw_tickers if t]

    log.info(
        "Lambda invocation: action=%s  date=%s  dry_run=%s  tickers=%s  function=%s",
        action,
        date_str or "today",
        dry_run,
        f"{len(explicit_tickers)} supplemental" if explicit_tickers else "full universe",
        getattr(context, "function_name", "local"),
    )

    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")

    # ── Coverage check (Step Function coverage-gap Choice state) ────────────
    if action == "check_coverage":
        from inference.coverage_check import compute_coverage_delta
        result = compute_coverage_delta(bucket=bucket, date_str=date_str)
        log.info(
            "Coverage check: %d buy_candidates, %d predictions, %d missing → %s",
            result["n_buy_candidates"], result["n_predictions"],
            result["missing_count"],
            ", ".join(result["missing_tickers"][:10]) + (
                "…" if len(result["missing_tickers"]) > 10 else ""
            ) if result["missing_tickers"] else "none",
        )
        return result

    # ── Deploy-drift check (Step Function first state) ──────────────────────
    if action == "check_deploy_drift":
        from inference.deploy_drift import check_deploy_drift
        account_id = (
            getattr(context, "invoked_function_arn", "").split(":")[4]
            if context is not None and getattr(context, "invoked_function_arn", "")
            else os.environ.get("AWS_ACCOUNT_ID", "")
        )
        result = check_deploy_drift(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            account_id=account_id,
        )
        log.info(
            "Deploy-drift check: upstream=%s  sf=%s(drift=%s)  cf=%s(drift=%s)",
            (result["upstream_sha"] or "?")[:12],
            (result["sf_sha"] or "missing")[:12], result["sf_drift"],
            (result["stack_sha"] or "missing")[:12], result["cf_drift"],
        )
        return result

    # ── Train (DEPRECATED — moved to EC2 spot instance) ─────────────────────
    if action == "train":
        log.warning(
            "action=train is deprecated on Lambda. Training now runs on EC2 spot "
            "via infrastructure/spot_train.sh (CatBoost + multi-horizon exceeds "
            "Lambda's 15-minute timeout)."
        )
        return {
            "statusCode": 400,
            "body": (
                "Training has moved to EC2 spot instance. "
                "Use infrastructure/spot_train.sh or the Saturday cron. "
                "Lambda is inference-only."
            ),
        }

    # ── Predict (default) ──────────────────────────────────────────────────────
    # Any failure must raise so Step Functions sees a real task failure and the
    # Catch branch blocks downstream executor. Returning statusCode:500 would
    # look like a successful Lambda response and let executor proceed on stale
    # predictions — the exact silent-failure mode that hit production on
    # 2026-04-13.
    from inference.daily_predict import main
    main(
        date_str=date_str,
        dry_run=dry_run,
        local=False,
        model_type="gbm",
        watchlist_path="auto",
        explicit_tickers=explicit_tickers,
    )
    log.info("Predictor Lambda completed successfully")
    return {
        "statusCode": 200,
        "body": (
            f"Supplemental predictions written for {date_str or 'today'} "
            f"({len(explicit_tickers)} tickers)"
            if explicit_tickers else
            f"Predictions written for {date_str or 'today'}"
        ),
    }
