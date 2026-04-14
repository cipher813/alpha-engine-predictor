"""
inference/handler.py — AWS Lambda handler (inference only).

Lambda runs daily GBM + CatBoost inference. Training has moved to EC2 spot
instance (infrastructure/spot_train.sh) due to CatBoost + multi-horizon
training exceeding Lambda's 15-minute timeout.

  action == "predict" (default, or omitted):
    Triggered by EventBridge Mon–Fri at 6:15am PT. Loads LightGBM + CatBoost
    models from S3, runs blended inference on the research watchlist, writes
    predictions to S3. Sends predictor email.

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
        action    (str)  : "predict" (default) | "train"
        date      (str)  : Override date YYYY-MM-DD.
        dry_run   (bool) : If True, skip S3 writes and email (for testing).
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
    from inference.preflight import PredictorPreflight
    PredictorPreflight(bucket=os.environ.get("S3_BUCKET", "alpha-engine-research")).run()

    fd = None

    action  = event.get("action", "predict")
    date_str = event.get("date", None)
    dry_run  = bool(event.get("dry_run", False))

    log.info(
        "Lambda invocation: action=%s  date=%s  dry_run=%s  function=%s",
        action,
        date_str or "today",
        dry_run,
        getattr(context, "function_name", "local"),
    )

    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")

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
    )
    log.info("Predictor Lambda completed successfully")
    return {
        "statusCode": 200,
        "body": f"Predictions written for {date_str or 'today'}",
    }
