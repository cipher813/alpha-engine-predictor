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

    # Structured logging: JSON on EC2/Lambda (ALPHA_ENGINE_JSON_LOGS=1), text locally.
    try:
        from log_config import setup_logging
        setup_logging("predictor")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logging.getLogger().setLevel(logging.INFO)

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
    try:
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

    except Exception as exc:
        log.exception("Predictor Lambda failed: %s", exc)
        return {
            "statusCode": 500,
            "body": f"Predictor Lambda failed: {exc}",
        }
