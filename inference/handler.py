"""
inference/handler.py — AWS Lambda handler.

Supports two actions via the event payload:

  action == "predict" (default, or omitted):
    Triggered by EventBridge Mon–Fri at 6:15am PT. Runs daily GBM inference
    on the research watchlist and writes predictions to S3. Sends predictor email.

  action == "train":
    Triggered by EventBridge Monday 07:00 UTC (Sun ~11pm PT). Downloads the price cache
    from S3, retrains GBMScorer, uploads new weights if IC gate passes,
    and sends a training summary email.

Lambda configuration:
  - Runtime: container image (public.ecr.aws/lambda/python:3.12)
  - Memory: 3072 MB  (training needs ~2 GB; inference fine with less but shared config)
  - Timeout: 900 seconds  (training takes ~8–12 min; inference takes ~2–3 min)
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

    # Lambda runtime pre-configures the root logger, making basicConfig a no-op.
    # Explicitly set the root logger level to ensure INFO lines reach CloudWatch.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
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

    # ── Train ──────────────────────────────────────────────────────────────────
    if action == "train":
        try:
            from training.train_handler import main as train_main
            result = train_main(bucket=bucket, date_str=date_str, dry_run=dry_run)
            log.info(
                "Training complete: test_IC=%.4f  promoted=%s",
                result.get("test_ic", float("nan")),
                result.get("promoted", False),
            )
            return {
                "statusCode": 200,
                "body": (
                    f"GBM training complete for {date_str or 'today'}: "
                    f"test_IC={result.get('test_ic', 'n/a')}  "
                    f"promoted={result.get('promoted', False)}"
                ),
            }
        except Exception as exc:
            log.exception("Training Lambda failed: %s", exc)
            return {
                "statusCode": 500,
                "body": f"Training Lambda failed: {exc}",
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
