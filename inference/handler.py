"""
inference/handler.py — AWS Lambda handler.

Triggered by EventBridge at 6:15am PT on trading days (30 minutes after the
research pipeline completes at 5:45am PT). Delegates to daily_predict.main()
which handles all model loading, inference, and S3 writes.

Lambda configuration:
  - Runtime: container image (public.ecr.aws/lambda/python:3.12)
  - Memory: 1024 MB (PyTorch CPU requires headroom)
  - Timeout: 300 seconds
  - Environment variables:
      S3_BUCKET   — override default bucket (optional)
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    event may contain:
        date      (str)  : Override prediction date YYYY-MM-DD.
        dry_run   (bool) : If True, skip S3 writes (for testing invocations).

    Returns HTTP-style response so EventBridge can log success/failure.
    """
    # Set S3 bucket from environment if not already configured
    os.environ.setdefault("S3_BUCKET", "alpha-engine-research")

    # Configure logging for Lambda (CloudWatch)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    date_str = event.get("date", None)
    dry_run = bool(event.get("dry_run", False))

    log.info(
        "Lambda invocation: date=%s  dry_run=%s  function=%s",
        date_str or "today",
        dry_run,
        getattr(context, "function_name", "local"),
    )

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
