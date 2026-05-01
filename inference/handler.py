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

# Ensure the project root is on sys.path so sibling modules (ssm_secrets)
# can be imported below. Cheap; safe at module-top.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all 5 entrypoints; see executor/main.py for reference).
# When FLOW_DOCTOR_ENABLED=1, attaches a FlowDoctorHandler at ERROR so
# every log.error() call routes through flow-doctor's dispatch (email +
# optional GitHub issue) without explicit fd.report() plumbing.
#
# Path resolution: LAMBDA_TASK_ROOT (=/var/task in the Lambda image,
# where Dockerfile COPYs flow-doctor.yaml) takes precedence; falls back
# to two-dirs-up from this file for local dev (inference/handler.py →
# repo root). Mirrors alpha-engine-research/lambda/handler.py.
#
# exclude_patterns starts empty by deliberate convention: add patterns
# only after observing real ERROR-level noise from a Lambda invocation.
#
# Why module-top is safe even though load_secrets is deferred below:
# flow-doctor.yaml only references EMAIL_SENDER / EMAIL_RECIPIENTS /
# GMAIL_APP_PASSWORD, all populated by Lambda's `--environment` block
# (set by deploy.sh from .env) BEFORE the Python interpreter starts.
# load_secrets() pulls separate SSM-backed secrets (e.g. ANTHROPIC_API_KEY)
# that flow_doctor.init() does not consult.
from alpha_engine_lib.logging import setup_logging
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "predictor",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

log = logging.getLogger(__name__)

# Expensive init is deferred to the first handler invocation to keep
# Lambda's cold-start init phase under the 10-second hard timeout.
# `load_secrets()` is an SSM round-trip (~1-2s) and the v72 canary
# failure (statusCode=0, FunctionError= empty) most plausibly traced
# to this plus the heavy LightGBM / CatBoost / scikit-learn module-top
# imports tipping over the 10s init wall — same class as the
# 2026-04-11 research-handler timeout that motivated this pattern.
# Moving load_secrets to the handler body pays the same cost on the
# first invocation but in the configurable 15-minute handler budget
# instead of the rigid 10s init wall. Idempotent via the `_init_done` flag.
# Mirrors alpha-engine-research/lambda/handler.py:_ensure_init.
_init_done = False


def _ensure_init() -> None:
    """Run expensive init once, on the first handler invocation."""
    global _init_done
    if _init_done:
        return
    from ssm_secrets import load_secrets
    load_secrets()
    _init_done = True


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
    # Run the deferred SSM secrets fetch on the first invocation. Warm
    # containers pay zero cost via the _init_done flag.
    _ensure_init()

    os.environ.setdefault("S3_BUCKET", "alpha-engine-research")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    # setup_logging already ran at module-top (see comment near the
    # alpha_engine_lib.logging import). Apply the standard log level here.
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
