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

  action == "check_trading_day":
    NYSE holiday gate for the weekday SF (pure calendar, no preflight).
    Returns {is_trading_day, check_date, day_name, marker, ...} computed
    from nousergon_lib.trading_calendar BEFORE StartExecutorEC2. Replaces
    the prior on-box SSM trading_calendar check whose stdout was unreliably
    captured on a cold-booted instance (config#1430).

  action == "check_pipeline_contract":
    Validate PIPELINE_CONTRACT.yaml's self-consistency + every artifact_id ∈
    ARTIFACT_REGISTRY.yaml, fetched from GitHub raw. Used by the Saturday Step
    Function as an early pre-spend gate (mirrors check_lib_pin_drift) so a
    contract break halts before any spot launch. Fail-open on a fetch/parse
    miss; halts only on a confirmed structural violation.

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

# Ensure the project root is on sys.path so sibling modules can be
# imported below. Cheap; safe at module-top.
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
# flow-doctor.yaml references EMAIL_SENDER / EMAIL_RECIPIENTS /
# GMAIL_APP_PASSWORD, all populated by Lambda's `--environment` block
# BEFORE the Python interpreter starts. Other secrets are pulled lazily
# from SSM via krepis.secrets.get_secret() (per-process cached).
from krepis.logging import setup_logging, monitor_handler
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


@monitor_handler
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
    # setup_logging already ran at module-top (see comment near the
    # krepis.logging import). Apply the standard log level here.
    logging.getLogger().setLevel(logging.INFO)

    # ── Trading-day gate (weekday SF, runs BEFORE StartExecutorEC2) ─────────
    # Pure NYSE-calendar math — deliberately returns BEFORE preflight so the
    # gate cannot be broken by env / connectivity / ArcticDB / model issues.
    # Replaces the prior on-box SSM trading_calendar check (config#1430).
    if event.get("action") == "check_trading_day":
        from inference.trading_day_gate import check_trading_day
        _r = check_trading_day(event.get("date"))
        log.info(
            "Trading-day gate: %s (%s) -> %s%s",
            _r["check_date"], _r["day_name"], _r["marker"],
            "" if _r["is_trading_day"] else f" (next: {_r.get('next_trading_day')})",
        )
        return _r

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
    # dry_run is re-parsed at its canonical site below; read it early here so
    # the canary (dry_run=true) can skip the deploy-drift assertion. A canary
    # writes/emails nothing, so drift-vs-main is the wrong invariant and a
    # false-failure source during merge bursts (config#1073). Production
    # (dry_run=false) and the SF drift gate below are unaffected.
    _dry_run = bool(event.get("dry_run", False))
    _pf = PredictorPreflight(bucket=_bucket)
    if _action in (
        "check_deploy_drift",
        "check_lib_pin_drift",
        "check_pipeline_contract",
    ):
        # All three are lightweight pre-spend SF gates (GitHub reads only) — run
        # the minimal preflight, not the full predictor bootstrap.
        _pf.run_for_drift_gate()
    elif _action == "check_drift":
        # Drift detection is a post-inference monitoring step: it reads the
        # day's predictions/features and writes drift_{date}.json. It needs env
        # + S3 reachability only — NOT model weights (it scores nothing) and NOT
        # the deploy-drift gate (a stale-code halt must never silence the
        # monitoring/observability arm). config#1282.
        _pf.check_env_vars("AWS_REGION")
        _pf.check_s3_bucket()
    else:
        _pf.run(skip_deploy_drift=_dry_run)

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

    # ── Drift detection (daily EOD Step Function state, config#1282) ─────────
    # Runs AFTER predictions are written so drift_{date}.json is produced every
    # weekday EOD — matching the artifact's registered `eod_sf` (daily) cadence.
    # Decouples the drift OUTPUT from the skippable, Saturday-only DriftDetection
    # state (which froze the artifact at drift_2026-06-13.json). check_drift
    # already writes predictor/metrics/drift_{date}.json and never raises on a
    # detected-drift condition (it returns a structured result); a real error
    # propagates so the SF Catch sees a true failure.
    if action == "check_drift":
        from monitoring.drift_detector import check_drift
        result = check_drift(bucket=bucket, date_str=date_str)
        log.info(
            "Drift check %s: status=%s severity=%s n_alerts=%d → drift_%s.json",
            result["date"], result["status"], result.get("severity"),
            result.get("n_alerts", 0), result["date"],
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

    # ── Lib-pin drift check (Saturday SF early state, L4517) ────────────────
    if action == "check_lib_pin_drift":
        from inference.lib_pin_drift import check_lib_pin_drift
        result = check_lib_pin_drift()
        log.info(
            "Lib-pin drift check: has_drift=%s reason=%s pins=%s%s",
            result["has_drift"], result["reason"], result["pins"],
            (" offenders=" + "; ".join(result["offenders"])) if result["offenders"] else "",
        )
        return result

    # ── Pipeline-contract preflight (Saturday SF early state, L4595) ─────────
    if action == "check_pipeline_contract":
        from inference.pipeline_contract_check import check_pipeline_contract
        result = check_pipeline_contract()
        log.info(
            "Pipeline-contract preflight: has_violation=%s reason=%s boundaries=%s%s",
            result["has_violation"], result["reason"], result["boundary_count"],
            (" violations=" + "; ".join(result["violations"]))
            if result["violations"] else "",
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
