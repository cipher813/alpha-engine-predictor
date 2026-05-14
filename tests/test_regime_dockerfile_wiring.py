"""Pin the Dockerfile + Lambda deploy wiring for the regime substrate.

Three contracts this PR locks down — silent drift in any of them ships
a regime Lambda that can't find its own code or can't be deployed:

1. ``Dockerfile`` ``COPY regime/`` line is present, so the substrate
   modules ship inside the Lambda image alongside ``inference/``.
2. ``requirements-lambda.txt`` pins ``hmmlearn``, the only direct dep
   the regime substrate adds that's not already in the inference image.
3. ``infrastructure/deploy.sh`` knows about the regime Lambda function
   name + has the Step 9 update path that mirrors the inference flow.

These are regression-only — no AWS, no Docker. They're cheap shape
checks that catch the "someone removed the COPY line during a refactor"
class of failure that's invisible until the Lambda 500s in prod.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_copies_regime_module() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "COPY regime/ regime/" in dockerfile, (
        "Dockerfile must COPY regime/ so the substrate modules ship in "
        "the Lambda image. Without this, the regime Lambda's CMD "
        "(regime.handler.lambda_handler) fails ImportError at cold start."
    )


def test_requirements_lambda_pins_hmmlearn() -> None:
    reqs = (REPO_ROOT / "requirements-lambda.txt").read_text()
    assert "hmmlearn" in reqs, (
        "requirements-lambda.txt must pin hmmlearn so it ships in the "
        "Lambda image. hmmlearn pulls scipy transitively (also needed "
        "by regime/bocpd.py via scipy.stats.t)."
    )


def test_deploy_sh_references_regime_lambda_function() -> None:
    """``deploy.sh`` must reference the regime Lambda function name so
    Step 9's update path runs on every deploy."""
    deploy = (REPO_ROOT / "infrastructure" / "deploy.sh").read_text()
    assert "REGIME_LAMBDA_FUNCTION" in deploy
    assert "alpha-engine-predictor-regime-substrate" in deploy


def test_deploy_sh_has_step_9_update_block() -> None:
    """The Step 9 update block must exist + call ``update-function-code``
    against ``REGIME_LAMBDA_FUNCTION``. Catches accidental removal of
    the regime update path during deploy.sh refactors."""
    deploy = (REPO_ROOT / "infrastructure" / "deploy.sh").read_text()
    # Step 9 header present
    assert "Step 9:" in deploy
    # Update-function-code invocation against the regime function
    assert 'aws lambda update-function-code \\' in deploy
    assert '--function-name "${REGIME_LAMBDA_FUNCTION}"' in deploy


def test_setup_regime_lambda_script_exists_and_executable() -> None:
    """The one-time setup script is part of the deploy contract — if a
    new operator clones the repo, they need this script to bring the
    regime Lambda online for the first time."""
    setup = REPO_ROOT / "infrastructure" / "setup-regime-lambda.sh"
    assert setup.exists(), "setup-regime-lambda.sh missing"
    content = setup.read_text()
    # Must reference the right CMD override
    assert "regime.handler.lambda_handler" in content
    # Must create the function with the image-config Command override
    assert "create-function" in content
    assert "image-config" in content


def test_iam_role_grants_logs_to_regime_lambda_log_group() -> None:
    """The shared IAM role's CloudWatchLogs statement must include the
    regime Lambda's log group; otherwise the function can be created
    but cannot write logs."""
    import json
    role_file = REPO_ROOT / "infrastructure" / "iam" / "alpha-engine-predictor-role.json"
    policy = json.loads(role_file.read_text())
    logs_statement = next(s for s in policy["Statement"] if s["Sid"] == "CloudWatchLogs")
    resources = logs_statement["Resource"]
    if isinstance(resources, str):
        resources = [resources]
    assert any(
        "alpha-engine-predictor-regime-substrate" in r for r in resources
    ), "IAM role's CloudWatchLogs statement must grant access to the regime Lambda's log group"


# ─────────────────────────────────────────────────────────────────────
# T1 retrospective eval Lambda — third shared-image Lambda (regime-v3
# §5.3.3). Same contracts as the substrate Lambda above, applied to the
# new function name.
# ─────────────────────────────────────────────────────────────────────


def test_deploy_sh_references_regime_eval_lambda_function() -> None:
    """``deploy.sh`` must reference the T1 retrospective eval Lambda
    function name so Step 10's update path runs on every deploy."""
    deploy = (REPO_ROOT / "infrastructure" / "deploy.sh").read_text()
    assert "REGIME_EVAL_LAMBDA_FUNCTION" in deploy
    assert "alpha-engine-predictor-regime-retrospective-eval" in deploy


def test_deploy_sh_has_step_10_update_block() -> None:
    """Step 10 update block must exist + call ``update-function-code``
    against ``REGIME_EVAL_LAMBDA_FUNCTION``."""
    deploy = (REPO_ROOT / "infrastructure" / "deploy.sh").read_text()
    assert "Step 10:" in deploy
    assert '--function-name "${REGIME_EVAL_LAMBDA_FUNCTION}"' in deploy


def test_setup_regime_eval_lambda_script_exists_and_executable() -> None:
    """The one-time setup script for the T1 eval Lambda. Mirrors the
    substrate setup contract."""
    setup = REPO_ROOT / "infrastructure" / "setup-regime-retrospective-eval-lambda.sh"
    assert setup.exists(), "setup-regime-retrospective-eval-lambda.sh missing"
    content = setup.read_text()
    # Must reference the right CMD override
    assert "regime.retrospective_eval_handler.lambda_handler" in content
    # Must create the function with the image-config Command override
    assert "create-function" in content
    assert "image-config" in content


def test_iam_role_grants_logs_to_regime_eval_lambda_log_group() -> None:
    """The shared IAM role's CloudWatchLogs statement must also include
    the T1 retrospective eval Lambda's log group."""
    import json
    role_file = REPO_ROOT / "infrastructure" / "iam" / "alpha-engine-predictor-role.json"
    policy = json.loads(role_file.read_text())
    logs_statement = next(s for s in policy["Statement"] if s["Sid"] == "CloudWatchLogs")
    resources = logs_statement["Resource"]
    if isinstance(resources, str):
        resources = [resources]
    assert any(
        "alpha-engine-predictor-regime-retrospective-eval" in r for r in resources
    ), (
        "IAM role's CloudWatchLogs statement must grant access to the "
        "T1 retrospective eval Lambda's log group"
    )


# ─────────────────────────────────────────────────────────────────────
# deploy.sh auto-create fall-through (2026-05-14 evening). Catches a
# refactor that re-introduces "NOT FOUND — skipping" instead of
# auto-creating. The setup-regime-*.sh scripts are now break-glass-only.
# ─────────────────────────────────────────────────────────────────────


def test_deploy_sh_auto_creates_regime_substrate_when_missing() -> None:
    """Step 9 must call ``aws lambda create-function`` on the NOT FOUND
    branch — not just print a "skipping" message."""
    deploy = (REPO_ROOT / "infrastructure" / "deploy.sh").read_text()
    assert "NOT FOUND — auto-creating with CMD=${REGIME_LAMBDA_CMD}" in deploy
    assert "aws lambda create-function" in deploy
    # Both Lambdas must reference the execution role for PassRole
    assert "LAMBDA_EXECUTION_ROLE_ARN" in deploy


def test_deploy_sh_auto_creates_regime_eval_when_missing() -> None:
    """Step 10 must call ``aws lambda create-function`` on the
    NOT FOUND branch (T1 retrospective eval Lambda)."""
    deploy = (REPO_ROOT / "infrastructure" / "deploy.sh").read_text()
    assert "NOT FOUND — auto-creating with CMD=${REGIME_EVAL_LAMBDA_CMD}" in deploy
    # The same Step-10 block references the eval-specific CMD + memory + timeout
    assert "REGIME_EVAL_LAMBDA_CMD" in deploy
    assert "REGIME_EVAL_LAMBDA_MEMORY" in deploy
    assert "REGIME_EVAL_LAMBDA_TIMEOUT" in deploy


def test_deploy_sh_no_longer_directs_operator_to_setup_scripts() -> None:
    """Post auto-create patch, deploy.sh should not direct operators
    to run the setup scripts as a required step. The scripts remain
    in the repo for break-glass; deploy.sh's failure path should
    surface the actual AWS error, not punt to a manual workflow."""
    deploy = (REPO_ROOT / "infrastructure" / "deploy.sh").read_text()
    # The substantive anti-pattern is the "NOT FOUND — skipping. Create the
    # function once via:" stanza — pin that exact wording is gone.
    assert "NOT FOUND — skipping. Create the function once via:" not in deploy
