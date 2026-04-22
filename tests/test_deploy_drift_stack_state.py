"""Regression tests for deploy_drift's stack-state distinctions.

Context (ROADMAP P1, 2026-04-22):
    Prior to this change, ``_read_stack_tag`` returned ``None`` for ALL
    non-happy-path outcomes:
      - stack doesn't exist
      - stack exists but is in ROLLBACK_COMPLETE / CREATE_FAILED / etc.
      - stack exists without a git-sha tag (legacy-deploy warn path)
      - describe_stacks itself raised (IAM, network)

    This conflation meant the SF ``DeployDriftGate`` silently passed
    when the stack was catastrophically broken — exactly the
    2026-04-20 orchestration-stack ROLLBACK_COMPLETE incident that
    required an emergency recovery. The only "None" path that should
    NOT trigger drift is the legacy-deploy warn path (stack healthy,
    no tag). Every other failure mode must surface as ``cf_drift=True``
    with a distinct reason code so the SF can route correctly and the
    operator can see WHICH failure mode fired.

The fix: ``_read_stack_tag`` now returns a tri-state:
    - ``str``                — healthy stack, tag present
    - ``None``               — healthy stack, tag absent (legacy)
    - ``StackStateError(reason=…)``  — every other failure mode

``check_deploy_drift`` interprets the sentinel and surfaces
``cf_drift_reason`` / ``cf_drift_detail`` / ``cf_stack_status`` so the
DeployDriftGate Choice state has the information it needs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

import inference.deploy_drift as dd


# ── _read_stack_tag tri-state behavior ────────────────────────────────────────


def _cfn_client_returning(stack_or_exc):
    """Build a mock boto3 client whose describe_stacks returns a fixed
    response or raises a fixed exception.
    """
    cfn = MagicMock()
    if isinstance(stack_or_exc, Exception):
        cfn.describe_stacks.side_effect = stack_or_exc
    else:
        cfn.describe_stacks.return_value = {"Stacks": [stack_or_exc]}
    return cfn


def test_happy_path_returns_sha_string():
    stack = {
        "StackName": "s", "StackStatus": "UPDATE_COMPLETE",
        "Tags": [{"Key": "git-sha", "Value": "abc123def"}],
    }
    with patch("boto3.client", return_value=_cfn_client_returning(stack)):
        result = dd._read_stack_tag("s")
    assert result == "abc123def"


def test_healthy_stack_missing_tag_returns_none():
    """Legacy-deploy warn path — stack ran fine but wasn't stamped."""
    stack = {"StackName": "s", "StackStatus": "CREATE_COMPLETE", "Tags": []}
    with patch("boto3.client", return_value=_cfn_client_returning(stack)):
        result = dd._read_stack_tag("s")
    assert result is None
    assert not isinstance(result, dd.StackStateError), (
        "Missing git-sha tag on a healthy stack is a legacy-deploy warn "
        "path, NOT a stack-state error — conflating them brings back the "
        "silent-pass the fix exists to prevent."
    )


def test_rollback_complete_returns_stack_state_error():
    """The 2026-04-20 incident mode."""
    stack = {"StackName": "s", "StackStatus": "ROLLBACK_COMPLETE", "Tags": []}
    with patch("boto3.client", return_value=_cfn_client_returning(stack)):
        result = dd._read_stack_tag("s")
    assert isinstance(result, dd.StackStateError)
    assert result.reason == "stack_in_terminal_state"
    assert result.stack_status == "ROLLBACK_COMPLETE"


@pytest.mark.parametrize("status", [
    "CREATE_FAILED", "ROLLBACK_FAILED", "DELETE_FAILED",
    "UPDATE_ROLLBACK_FAILED", "UPDATE_ROLLBACK_COMPLETE",
    "IMPORT_ROLLBACK_FAILED", "IMPORT_ROLLBACK_COMPLETE",
])
def test_every_terminal_failed_state_returns_stack_state_error(status):
    stack = {"StackName": "s", "StackStatus": status, "Tags": []}
    with patch("boto3.client", return_value=_cfn_client_returning(stack)):
        result = dd._read_stack_tag("s")
    assert isinstance(result, dd.StackStateError)
    assert result.reason == "stack_in_terminal_state"
    assert result.stack_status == status


def test_stack_not_exists_returns_stack_state_error():
    err = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "Stack with id foo does not exist"}},
        "DescribeStacks",
    )
    with patch("boto3.client", return_value=_cfn_client_returning(err)):
        result = dd._read_stack_tag("foo")
    assert isinstance(result, dd.StackStateError)
    assert result.reason == "stack_not_exists"


def test_describe_stacks_iam_error_returns_stack_state_error():
    """IAM failure is 'can't prove healthy' — must route to hard-fail."""
    err = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "not authorized"}},
        "DescribeStacks",
    )
    with patch("boto3.client", return_value=_cfn_client_returning(err)):
        result = dd._read_stack_tag("foo")
    assert isinstance(result, dd.StackStateError)
    assert result.reason == "describe_stacks_error"
    assert "AccessDenied" in result.detail or "ClientError" in result.detail


def test_non_client_error_returns_stack_state_error():
    """SDK-level RuntimeError (unusual but possible) — still routes hard-fail."""
    with patch("boto3.client", return_value=_cfn_client_returning(RuntimeError("boto3 broke"))):
        result = dd._read_stack_tag("foo")
    assert isinstance(result, dd.StackStateError)
    assert result.reason == "describe_stacks_error"


def test_zero_stacks_in_response_returns_stack_state_error():
    """Response has no Stacks — distinct from ValidationError but similar intent."""
    cfn = MagicMock()
    cfn.describe_stacks.return_value = {"Stacks": []}
    with patch("boto3.client", return_value=cfn):
        result = dd._read_stack_tag("foo")
    assert isinstance(result, dd.StackStateError)
    assert result.reason == "stack_not_exists"


# ── check_deploy_drift surfaces reason codes ──────────────────────────────────


def _patch_all(stack_read, sf_comment="[git:abc123d] ok",
               upstream_sha="abc123d4567890"):
    """Context manager composing all patches needed for check_deploy_drift."""
    return (
        patch("inference.deploy_drift._read_stack_tag", return_value=stack_read),
        patch("inference.deploy_drift._read_sf_comment", return_value=sf_comment),
        patch("inference.deploy_drift._fetch_origin_main_sha", return_value=upstream_sha),
    )


def test_check_deploy_drift_happy_path_no_drift():
    patches = _patch_all(stack_read="abc123d4567890")
    with patches[0], patches[1], patches[2]:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["has_drift"] is False
    assert result["cf_drift"] is False
    assert result["cf_drift_reason"] == "in_sync"
    assert result["cf_stack_status"] is None


def test_check_deploy_drift_rollback_complete_fires_drift():
    """Regression: the exact scenario the fix exists for."""
    sse = dd.StackStateError(
        reason="stack_in_terminal_state",
        detail="orchestration-stack is in ROLLBACK_COMPLETE",
        stack_status="ROLLBACK_COMPLETE",
    )
    patches = _patch_all(stack_read=sse)
    with patches[0], patches[1], patches[2]:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["cf_drift"] is True
    assert result["has_drift"] is True
    assert result["cf_drift_reason"] == "stack_in_terminal_state"
    assert result["cf_stack_status"] == "ROLLBACK_COMPLETE"
    assert "ROLLBACK_COMPLETE" in result["cf_drift_detail"]


def test_check_deploy_drift_stack_not_exists_fires_drift():
    sse = dd.StackStateError(reason="stack_not_exists", detail="foo not found")
    patches = _patch_all(stack_read=sse)
    with patches[0], patches[1], patches[2]:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["cf_drift"] is True
    assert result["cf_drift_reason"] == "stack_not_exists"


def test_check_deploy_drift_describe_error_fires_drift():
    sse = dd.StackStateError(
        reason="describe_stacks_error",
        detail="AccessDenied: not authorized",
    )
    patches = _patch_all(stack_read=sse)
    with patches[0], patches[1], patches[2]:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["cf_drift"] is True
    assert result["cf_drift_reason"] == "describe_stacks_error"
    assert "AccessDenied" in result["cf_drift_detail"]


def test_check_deploy_drift_legacy_no_tag_does_not_fire_drift():
    """Stack healthy, no git-sha tag. Older deploys predate the tagging;
    must warn, not fire drift.
    """
    patches = _patch_all(stack_read=None)  # healthy + no tag
    with patches[0], patches[1], patches[2]:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["cf_drift"] is False
    assert result["has_drift"] is False
    assert result["cf_drift_reason"] == "no_git_sha_tag_legacy"
    assert result["stack_stamp_present"] is False


def test_check_deploy_drift_sha_mismatch_reason_code():
    """Healthy stack with tag that doesn't match upstream SHA."""
    patches = _patch_all(stack_read="different_sha_xxx", upstream_sha="abc123d4567890")
    with patches[0], patches[1], patches[2]:
        result = dd.check_deploy_drift(region="us-east-1", account_id="123")
    assert result["cf_drift"] is True
    assert result["cf_drift_reason"] == "sha_mismatch"
