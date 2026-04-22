"""Deploy-drift probe: confirm Step Function + CloudFormation deployed SHAs
match `origin/main` HEAD for their source repos.

Exposed as `action=check_deploy_drift` on the predictor Lambda handler so
the weekday Step Function can invoke it as a Task state before any real
work runs. Keeps the check-surface in one Lambda rather than deploying a
new one for a ~100-line concern; architectural split can happen later if
this Lambda's surface grows.

Returns a JSON-serializable dict the SF consumes via a Choice state.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Union

log = logging.getLogger(__name__)

_SF_ARN_TEMPLATE = (
    "arn:aws:states:{region}:{account}:stateMachine:{name}"
)

# `[git:<40 hex>]` prefix injected by alpha-engine-data/infrastructure/
# deploy-infrastructure.sh into the SF Comment field at deploy time.
_GIT_PREFIX_RE = re.compile(r"^\[git:([0-9a-f]{7,40})\]")

# CloudFormation terminal / failed states. A stack sitting in any of these
# cannot be trusted to actually hold the resources its template declares,
# so the drift probe must distinguish this condition from "stack exists
# with no git-sha tag" (legacy deploy warn-path) and fire the hard-fail
# branch of the SF DeployDriftGate.
#
# Keep in sync with CloudFormation's published state machine
# (https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-describing-stacks.html).
_CFN_TERMINAL_FAILED_STATES = frozenset({
    "CREATE_FAILED",
    "ROLLBACK_COMPLETE",       # ← 2026-04-20 incident mode
    "ROLLBACK_FAILED",
    "DELETE_FAILED",
    "UPDATE_ROLLBACK_FAILED",
    "UPDATE_ROLLBACK_COMPLETE",
    "IMPORT_ROLLBACK_FAILED",
    "IMPORT_ROLLBACK_COMPLETE",
})


@dataclass(frozen=True)
class StackStateError:
    """Sentinel for ``_read_stack_tag`` return when we can't trust the stack.

    Distinguishes:
      - ``stack_not_exists`` — describe_stacks found nothing (or raised
        ValidationError "does not exist"). Stack is absent entirely.
      - ``stack_in_terminal_state`` — stack exists but is in a terminal
        failed state (ROLLBACK_COMPLETE etc.) and cannot be trusted to
        hold its declared resources. This is what tripped the
        2026-04-20 orchestration-stack recovery.
      - ``describe_stacks_error`` — boto3 itself raised (IAM, network).
        Treated as "can't prove healthy" → route to hard-fail.

    The ``no_git_sha_tag_legacy`` case is NOT a StackStateError — that's
    a legitimate legacy-deploy warn path where the stack is healthy and
    tagged correctly by older tooling; returned as a plain ``None``.
    """
    reason: str           # stack_not_exists | stack_in_terminal_state | describe_stacks_error
    detail: str = ""      # stack status name or exception class name
    stack_status: Optional[str] = None  # populated for terminal-state

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "stack_status": self.stack_status,
        }


def _fetch_origin_main_sha(
    repo: str, branch: str = "main", timeout: float = 5.0,
) -> Optional[str]:
    """Public-repo GitHub branch-HEAD lookup. Returns None on network error."""
    url = f"https://api.github.com/repos/{repo}/branches/{branch}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        return payload.get("commit", {}).get("sha")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        log.warning("GitHub API unreachable (%s) for %s@%s", exc, repo, branch)
        return None


def _extract_sf_sha(comment: str) -> Optional[str]:
    """Pull `<sha>` out of a `[git:<sha>] rest…` Comment string."""
    if not comment:
        return None
    m = _GIT_PREFIX_RE.match(comment.strip())
    return m.group(1) if m else None


def _read_sf_comment(state_machine_arn: str) -> Optional[str]:
    """describe-state-machine → definition.Comment. None on error."""
    import boto3
    try:
        sfn = boto3.client("stepfunctions")
        resp = sfn.describe_state_machine(stateMachineArn=state_machine_arn)
        definition = json.loads(resp["definition"])
        return definition.get("Comment", "")
    except Exception as exc:  # noqa: BLE001 — any failure routes to None
        log.warning("describe_state_machine failed for %s: %s",
                    state_machine_arn, exc)
        return None


def _read_stack_tag(
    stack_name: str, tag_key: str = "git-sha",
) -> Union[str, StackStateError, None]:
    """describe-stacks → Tags[git-sha], distinguishing four states:

    - ``str``                       stack healthy, tag present → return SHA
    - ``None``                      stack healthy, tag absent → legacy-deploy
                                    warn path (NOT drift)
    - ``StackStateError`` (reason=stack_not_exists)
                                    stack absent entirely — SF must hard-fail
    - ``StackStateError`` (reason=stack_in_terminal_state)
                                    stack in ROLLBACK_COMPLETE / *_FAILED →
                                    cannot be trusted; SF must hard-fail
    - ``StackStateError`` (reason=describe_stacks_error)
                                    boto3 raised (IAM, network); can't prove
                                    healthy — SF must hard-fail

    Previously this function conflated all four into ``None``, which made
    the SF's DeployDriftGate silently pass when the stack was catastrophically
    broken — exactly the 2026-04-20 orchestration-stack ROLLBACK_COMPLETE
    scenario that required an emergency recovery.
    """
    import boto3
    from botocore.exceptions import ClientError
    try:
        cfn = boto3.client("cloudformation")
        resp = cfn.describe_stacks(StackName=stack_name)
    except ClientError as exc:
        err_code = exc.response.get("Error", {}).get("Code", "")
        err_msg = str(exc)
        # boto3 surfaces "stack does not exist" as ValidationError with this
        # exact substring. Distinguish it from generic API failures so the
        # SF can report the clearer "stack_not_exists" reason code.
        if (err_code == "ValidationError" and "does not exist" in err_msg):
            log.warning("describe_stacks: %s does not exist", stack_name)
            return StackStateError(
                reason="stack_not_exists",
                detail=f"{stack_name} not found",
            )
        log.warning("describe_stacks failed for %s: %s", stack_name, exc)
        return StackStateError(
            reason="describe_stacks_error",
            detail=f"{exc.__class__.__name__}: {err_msg[:200]}",
        )
    except Exception as exc:  # noqa: BLE001 — non-ClientError (SDK bug, etc.)
        log.warning("describe_stacks failed for %s: %s", stack_name, exc)
        return StackStateError(
            reason="describe_stacks_error",
            detail=f"{exc.__class__.__name__}: {str(exc)[:200]}",
        )

    stacks = resp.get("Stacks") or []
    if not stacks:
        # describe_stacks returned 0 stacks without raising — shouldn't happen
        # with a StackName lookup, but preserve the distinct reason.
        return StackStateError(
            reason="stack_not_exists",
            detail=f"{stack_name} not returned by describe_stacks",
        )

    stack = stacks[0]
    stack_status = stack.get("StackStatus", "")
    if stack_status in _CFN_TERMINAL_FAILED_STATES:
        log.error(
            "Stack %s is in terminal state %s — describe_stacks resources "
            "cannot be trusted. Probe will report cf_drift=true with "
            "reason=stack_in_terminal_state so the SF DeployDriftGate fires.",
            stack_name, stack_status,
        )
        return StackStateError(
            reason="stack_in_terminal_state",
            detail=f"{stack_name} is in {stack_status}",
            stack_status=stack_status,
        )

    for tag in stack.get("Tags") or []:
        if tag.get("Key") == tag_key:
            return tag.get("Value")
    # Stack healthy, tag absent — legacy-deploy warn path, NOT drift.
    return None


def _shas_match(deployed: Optional[str], upstream: Optional[str]) -> bool:
    """Compare a deployed SHA stamp (may be 7-40 chars) to full upstream SHA.
    Missing either side → return True (can't prove drift → don't raise).
    """
    if not deployed or not upstream:
        return True
    if len(deployed) < 7:
        return True  # malformed stamp — warn elsewhere, don't block
    return upstream.startswith(deployed) or deployed.startswith(upstream)


def check_deploy_drift(
    region: str,
    account_id: str,
    sf_name: str = "alpha-engine-weekday-pipeline",
    stack_name: str = "alpha-engine-orchestration",
    repo: str = "cipher813/alpha-engine-data",
    branch: str = "main",
) -> dict:
    """Compare deployed SF + CF SHAs against GitHub `repo@branch` HEAD.

    Returns a dict with per-artifact stamps, the upstream SHA, and
    booleans flagging drift. The Step Function's Choice state uses
    `has_drift` to decide whether to proceed or route to HandleFailure.
    Degraded modes (GitHub outage, missing stamps) set `has_drift=false`
    with diagnostic fields populated — see sibling functions for the
    "why" of each None value.
    """
    sf_arn = _SF_ARN_TEMPLATE.format(
        region=region, account=account_id, name=sf_name,
    )

    sf_comment = _read_sf_comment(sf_arn) or ""
    sf_sha = _extract_sf_sha(sf_comment)
    stack_read = _read_stack_tag(stack_name)
    upstream = _fetch_origin_main_sha(repo, branch=branch)

    sf_drift = (
        upstream is not None
        and sf_sha is not None
        and not _shas_match(sf_sha, upstream)
    )

    # Interpret stack_read tri-state:
    #   str               → healthy stack with tag; compare to upstream
    #   None              → healthy stack, tag absent (legacy-deploy warn)
    #   StackStateError   → stack not usable; cf_drift=true with reason code
    if isinstance(stack_read, StackStateError):
        stack_sha = None
        stack_stamp_present = False
        cf_drift = True
        cf_drift_reason = stack_read.reason
        cf_drift_detail = stack_read.detail
        cf_stack_status = stack_read.stack_status
    else:
        stack_sha = stack_read  # str | None
        stack_stamp_present = stack_sha is not None
        cf_drift = (
            upstream is not None
            and stack_sha is not None
            and not _shas_match(stack_sha, upstream)
        )
        cf_drift_reason = (
            "sha_mismatch" if cf_drift else
            ("no_git_sha_tag_legacy" if stack_sha is None else "in_sync")
        )
        cf_drift_detail = ""
        cf_stack_status = None

    return {
        "repo": repo,
        "branch": branch,
        "upstream_sha": upstream,
        "sf_sha": sf_sha,
        "sf_stamp_present": sf_sha is not None,
        "stack_sha": stack_sha,
        "stack_stamp_present": stack_stamp_present,
        "sf_drift": sf_drift,
        "cf_drift": cf_drift,
        "cf_drift_reason": cf_drift_reason,
        "cf_drift_detail": cf_drift_detail,
        "cf_stack_status": cf_stack_status,
        "has_drift": sf_drift or cf_drift,
    }
