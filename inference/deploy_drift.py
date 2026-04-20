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
from typing import Optional

log = logging.getLogger(__name__)

_SF_ARN_TEMPLATE = (
    "arn:aws:states:{region}:{account}:stateMachine:{name}"
)

# `[git:<40 hex>]` prefix injected by alpha-engine-data/infrastructure/
# deploy-infrastructure.sh into the SF Comment field at deploy time.
_GIT_PREFIX_RE = re.compile(r"^\[git:([0-9a-f]{7,40})\]")


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


def _read_stack_tag(stack_name: str, tag_key: str = "git-sha") -> Optional[str]:
    """describe-stacks → Tags[git-sha]. None on error or missing tag."""
    import boto3
    try:
        cfn = boto3.client("cloudformation")
        resp = cfn.describe_stacks(StackName=stack_name)
        stacks = resp.get("Stacks") or []
        if not stacks:
            return None
        for tag in stacks[0].get("Tags") or []:
            if tag.get("Key") == tag_key:
                return tag.get("Value")
    except Exception as exc:  # noqa: BLE001
        log.warning("describe_stacks failed for %s: %s", stack_name, exc)
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
    stack_sha = _read_stack_tag(stack_name)
    upstream = _fetch_origin_main_sha(repo, branch=branch)

    sf_drift = (
        upstream is not None
        and sf_sha is not None
        and not _shas_match(sf_sha, upstream)
    )
    cf_drift = (
        upstream is not None
        and stack_sha is not None
        and not _shas_match(stack_sha, upstream)
    )

    return {
        "repo": repo,
        "branch": branch,
        "upstream_sha": upstream,
        "sf_sha": sf_sha,
        "sf_stamp_present": sf_sha is not None,
        "stack_sha": stack_sha,
        "stack_stamp_present": stack_sha is not None,
        "sf_drift": sf_drift,
        "cf_drift": cf_drift,
        "has_drift": sf_drift or cf_drift,
    }
