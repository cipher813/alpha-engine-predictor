"""Pin the lib-chokepoint lift in infrastructure/spot_train.sh.

Origin: ROADMAP L342 PR 4 — the 2026-05-27 retire-inline-run_ssm-helper
migration replaced the 54-line inline ``aws ssm send-command`` +
poll + stream bash function (originally shipped in predictor #168
2026-05-15) with a thin wrapper around the lib chokepoint
``python -m alpha_engine_lib.ssm_dispatcher run`` (lib v0.35.0+).

This file's helper was what the entire L342 arc was lifted from: it
appeared first in predictor #168 as the institutional precedent, then
the same pattern would have been mirrored into alpha-engine-data and
alpha-engine-backtester — but per CLAUDE.md SOTA / institutional-default
sub-sub-rule, the 2nd occurrence triggered the lift to lib. PR 4 of
the L342 arc closes the original site at the chokepoint level.

The chokepoint tests:
  * forbid re-introducing the inline ``aws ssm send-command`` body
  * pin the lib CLI as the dispatch surface
  * pin the 5 historical call-site descriptions so future refactors
    can't silently rename them out of operator-recognized logs
  * mirror alpha-engine-data PR 2 (#330) and alpha-engine-backtester
    PR 3 (#251) test files for the same regression class

PR 5 will revoke the port-22 SG inbound rule once 1 clean Saturday
SF runs on the new transport across all three spots; that's the
operator-side soak this CI guard doesn't cover.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "spot_train.sh"


def _script_lines() -> list[tuple[int, str]]:
    """Return (line_no, line) tuples, comment lines stripped.

    Comment lines may legitimately reference ``aws ssm send-command``
    in historical-context prose (e.g. the helper's docstring documents
    what the pre-lift inline form did); only non-comment lines are
    subject to the forbidden-phrase chokepoint.
    """
    assert _SCRIPT.exists(), f"spot_train.sh missing at {_SCRIPT}"
    out: list[tuple[int, str]] = []
    for i, raw in enumerate(_SCRIPT.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        out.append((i, raw))
    return out


def test_spot_train_script_exists():
    """Guards against accidental script deletion. Without the script,
    the chokepoint assertions below silently no-op."""
    assert _SCRIPT.exists(), (
        f"infrastructure/spot_train.sh missing at {_SCRIPT}. "
        "This script drives the Saturday SF PredictorTraining spot; "
        "CI cannot validate its lib-chokepoint invariant without it."
    )


def test_run_ssm_uses_lib_dispatcher():
    """The ``run_ssm`` helper MUST delegate to ``python -m
    alpha_engine_lib.ssm_dispatcher`` (lib v0.35.0+). Pinning this
    catches a regression that re-introduces the pre-2026-05-27 inline
    ``aws ssm send-command`` + poll bash body."""
    text = _SCRIPT.read_text()
    m = re.search(r"^run_ssm\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, (
        "no run_ssm() helper found in spot_train.sh — the lib chokepoint "
        "lift removed the function. The L342 PR 4 contract is to KEEP "
        "the helper signature (so the 5 call sites don't need to change) "
        "and delegate the body to the lib CLI."
    )
    body = m.group(0)
    assert "alpha_engine_lib.ssm_dispatcher" in body, (
        "run_ssm() body does not reference alpha_engine_lib.ssm_dispatcher. "
        "The 2026-05-27 L342 PR 4 lift replaced the inline aws-ssm-send-command "
        "+ poll body with `python -m alpha_engine_lib.ssm_dispatcher run "
        "--script-stdin`. Re-introducing the inline form undoes the lift."
    )
    assert "--script-stdin" in body, (
        "run_ssm() must pipe the script body via --script-stdin so the "
        "lib reads it verbatim (no command-substitution scanning). This "
        "is the pattern alpha-engine-data PR 2 (#330) and "
        "alpha-engine-backtester PR 3 (#251) adopted for the same lift."
    )


def test_no_inline_aws_ssm_send_command():
    """The script MUST NOT call ``aws ssm send-command`` directly in a
    non-comment line. That's the pre-lift pattern L342 explicitly
    retired at the chokepoint level. The lib CLI wraps that exact API
    call with InvocationDoesNotExist registration grace + stdout
    streaming + consistent S3 output-key layout; bypassing it loses
    those guarantees."""
    offenders = [
        (n, line)
        for n, line in _script_lines()
        if "aws ssm send-command" in line
    ]
    assert not offenders, (
        f"Found {len(offenders)} non-comment ``aws ssm send-command`` "
        f"invocations in spot_train.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
        + "\n\nRoute through the lib chokepoint: "
        "``python -m alpha_engine_lib.ssm_dispatcher run --script-stdin``."
    )


def test_no_inline_aws_ssm_get_command_invocation():
    """Same chokepoint as send-command: the pre-lift inline helper also
    polled via ``aws ssm get-command-invocation``. The lib CLI absorbs
    that polling loop (including the InvocationDoesNotExist registration
    race, 2026-05-23 SF event-16 substrate weakness)."""
    offenders = [
        (n, line)
        for n, line in _script_lines()
        if "aws ssm get-command-invocation" in line
    ]
    assert not offenders, (
        f"Found {len(offenders)} non-comment ``aws ssm get-command-invocation`` "
        f"polls in spot_train.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
        + "\n\nThe lib CLI absorbs the polling loop — the only caller "
        "should be inside alpha_engine_lib.ssm_dispatcher, not this script."
    )


# Pinned call-site descriptions: each appears in CloudWatch SSM history,
# operator dashboards, and the lib CLI's `--description` flag. A future
# refactor that renames these would break the operator-recognized
# vocabulary for the Saturday SF PredictorTraining stages.
_EXPECTED_DESCRIPTIONS = frozenset(
    {
        "bootstrap",
        "deps",
        "preflight-only",
        "smoke",
        "full-training",
    }
)


def test_run_ssm_call_sites_present():
    """All 5 historical run_ssm call-site descriptions are present.
    Pinning the registry catches both:
      * a refactor that removes a stage (e.g. drops 'preflight-only')
      * a rename that drifts the operator vocabulary
    """
    body = _SCRIPT.read_text()
    missing = sorted(
        desc
        for desc in _EXPECTED_DESCRIPTIONS
        if f'run_ssm "{desc}"' not in body
    )
    assert not missing, (
        f"run_ssm call sites missing from spot_train.sh: {missing}. "
        f"Expected descriptions (pre-2026-05-27 stable vocabulary): "
        f"{sorted(_EXPECTED_DESCRIPTIONS)}. If the change is deliberate, "
        "update _EXPECTED_DESCRIPTIONS in this test file in the SAME PR "
        "and document the rename in operator notes."
    )


def test_run_ssm_signature_unchanged():
    """The L342 PR 4 lift kept the caller-facing signature:
    ``run_ssm "<description>" "<bash script>" [timeout_seconds]``.
    Pinning this catches a regression where someone flips to the
    stdin-driven shape alpha-engine-data #330 / -backtester #251 use,
    which would require rewriting all 5 call sites — a deliberate
    architectural choice that should land as a separate PR with a clear
    rationale."""
    text = _SCRIPT.read_text()
    m = re.search(r"^run_ssm\(\)\s*\{\s*\n\s*([^\n]+)", text, re.MULTILINE)
    assert m, "no run_ssm() opening found"
    first_line = m.group(1)
    # Expected: local description="$1" script="$2" timeout_s="${3:-3600}"
    # Note: ``${3:-default}`` is a bash default-value expansion, not the
    # bare ``$3``. Accept either spelling.
    has_arg3 = "$3" in first_line or "${3" in first_line
    assert "$1" in first_line and "$2" in first_line and has_arg3, (
        f"run_ssm() first body line should declare description=$1, "
        f"script=$2, timeout_s=$3 — got: {first_line!r}. The L342 PR 4 "
        "contract preserves the 3-positional-arg signature so the 5 "
        "existing call sites don't need to change."
    )
