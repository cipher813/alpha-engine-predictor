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


def test_run_ssm_passes_diagnostics_flags():
    """L394 cascade — ``run_ssm`` MUST pass both ``--diagnostics-bucket``
    and ``--diagnostics-prefix`` so terminal non-Success in any spot
    SSM step writes a JSON failure record to
    ``s3://${S3_BUCKET}/_spot_diagnostics/ae-predictor/{date}.json`` per
    the lib v0.39.0 contract. Both flags must be present — lib's partial-
    config guard makes a missing flag a silent no-op."""
    text = _SCRIPT.read_text()
    m = re.search(r"^run_ssm\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no run_ssm() helper found in spot_train.sh"
    body = m.group(0)
    assert "--diagnostics-bucket" in body, (
        "spot_train.sh run_ssm() does not pass --diagnostics-bucket to "
        "the lib CLI. L394 cascade requires both --diagnostics-bucket "
        "and --diagnostics-prefix together; without --diagnostics-bucket "
        "the lib's partial-config guard makes the diagnostics-write a "
        "silent no-op even on terminal non-Success."
    )
    assert "--diagnostics-prefix" in body, (
        "spot_train.sh run_ssm() does not pass --diagnostics-prefix to "
        "the lib CLI."
    )
    # Per-repo subprefix discriminates cascade A (ae-data) + cascade B
    # (ae-backtester) sibling writes — lib's {date}.json key shape would
    # otherwise clobber within a shared prefix.
    assert "_spot_diagnostics/ae-predictor" in body, (
        "spot_train.sh --diagnostics-prefix must scope to "
        "_spot_diagnostics/ae-predictor so ae-data + ae-backtester "
        "cascade siblings write to disjoint S3 namespaces."
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


# ── Spot-side log durability (ship-on-exit) ──────────────────────────────────
# Each spot-side WORKLOAD heredoc (smoke / model-zoo-weekly / full-training)
# previously ran its python inline via ``$PY - <<'PYEOF'`` — so the spot's own
# stdout/stderr lived ONLY in SSM ``get-command-invocation``, which returns
# EMPTY when the spot dies mid-run (e.g. an OOM SIGKILL, RC=-1) and is destroyed
# the moment the dispatcher's ``trap cleanup EXIT`` terminates the box. The fix
# routes each workload through ``alpha_engine_lib.ssm_log_capture run`` (the lib
# chokepoint for tee-to-logfile + ship-to-S3-on-EXIT, same primitive the data /
# backtester Saturday-SF spot states use), so the FULL workload log always
# reaches S3 before teardown. These tests pin that wrap so a future refactor
# can't silently revert to the inline ``$PY -`` form and re-lose the log.
_SPOT_WORKLOAD_SLUGS = (
    "spot-smoke",
    "spot-model-zoo-weekly",
    "spot-full-training",
)


def test_spot_workloads_ship_log_on_exit_via_lib():
    """Every spot-side workload MUST run through
    ``python -m alpha_engine_lib.ssm_log_capture run`` so its combined
    stdout+stderr is tee'd to a spot-local logfile AND shipped to S3 on
    EXIT (success, non-zero, or SIGKILL) BEFORE the dispatcher terminates
    the spot. This is the durability fix for the off-cycle full-only OOM
    (RC=-1) incident whose full training log was lost."""
    text = _SCRIPT.read_text()
    for slug in _SPOT_WORKLOAD_SLUGS:
        wrap = (
            f"alpha_engine_lib.ssm_log_capture run --slug {slug} "
            f"--log /var/log/{slug}.log"
        )
        assert wrap in text, (
            f"spot workload {slug!r} is not wrapped in the lib log-capture "
            f"chokepoint. Expected a line containing:\n  {wrap}\n"
            "Without it the workload's stdout/stderr lives only in SSM "
            "get-command-invocation, which returns EMPTY on instance death "
            "(OOM RC=-1) and is destroyed when the dispatcher cleanup EXIT "
            "trap terminates the spot. Route the workload through "
            "`$PY -m alpha_engine_lib.ssm_log_capture run --slug <X> "
            "--log /var/log/<X>.log --bucket \"$S3_BUCKET\" -- $PY /tmp/<X>.py`."
        )
        # The wrapper must pass --bucket so the ship targets the right bucket.
        assert f"--slug {slug} --log /var/log/{slug}.log --bucket" in text, (
            f"spot workload {slug!r} ssm_log_capture wrap is missing the "
            "--bucket argument."
        )


def test_no_inline_py_dash_heredoc_for_workloads():
    """The workload python MUST be written to a spot-local file and run
    via the log-capture wrap — NOT inline via ``$PY - <<'PYEOF'`` (the
    inline form is what lost the log on the OOM incident). Pin that the
    only ``$PY -`` style invocations left are the wrapped ones; the
    workload bodies are now ``cat > /tmp/<slug>.py <<'PYEOF'``."""
    text = _SCRIPT.read_text()
    # The preflight-only step still legitimately uses an inline `$PY - <<`
    # because it is a fast read-only probe that cannot OOM the box and exits
    # 0 cleanly; durability there is not load-bearing. Assert the three
    # heavy workloads each stage their python to a file instead.
    for slug in _SPOT_WORKLOAD_SLUGS:
        assert f"cat > /tmp/{slug}.py <<'PYEOF'" in text, (
            f"spot workload {slug!r} should stage its python body to "
            f"/tmp/{slug}.py via `cat > /tmp/{slug}.py <<'PYEOF'` so it can "
            "be run under the log-capture wrap, not inline via `$PY -`."
        )


def test_cleanup_confirms_spot_log_before_terminate():
    """The dispatcher ``cleanup()`` (EXIT trap) MUST make a bounded
    best-effort confirmation of the spot-side log location in S3 BEFORE
    ``terminate-instances`` — so the dispatcher log → spot log is one hop
    for an operator triaging a failure. Secondary to the self-ship; must
    not block teardown."""
    text = _SCRIPT.read_text()
    m = re.search(r"^cleanup\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no cleanup() function found in spot_train.sh"
    body = m.group(0)
    # The confirmation (s3 ls into _ssm_logs/) must appear BEFORE the
    # terminate-instances call within the function body.
    ls_idx = body.find("_ssm_logs/")
    term_idx = body.find("terminate-instances")
    assert ls_idx != -1, (
        "cleanup() does not reference the _ssm_logs/ tree — the STEP 3 "
        "belt-and-suspenders spot-log confirmation is missing."
    )
    assert term_idx != -1, "cleanup() no longer calls terminate-instances?"
    assert ls_idx < term_idx, (
        "cleanup() must confirm the spot log in S3 BEFORE terminating the "
        "instance — otherwise the pointer is logged after the box (and any "
        "still-unshipped state) is gone."
    )
    # One-hop pointer to the failure diagnostics record too (STEP 4).
    assert "_spot_diagnostics/ae-predictor" in body, (
        "cleanup() should echo the _spot_diagnostics/ae-predictor/{date}.json "
        "pointer so the dispatcher log → failure record is one hop."
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
