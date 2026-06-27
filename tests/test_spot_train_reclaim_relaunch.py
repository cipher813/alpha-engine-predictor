"""Pin the #883 mid-run spot-reclaim relaunch adoption in spot_train.sh.

Origin: nousergon/alpha-engine-config#883 — the 2026-05-30 Saturday SF
failed in DataPhase1 when a nested data spot was reclaimed by AWS
*mid-workload*. The relaunch fix shipped first in alpha-engine-data #349,
then the classify→decide DECISION was lifted into the lib chokepoint
``python -m nousergon_lib.ec2_spot relaunch-decision`` (lib v0.65.0+, now
``krepis.ec2_spot``). The predictor Saturday retrain shared the identical
spot+SSM orchestration but carried the SAME gap — its ``cleanup()`` EXIT
trap terminated the box on a reclaim with NO relaunch at all (the #883
PRIMARY gap; the missing adopter). This PR adopts the lib chokepoint.

These tests pin that adoption so a future refactor can't silently revert it:
  * the lib ``relaunch-decision`` chokepoint is the DECISION surface
    (NOT a re-introduced inline reclaim classifier)
  * ``cleanup()`` captures the exit status FIRST and re-exits with it
    (so a recovered cleanup path never masks a real failure as rc=0)
  * the relaunch ``exec``s a FRESH spot with the SAME argv, threading
    SPOT_ATTEMPT, bounded by MAX_SPOT_ATTEMPTS
  * the classify (describe-instances, inside the lib) happens BEFORE
    terminate-instances, the relaunch exec AFTER teardown
  * only a confirmed reclaim relaunches — fail-loud is preserved because
    the lib classifies a genuine workload failure as "other"/"unknown"
    and exits NO_RELAUNCH_EXIT_CODE (no blind retry)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "spot_train.sh"


def _cleanup_body() -> str:
    text = _SCRIPT.read_text()
    m = re.search(r"^cleanup\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no cleanup() function found in spot_train.sh"
    return m.group(0)


def test_script_exists():
    assert _SCRIPT.exists(), f"spot_train.sh missing at {_SCRIPT}"


def test_script_syntactically_valid():
    """``bash -n`` must accept the script — the relaunch edits add a
    command-substitution + array re-exec that must parse under bash."""
    r = subprocess.run(
        ["bash", "-n", str(_SCRIPT)], capture_output=True, text=True
    )
    assert r.returncode == 0, f"bash -n rejected spot_train.sh:\n{r.stderr}"


def test_orig_args_captured_before_parse():
    """``_ORIG_ARGS`` must be captured BEFORE the flag-parse loop consumes
    "$@", so the relaunch re-execs with the IDENTICAL argv (--full-only /
    --model-zoo-spec <id> / --instance-type X). Capturing after the parse
    loop (which `shift`s every arg away) would relaunch with an EMPTY argv,
    silently downgrading a --full-only champion retrain to the default mode."""
    text = _SCRIPT.read_text()
    cap = text.find("_ORIG_ARGS=(")
    parse = text.find("while [ $# -gt 0 ]")
    assert cap != -1, (
        "spot_train.sh does not capture _ORIG_ARGS — the #883 relaunch "
        "cannot re-exec with the original argv."
    )
    assert parse != -1, "flag-parse loop (while [ $# -gt 0 ]) not found"
    assert cap < parse, (
        "_ORIG_ARGS must be captured BEFORE the flag-parse loop — capturing "
        "after the loop (which shifts every arg away) relaunches with an "
        "empty argv and silently drops --full-only / --model-zoo-spec."
    )


def test_cleanup_uses_lib_relaunch_decision_chokepoint():
    """The relaunch DECISION MUST come from the lib chokepoint
    ``python -m nousergon_lib.ec2_spot relaunch-decision`` (lib v0.65.0+),
    NOT a re-introduced inline reclaim classifier. The lib owns the
    classify (describe-instances) + the bounded/SF-coupled decision; the
    launcher branches on the exit code."""
    body = _cleanup_body()
    assert "nousergon_lib.ec2_spot relaunch-decision" in body, (
        "cleanup() does not call the lib chokepoint "
        "`python -m nousergon_lib.ec2_spot relaunch-decision`. #883 lifts the "
        "classify→decide logic into the lib (krepis.ec2_spot); the launcher "
        "must consume it, not re-inline a Server.SpotInstanceTermination grep."
    )
    # Must pass the bounded-attempt args the lib decision needs.
    for flag in ("--instance-id", "--attempt", "--max-attempts"):
        assert flag in body, (
            f"cleanup() relaunch-decision call is missing {flag} — the lib "
            "decision needs the 1-based attempt + total-attempts budget."
        )


def test_cleanup_no_inline_reclaim_classifier():
    """The launcher MUST NOT re-inline the spot-reclaim classification
    (that's exactly what #883 lifted to the lib). A non-comment line that
    greps Server.SpotInstanceTermination / the spot-request status codes is
    the divergent-copy regression the chokepoint exists to prevent."""
    forbidden = (
        "Server.SpotInstanceTermination",
        "instance-terminated-no-capacity",
        "describe-spot-instance-requests",
    )
    offenders = []
    for i, raw in enumerate(_SCRIPT.read_text().splitlines(), start=1):
        if raw.strip().startswith("#"):
            continue
        for tok in forbidden:
            if tok in raw:
                offenders.append((i, tok, raw.strip()))
    assert not offenders, (
        "cleanup() re-inlines reclaim classification instead of using the lib "
        "chokepoint:\n"
        + "\n".join(f"  line {n}: {tok} :: {ln}" for n, tok, ln in offenders)
        + "\n\nRoute through `python -m nousergon_lib.ec2_spot relaunch-decision`."
    )


def test_cleanup_captures_and_reexits_status():
    """cleanup() must capture ``exit_code=$?`` as its FIRST statement and end
    on ``exit "$exit_code"`` so a recovered cleanup path (echos + `|| true`
    teardown all succeed) can never mask a real workload failure as rc=0 to
    the cron/orchestration wrapper (the L4485 status-masking class)."""
    body = _cleanup_body()
    first = body.split("{", 1)[1].lstrip().splitlines()
    # First non-blank, non-comment statement should set exit_code from $?.
    first_stmt = next(
        ln.strip() for ln in first if ln.strip() and not ln.strip().startswith("#")
    )
    assert "exit_code=$?" in first_stmt, (
        f"cleanup()'s first statement must capture exit_code=$? — got "
        f"{first_stmt!r}. Any earlier command overwrites $? and the relaunch "
        "decision + final exit then run against the wrong status."
    )
    assert 'exit "$exit_code"' in body, (
        "cleanup() must end on `exit \"$exit_code\"` so the captured status is "
        "re-raised — otherwise a successful teardown command leaves the script "
        "exiting 0 and a failed run reads as success."
    )


def test_relaunch_execs_fresh_spot_with_orig_args():
    """On a classified reclaim, cleanup() must `exec bash "$0"` with the
    captured _ORIG_ARGS and an incremented SPOT_ATTEMPT, after `trap - EXIT`.
    The exec re-runs the launcher in place (bounded by MAX_SPOT_ATTEMPTS)."""
    body = _cleanup_body()
    assert "trap - EXIT" in body, (
        "cleanup() must `trap - EXIT` before the relaunch exec so the exec'd "
        "process installs its own EXIT trap cleanly."
    )
    m = re.search(r'SPOT_ATTEMPT=\$\(\(SPOT_ATTEMPT \+ 1\)\) exec bash "\$0"', body)
    assert m, (
        "cleanup() must relaunch via "
        '`SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ...` threading the '
        "incremented attempt across the re-exec."
    )
    assert "_ORIG_ARGS" in body, (
        "the relaunch exec must forward _ORIG_ARGS so the fresh attempt re-runs "
        "under the same mode (--full-only / --model-zoo-spec / --instance-type)."
    )


def test_classify_happens_before_terminate_relaunch_after():
    """The lib classify (describe-instances) must run while the instance still
    exists — BEFORE terminate-instances; the relaunch exec must run AFTER
    teardown so the dead worker + its S3 staging are already cleaned when the
    fresh attempt starts."""
    body = _cleanup_body()
    # Match the COMMANDS, not prose comments that also name these tokens.
    decide_idx = body.find("ec2_spot relaunch-decision")
    term_idx = body.find("aws ec2 terminate-instances")
    exec_idx = body.find("exec bash")
    assert decide_idx != -1 and term_idx != -1 and exec_idx != -1, (
        "cleanup() must contain the relaunch-decision call, the "
        "terminate-instances call, and the relaunch exec."
    )
    assert decide_idx < term_idx, (
        "the lib relaunch-decision (which describe-instances the spot) must run "
        "BEFORE terminate-instances — after teardown the instance is gone and "
        "the reclaim is unclassifiable."
    )
    assert term_idx < exec_idx, (
        "the relaunch exec must run AFTER terminate-instances + staging cleanup "
        "so the fresh attempt does not race the dead worker's S3 staging."
    )


def test_relaunch_bounded_by_max_spot_attempts():
    """The relaunch must be gated on SPOT_ATTEMPT < MAX_SPOT_ATTEMPTS so a
    persistent-reclaim loop can't relaunch unboundedly. Both the env defaults
    and the guard must be present."""
    text = _SCRIPT.read_text()
    assert re.search(r'MAX_SPOT_ATTEMPTS="\$\{MAX_SPOT_ATTEMPTS:-\d+\}"', text), (
        "MAX_SPOT_ATTEMPTS default not declared."
    )
    assert re.search(r'SPOT_ATTEMPT="\$\{SPOT_ATTEMPT:-1\}"', text), (
        "SPOT_ATTEMPT must default to 1 (first run)."
    )
    body = _cleanup_body()
    assert "SPOT_ATTEMPT" in body and "MAX_SPOT_ATTEMPTS" in body, (
        "cleanup() must gate the relaunch on SPOT_ATTEMPT < MAX_SPOT_ATTEMPTS."
    )


def test_absorbed_interruption_emits_metric():
    """A relaunch is fail-loud-but-recovering: it must emit the
    AlphaEngine/SpotInterruptionRetry CloudWatch metric (Process=
    predictor-training) so the absorbed interruption is observable, never
    silent — mirroring the #349 data-launcher reference."""
    body = _cleanup_body()
    assert "SpotInterruptionRetry" in body, (
        "cleanup() must emit the AlphaEngine/SpotInterruptionRetry metric on a "
        "relaunch so the absorbed interruption is observable."
    )
    assert "Process=predictor-training" in body, (
        "the SpotInterruptionRetry metric must carry "
        "Process=predictor-training to discriminate it from the data/backtester "
        "siblings on the shared AlphaEngine namespace."
    )
