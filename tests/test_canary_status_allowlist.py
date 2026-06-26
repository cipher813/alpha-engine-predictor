"""Unit tests for the deploy.sh canary status-acceptance allowlist (config#853).

The post-deploy canaries in ``infrastructure/deploy.sh`` gate promotion of a
freshly-published Lambda version on the ``statusCode`` returned by a dry_run
invocation. The acceptance decision is a single Bash helper,
``canary_status_ok``, shared by all three canary sites (inference, regime
substrate, regime retrospective-eval).

These tests source that helper directly (no AWS, no Docker — the deploy body is
skipped via ``CANARY_LIB_SOURCE_ONLY=1``) and assert the producer-status
contract: the legitimate no-op statuses are ACCEPTED (so a future skip-path
cannot false-red a clean deploy) while genuine error codes are REJECTED (so a
broken image still rolls back). Mirrors nousergon-data #295's explicit
``status in ('OK','SKIPPED')`` allowlist.

A full canary requires a live Lambda invoke, which is unavailable in CI; this
suite covers the pure status-acceptance logic that the live promote path
depends on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

DEPLOY_SH = Path(__file__).resolve().parents[1] / "infrastructure" / "deploy.sh"


def _canary_status_ok(status: str) -> bool:
    """Source deploy.sh and run ``canary_status_ok <status>``; return its exit.

    Exit 0 (accept → promote) maps to True, any non-zero (reject → roll back)
    maps to False, matching the ``if ! canary_status_ok ...`` use sites.
    """
    script = (
        f'source "{DEPLOY_SH}"\n'
        'if canary_status_ok "$1"; then exit 0; else exit 1; fi\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script, "_", status],
        env={"CANARY_LIB_SOURCE_ONLY": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    # Any exit code other than the helper's clean 0/1 means the source step
    # itself blew up — surface it rather than silently treating it as reject.
    assert proc.returncode in (0, 1), (
        f"unexpected exit {proc.returncode} for status={status!r}: "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return proc.returncode == 0


# ── Allowlisted statuses (200 + legitimate no-ops) → accept / promote ────────

@pytest.mark.parametrize("status", ["200", "204", "skipped"])
def test_allowlisted_statuses_accepted(status):
    assert _canary_status_ok(status) is True


# ── Genuine error statuses → reject / roll back ──────────────────────────────

@pytest.mark.parametrize("status", ["400", "500", "502", "418"])
def test_error_statuses_rejected(status):
    # 400 = deprecated train route / unknown regime action; 5xx = real failure.
    assert _canary_status_ok(status) is False


# ── Degenerate parser outputs → reject (fail closed) ─────────────────────────

@pytest.mark.parametrize("status", ["", "0", "UNKNOWN", "PARSE_ERROR"])
def test_degenerate_status_rejected(status):
    # The payload parsers emit "" / "0" on a malformed or missing statusCode;
    # those must NOT be mistaken for a no-op success.
    assert _canary_status_ok(status) is False


# ── Near-miss success codes are NOT silently accepted ────────────────────────

@pytest.mark.parametrize("status", ["201", "200 ", " 200", "20", "2000"])
def test_near_miss_codes_rejected(status):
    # Only the exact enumerated strings are no-ops; the allowlist must not
    # widen to "any 2xx" or tolerate stray whitespace.
    assert _canary_status_ok(status) is False
