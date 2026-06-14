"""Regression test pinning that every ``analysis/*`` CLI invokes ``--help``
without an ``AttributeError`` on a stale ``cfg.<MISSING>`` reference at
argparse default-setup time.

Caught 2026-05-28 (operator audit while scoping L2878 / the regime-
conditioning rebuild): three diagnostic CLIs that gate Stages 1d / 2d /
3 cutover decisions — ``variant_cutover_gate``, ``horizon_battery``,
``triple_barrier_cutover_runner`` — were ALL referencing
``cfg.RESEARCH_BUCKET``, an attribute that has never been defined in
``config.py`` (canonical name is ``cfg.S3_BUCKET``). The stale symbol
was introduced 2026-05-10 by PR #117 (variant gate substrate) and
copied into the two sibling CLIs since. The CLIs died at import time
during the argparse default lookup before any user-facing message.

Programmatic API was unaffected — the bug only surfaced when an
operator actually ran the CLI for the first time. Latent for 18 days.

This test closes the bug CLASS at PR time: every analysis-CLI module
with a ``main()`` entry point is invoked via ``python -m`` with
``--help`` and the test asserts exit code 0 + non-empty stdout.
``argparse``'s ``--help`` exits with code 0; a missing-attr error
surfaces as a non-zero exit code + stderr trace.

Composes with the lib-version-bump-check chokepoint pattern (mirror
test ``test_version_bump_workflow.py`` in alpha-engine-lib #82) —
when a class of mistake recurs in more than one file, lift the
chokepoint to a test that fails at PR time. See
[[feedback_lift_invariants_to_chokepoint_after_second_recurrence]].
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every analysis CLI with a ``main()`` entry. New CLIs added under
# ``analysis/`` MUST be appended here (or excluded via a comment if
# explicitly internal-only). The walker below would auto-discover them
# but the explicit list also serves as documentation.
ANALYSIS_CLIS = [
    "analysis.variant_cutover_gate",
    "analysis.horizon_battery",
    "analysis.triple_barrier_cutover_runner",
    "analysis.compare_modes",
    "analysis.observe_leaderboard",
]


@pytest.mark.parametrize("module", ANALYSIS_CLIS)
def test_analysis_cli_help_works(module: str) -> None:
    """Pin ``python -m <module> --help`` exit 0 + non-empty usage line.

    Catches stale ``cfg.<MISSING_ATTR>`` references in argparse
    defaults — the failure mode of the original 2026-05-28 incident.
    """
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"`python -m {module} --help` exited {result.returncode}.\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    assert result.stdout.strip(), (
        f"`python -m {module} --help` produced empty stdout — argparse "
        f"should always print a usage line."
    )
    assert "usage:" in result.stdout.lower(), (
        f"`python -m {module} --help` stdout missing 'usage:' line "
        f"(argparse default formatter).\nstdout: {result.stdout[:500]}"
    )


def test_analysis_cli_list_covers_every_module_with_main() -> None:
    """Walk ``analysis/*.py`` looking for any module that defines a
    top-level ``main()`` function; assert it appears in
    ``ANALYSIS_CLIS`` above.

    Catches the case where someone adds a new analysis CLI but forgets
    to register it in the parametrize list above, leaving the new CLI
    un-smoke-tested.
    """
    analysis_dir = REPO_ROOT / "analysis"
    discovered: list[str] = []
    for py_path in sorted(analysis_dir.glob("*.py")):
        if py_path.name.startswith("_"):
            continue
        source = py_path.read_text()
        if "\ndef main(" not in source and not source.startswith("def main("):
            continue
        mod_name = f"analysis.{py_path.stem}"
        discovered.append(mod_name)

    missing = sorted(set(discovered) - set(ANALYSIS_CLIS))
    assert not missing, (
        "Newly-discovered analysis CLIs missing from ANALYSIS_CLIS: "
        f"{missing}. Append them to the list above (or comment why "
        "they're excluded from the help-smoke gate)."
    )
