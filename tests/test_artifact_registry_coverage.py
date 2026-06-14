"""
Artifact-registry coverage CI guard.

Phase 4 PR 6 of the artifact-freshness-monitor arc (plan doc:
``~/Development/alpha-engine-docs/private/artifact-freshness-monitor-260527.md``).
Mirrors ``alpha-engine-data/tests/test_artifact_registry_coverage.py``
(PR 4, merged 2026-05-27); the cascade closes producer-side coverage
of the registry across all 4 producing repos (ae-data, ae-research,
this repo, ae-backtester).

**What this catches.** A new ``s3.put_object(...)`` or
``s3.upload_file(...)`` site in ae-predictor production code that
hasn't been registered in
``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml`` (or
explicitly grandfathered). Forces operator attention at every new
producer addition — the silent absence-of-artifact bug class
can't slip past PR review without an explicit register-or-grandfather
decision.

**Design choice — per-file count rather than per-key-template
extraction.** Statically extracting key templates from f-string
``put_object(Key=...)`` calls is fragile (keys are often constructed
from surrounding context). Per-file count is stable across refactors
and sufficient to force operator review. See the ae-data PR 4 commit
message for the full rationale.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-file PUT-site counts. Pinning enforces operator attention on
# every new producer addition. Captured 2026-05-27.
EXPECTED_PER_FILE_PUT_COUNTS: dict[str, int] = {
    "analysis/observe_leaderboard.py": 2,  # observe_leaderboard/{date}.json + latest.json — realized-edge measurement artifact, observe-only, no freshness SLA (config#671/#702 L4539)
    "analysis/triple_barrier_cutover_runner.py": 2,
    "data/earnings_fetcher.py": 2,
    "health_status.py": 2,
    "inference/s3_io.py": 2,
    "inference/stages/shadow_versions.py": 1,  # predictions_shadow/{vid}/{date}.json — observe-only, best-effort, no freshness SLA (L4469 Phase 1)
    "model/registry.py": 4,  # lineage write + promote-in-place patch + _patch_stage + L4540 served-identity restamp of live manifest.json (already a registered artifact; L4469; copy_object not counted)
    "monitoring/drift_detector.py": 1,
    "regime/retrospective_eval_handler.py": 2,
    "regime/substrate.py": 2,
    "training/meta_trainer.py": 7,
    "training/model_zoo.py": 3,  # G2 live-contract RESTORE (re-puts existing champion keys, not a new artifact) + model_zoo/leaderboard/{date}.json (observe-only, best-effort, no freshness SLA — L4544) + model_zoo/trial_log.json (cumulative trial ledger, observe-only, failure recorded in leaderboard trial_log_status — L4582)
    "training/risk_model_persist.py": 2,
    "training/train_handler.py": 2,
}


_SCAN_EXEMPT_PREFIXES: tuple[str, ...] = (
    "tests/",
    "infrastructure/lambdas/",
    ".claude/",
    ".venv/",
    "build/",
)


def _enumerate_put_sites() -> dict[str, int]:
    """Return ``{relative_path: count}`` of production files with PUT sites."""
    result = subprocess.run(
        [
            "git", "grep", "-l", "-E",
            r"(put_object|upload_file)\(",
            "--", "*.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = [
        line for line in result.stdout.splitlines()
        if line and not any(line.startswith(p) for p in _SCAN_EXEMPT_PREFIXES)
    ]
    counts: dict[str, int] = {}
    for rel in files:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        counts[rel] = len(re.findall(r"(?:put_object|upload_file)\(", text))
    return counts


def test_every_producer_file_is_pinned():
    actual = _enumerate_put_sites()
    unpinned = sorted(set(actual.keys()) - set(EXPECTED_PER_FILE_PUT_COUNTS.keys()))
    assert not unpinned, (
        "New producer file(s) with S3 PUT sites detected but not pinned:\n"
        + "\n".join(f"  - {f} ({actual[f]} PUT call(s))" for f in unpinned)
        + "\n\nResolution:\n"
        "  1. Register the new artifact(s) in alpha-engine-config/"
        "private-docs/ARTIFACT_REGISTRY.yaml (or add the prefix to "
        "grandfathered_paths with a one-line reason).\n"
        "  2. Add the file(s) to EXPECTED_PER_FILE_PUT_COUNTS in "
        "tests/test_artifact_registry_coverage.py with the per-file count.\n"
        "  3. Re-run this test."
    )


def test_every_pinned_file_still_exists():
    actual = _enumerate_put_sites()
    stale = sorted(set(EXPECTED_PER_FILE_PUT_COUNTS.keys()) - set(actual.keys()))
    assert not stale, (
        "Pinned file(s) no longer have PUT sites (or no longer exist):\n"
        + "\n".join(f"  - {f}" for f in stale)
        + "\n\nResolution: remove the file from EXPECTED_PER_FILE_PUT_COUNTS. "
        "If the artifact was retired, also retire its row in "
        "alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml."
    )


def test_pinned_counts_match_actual():
    actual = _enumerate_put_sites()
    deltas = []
    for path, expected_count in sorted(EXPECTED_PER_FILE_PUT_COUNTS.items()):
        actual_count = actual.get(path, 0)
        if actual_count != expected_count:
            deltas.append(f"  - {path}: expected={expected_count}, actual={actual_count}")
    assert not deltas, (
        "PUT-site count drift detected:\n"
        + "\n".join(deltas)
        + "\n\nResolution: for each delta, either (a) the PUT count changed "
        "legitimately — register the new artifact in alpha-engine-config/"
        "private-docs/ARTIFACT_REGISTRY.yaml (or grandfather), then bump "
        "the pinned count; or (b) the change was inadvertent — revert."
    )
