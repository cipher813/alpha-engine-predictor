"""Pins spot_train.sh `--preflight-only` to the Friday shell_run dry-path
hard invariant: boot + import/lib-pin + read-only ArcticDB connectivity
probe, then `exit 0` BEFORE the smoke step and BEFORE the full-training
step — with NO model training, NO weight promotion, and ZERO S3/config
writes.

Owed-item #2 of ROADMAP "Friday shell-run — per-module dry-path
activation" (P1). Static-analysis test (mirrors
test_spot_train_aws_region.py) — the spot_train.sh path is SSM/EC2 and
cannot be exercised in CI; these assertions guard the structural
invariant against a future edit that would let preflight-only fall
through into training.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"


def _text() -> str:
    return _SCRIPT.read_text()


def test_spot_train_exists():
    assert _SCRIPT.is_file()


def test_preflight_only_flag_parses():
    text = _text()
    assert "--preflight-only) MODE=\"preflight-only\" ;;" in text, (
        "--preflight-only flag not wired into the flag parser"
    )


def test_preflight_only_branch_exists_and_exits_zero():
    text = _text()
    assert 'if [ "$MODE" = "preflight-only" ]; then' in text, (
        "no dedicated preflight-only branch found"
    )
    # The branch must terminate with `exit 0` (clean dispatcher exit;
    # trap cleanup still terminates the spot + clears S3 staging).
    branch = text.split('if [ "$MODE" = "preflight-only" ]; then', 1)[1]
    branch = branch.split("# ── Smoke test", 1)[0]
    assert "exit 0" in branch, "preflight-only branch must exit 0"


def test_preflight_only_runs_before_smoke_and_full_training():
    text = _text()
    i_branch = text.index('if [ "$MODE" = "preflight-only" ]; then')
    i_smoke = text.index("# ── Smoke test (dry_run=True)")
    i_full = text.index("# ── Full training (dry_run=False)")
    assert i_branch < i_smoke < i_full, (
        "preflight-only branch must precede the smoke and full-training "
        "sections so its exit 0 short-circuits before any training"
    )


def test_preflight_only_does_not_invoke_training_or_promotion():
    """The preflight-only SSM step body must not call run_meta_training /
    train_handler.main / dry_run training — those are the train+promote
    entrypoints. It may only import the package + run TrainingPreflight +
    the read-only ArcticDB probe."""
    text = _text()
    start = text.index('run_ssm "preflight-only"')
    # End of the preflight-only run_ssm heredoc payload.
    end = text.index("PREFLIGHT\n)\"", start)
    payload = text[start:end]

    # Forbidden as *executable references* (call/import forms), not as
    # mentions inside the human-readable proof print() strings. We strip
    # the print(...) / log.info(...) diagnostic lines first so the proof
    # text ("no run_meta_training call") doesn't false-positive.
    code_lines = [
        ln
        for ln in payload.splitlines()
        if not ln.lstrip().startswith(("print(", "log.info(", "#"))
    ]
    code = "\n".join(code_lines)
    forbidden = [
        "run_meta_training(",
        "import run_meta_training",
        "train_main(",
        "from training.train_handler import main",
        "download_from_arctic",  # that is the training data DOWNLOAD, not a probe
        "put_object",
        "upload_file",
    ]
    for token in forbidden:
        assert token not in code, (
            f"preflight-only step must NOT reference {token!r} — "
            f"it would break the no-train/no-promote/no-write invariant"
        )

    # Positive: it must do the reuse-not-rebuild things.
    assert "TrainingPreflight" in payload, (
        "preflight-only must reuse the existing training/preflight.py "
        "TrainingPreflight (do not rebuild a parallel preflight)"
    )
    assert "import alpha_engine_lib" in payload, (
        "preflight-only must import alpha_engine_lib to catch lib-pin drift"
    )
    assert "list_symbols()" in payload, (
        "preflight-only must do a read-only ArcticDB universe probe"
    )


def test_preflight_only_step_keeps_aws_region_export():
    """Same #247 regression guard as test_spot_train_aws_region.py — the
    new step's env line must still export AWS_REGION/AWS_DEFAULT_REGION
    (TrainingPreflight.check_env_vars('AWS_REGION') hard-requires it)."""
    text = _text()
    start = text.index('run_ssm "preflight-only"')
    end = text.index("PREFLIGHT\n)\"", start)
    payload = text[start:end]
    export_lines = [
        ln
        for ln in payload.splitlines()
        if ln.startswith("export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp")
    ]
    assert export_lines, "preflight-only step missing its per-step export line"
    for ln in export_lines:
        assert "AWS_REGION=" in ln and "AWS_DEFAULT_REGION=" in ln, (
            f"preflight-only export line missing AWS region vars: {ln!r}"
        )
