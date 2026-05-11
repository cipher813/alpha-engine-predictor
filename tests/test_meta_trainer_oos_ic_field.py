"""
tests/test_meta_trainer_oos_ic_field.py — Pin the meta_model_oos_ic
field added 2026-05-11 in response to the false-positive retrain alert.

The L2 Ridge's `_val_ic` is its IN-SAMPLE Pearson fit (no held-out
split inside `MetaModel.fit`). Publishing only that as the headline
`meta_model_ic` misled the backtester production_health degradation
gate into reading an inflated 0.4634 reference against a -0.10
rolling IC, tripping a HIGH-severity false-positive alert.

This PR adds `meta_model_in_sample_ic` (explicit alias) and
`meta_model_oos_ic` (walk-forward Spearman at cfg.FORWARD_DAYS, the
active label horizon — already computed for horizon_diagnostic).
Consumer-side fix: alpha-engine-backtester #181.

Source-text invariants only — a full behavioral test would require
spinning up the entire trainer with synthetic data, which is out of
scope. The behavioral guarantee (oos_ic value matches
horizon_diagnostic.curve.{FORWARD_DAYS}d.spearman) is structural: both
fields read from the same `horizon_ics` dict literal in the same
return statement.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_META_TRAINER = (
    Path(__file__).resolve().parent.parent
    / "training" / "meta_trainer.py"
)


@pytest.fixture(scope="module")
def meta_trainer_source() -> str:
    return _META_TRAINER.read_text()


# ── New OOS / in-sample fields are published ────────────────────────────────


def test_publishes_meta_model_in_sample_ic_key(meta_trainer_source):
    """The explicit alias must appear in the result-dict so consumers
    can distinguish the in-sample fit from the OOS measurement by
    name alone."""
    assert '"meta_model_in_sample_ic"' in meta_trainer_source, (
        "meta_model_in_sample_ic key removed from training_summary — "
        "downstream consumers rely on the explicit alias for naming clarity"
    )


def test_publishes_meta_model_oos_ic_key(meta_trainer_source):
    """The honest OOS field — the canonical reference for downstream
    degradation comparisons in alpha-engine-backtester
    production_health._load_training_ic."""
    assert '"meta_model_oos_ic"' in meta_trainer_source, (
        "meta_model_oos_ic key removed from training_summary — "
        "alpha-engine-backtester production_health depends on it as "
        "the OOS reference for retrain-alert degradation comparisons"
    )


def test_oos_ic_sources_from_forward_days_not_hardcoded_21d(meta_trainer_source):
    """The OOS field must be derived from `cfg.FORWARD_DAYS` so it
    tracks any future horizon migration in lockstep with the active
    label horizon. A hardcoded "21d" literal would silently drift if
    forward_days bumped."""
    # The expected pattern: `horizon_ics[f"{cfg.FORWARD_DAYS}d"]`
    pat = re.compile(
        r'meta_model_oos_ic.*?horizon_ics\s*\[\s*f"\{cfg\.FORWARD_DAYS\}d"',
        re.DOTALL,
    )
    assert pat.search(meta_trainer_source), (
        "meta_model_oos_ic source string does not match the expected "
        "`horizon_ics[f\"{cfg.FORWARD_DAYS}d\"]` pattern. Either the "
        "field was removed or it's now reading a hardcoded horizon, "
        "which would silently drift on the next horizon migration."
    )


def test_in_sample_warning_comment_present(meta_trainer_source):
    """The block comment warning consumers about the in-sample vs
    OOS distinction must remain — without it the next reader will
    re-introduce the same false-alarm path. The wording is allowed
    to evolve; pin only the load-bearing phrase."""
    assert "IN-SAMPLE Pearson fit" in meta_trainer_source, (
        "Lost the 'IN-SAMPLE Pearson fit' warning comment — re-add "
        "documentation so future readers know meta_model_ic is the "
        "Ridge's training fit, not its OOS skill"
    )


def test_legacy_meta_model_ic_preserved_for_backward_compat(meta_trainer_source):
    """S3 contract safety: schema changes are additive only. Removing
    `meta_model_ic` would break v2 consumers (train_handler email,
    dry_run_meta_training sanity check) that read the unsuffixed key
    directly. Keep it published even though it's deprecated."""
    # The unsuffixed key must appear (alongside the new ones)
    assert re.search(
        r'^\s*"meta_model_ic"\s*:\s*round\(meta_model\._val_ic',
        meta_trainer_source,
        re.MULTILINE,
    ), (
        "meta_model_ic removed from training_summary — this breaks the "
        "S3 schema additive-only contract. Keep the field published; "
        "add new ones alongside, don't rename in place."
    )
