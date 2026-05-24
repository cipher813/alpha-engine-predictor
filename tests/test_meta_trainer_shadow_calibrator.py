"""Regression tests for the shadow calibrator infrastructure (2026-05-24
calibrator-shadow arc, in response to the 2026-05-23 Platt-collapse incident).

The shadow calibrator is fit alongside the live calibrator on the SAME
OOS rows and labels, runs through both promotion gates, and persists
results in the training manifest. It is observability-only — does NOT
gate promotion, does NOT feed inference, does NOT overwrite the live
calibrator's S3 key.

These tests pin the wiring so future refactors can't silently retire the
shadow path. They are source-text + AST invariants (full behavioral
coverage requires a synthetic-data trainer run, which is out of scope
for the unit suite — covered by the per-component tests in
``test_calibrator.py`` plus the live Saturday SF observation cycle).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import config as cfg


_META_TRAINER = Path(__file__).resolve().parents[1] / "training" / "meta_trainer.py"
_CONFIG = Path(__file__).resolve().parents[1] / "config.py"
_TRAIN_HANDLER = Path(__file__).resolve().parents[1] / "training" / "train_handler.py"


@pytest.fixture(scope="module")
def meta_trainer_source() -> str:
    return _META_TRAINER.read_text()


@pytest.fixture(scope="module")
def config_source() -> str:
    return _CONFIG.read_text()


@pytest.fixture(scope="module")
def train_handler_source() -> str:
    return _TRAIN_HANDLER.read_text()


# ── Config knobs are exposed ─────────────────────────────────────────────────


def test_config_exposes_shadow_method_knob():
    """``cfg.CALIBRATION_SHADOW_METHOD`` defaults to None (shadow disabled)
    so adopting the trainer on a yaml that hasn't been updated yet is a
    no-op. The shadow path activates only when a yaml sets it explicitly.
    """
    assert hasattr(cfg, "CALIBRATION_SHADOW_METHOD"), (
        "shadow infrastructure requires cfg.CALIBRATION_SHADOW_METHOD; "
        "PR B retired without wiring the config knob?"
    )


def test_config_exposes_shadow_tuning_knobs():
    """Shadow's whole point is to let operators iterate Platt's
    class-balance + regularisation knobs across Saturday cycles. Without
    these knobs the shadow path can only test method swaps, not the
    actual mechanism that caused the 5/23 collapse (unbalanced intercept).
    """
    assert hasattr(cfg, "CALIBRATION_SHADOW_CLASS_WEIGHT")
    assert hasattr(cfg, "CALIBRATION_SHADOW_C")
    # C must always be float so the LogisticRegression constructor doesn't
    # raise on a yaml that left it as a string.
    assert isinstance(cfg.CALIBRATION_SHADOW_C, float)


def test_live_calibrator_also_honours_class_weight_and_C_knobs():
    """When shadow is eventually promoted → live, the operator just flips
    ``calibration.method`` in predictor.yaml — the class_weight/C knobs the
    shadow validated must carry over to the live path. Both surfaces must
    parse the same keys from the same config block.
    """
    assert hasattr(cfg, "CALIBRATION_CLASS_WEIGHT")
    assert hasattr(cfg, "CALIBRATION_C")
    assert isinstance(cfg.CALIBRATION_C, float)


# ── meta_trainer.py wires shadow correctly ───────────────────────────────────


def test_meta_trainer_fits_shadow_when_enabled(meta_trainer_source):
    """Shadow fit lives in Step 7b and is gated on the config knob."""
    assert "shadow_method = cfg.CALIBRATION_SHADOW_METHOD" in meta_trainer_source, (
        "Shadow fit must read cfg.CALIBRATION_SHADOW_METHOD; otherwise "
        "operators have no way to disable the shadow path."
    )
    assert "if shadow_method:" in meta_trainer_source


def test_shadow_calibrator_uses_same_oos_inputs_as_live(meta_trainer_source):
    """Shadow's value comes from comparing apples-to-apples with live. If
    shadow is fit on different inputs the comparison is uninformative.
    """
    # The shadow fit must reuse the same oos_meta_preds / oos_up_labels
    # the live calibrator just consumed. AST-walk would be cleaner, but
    # the source-text assertion is sufficient — these are local variable
    # names that don't get renamed silently.
    fit_lines = [
        ln for ln in meta_trainer_source.splitlines()
        if "shadow_calibrator.fit(" in ln
        or (ln.lstrip().startswith("oos_meta_preds, oos_up_labels")
            and "shadow_calibrator" in meta_trainer_source.split(ln)[0][-200:])
    ]
    assert any("oos_meta_preds" in ln for ln in fit_lines), (
        "shadow_calibrator.fit must take oos_meta_preds, not a different "
        "row set — apples-to-apples vs the live calibrator is the point."
    )


def test_shadow_does_not_feed_promotion_logic(meta_trainer_source):
    """The ``promoted = ...`` and clause must not reference shadow gates."""
    # Find the promoted assignment expression.
    m = re.search(
        r"promoted\s*=\s*\(([^)]*)\)",
        meta_trainer_source,
        re.DOTALL,
    )
    assert m is not None, "could not locate 'promoted = (...)' expression"
    promoted_expr = m.group(1)
    assert "shadow" not in promoted_expr.lower(), (
        "shadow calibrator must NOT influence the promoted boolean — "
        "observability only. Found 'shadow' in:\n" + promoted_expr
    )


def test_shadow_persists_in_manifest_under_distinct_key(meta_trainer_source):
    """The manifest block carries shadow under its own key so dashboards /
    diagnostics can render shadow-vs-live without special-casing the
    live block's shape."""
    assert '"shadow_calibrator":' in meta_trainer_source, (
        "manifest must publish a 'shadow_calibrator' key when shadow is fit"
    )


def test_shadow_calibrator_persists_to_distinct_s3_key(meta_trainer_source):
    """Shadow's pickle is uploaded under isotonic_calibrator_shadow.pkl
    so the backtester's grace-period check on isotonic_calibrator.pkl.meta.json
    is untouched and inference never picks up the shadow artifact by
    accident.
    """
    assert "isotonic_calibrator_shadow.pkl" in meta_trainer_source, (
        "shadow pickle must use a distinct S3 key from the live calibrator"
    )


def test_shadow_calibration_metrics_published_in_result(meta_trainer_source):
    """train_handler's email reads result['shadow_calibration'] — the
    trainer must publish that key so the email row renders."""
    assert '"shadow_calibration":' in meta_trainer_source


# ── train_handler.py renders the shadow row ──────────────────────────────────


def test_train_handler_renders_shadow_row(train_handler_source):
    """The shadow row is what tells the operator whether shadow is ready
    to promote → live."""
    assert "_build_shadow_calibration_html" in train_handler_source, (
        "train_handler must call _build_shadow_calibration_html"
    )
    assert "Shadow Calibrator" in train_handler_source, (
        "email must label the shadow section so operators can identify it"
    )
