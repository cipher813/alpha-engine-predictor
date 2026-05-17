"""Regression: the meta-trainer's direction calibrator must honour
``cfg.CALIBRATION_METHOD`` rather than hardcoding the method.

ROADMAP L1873 — the 2026-05-09 75%+ DOWN-veto inversion was isotonic-tail
overconfidence; the prescribed fix is the tail-robust ``platt`` method. The
config knob (``predictor.yaml::calibration.method`` → ``cfg.CALIBRATION_METHOD``)
existed and defaulted to ``platt`` since 2026-04-01, but ``meta_trainer.py``'s
Step-7b call site hardcoded ``PlattCalibrator(method="isotonic")`` and never
read the config — a dead knob that silently ignored the committed default.

This is an AST pin-test: it asserts every ``PlattCalibrator(...)``
construction in ``training/meta_trainer.py`` that passes ``method=`` does so
via a config reference (Name/Attribute), never a string literal. It is the
tripwire that would have caught the dead-knob bug at the source.
"""

import ast
from pathlib import Path

import config as cfg


META_TRAINER = Path(__file__).resolve().parents[1] / "training" / "meta_trainer.py"


def _platt_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (
            fn.id if isinstance(fn, ast.Name)
            else fn.attr if isinstance(fn, ast.Attribute)
            else None
        )
        if name == "PlattCalibrator":
            calls.append(node)
    return calls


def test_meta_trainer_constructs_direction_calibrator():
    tree = ast.parse(META_TRAINER.read_text())
    calls = _platt_calls(tree)
    assert calls, "expected at least one PlattCalibrator(...) construction in meta_trainer.py"


def test_direction_calibrator_method_is_config_driven_not_literal():
    tree = ast.parse(META_TRAINER.read_text())
    for call in _platt_calls(tree):
        method_kw = next(
            (kw for kw in call.keywords if kw.arg == "method"), None
        )
        assert method_kw is not None, (
            "PlattCalibrator(...) must pass an explicit method= keyword"
        )
        value = method_kw.value
        assert not isinstance(value, ast.Constant), (
            "PlattCalibrator method= must be config-driven "
            "(cfg.CALIBRATION_METHOD), not a hardcoded literal "
            f"{getattr(value, 'value', value)!r} — the dead-knob bug "
            "(ROADMAP L1873) was exactly a hardcoded method= string."
        )
        # Accept either a direct attribute ref (cfg.CALIBRATION_METHOD) or a
        # local bound to it (e.g. ``cal_method = cfg.CALIBRATION_METHOD``).
        assert isinstance(value, (ast.Attribute, ast.Name)), (
            f"unexpected method= expression type: {type(value).__name__}"
        )


def test_calibration_method_config_is_valid():
    # The wired-through value must be something PlattCalibrator accepts,
    # else every Saturday retrain hard-fails at the calibrator step.
    assert cfg.CALIBRATION_METHOD in ("platt", "isotonic"), (
        f"cfg.CALIBRATION_METHOD={cfg.CALIBRATION_METHOD!r} is not a valid "
        "PlattCalibrator method"
    )
