"""Guard: cross-sectional rescaling must not clobber calibrator output.

When a fitted isotonic calibrator is loaded, per-ticker
``calibrator.calibrate_prediction()`` already produces properly-scaled
probabilities. The cross-sectional rescaling tail of
``_run_meta_inference`` must be bypassed or it silently overwrites
calibrator output with a linear heuristic — a silent regression of the
calibration work.

This regression potential was live from the v3 meta-inference path
(shipped 2026-04-01) until the ROADMAP P1 binary UP/DOWN + isotonic
calibration migration (2026-04-15). It went undetected because the
calibrator was never fit in the v3 training path (``meta_trainer.py``)
so the guard was never exercised.
"""

from __future__ import annotations

import pytest


class _StubCalibrator:
    """Minimal stand-in for PlattCalibrator — just the attrs the guard reads."""
    method = "isotonic"
    is_fitted = True
    _ece_after = 0.04


class _StubCtx:
    """Stand-in for PipelineContext with only the attrs the tail touches."""
    def __init__(self, calibrator, predictions):
        self.calibrator = calibrator
        self.predictions = predictions


def _sample_predictions():
    """Calibrator-assigned probabilities that a linear heuristic would NOT produce.

    BBB has alpha=0.0 yet prediction_confidence=0.72 — isotonic learned
    a non-symmetric mapping. The linear heuristic at alpha=0 would give
    confidence≈0.5, so if the guard fails this value gets clobbered.
    """
    return [
        {"ticker": "AAA", "predicted_alpha": 0.008,
         "p_up": 0.81, "p_down": 0.19,
         "predicted_direction": "UP", "prediction_confidence": 0.81},
        {"ticker": "BBB", "predicted_alpha": 0.000,
         "p_up": 0.72, "p_down": 0.28,
         "predicted_direction": "UP", "prediction_confidence": 0.72},
        {"ticker": "CCC", "predicted_alpha": -0.004,
         "p_up": 0.35, "p_down": 0.65,
         "predicted_direction": "DOWN", "prediction_confidence": 0.65},
    ]


def test_rescaling_is_noop_with_calibrator():
    from inference.stages.run_inference import _rescale_cross_sectional

    ctx = _StubCtx(calibrator=_StubCalibrator(), predictions=_sample_predictions())
    original = [dict(p) for p in ctx.predictions]

    _rescale_cross_sectional(ctx)

    assert ctx.predictions == original, (
        "Calibrator was loaded; cross-sectional rescaling must not rewrite predictions."
    )


def test_rescaling_runs_without_calibrator():
    """Fallback path: calibrator absent → linear heuristic fires and reshapes values."""
    from inference.stages.run_inference import _rescale_cross_sectional

    preds = _sample_predictions()
    ctx = _StubCtx(calibrator=None, predictions=preds)
    _rescale_cross_sectional(ctx)

    # Linear heuristic produces symmetric output: alpha=0 → p_up=0.5, confidence=0.5.
    bbb = next(p for p in ctx.predictions if p["ticker"] == "BBB")
    assert abs(bbb["p_up"] - 0.5) < 1e-6, (
        f"Fallback rescaling expected p_up≈0.5 at alpha=0, got {bbb['p_up']}"
    )

    # With max_abs=0.008 and META_ALPHA_CLIP floor=0.02, meta_clip=0.02.
    # AAA at alpha=0.008 → p_up = 0.5 + 0.008/(2*0.02) = 0.70.
    aaa = next(p for p in ctx.predictions if p["ticker"] == "AAA")
    assert abs(aaa["p_up"] - 0.70) < 1e-4, (
        f"Fallback rescaling with floor clip expected p_up≈0.70 for alpha=0.008, got {aaa['p_up']}"
    )
    assert aaa["predicted_direction"] == "UP"


def test_rescaling_handles_empty_predictions():
    from inference.stages.run_inference import _rescale_cross_sectional

    ctx = _StubCtx(calibrator=None, predictions=[])
    _rescale_cross_sectional(ctx)  # Must not raise


def test_rescaling_noop_for_unfitted_calibrator():
    """A calibrator object without is_fitted=True must route to the heuristic path."""
    from inference.stages.run_inference import _rescale_cross_sectional

    class _Unfitted:
        method = "isotonic"
        is_fitted = False

    preds = _sample_predictions()
    ctx = _StubCtx(calibrator=_Unfitted(), predictions=preds)
    _rescale_cross_sectional(ctx)

    # Heuristic should have overwritten — BBB should now be at p_up=0.5, not 0.72.
    bbb = next(p for p in ctx.predictions if p["ticker"] == "BBB")
    assert abs(bbb["p_up"] - 0.5) < 1e-6, (
        "Unfitted calibrator must be treated as absent; heuristic should have rewritten."
    )
