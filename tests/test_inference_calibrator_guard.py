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


# ── Variance fallback (2026-04-29, ROADMAP P1 calibrator-collapse step #5) ──
#
# When the calibrator IS loaded but its batch outputs collapse to fewer than
# _MIN_UNIQUE_P_UP_BINS unique values (the 2026-04-28 pathology — 27 tickers
# all at p_up=0.5119), the skip is suppressed and the batch falls through to
# the linear heuristic. Test count distinguishes:
#   * collapse case   → fallback engages, values rewritten
#   * small batch     → fallback suppressed (would false-engage on 4-ticker
#                       holiday batches with calibrator plateau)
#   * healthy batch   → fallback clears, calibrator output preserved


def _collapsed_predictions(n: int = 27, p_up_value: float = 0.5119):
    """Reproduce the 2026-04-28 pathology: N tickers all assigned the same
    calibrator-output p_up value. predicted_alpha varies (so the linear
    fallback CAN produce variance), but per-ticker p_up has collapsed."""
    preds = []
    for i in range(n):
        # Vary predicted_alpha symmetrically so the linear fallback's
        # cross-sectional rescale produces real variance — that's the
        # whole point of the fallback engaging.
        alpha = (i - n / 2) * 0.001  # spans roughly [-0.013, +0.013]
        preds.append({
            "ticker": f"T{i:02d}",
            "predicted_alpha": alpha,
            "p_up": p_up_value,
            "p_down": 1.0 - p_up_value,
            "predicted_direction": "UP",
            "prediction_confidence": p_up_value,
        })
    return preds


def test_variance_fallback_engages_on_collapse(caplog):
    """Calibrator loaded but batch p_up collapses to 1 unique bin →
    variance fallback engages, linear heuristic rewrites values, loud
    error log fires."""
    import logging
    from inference.stages.run_inference import _rescale_cross_sectional

    preds = _collapsed_predictions(n=27, p_up_value=0.5119)
    ctx = _StubCtx(calibrator=_StubCalibrator(), predictions=preds)

    with caplog.at_level(logging.ERROR):
        _rescale_cross_sectional(ctx)

    # Loud error log fires — operator must see this.
    assert any(
        "VARIANCE FALLBACK ENGAGED" in rec.message
        for rec in caplog.records
    ), (
        "Expected loud ERROR log when calibrator collapsed to 1 unique bin "
        f"across 27 tickers. Captured logs: {[r.message for r in caplog.records]}"
    )

    # Values must have been rewritten — p_up across the batch should now
    # span a range, not be stuck at 0.5119.
    p_ups = sorted({p["p_up"] for p in ctx.predictions})
    assert len(p_ups) > 1, (
        f"Variance fallback engaged but batch p_up still has {len(p_ups)} "
        f"unique value(s) — linear heuristic didn't fire. p_ups={p_ups}"
    )
    # Linear heuristic uses META_ALPHA_CLIP=0.02 floor; with alphas spanning
    # [-0.013, +0.013] meta_clip=0.02 → p_up ≈ 0.5 + a/0.04. Min/max should
    # span ~[0.175, 0.825].
    assert min(p_ups) < 0.4 and max(p_ups) > 0.6, (
        f"Linear fallback should produce p_up spanning [<0.4, >0.6], got "
        f"[{min(p_ups):.4f}, {max(p_ups):.4f}]"
    )


def test_variance_fallback_suppressed_for_small_batch():
    """Small batches (N < 5) skip the variance gate entirely. A 4-ticker
    holiday batch where calibrator outputs naturally cluster in 1-2
    isotonic plateaus must NOT fire the fallback — that would be a
    false-engage."""
    from inference.stages.run_inference import _rescale_cross_sectional

    # 4 tickers, 1 unique bin — would trip the gate if not for the
    # _MIN_BATCH_SIZE_FOR_VARIANCE_GATE=5 floor.
    preds = _collapsed_predictions(n=4, p_up_value=0.5119)
    ctx = _StubCtx(calibrator=_StubCalibrator(), predictions=preds)
    original = [dict(p) for p in ctx.predictions]

    _rescale_cross_sectional(ctx)

    # No rewrite — calibrator's collapsed output is preserved untouched.
    assert ctx.predictions == original, (
        "Variance gate should be suppressed for small batches (N < 5). "
        "A 4-ticker batch with 1 unique p_up bin must NOT trigger the "
        "linear-fallback rewrite — that would clobber calibrator output "
        "on holiday batches where small N + isotonic plateau is normal."
    )


def test_variance_fallback_clears_for_healthy_batch():
    """Calibrator loaded with healthy variance (>=3 unique bins) →
    fallback clears, existing skip-rescale behavior preserved."""
    from inference.stages.run_inference import _rescale_cross_sectional

    # 27 tickers spanning 5 distinct p_up bins — comfortably above
    # _MIN_UNIQUE_P_UP_BINS=3.
    p_up_values = [0.45, 0.48, 0.51, 0.55, 0.62]
    preds = []
    for i in range(27):
        p_up = p_up_values[i % len(p_up_values)]
        preds.append({
            "ticker": f"T{i:02d}",
            "predicted_alpha": (p_up - 0.5) * 0.04,
            "p_up": p_up,
            "p_down": 1.0 - p_up,
            "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
            "prediction_confidence": max(p_up, 1.0 - p_up),
        })
    ctx = _StubCtx(calibrator=_StubCalibrator(), predictions=preds)
    original = [dict(p) for p in ctx.predictions]

    _rescale_cross_sectional(ctx)

    # Healthy batch — calibrator output preserved, no rewrite.
    assert ctx.predictions == original, (
        "Healthy batch (5 unique p_up bins, N=27) should skip "
        "cross-sectional rescaling and preserve calibrator output. "
        f"Got {len({p['p_up'] for p in ctx.predictions})} unique "
        f"p_up bins post-call."
    )
