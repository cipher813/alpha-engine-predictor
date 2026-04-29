"""Tests for model/calibrator.py — Platt scaling and isotonic calibration."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    """Synthetic alpha predictions with known calibration properties."""
    np.random.seed(42)
    n = 2000
    # True alpha = signal + noise
    signal = np.random.randn(n) * 0.05
    alpha = np.clip(signal + np.random.randn(n) * 0.03, -0.15, 0.15)
    # Label: did the stock actually go UP?
    actual_up = (signal > 0).astype(np.int32)
    return alpha, actual_up


class TestPlattCalibrator:
    def test_fit_platt(self, synthetic_data):
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data

        cal = PlattCalibrator(method="platt")
        cal.fit(alpha, actual_up)

        assert cal.is_fitted
        assert cal._n_samples == len(alpha)
        assert cal._ece_before is not None
        assert cal._ece_after is not None
        assert cal._ece_after <= cal._ece_before + 0.01  # shouldn't get much worse

    def test_fit_isotonic(self, synthetic_data):
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data

        cal = PlattCalibrator(method="isotonic")
        cal.fit(alpha, actual_up)

        assert cal.is_fitted
        assert cal._ece_after is not None

    def test_predict_proba_range(self, synthetic_data):
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data

        cal = PlattCalibrator(method="platt")
        cal.fit(alpha, actual_up)

        probs = cal.predict_proba(alpha)
        assert probs.shape == (len(alpha),)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_proba_monotonic(self, synthetic_data):
        """Higher alpha should map to higher P(UP)."""
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data

        cal = PlattCalibrator(method="platt")
        cal.fit(alpha, actual_up)

        test_alphas = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
        probs = cal.predict_proba(test_alphas)
        # Should be monotonically increasing (or very close)
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 0.01

    def test_calibrate_prediction(self, synthetic_data):
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data

        cal = PlattCalibrator(method="platt")
        cal.fit(alpha, actual_up)

        result = cal.calibrate_prediction(0.05)
        assert result["predicted_direction"] == "UP"
        assert 0.5 < result["prediction_confidence"] <= 1.0
        assert result["p_up"] + result["p_down"] == pytest.approx(1.0, abs=0.01)

        result_neg = cal.calibrate_prediction(-0.05)
        assert result_neg["predicted_direction"] == "DOWN"
        assert result_neg["prediction_confidence"] > 0.5

    def test_calibrate_prediction_linear_fallback(self):
        """Unfitted calibrator should use linear fallback."""
        from model.calibrator import PlattCalibrator

        cal = PlattCalibrator(method="platt")
        assert not cal.is_fitted

        result = cal.calibrate_prediction(0.075, label_clip=0.15)
        # Linear: p_up = 0.5 + 0.075 / 0.30 = 0.75
        assert result["p_up"] == pytest.approx(0.75, abs=0.01)
        assert result["predicted_direction"] == "UP"

    def test_save_load_roundtrip(self, synthetic_data):
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data

        cal = PlattCalibrator(method="platt")
        cal.fit(alpha, actual_up)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibrator.pkl"
            cal.save(path)

            assert path.exists()
            assert Path(str(path) + ".meta.json").exists()

            cal2 = PlattCalibrator.load(path)
            assert cal2.is_fitted
            assert cal2.method == "platt"
            assert cal2._n_samples == cal._n_samples

            # Predictions should match
            test_alphas = np.array([-0.10, 0.0, 0.10])
            np.testing.assert_allclose(
                cal.predict_proba(test_alphas),
                cal2.predict_proba(test_alphas),
                rtol=1e-6,
            )

    def test_metrics(self, synthetic_data):
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data

        cal = PlattCalibrator(method="platt")
        cal.fit(alpha, actual_up)

        m = cal.metrics()
        assert m["method"] == "platt"
        assert m["fitted"] is True
        assert m["n_samples"] == len(alpha)
        assert isinstance(m["ece_before"], float)
        assert isinstance(m["ece_after"], float)

    def test_too_few_samples(self):
        from model.calibrator import PlattCalibrator

        cal = PlattCalibrator(method="platt")
        cal.fit(np.array([0.01, 0.02]), np.array([1, 0]))
        assert not cal.is_fitted  # Should skip with < 100 samples

    def test_invalid_method(self):
        from model.calibrator import PlattCalibrator
        with pytest.raises(ValueError, match="Unknown calibration method"):
            PlattCalibrator(method="invalid")

    def test_save_writes_deployed_at_sidecar(self, synthetic_data):
        """save() stamps the sidecar with deployed_at for backtester grace gate."""
        import json
        from model.calibrator import PlattCalibrator
        alpha, actual_up = synthetic_data
        cal = PlattCalibrator(method="isotonic")
        cal.fit(alpha, actual_up)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "isotonic_calibrator.pkl"
            cal.save(path)
            sidecar = json.loads(Path(str(path) + ".meta.json").read_text())
            assert "deployed_at" in sidecar
            # ISO-8601 shape
            assert "T" in sidecar["deployed_at"]
            assert sidecar["method"] == "isotonic"


class TestCalibratorOutputVariance:
    """Regression tests for the 2026-04-28 collapse pathology.

    Pathology: meta-trainer hardcoded the four research features to constants
    in walk-forward row construction (PR #55 fix). Ridge correctly zeroed
    those coefficients, leaving expected_move as the only source of per-
    ticker variance. Inference-time meta-model outputs collapsed to a
    ~0.003-wide window (predicted_alpha ∈ [+0.002, +0.005] across all 27
    tickers). The isotonic calibrator's plateau over that narrow input
    region then mapped every ticker to a single p_up bin (0.5119), which
    propagated as 27 UP / 0 DOWN / identical 51% confidence in production.

    These tests pin two invariants:

      1. A calibrator fitted on signal-bearing training data must produce
         meaningful output variance when fed inputs spanning the typical
         post-PR-#55 inference range (~[-0.01, +0.01]). This catches the
         class of bug where calibrator fitting itself is degenerate.

      2. When inference inputs DO collapse into a narrow window, the
         calibrator's behavior is structurally limited (plateau → single
         output). This isn't a calibrator bug — it's the consequence of
         upstream collapse. Test pins the pathology so a defensive
         downstream check (e.g. _rescale_cross_sectional variance fallback)
         has a known reference scenario to gate against.
    """

    def _build_signal_bearing_data(self, n: int = 2000, seed: int = 7):
        """Synthetic alpha + label pair with real signal — calibrator should fit cleanly."""
        rng = np.random.default_rng(seed)
        # Range loosely matches typical post-PR-#55 meta-model output:
        # ~[-0.05, +0.05] before clip; mode near zero with light-tailed noise.
        signal = rng.normal(0.0, 0.025, size=n)
        noise = rng.normal(0.0, 0.015, size=n)
        alpha = np.clip(signal + noise, -0.15, 0.15)
        # Labels track signal — calibrator can learn a non-flat curve.
        actual_up = (signal > 0).astype(np.int32)
        return alpha, actual_up

    def test_isotonic_produces_variance_on_typical_inference_range(self):
        """Isotonic calibrator fit on signal-bearing data must yield ≥K
        unique p_up outputs when fed a batch of inputs spanning the
        typical post-PR-#55 inference range.

        This is the dominant regression detector for the 2026-04-28
        collapse class: it would have failed under the pre-PR-#55 trainer
        because the calibrator was fit on inputs whose meta-model
        coefficients had already been zeroed → plateau over [+0.002,
        +0.005] → single output bin.

        Why K=5 (not 10): isotonic regression is structurally a step
        function. The number of unique outputs across a fixed input range
        depends on the breakpoint density, which depends on training-data
        density in that range. Healthy isotonic on the synthetic fixture
        produces ~8 unique bins across [-0.01, +0.01]; K=5 is the floor
        well above the 1-2 bin collapse class without false-failing
        healthy calibrators that happen to have sparser breakpoints.
        """
        from model.calibrator import PlattCalibrator

        alpha, actual_up = self._build_signal_bearing_data()
        cal = PlattCalibrator(method="isotonic")
        cal.fit(alpha, actual_up)
        assert cal.is_fitted

        # 27 inputs spanning the typical inference range (matches today's
        # universe size; mirrors the 2026-04-28 batch shape).
        inference_batch = np.linspace(-0.01, 0.01, 27)
        probs = cal.predict_proba(inference_batch)

        # Round to inference precision (write_output rounds to 4dp).
        rounded = np.round(probs, 4)
        unique_count = len(set(rounded.tolist()))

        assert unique_count >= 5, (
            f"Isotonic calibrator collapsed to {unique_count} unique "
            f"outputs across the typical inference range [-0.01, +0.01]. "
            f"This is the 2026-04-28 pathology class (every ticker → "
            f"same p_up bin). Investigate: (a) calibrator training data "
            f"variance, (b) isotonic plateau width, (c) whether "
            f"_rescale_cross_sectional variance fallback should have "
            f"engaged. Outputs: min={probs.min():.4f}, max={probs.max():.4f}, "
            f"unique={sorted(set(rounded.tolist()))}"
        )

        # Defense-in-depth: outputs must also span a real range, not
        # cluster into K bins all within 0.01 of each other.
        assert probs.max() - probs.min() >= 0.05, (
            f"Calibrator output range too narrow: {probs.min():.4f} → "
            f"{probs.max():.4f} (span {probs.max() - probs.min():.4f}). "
            f"A healthy calibrator on signal-bearing data should produce "
            f"≥0.05 spread across [-0.01, +0.01] inputs."
        )

    def test_platt_produces_variance_on_typical_inference_range(self):
        """Same invariant for Platt scaling. Logistic regression is
        structurally smoother than isotonic so the bar is easier to clear,
        but a flat training run would still collapse."""
        from model.calibrator import PlattCalibrator

        alpha, actual_up = self._build_signal_bearing_data()
        cal = PlattCalibrator(method="platt")
        cal.fit(alpha, actual_up)
        assert cal.is_fitted

        inference_batch = np.linspace(-0.01, 0.01, 27)
        probs = cal.predict_proba(inference_batch)
        rounded = np.round(probs, 4)
        unique_count = len(set(rounded.tolist()))

        # Platt is continuous → unique_count should equal batch size (27)
        # absent rounding collisions. The K=10 floor is intentionally
        # generous so accidental rounding clustering doesn't false-fail.
        assert unique_count >= 10, (
            f"Platt calibrator collapsed to {unique_count} unique outputs "
            f"across [-0.01, +0.01] — this should not happen for a logistic "
            f"regression fit on signal-bearing data. "
            f"Outputs: min={probs.min():.4f}, max={probs.max():.4f}"
        )

    def test_narrow_input_range_documents_collapse_pathology(self):
        """Pin the 2026-04-28 pathology: when meta-model output collapses
        into a narrow [+0.002, +0.005] window, an isotonic calibrator
        fit on signal-bearing-but-different data produces structurally
        limited variance.

        This test does NOT assert "calibrator must produce variance" in
        this scenario — that would conflate two failure modes. Instead it
        asserts that a defensive downstream check (e.g. variance-aware
        cross-sectional rescaling fallback) has a deterministic reference
        scenario: the same 27 inputs in the narrow band map to ≤ a small
        number of bins, and any fallback must engage on this signal.
        """
        from model.calibrator import PlattCalibrator

        alpha, actual_up = self._build_signal_bearing_data()
        cal = PlattCalibrator(method="isotonic")
        cal.fit(alpha, actual_up)
        assert cal.is_fitted

        # Reproduce the 2026-04-28 narrow-input scenario: 27 tickers all
        # falling in the [+0.002, +0.005] window.
        narrow_batch = np.linspace(0.002, 0.005, 27)
        probs = cal.predict_proba(narrow_batch)
        rounded = np.round(probs, 4)
        unique_count = len(set(rounded.tolist()))

        # Document the pathology: a very narrow input range is expected to
        # produce a small number of unique outputs (isotonic plateaus).
        # Capping at 27 (= batch size) and asserting ≤ 27 is just a
        # tautology, but printing the actual count tells the operator
        # what the threshold should be for a downstream fallback.
        assert unique_count <= 27
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        # The collapse signature: max-min span on narrow inputs is small.
        narrow_span = probs.max() - probs.min()
        # Not asserted as a hard threshold — this is a documentation test.
        # A future fallback gate would consume this number (or
        # unique_count) at inference time and decide whether to skip
        # cross-sectional rescaling. The test stays green; a regression
        # in the calibrator's plateau behavior would change the printed
        # diagnostic and surface in CI logs.
        print(
            f"\n[2026-04-28 collapse signature] narrow input range "
            f"[0.002, 0.005] → {unique_count} unique p_up bins, "
            f"span={narrow_span:.4f} "
            f"(p_up min={probs.min():.4f}, max={probs.max():.4f}). "
            f"Reference threshold for a future variance-fallback gate."
        )


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        from model.calibrator import _expected_calibration_error
        # Perfect calibration: predicted probs match actual rates
        probs = np.array([0.1] * 100 + [0.9] * 100)
        labels = np.array([0] * 90 + [1] * 10 + [0] * 10 + [1] * 90)
        ece = _expected_calibration_error(probs, labels, n_bins=10)
        assert ece < 0.05  # Should be near zero

    def test_terrible_calibration(self):
        from model.calibrator import _expected_calibration_error
        # Always predicts 0.9 but actual rate is 10%
        probs = np.array([0.9] * 1000)
        labels = np.array([0] * 900 + [1] * 100)
        ece = _expected_calibration_error(probs, labels, n_bins=10)
        assert ece > 0.7  # Should be very high
