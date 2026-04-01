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
