"""Tests for inference/stages/write_output.py — veto threshold logic."""

from unittest.mock import patch

import pytest

from inference.stages.write_output import get_veto_threshold, _load_predictor_params_from_s3
import inference.stages.write_output as wo


class TestGetVetoThreshold:
    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_neutral_regime(self, _mock):
        result = get_veto_threshold("bucket", "neutral")
        assert result == pytest.approx(0.65)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_bear_regime_lowers(self, _mock):
        result = get_veto_threshold("bucket", "bear")
        assert result == pytest.approx(0.55)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_caution_regime(self, _mock):
        result = get_veto_threshold("bucket", "caution")
        assert result == pytest.approx(0.60)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_bull_regime_raises(self, _mock):
        result = get_veto_threshold("bucket", "bull")
        assert result == pytest.approx(0.70)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_bullish_alias(self, _mock):
        result = get_veto_threshold("bucket", "bullish")
        assert result == pytest.approx(0.70)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_bearish_alias(self, _mock):
        result = get_veto_threshold("bucket", "bearish")
        assert result == pytest.approx(0.55)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_unknown_regime_no_adjustment(self, _mock):
        result = get_veto_threshold("bucket", "unknown")
        assert result == pytest.approx(0.65)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_empty_regime(self, _mock):
        result = get_veto_threshold("bucket", "")
        assert result == pytest.approx(0.65)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.45})
    def test_bear_clamped_at_040(self, _mock):
        result = get_veto_threshold("bucket", "bear")
        assert result == pytest.approx(0.40)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.88})
    def test_bull_clamped_at_090(self, _mock):
        result = get_veto_threshold("bucket", "bull")
        assert result == pytest.approx(0.90)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value=None)
    def test_no_s3_uses_config_default(self, _mock):
        result = get_veto_threshold("bucket", "neutral")
        assert 0.40 <= result <= 0.90


class TestLoadPredictorParams:
    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    def test_no_s3_returns_none(self):
        result = _load_predictor_params_from_s3("nonexistent-bucket")
        assert result is None

    def test_caches_after_first_call(self):
        wo._predictor_params_loaded = True
        wo._predictor_params_cache = {"veto_confidence": 0.70}
        result = _load_predictor_params_from_s3("bucket")
        assert result == {"veto_confidence": 0.70}
