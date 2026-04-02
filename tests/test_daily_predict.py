"""
tests/test_daily_predict.py — Unit tests for inference/daily_predict.py.

Tests standalone utility functions: email builder, veto threshold logic.
Does NOT test the full prediction pipeline (which requires S3 + model weights).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tests: _build_predictor_email
# ---------------------------------------------------------------------------

class TestBuildPredictorEmail:
    """Tests for _build_predictor_email() — email formatting helper."""

    def _make_predictions(self, n: int = 5) -> list[dict]:
        """Create synthetic prediction dicts."""
        directions = ["UP", "FLAT", "DOWN"]
        preds = []
        for i in range(n):
            preds.append({
                "ticker": f"TICK{i}",
                "predicted_alpha": 0.05 - i * 0.025,
                "predicted_direction": directions[i % 3],
                "prediction_confidence": 0.7 + i * 0.05,
                "mse_rank": i + 1,
                "model_rank": n - i,
                "watchlist_source": "tracked" if i % 2 == 0 else "buy_candidate",
            })
        return preds

    def _make_metrics(self) -> dict:
        return {
            "model_version": "gbm_v1.0",
            "ic_30d": 0.0812,
            "inference_mode": "mse",
        }

    def test_returns_three_strings(self):
        """_build_predictor_email should return (subject, html, plain)."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions()
        metrics = self._make_metrics()

        result = _build_predictor_email(preds, metrics, "2024-06-01")
        assert isinstance(result, tuple)
        assert len(result) == 3

        subject, html_body, plain_body = result
        assert isinstance(subject, str)
        assert isinstance(html_body, str)
        assert isinstance(plain_body, str)

    def test_subject_contains_date(self):
        """Subject line should include the prediction date."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions()
        metrics = self._make_metrics()

        subject, _, _ = _build_predictor_email(preds, metrics, "2024-06-01")
        assert "2024-06-01" in subject

    def test_subject_contains_direction_counts(self):
        """Subject line should include UP and DOWN counts."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions(6)  # 2 UP, 2 FLAT, 2 DOWN
        metrics = self._make_metrics()

        subject, _, _ = _build_predictor_email(preds, metrics, "2024-06-01")
        assert "UP" in subject
        assert "DOWN" in subject

    def test_html_contains_tickers(self):
        """HTML body should contain all ticker names."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions(3)
        metrics = self._make_metrics()

        _, html, _ = _build_predictor_email(preds, metrics, "2024-06-01")
        for p in preds:
            assert p["ticker"] in html

    def test_plain_contains_tickers(self):
        """Plain text body should contain all ticker names."""
        from inference.daily_predict import _build_predictor_email

        preds = self._make_predictions(3)
        metrics = self._make_metrics()

        _, _, plain = _build_predictor_email(preds, metrics, "2024-06-01")
        for p in preds:
            assert p["ticker"] in plain

    def test_veto_section_appears_for_high_confidence_down(self):
        """Vetoes should appear when a DOWN prediction has negative alpha and low rank."""
        from inference.daily_predict import _build_predictor_email

        # Veto logic: predicted_alpha < 0 AND combined_rank > n_preds / 2
        preds = [
            {
                "ticker": "GOOD",
                "predicted_alpha": 0.05,
                "predicted_direction": "UP",
                "prediction_confidence": 0.70,
                "mse_rank": 1,
                "model_rank": 1,
                "combined_rank": 1.0,
                "watchlist_source": "tracked",
            },
            {
                "ticker": "VETOME",
                "predicted_alpha": -0.05,
                "predicted_direction": "DOWN",
                "prediction_confidence": 0.95,
                "mse_rank": 2,
                "model_rank": 2,
                "combined_rank": 2.0,  # > n_preds/2 (2 > 1)
                "watchlist_source": "tracked",
            },
        ]
        metrics = self._make_metrics()

        subject, html, plain = _build_predictor_email(
            preds, metrics, "2024-06-01", veto_threshold=0.65
        )
        assert "veto" in subject.lower()
        assert "VETOME" in html
        assert "VETO" in plain or "VETOME" in plain

    def test_empty_predictions(self):
        """Should handle empty prediction list gracefully."""
        from inference.daily_predict import _build_predictor_email

        subject, html, plain = _build_predictor_email([], self._make_metrics(), "2024-06-01")
        assert isinstance(subject, str)
        assert "0 UP" in subject
        assert "No predictions" in html or "none" in html.lower() or len(html) > 0

    def test_with_signals_data(self):
        """Research section should appear when signals_data is provided."""
        from inference.daily_predict import _build_predictor_email

        signals = {
            "market_regime": "bullish",
            "universe": [
                {"ticker": "AAPL", "score": 85.5, "conviction": "HIGH",
                 "signal": "BUY", "sector": "Technology"},
            ],
            "buy_candidates": [],
            "sector_ratings": {
                "Technology": {"rating": 80, "modifier": 1.15},
                "Healthcare": {"rating": 65, "modifier": 1.0},
            },
        }
        preds = self._make_predictions(2)
        metrics = self._make_metrics()

        subject, html, plain = _build_predictor_email(
            preds, metrics, "2024-06-01", signals_data=signals
        )
        assert "BULLISH" in subject
        assert "AAPL" in html
        assert "Research" in html or "research" in html.lower()
        assert "AAPL" in plain

    def test_missing_ic_metric(self):
        """Should handle missing ic_30d metric without crashing."""
        from inference.daily_predict import _build_predictor_email

        metrics = {"model_version": "gbm_v1.0"}
        preds = self._make_predictions(1)

        subject, html, plain = _build_predictor_email(preds, metrics, "2024-06-01")
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# Tests: get_veto_threshold
# ---------------------------------------------------------------------------

class TestGetVetoThreshold:
    """Tests for get_veto_threshold() — regime-adaptive veto threshold."""

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_base_threshold_from_s3(self, mock_load):
        """Should use veto_confidence from S3 params when available."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "neutral")
        assert result == 0.65

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_fallback_to_config(self, mock_load):
        """Should fall back to cfg.MIN_CONFIDENCE when S3 params are None."""
        from inference.daily_predict import get_veto_threshold
        import config as cfg

        mock_load.return_value = None
        result = get_veto_threshold("test-bucket", "neutral")
        assert result == cfg.MIN_CONFIDENCE

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_bear_regime_lowers_threshold(self, mock_load):
        """Bear regime should lower the threshold by 0.10."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "bear")
        assert result == pytest.approx(0.55)

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_bullish_regime_raises_threshold(self, mock_load):
        """Bullish regime should raise the threshold by 0.05."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "bullish")
        assert result == pytest.approx(0.70)

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_caution_regime(self, mock_load):
        """Caution regime should lower threshold by 0.05."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "caution")
        assert result == pytest.approx(0.60)

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_threshold_clamped_floor(self, mock_load):
        """Threshold should not go below 0.40 even with bear adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.42}
        result = get_veto_threshold("test-bucket", "bear")
        assert result == 0.40

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_threshold_clamped_ceiling(self, mock_load):
        """Threshold should not exceed 0.90 even with bullish adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.88}
        result = get_veto_threshold("test-bucket", "bullish")
        assert result == 0.90

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_empty_regime_string(self, mock_load):
        """Empty regime string should apply no adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "")
        assert result == 0.65

    @patch("inference.daily_predict._load_predictor_params_from_s3")
    def test_unknown_regime_no_adjustment(self, mock_load):
        """Unknown regime should apply no adjustment."""
        from inference.daily_predict import get_veto_threshold

        mock_load.return_value = {"veto_confidence": 0.65}
        result = get_veto_threshold("test-bucket", "sideways")
        assert result == 0.65
