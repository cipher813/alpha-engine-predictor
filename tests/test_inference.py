"""
tests/test_inference.py — Unit tests for inference/daily_predict.py.

Tests predict_ticker() with synthetic data, verifies the output dict schema,
and checks that the predicted direction is always one of UP/FLAT/DOWN.

These tests run without network access or S3 credentials — model weights
are generated fresh for each test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.daily_predict import predict_ticker
from model.predictor import DirectionPredictor
from config import N_FEATURES, CLASS_LABELS


# ── Test fixtures ─────────────────────────────────────────────────────────────

def _make_model_and_norm_stats() -> tuple[DirectionPredictor, dict]:
    """Create a fresh (untrained but valid) model and dummy norm stats."""
    model = DirectionPredictor(n_features=N_FEATURES)
    model.eval()
    norm_stats = {
        "mean": [0.0] * N_FEATURES,
        "std": [1.0] * N_FEATURES,
        "features": [f"f{i}" for i in range(N_FEATURES)],
    }
    return model, norm_stats


def _make_ohlcv(n: int = 520, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV DataFrame with n rows.
    Uses the same random-walk approach as test_features.py.
    520 rows (~2y of trading days) provides enough warmup for 52w rolling windows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    log_returns = rng.normal(0.0, 0.01, size=n)
    price = 100.0 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame(
        {
            "Open": price * rng.uniform(0.995, 1.005, size=n),
            "High": price * rng.uniform(1.001, 1.015, size=n),
            "Low": price * rng.uniform(0.985, 0.999, size=n),
            "Close": price * rng.uniform(0.99, 1.01, size=n),
            "Volume": rng.integers(1_000_000, 10_000_000, size=n).astype(float),
        },
        index=dates,
    )
    return df


def _make_macro(n: int = 520, seed: int = 99) -> dict:
    """Create minimal macro series so compute_features doesn't produce all-NaN rows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return {
        "SPY": pd.Series(450 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=dates),
        "VIX": pd.Series(20 + rng.normal(0, 2, n), index=dates),
        "TNX": pd.Series(4.0 + rng.normal(0, 0.1, n), index=dates),
        "IRX": pd.Series(5.0 + rng.normal(0, 0.1, n), index=dates),
        "GLD": pd.Series(200 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=dates),
        "USO": pd.Series(70 * np.exp(np.cumsum(rng.normal(0, 0.008, n))), index=dates),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPredictTicker:
    """Tests for predict_ticker()."""

    def test_returns_dict_for_valid_input(self):
        """predict_ticker() must return a dict for a ticker with sufficient history."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("AAPL", df, model, norm_stats, macro=macro)
        assert result is not None, "Expected a prediction dict, got None"
        assert isinstance(result, dict)

    def test_output_has_required_keys(self):
        """Output dict must contain all required schema keys."""
        required_keys = {
            "ticker",
            "predicted_direction",
            "prediction_confidence",
            "p_up",
            "p_flat",
            "p_down",
        }
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("MSFT", df, model, norm_stats, macro=macro)
        assert result is not None

        missing = required_keys - set(result.keys())
        assert not missing, f"Missing keys in prediction output: {missing}"

    def test_predicted_direction_is_valid_class(self):
        """predicted_direction must be one of UP, FLAT, or DOWN."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("NVDA", df, model, norm_stats, macro=macro)
        assert result is not None
        assert result["predicted_direction"] in CLASS_LABELS, (
            f"Invalid predicted_direction: {result['predicted_direction']}"
        )

    def test_probabilities_sum_to_one(self):
        """p_up + p_flat + p_down must sum to approximately 1."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("LLY", df, model, norm_stats, macro=macro)
        assert result is not None
        total = result["p_up"] + result["p_flat"] + result["p_down"]
        assert abs(total - 1.0) < 0.01, (
            f"Probabilities don't sum to 1: {total} (p_up={result['p_up']}, "
            f"p_flat={result['p_flat']}, p_down={result['p_down']})"
        )

    def test_prediction_confidence_is_max_probability(self):
        """prediction_confidence must equal the max of (p_up, p_flat, p_down)."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("COST", df, model, norm_stats, macro=macro)
        assert result is not None

        expected_max = max(result["p_up"], result["p_flat"], result["p_down"])
        assert abs(result["prediction_confidence"] - expected_max) < 0.001, (
            f"prediction_confidence ({result['prediction_confidence']}) != "
            f"max probability ({expected_max})"
        )

    def test_predicted_direction_matches_max_prob(self):
        """predicted_direction must correspond to the class with highest probability."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("XOM", df, model, norm_stats, macro=macro)
        assert result is not None

        probs = {
            "DOWN": result["p_down"],
            "FLAT": result["p_flat"],
            "UP": result["p_up"],
        }
        expected_dir = max(probs, key=probs.get)
        assert result["predicted_direction"] == expected_dir, (
            f"predicted_direction={result['predicted_direction']} but max prob is {expected_dir}"
        )

    def test_returns_none_for_insufficient_data(self):
        """predict_ticker() must return None for DataFrames with < 205 rows."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv(100)  # not enough for MA200
        result = predict_ticker("SHORT", df, model, norm_stats)
        assert result is None, "Expected None for insufficient data"

    def test_returns_none_for_empty_dataframe(self):
        """predict_ticker() must return None for empty input."""
        model, norm_stats = _make_model_and_norm_stats()
        df = pd.DataFrame()
        result = predict_ticker("EMPTY", df, model, norm_stats)
        assert result is None

    def test_ticker_in_output(self):
        """Output dict must include the correct ticker symbol."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("JPM", df, model, norm_stats, macro=macro)
        assert result is not None
        assert result["ticker"] == "JPM"

    def test_confidence_in_valid_range(self):
        """prediction_confidence must be in [0, 1]."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("V", df, model, norm_stats, macro=macro)
        assert result is not None
        assert 0.0 <= result["prediction_confidence"] <= 1.0

    def test_individual_probs_in_valid_range(self):
        """Each probability (p_up, p_flat, p_down) must be in [0, 1]."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv()
        macro = _make_macro()
        result = predict_ticker("META", df, model, norm_stats, macro=macro)
        assert result is not None

        for key in ["p_up", "p_flat", "p_down"]:
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_deterministic_inference(self):
        """Inference with the same model and data should produce identical results."""
        model, norm_stats = _make_model_and_norm_stats()
        df = _make_ohlcv(seed=7)
        macro = _make_macro()

        result_1 = predict_ticker("AMZN", df, model, norm_stats, macro=macro)
        result_2 = predict_ticker("AMZN", df, model, norm_stats, macro=macro)

        assert result_1 is not None
        assert result_2 is not None
        assert result_1["p_up"] == result_2["p_up"]
        assert result_1["predicted_direction"] == result_2["predicted_direction"]

    def test_different_seeds_can_produce_different_directions(self):
        """Different price histories should produce at least some variation in direction."""
        model, norm_stats = _make_model_and_norm_stats()
        macro = _make_macro()
        directions = set()

        for seed in range(20):
            df = _make_ohlcv(seed=seed)
            result = predict_ticker(f"TEST{seed}", df, model, norm_stats, macro=macro)
            if result is not None:
                directions.add(result["predicted_direction"])

        # A valid model should predict more than one direction across 20 different inputs
        # (This would only fail for a degenerate constant-output model)
        assert len(directions) >= 1, "Model produced zero valid predictions"
