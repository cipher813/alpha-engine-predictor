"""
tests/test_gbm_scorer.py — Unit tests for model/gbm_scorer.py.

Tests validation logic, feature count gate, and predict/save guards.
Uses mocked LightGBM booster to avoid needing actual model files.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gbm_scorer import GBMScorer, GBM_VERSION


# ---------------------------------------------------------------------------
# Tests: initialization and defaults
# ---------------------------------------------------------------------------

class TestGBMScorerInit:
    """Tests for GBMScorer initialization and default parameters."""

    def test_default_params(self):
        """Default params should use regression objective with MSE metric."""
        scorer = GBMScorer()
        assert scorer.params["objective"] == "regression"
        assert scorer.params["metric"] == "mse"
        assert scorer._booster is None

    def test_custom_params(self):
        """Custom params should override defaults."""
        custom = {"objective": "regression", "metric": "mae", "num_leaves": 31}
        scorer = GBMScorer(params=custom)
        assert scorer.params["metric"] == "mae"
        assert scorer.params["num_leaves"] == 31

    def test_ranking_objective_overrides(self):
        """ranking_objective=True should switch to lambdarank."""
        scorer = GBMScorer(ranking_objective=True)
        assert scorer.params["objective"] == "lambdarank"
        assert scorer.params["metric"] == "ndcg"

    def test_repr_not_fitted(self):
        """repr should indicate 'not fitted' before training."""
        scorer = GBMScorer()
        assert "not fitted" in repr(scorer)

    def test_repr_fitted(self):
        """repr should show stats when a booster is set."""
        scorer = GBMScorer()
        scorer._booster = MagicMock()
        scorer._best_iteration = 100
        scorer._val_ic = 0.0732
        r = repr(scorer)
        assert "best_iter=100" in r
        assert "0.0732" in r


# ---------------------------------------------------------------------------
# Tests: predict validation
# ---------------------------------------------------------------------------

class TestGBMScorerPredict:
    """Tests for GBMScorer.predict() validation gates."""

    def test_predict_raises_when_not_fitted(self):
        """predict() should raise RuntimeError if model hasn't been fitted."""
        scorer = GBMScorer()
        X = np.random.randn(5, 10).astype(np.float32)

        with pytest.raises(RuntimeError, match="not been fitted"):
            scorer.predict(X)

    def test_predict_raises_on_feature_count_mismatch(self):
        """predict() should raise ValueError if input has wrong feature count."""
        scorer = GBMScorer()
        mock_booster = MagicMock()
        mock_booster.num_feature.return_value = 36
        scorer._booster = mock_booster

        # Input has 10 features, model expects 36
        X = np.random.randn(5, 10).astype(np.float32)

        with pytest.raises(ValueError, match="Feature count mismatch"):
            scorer.predict(X)

    def test_predict_passes_with_correct_feature_count(self):
        """predict() should succeed when feature count matches."""
        scorer = GBMScorer()
        mock_booster = MagicMock()
        mock_booster.num_feature.return_value = 36
        mock_booster.predict.return_value = np.zeros(5)
        scorer._booster = mock_booster
        scorer._best_iteration = 50

        X = np.random.randn(5, 36).astype(np.float32)
        result = scorer.predict(X)

        mock_booster.predict.assert_called_once()
        assert result.shape == (5,)

    def test_predict_uses_best_iteration(self):
        """predict() should pass num_iteration=best_iteration to booster."""
        scorer = GBMScorer()
        mock_booster = MagicMock()
        mock_booster.num_feature.return_value = 10
        mock_booster.predict.return_value = np.ones(3)
        scorer._booster = mock_booster
        scorer._best_iteration = 42

        X = np.random.randn(3, 10).astype(np.float32)
        scorer.predict(X)

        call_kwargs = mock_booster.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 42


# ---------------------------------------------------------------------------
# Tests: feature importance
# ---------------------------------------------------------------------------

class TestGBMScorerFeatureImportance:
    """Tests for feature_importance()."""

    def test_feature_importance_raises_when_not_fitted(self):
        """feature_importance() should raise RuntimeError before fit."""
        scorer = GBMScorer()
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.feature_importance()

    def test_feature_importance_returns_dict(self):
        """feature_importance() should return a dict of name -> score."""
        scorer = GBMScorer()
        mock_booster = MagicMock()
        mock_booster.feature_importance.return_value = np.array([10.0, 20.0, 30.0])
        scorer._booster = mock_booster
        scorer._feature_names = ["f0", "f1", "f2"]

        result = scorer.feature_importance()
        assert isinstance(result, dict)
        assert result["f0"] == 10.0
        assert result["f2"] == 30.0


# ---------------------------------------------------------------------------
# Tests: save guards
# ---------------------------------------------------------------------------

class TestGBMScorerSave:
    """Tests for save() validation."""

    def test_save_raises_when_not_fitted(self):
        """save() should raise RuntimeError if model hasn't been fitted."""
        scorer = GBMScorer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scorer.save("/tmp/test_model.txt")


# ---------------------------------------------------------------------------
# Tests: relevance grade conversion
# ---------------------------------------------------------------------------

class TestRelevanceGrades:
    """Tests for _to_relevance_grades()."""

    def test_output_range(self):
        """Grades should be in [0, n_grades-1]."""
        y = np.array([0.01, -0.02, 0.05, -0.01, 0.0, 0.03, -0.03, 0.02, 0.04, -0.05])
        grades = GBMScorer._to_relevance_grades(y, n_grades=5)

        assert grades.min() >= 0
        assert grades.max() <= 4
        assert grades.dtype == np.int32

    def test_output_length(self):
        """Output length should match input length."""
        y = np.random.randn(100)
        grades = GBMScorer._to_relevance_grades(y)
        assert len(grades) == len(y)

    def test_monotonic_mapping(self):
        """Higher returns should map to higher (or equal) grades."""
        y = np.linspace(-0.1, 0.1, 20)
        grades = GBMScorer._to_relevance_grades(y, n_grades=5)

        # The sorted input should produce non-decreasing grades
        for i in range(1, len(grades)):
            assert grades[i] >= grades[i - 1]


# ---------------------------------------------------------------------------
# Tests: version constant
# ---------------------------------------------------------------------------

def test_gbm_version_format():
    """GBM_VERSION should be a semver-like string."""
    assert isinstance(GBM_VERSION, str)
    parts = GBM_VERSION.lstrip("v").split(".")
    assert len(parts) == 3
    for part in parts:
        int(part)  # should not raise
