"""
tests/test_gbm_scorer.py — Unit tests for model/gbm_scorer.py.

Tests validation logic, feature count gate, and predict/save guards.
Uses mocked LightGBM booster to avoid needing actual model files.
"""

import json
import sys
import tempfile
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


# ---------------------------------------------------------------------------
# Tests: source-of-truth load contract
# ---------------------------------------------------------------------------
#
# These tests exercise real LightGBM save/load roundtrips (no mocks) to verify
# the contract that the booster file is the source of truth for feature_names
# and best_iteration. Stubbed/missing/mismatched sidecar metadata must NOT
# corrupt the loaded scorer's schema.
#
# Regression target: 2026-05-07 Sat-SF Backtester runtime_smoke failure where
# a stub `.meta.json` sidecar (force-promoted with feature_names=[]) caused
# `if not scorer.feature_names: raise` to trip even though the booster file
# itself had the correct names embedded. Root cause was load() reading
# feature_names from the sidecar instead of the canonical booster format.


def _train_tiny_scorer():
    """Train a minimal GBMScorer so save/load round-trips have real bytes."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(42)
    n, d = 256, 4
    X_train = rng.standard_normal((n, d)).astype(np.float32)
    y_train = X_train[:, 0] * 0.5 + rng.standard_normal(n).astype(np.float32) * 0.1
    X_val = rng.standard_normal((64, d)).astype(np.float32)
    y_val = X_val[:, 0] * 0.5 + rng.standard_normal(64).astype(np.float32) * 0.1
    scorer = GBMScorer(n_estimators=20, early_stopping_rounds=5)
    scorer.fit(
        X_train, y_train, X_val, y_val,
        feature_names=["alpha_a", "alpha_b", "alpha_c", "alpha_d"],
    )
    return scorer


class TestGBMScorerSourceOfTruth:
    """Booster file is canonical for feature_names + best_iteration."""

    def test_load_recovers_feature_names_from_booster(self):
        """A roundtrip save/load preserves feature_names — sourced from booster, not sidecar."""
        scorer = _train_tiny_scorer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.txt"
            scorer.save(path)
            loaded = GBMScorer.load(path)
        assert loaded.feature_names == ["alpha_a", "alpha_b", "alpha_c", "alpha_d"]

    def test_save_does_not_persist_feature_names_in_sidecar(self):
        """save() must NOT write feature_names to the sidecar — booster owns them.

        Two-source-of-truth was the root cause of the 2026-05-07 force-promote
        incident; the contract is that the sidecar carries diagnostic metadata
        only.
        """
        scorer = _train_tiny_scorer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.txt"
            scorer.save(path)
            sidecar = json.loads(Path(str(path) + ".meta.json").read_text())
        assert "feature_names" not in sidecar
        assert "best_iteration" not in sidecar
        # Diagnostic fields the sidecar still owns:
        assert "val_ic" in sidecar
        assert "params" in sidecar
        assert "gbm_version" in sidecar

    def test_load_without_sidecar_still_works(self):
        """Missing sidecar must NOT break load — booster has everything inference needs.

        Force-promote from a dated archive that lacks a sidecar (current
        meta_trainer.py:1211 archive upload omits sidecars) should produce a
        fully usable scorer.
        """
        scorer = _train_tiny_scorer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.txt"
            scorer.save(path)
            Path(str(path) + ".meta.json").unlink()
            loaded = GBMScorer.load(path)
        assert loaded.feature_names == ["alpha_a", "alpha_b", "alpha_c", "alpha_d"]
        assert loaded._best_iteration > 0
        # Diagnostic field defaults to 0.0 when sidecar is absent:
        assert loaded._val_ic == 0.0
        # Inference path still works end-to-end:
        preds = loaded.predict(np.zeros((3, 4), dtype=np.float32))
        assert preds.shape == (3,)

    def test_load_ignores_stub_empty_feature_names_in_sidecar(self):
        """Stub sidecar with feature_names=[] (the 2026-05-07 incident) must not corrupt load.

        Direct regression test for the failure email. The booster file is the
        source of truth; whatever the sidecar claims about feature_names is
        ignored.
        """
        scorer = _train_tiny_scorer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.txt"
            scorer.save(path)
            # Hand-craft a stub sidecar matching the 2026-05-07 force-promote
            # backfill (operator wrote feature_names=[], best_iteration=null
            # because dated archive didn't include the original sidecar).
            Path(str(path) + ".meta.json").write_text(json.dumps({
                "gbm_version": "v1.0.0",
                "feature_names": [],
                "best_iteration": None,
                "val_ic": 0.008392,
                "params": {},
                "n_estimators": 2000,
            }))
            loaded = GBMScorer.load(path)
        # feature_names recovered from booster despite stub sidecar:
        assert loaded.feature_names == ["alpha_a", "alpha_b", "alpha_c", "alpha_d"]
        # best_iteration is a real int (not None — the stub's null is ignored):
        assert isinstance(loaded._best_iteration, int)
        assert loaded._best_iteration > 0
        # val_ic from sidecar is preserved (still diagnostic-only):
        assert loaded._val_ic == 0.008392
        # And predict works — this is what the backtester runtime_smoke needs:
        preds = loaded.predict(np.zeros((3, 4), dtype=np.float32))
        assert preds.shape == (3,)
