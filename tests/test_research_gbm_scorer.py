"""Tests for the ResearchGBMScorer (audit Phase 3 PR 1).

Covers fit / predict / save / load / metrics + interface contract that
meta_trainer + inference will depend on in subsequent PRs. No training
or inference integration in this PR — module-only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.research_gbm import RESEARCH_GBM_FEATURES, ResearchGBMScorer


def _synthetic_training_data(n: int = 800, seed: int = 0):
    """Build (X, y) with a real signal so the LGB has something to learn.

    Label is binary beat-SPY-at-10d with the actual signal coming from
    the first three features (research-derived) and small contributions
    from macro context.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, len(RESEARCH_GBM_FEATURES))).astype(np.float32)
    # Real signal: composite + conviction matter most; macro adds noise
    score_signal = 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * X[:, 2]
    macro_noise = 0.05 * X[:, 3:].sum(axis=1)
    raw = score_signal + macro_noise + rng.normal(0, 0.5, n)
    # Convert to binary beat-SPY label via threshold at zero
    y = (raw > 0).astype(np.float32)
    return X, y


class TestConstructorAndContract:

    def test_default_construction(self):
        scorer = ResearchGBMScorer()
        assert scorer.fitted is False
        assert scorer.feature_names == RESEARCH_GBM_FEATURES

    def test_feature_names_are_canonical_9(self):
        # Audit §8 Phase 3 deliverable: research_composite_score +
        # research_conviction + sector_macro_modifier + 6 macro features
        # = 9 features total.
        assert len(RESEARCH_GBM_FEATURES) == 9
        assert "research_composite_score" in RESEARCH_GBM_FEATURES
        assert "research_conviction" in RESEARCH_GBM_FEATURES
        assert "sector_macro_modifier" in RESEARCH_GBM_FEATURES
        macro_count = sum(1 for f in RESEARCH_GBM_FEATURES if f.startswith("macro_"))
        assert macro_count == 6


# ── Fit ─────────────────────────────────────────────────────────────────

class TestFit:

    def test_fit_happy_path(self):
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer(n_estimators=50)
        scorer.fit(X, y)
        assert scorer.fitted is True
        assert scorer._n_samples == len(X)
        assert scorer._best_iteration > 0

    def test_fit_with_validation_set_records_ic(self):
        X, y = _synthetic_training_data(n=600)
        # Split 70/30
        split = int(len(X) * 0.7)
        scorer = ResearchGBMScorer(n_estimators=200, early_stopping_rounds=20)
        scorer.fit(X[:split], y[:split], X[split:], y[split:])
        # Synthetic data has real signal; val_ic should be meaningfully
        # above zero (not asserting a tight bound — LightGBM stochasticity).
        assert scorer._val_ic > 0.05

    def test_fit_rejects_non_2d_input(self):
        scorer = ResearchGBMScorer()
        with pytest.raises(ValueError, match="2-D"):
            scorer.fit(np.array([1.0, 2.0, 3.0]), np.array([0]))

    def test_fit_rejects_x_y_length_mismatch(self):
        scorer = ResearchGBMScorer()
        X = np.zeros((10, 9))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="rows.*length.*disagree"):
            scorer.fit(X, y)

    def test_fit_with_custom_feature_names(self):
        X, y = _synthetic_training_data()
        custom_names = [f"custom_f{i}" for i in range(X.shape[1])]
        scorer = ResearchGBMScorer(n_estimators=20)
        scorer.fit(X, y, feature_names=custom_names)
        assert scorer.feature_names == custom_names

    def test_fit_rejects_feature_names_length_mismatch(self):
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer()
        with pytest.raises(ValueError, match="length.*does not match"):
            scorer.fit(X, y, feature_names=["only_one"])


# ── Predict ─────────────────────────────────────────────────────────────

class TestPredict:

    def test_predict_returns_in_unit_interval(self):
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer(n_estimators=50).fit(X, y)
        preds = scorer.predict(X[:50])
        assert preds.shape == (50,)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_predict_unfitted_raises(self):
        scorer = ResearchGBMScorer()
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.predict(np.zeros((5, 9)))

    def test_predict_rejects_wrong_feature_count(self):
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer(n_estimators=20).fit(X, y)
        with pytest.raises(ValueError, match="model expects"):
            scorer.predict(np.zeros((5, 3)))  # only 3 features, not 9

    def test_predict_from_dict_unfitted_returns_neutral(self):
        scorer = ResearchGBMScorer()
        result = scorer.predict_from_dict({})
        assert result == 0.5  # DEFAULT_PRIOR

    def test_predict_from_dict_fitted(self):
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer(n_estimators=30).fit(X, y)
        # Build a dict with all 9 expected keys
        feats = {name: 0.5 for name in RESEARCH_GBM_FEATURES}
        result = scorer.predict_from_dict(feats)
        assert 0.0 <= result <= 1.0

    def test_predict_from_dict_missing_keys_imputed_to_zero(self):
        # Missing keys default to 0.0; predict should not crash.
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer(n_estimators=30).fit(X, y)
        result = scorer.predict_from_dict({"research_composite_score": 0.7})
        assert 0.0 <= result <= 1.0


# ── Save / Load roundtrip ──────────────────────────────────────────────

class TestSaveLoad:

    def test_save_load_roundtrip(self, tmp_path: Path):
        X, y = _synthetic_training_data(n=300)
        scorer = ResearchGBMScorer(n_estimators=30).fit(X, y)
        path = tmp_path / "research_gbm.pkl"
        scorer.save(path)

        # Sidecar exists with the expected schema.
        sidecar = tmp_path / "research_gbm.pkl.meta.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["type"] == "research_gbm_v1"
        assert meta["fitted"] is True
        assert meta["n_features"] == 9
        assert meta["feature_names"] == RESEARCH_GBM_FEATURES
        assert "deployed_at" in meta

        # Load and verify predictions match.
        loaded = ResearchGBMScorer.load(path)
        assert loaded.fitted is True
        assert loaded.feature_names == RESEARCH_GBM_FEATURES
        original_preds = scorer.predict(X[:20])
        loaded_preds = loaded.predict(X[:20])
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_unfitted_raises(self, tmp_path: Path):
        scorer = ResearchGBMScorer()
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            scorer.save(tmp_path / "should_not_exist.pkl")

    def test_load_tolerates_missing_sidecar(self, tmp_path: Path):
        # If the sidecar is gone (e.g., legacy upload), load should
        # fall back to booster-internal feature names without crashing.
        # Mirrors the gbm_scorer.load tolerance pattern.
        X, y = _synthetic_training_data(n=300)
        scorer = ResearchGBMScorer(n_estimators=30).fit(X, y)
        path = tmp_path / "research_gbm.pkl"
        scorer.save(path)
        sidecar = tmp_path / "research_gbm.pkl.meta.json"
        sidecar.unlink()  # simulate missing sidecar

        loaded = ResearchGBMScorer.load(path)
        assert loaded.fitted is True
        assert len(loaded.feature_names) == 9

    def test_load_tolerates_corrupt_sidecar(self, tmp_path: Path):
        X, y = _synthetic_training_data(n=300)
        scorer = ResearchGBMScorer(n_estimators=30).fit(X, y)
        path = tmp_path / "research_gbm.pkl"
        scorer.save(path)
        sidecar = tmp_path / "research_gbm.pkl.meta.json"
        sidecar.write_text("{not valid json")  # corrupt

        loaded = ResearchGBMScorer.load(path)
        assert loaded.fitted is True
        assert len(loaded.feature_names) == 9


# ── Metrics + importance ───────────────────────────────────────────────

class TestMetrics:

    def test_metrics_unfitted(self):
        scorer = ResearchGBMScorer()
        m = scorer.metrics()
        assert m["fitted"] is False
        assert m["n_samples"] == 0
        assert m["best_iteration"] is None

    def test_metrics_fitted_complete(self):
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer(n_estimators=50).fit(X, y)
        m = scorer.metrics()
        assert m["type"] == "research_gbm_v1"
        assert m["fitted"] is True
        assert m["n_samples"] == len(X)
        assert m["n_features"] == 9
        assert m["feature_names"] == RESEARCH_GBM_FEATURES
        assert m["best_iteration"] is not None
        assert isinstance(m["val_ic"], float)

    def test_feature_importance_sums_meaningfully(self):
        X, y = _synthetic_training_data()
        scorer = ResearchGBMScorer(n_estimators=50).fit(X, y)
        imp = scorer.feature_importance("gain")
        # All 9 features should be in the importance dict.
        assert set(imp.keys()) == set(RESEARCH_GBM_FEATURES)
        # At least one feature has nonzero importance (synthetic signal
        # makes research_composite_score the strongest by construction).
        assert sum(imp.values()) > 0

    def test_feature_importance_unfitted_empty(self):
        scorer = ResearchGBMScorer()
        assert scorer.feature_importance() == {}
