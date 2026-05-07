"""Tests for RegimeConditionedMeta (audit Phase 4 PR 3).

Three Ridges (one per regime) + an unconditioned fallback Ridge. At
inference, the regime detector routes to one Ridge; under-sampled
regimes fall back to the unconditioned Ridge.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.regime_conditioned_meta import REGIME_LABELS, RegimeConditionedMeta


def _balanced_training_data(n_per_regime: int = 250, seed: int = 0):
    """Build (X, y, regimes) with all three regimes well-represented.

    Each regime has its own coefficient on the first feature so the
    per-regime Ridges can learn distinct weightings (testing the
    specialization the audit Phase 4 design predicts).
    """
    rng = np.random.default_rng(seed)
    n_features = 12  # matches META_FEATURES count
    n_total = n_per_regime * 3
    X = rng.normal(0, 1, (n_total, n_features)).astype(np.float64)
    y = np.zeros(n_total)
    regimes: list[str] = []
    # Bear regime: y = -0.4 * X[:,0] + noise
    bear_idx = slice(0, n_per_regime)
    y[bear_idx] = -0.4 * X[bear_idx, 0] + rng.normal(0, 0.5, n_per_regime)
    regimes.extend(["bear"] * n_per_regime)
    # Neutral regime: y = 0.1 * X[:,0] + noise
    neut_idx = slice(n_per_regime, 2 * n_per_regime)
    y[neut_idx] = 0.1 * X[neut_idx, 0] + rng.normal(0, 0.5, n_per_regime)
    regimes.extend(["neutral"] * n_per_regime)
    # Bull regime: y = 0.4 * X[:,0] + noise
    bull_idx = slice(2 * n_per_regime, 3 * n_per_regime)
    y[bull_idx] = 0.4 * X[bull_idx, 0] + rng.normal(0, 0.5, n_per_regime)
    regimes.extend(["bull"] * n_per_regime)
    return X, y, regimes


# ── Constructor + contract ──────────────────────────────────────────────

class TestConstructorAndContract:

    def test_default_construction(self):
        m = RegimeConditionedMeta()
        assert m.fitted is False
        assert m.regime_labels == ["bear", "neutral", "bull"]

    def test_custom_alpha_and_threshold(self):
        m = RegimeConditionedMeta(alpha=2.5, min_per_regime_rows=50)
        assert m.alpha == 2.5
        assert m.min_per_regime_rows == 50


# ── Fit ─────────────────────────────────────────────────────────────────

class TestFit:

    def test_fit_with_all_three_regimes(self):
        X, y, regimes = _balanced_training_data()
        m = RegimeConditionedMeta().fit(X, y, regimes)
        assert m.fitted is True
        # All three regimes should have dedicated Ridges (each has 250 rows
        # which exceeds min_per_regime_rows=200).
        assert "bear" in m._ridges
        assert "neutral" in m._ridges
        assert "bull" in m._ridges
        assert m._fallback_ridge is not None

    def test_under_sampled_regime_uses_fallback(self):
        X, y, regimes = _balanced_training_data(n_per_regime=250)
        # Override bear regime to have only 50 rows — below default 200.
        regimes = ["neutral"] * 200 + regimes[200:]
        m = RegimeConditionedMeta().fit(X, y, regimes)
        # bear regime is now under-sampled (0 rows after override) so its
        # ridge isn't fit; bear lookups route to fallback.
        assert "bear" not in m._ridges
        # Fallback is always fit.
        assert m._fallback_ridge is not None

    def test_fit_rejects_non_2d_X(self):
        with pytest.raises(ValueError, match="2-D"):
            RegimeConditionedMeta().fit(np.array([1.0, 2.0]), np.array([0.1]), ["bear"])

    def test_fit_rejects_length_mismatch(self):
        X, y, regimes = _balanced_training_data(n_per_regime=100)
        with pytest.raises(ValueError, match="must all agree"):
            RegimeConditionedMeta().fit(X, y[:50], regimes)


# ── Predict routing ─────────────────────────────────────────────────────

class TestPredictRouting:

    def test_predict_for_regime_routes_to_correct_ridge(self):
        X, y, regimes = _balanced_training_data()
        m = RegimeConditionedMeta().fit(X, y, regimes)
        # Each regime's Ridge should produce different predictions on
        # the same input — they were fit on data with different sign.
        sample = X[:5]
        bear_preds = m.predict_for_regime(sample, "bear")
        bull_preds = m.predict_for_regime(sample, "bull")
        # Bear and bull Ridges had opposite-sign relationships with X[:,0],
        # so their predictions should be meaningfully different.
        assert not np.allclose(bear_preds, bull_preds, atol=0.05)

    def test_unrecognized_regime_falls_back(self):
        X, y, regimes = _balanced_training_data()
        m = RegimeConditionedMeta().fit(X, y, regimes)
        # An unrecognized label (e.g., "transition") should route to fallback.
        sample = X[:3]
        fallback_preds = m.predict_unconditioned(sample)
        unrecognized_preds = m.predict_for_regime(sample, "transition")
        np.testing.assert_array_almost_equal(fallback_preds, unrecognized_preds)

    def test_under_sampled_regime_falls_back(self):
        X, y, regimes = _balanced_training_data(n_per_regime=250)
        # Force bear regime to have 0 rows.
        regimes = ["neutral" if r == "bear" else r for r in regimes]
        m = RegimeConditionedMeta().fit(X, y, regimes)
        # bear has no dedicated Ridge → routes to fallback.
        sample = X[:3]
        fallback_preds = m.predict_unconditioned(sample)
        bear_preds = m.predict_for_regime(sample, "bear")
        np.testing.assert_array_almost_equal(fallback_preds, bear_preds)

    def test_predict_unfitted_raises(self):
        m = RegimeConditionedMeta()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_for_regime(np.zeros((3, 12)), "bull")


# ── predict_single_for_regime ──────────────────────────────────────────

class TestPredictSingle:

    def test_predict_single_for_regime_returns_scalar(self):
        X, y, regimes = _balanced_training_data()
        m = RegimeConditionedMeta().fit(X, y, regimes)
        # Build a feature dict
        feats = {f"f{i}": float(X[0, i]) for i in range(X.shape[1])}
        # Match the Ridge's expected feature names; if MetaModel uses
        # default f0/f1/... names this should work.
        result = m.predict_single_for_regime(feats, "bull")
        assert isinstance(result, float)

    def test_predict_single_unfitted_returns_zero(self):
        m = RegimeConditionedMeta()
        assert m.predict_single_for_regime({}, "bull") == 0.0


# ── predict_unconditioned ──────────────────────────────────────────────

class TestPredictUnconditioned:

    def test_predict_unconditioned_returns_fallback_predictions(self):
        X, y, regimes = _balanced_training_data()
        m = RegimeConditionedMeta().fit(X, y, regimes)
        preds = m.predict_unconditioned(X[:5])
        # Same Ridge as the unconditioned fallback — direct call.
        fallback_preds = m._fallback_ridge.predict(X[:5])
        np.testing.assert_array_almost_equal(preds, fallback_preds)

    def test_predict_unconditioned_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            RegimeConditionedMeta().predict_unconditioned(np.zeros((3, 12)))


# ── Metrics + save/load ─────────────────────────────────────────────────

class TestMetrics:

    def test_metrics_unfitted(self):
        m = RegimeConditionedMeta().metrics()
        assert m["fitted"] is False
        assert m["regimes_with_dedicated_ridge"] == []
        # All three regimes appear in the fallback list when none have ridges.
        assert set(m["regimes_using_fallback"]) == set(REGIME_LABELS)

    def test_metrics_fitted_complete(self):
        X, y, regimes = _balanced_training_data()
        m = RegimeConditionedMeta().fit(X, y, regimes)
        meta = m.metrics()
        assert meta["type"] == "regime_conditioned_meta_v1"
        assert meta["fitted"] is True
        assert set(meta["regimes_with_dedicated_ridge"]) == set(REGIME_LABELS)
        assert meta["regimes_using_fallback"] == []
        assert "fallback_val_ic" in meta
        for regime in REGIME_LABELS:
            assert regime in meta["n_samples_per_regime"]

    def test_metrics_distinguishes_dedicated_vs_fallback(self):
        X, y, regimes = _balanced_training_data(n_per_regime=250)
        # Force bear under-sampled.
        regimes = ["neutral" if r == "bear" else r for r in regimes]
        m = RegimeConditionedMeta().fit(X, y, regimes)
        meta = m.metrics()
        assert "bear" not in meta["regimes_with_dedicated_ridge"]
        assert "bear" in meta["regimes_using_fallback"]


class TestSaveLoad:

    def test_save_load_roundtrip(self, tmp_path: Path):
        X, y, regimes = _balanced_training_data()
        m = RegimeConditionedMeta().fit(X, y, regimes)
        path = tmp_path / "regime_conditioned_meta.pkl"
        m.save(path)

        sidecar = tmp_path / "regime_conditioned_meta.pkl.meta.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["type"] == "regime_conditioned_meta_v1"
        assert meta["fitted"] is True
        assert "deployed_at" in meta

        loaded = RegimeConditionedMeta.load(path)
        assert loaded.fitted is True
        # Predictions match original.
        sample = X[:10]
        for regime in REGIME_LABELS:
            np.testing.assert_array_almost_equal(
                m.predict_for_regime(sample, regime),
                loaded.predict_for_regime(sample, regime),
            )

    def test_save_unfitted_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            RegimeConditionedMeta().save(tmp_path / "should_not_exist.pkl")

    def test_load_preserves_regime_routing_state(self, tmp_path: Path):
        X, y, regimes = _balanced_training_data(n_per_regime=250)
        regimes = ["neutral" if r == "bear" else r for r in regimes]  # bear under-sampled
        m = RegimeConditionedMeta().fit(X, y, regimes)
        path = tmp_path / "regime_conditioned_meta.pkl"
        m.save(path)
        loaded = RegimeConditionedMeta.load(path)
        # Bear should still route to fallback after load.
        sample = X[:3]
        np.testing.assert_array_almost_equal(
            loaded.predict_for_regime(sample, "bear"),
            loaded.predict_unconditioned(sample),
        )
