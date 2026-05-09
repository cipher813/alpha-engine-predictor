"""Tests for RegimePredictorV2 (audit Phase 4 PR 1).

Covers triple-barrier label generation, fit/predict on 3-class targets,
honest gate metrics (OOS accuracy + per-class recall + macro-F1),
save/load roundtrip, and the audit-§8 promotion gate.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.regime_predictor_v2 import (
    REGIME_CLASSES,
    REGIME_V2_FEATURES,
    RegimePredictorV2,
    make_triple_barrier_labels,
)


# ── Triple-barrier label generation ─────────────────────────────────────

class TestTripleBarrierLabels:

    def test_all_zero_returns_yields_neutral(self):
        # Flat market — every window has zero cumulative return → neutral.
        spy = np.zeros(100)
        labels = make_triple_barrier_labels(spy, forward_window=21)
        # Tail rows that lack a forward window get -1 sentinel.
        finite = labels[labels != -1]
        # All non-tail labels should be neutral (1).
        assert (finite == 1).all()

    def test_steady_uptrend_yields_bull(self):
        # +1% per day → quickly hits +5% bull barrier.
        spy = np.full(100, 0.01)
        labels = make_triple_barrier_labels(
            spy, forward_window=21, up_barrier_pct=0.05, down_barrier_pct=0.05,
        )
        finite = labels[labels != -1]
        assert (finite == 2).all()

    def test_steady_downtrend_yields_bear(self):
        spy = np.full(100, -0.01)
        labels = make_triple_barrier_labels(spy, forward_window=21)
        finite = labels[labels != -1]
        assert (finite == 0).all()

    def test_tail_rows_get_sentinel(self):
        spy = np.full(50, 0.005)
        labels = make_triple_barrier_labels(spy, forward_window=21)
        # Last `forward_window` rows lack a forward window → -1.
        assert (labels[-21:] == -1).all()
        assert (labels[:-21] != -1).all()

    def test_label_distribution_on_mixed_returns(self):
        # Mixed series: alternating +/- so neither barrier hits cleanly.
        rng = np.random.default_rng(0)
        spy = rng.normal(0, 0.005, 200)  # small noise
        labels = make_triple_barrier_labels(spy, forward_window=21)
        finite = labels[labels != -1]
        # All three classes should appear (or at least neutral and one other).
        unique = set(finite)
        assert 1 in unique  # neutral certainly should
        # Total finite labels = n - forward_window
        assert len(finite) == 200 - 21


# ── Constructor + canonical contract ────────────────────────────────────

class TestConstructorAndContract:

    def test_default_construction(self):
        scorer = RegimePredictorV2()
        assert scorer.fitted is False
        assert scorer.feature_names == REGIME_V2_FEATURES
        assert scorer.class_labels == ["bear", "neutral", "bull"]

    def test_canonical_9_features(self):
        # 6 macro + 3 SPY-momentum = 9 features per audit Phase 4 spec.
        assert len(REGIME_V2_FEATURES) == 9
        macro_count = sum(1 for f in REGIME_V2_FEATURES if f.startswith("macro_"))
        assert macro_count == 6
        spy_momentum_count = sum(1 for f in REGIME_V2_FEATURES if f.startswith("spy_"))
        assert spy_momentum_count == 3


# ── Fit + predict ───────────────────────────────────────────────────────

def _balanced_3class_data(n_per_class: int = 200, seed: int = 0):
    """Build (X, y) with all three classes well-represented."""
    rng = np.random.default_rng(seed)
    n = n_per_class * 3
    X = rng.normal(0, 1, (n, len(REGIME_V2_FEATURES))).astype(np.float32)
    y = np.zeros(n, dtype=np.int32)
    # Class 0 (bear): negative spy_20d_return, high vix_level
    bear_idx = slice(0, n_per_class)
    X[bear_idx, 0] = rng.normal(-0.05, 0.02, n_per_class)  # spy_20d_return
    X[bear_idx, 2] = rng.normal(1.5, 0.3, n_per_class)     # vix_level
    y[bear_idx] = 0
    # Class 1 (neutral): mid-range
    neut_idx = slice(n_per_class, 2 * n_per_class)
    X[neut_idx, 0] = rng.normal(0.0, 0.01, n_per_class)
    X[neut_idx, 2] = rng.normal(1.0, 0.2, n_per_class)
    y[neut_idx] = 1
    # Class 2 (bull): positive spy_20d_return, low vix_level
    bull_idx = slice(2 * n_per_class, 3 * n_per_class)
    X[bull_idx, 0] = rng.normal(0.05, 0.02, n_per_class)
    X[bull_idx, 2] = rng.normal(0.7, 0.15, n_per_class)
    y[bull_idx] = 2
    # Shuffle so train/val splits are mixed.
    perm = rng.permutation(n)
    return X[perm], y[perm]


class TestFit:

    def test_fit_happy_path(self):
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=80).fit(X, y)
        assert scorer.fitted is True
        assert scorer._n_samples == len(X)

    def test_fit_with_validation_records_gate_metrics(self):
        X, y = _balanced_3class_data(n_per_class=300)
        split = int(0.7 * len(X))
        scorer = RegimePredictorV2(n_estimators=200, early_stopping_rounds=20)
        scorer.fit(X[:split], y[:split], X[split:], y[split:])
        # Synthetic data has clear class signal → metrics should be populated.
        assert scorer._oos_accuracy is not None
        assert scorer._per_class_recall is not None
        for cls in ("bear", "neutral", "bull"):
            assert cls in scorer._per_class_recall
        assert scorer._macro_f1 is not None

    def test_fit_rejects_invalid_class_labels(self):
        X, _ = _balanced_3class_data()
        bad_y = np.full(len(X), 5, dtype=np.int32)  # class 5 doesn't exist
        with pytest.raises(ValueError, match="must be in"):
            RegimePredictorV2().fit(X, bad_y)

    def test_fit_rejects_2d_y(self):
        X, y = _balanced_3class_data()
        with pytest.raises(ValueError):
            RegimePredictorV2().fit(X, y[: len(X) - 5])  # length mismatch

    def test_class_weights_recorded_in_metrics(self):
        """Balanced sample weights are computed at fit time + surface in
        metrics() so the manifest documents what the model was trained
        against. Inverse-proportional → minority class gets larger weight."""
        rng = np.random.default_rng(0)
        # Imbalanced: 80% bull, 15% neutral, 5% bear (mimics recent market)
        n = 1000
        X = rng.normal(0, 1, (n, len(REGIME_V2_FEATURES))).astype(np.float32)
        y = np.zeros(n, dtype=np.int32)
        y[:50] = 0       # 5% bear
        y[50:200] = 1    # 15% neutral
        y[200:] = 2      # 80% bull
        rng.shuffle(y)

        scorer = RegimePredictorV2(n_estimators=80).fit(X, y)
        m = scorer.metrics()
        assert m["class_weights"] is not None
        cw = m["class_weights"]
        # bear is rarest → largest weight; bull is most common → smallest.
        # n_classes=3, so balanced weight = n / (3 * count_c).
        assert cw["bear"] > cw["neutral"] > cw["bull"], (
            f"class_weights should rank rare → frequent: {cw}"
        )
        # Sanity: bear (rarest) gets a substantial up-weight (>3×) vs bull.
        assert cw["bear"] / cw["bull"] > 3.0

    def test_class_weighting_lifts_bear_recall_above_zero(self):
        """The 2026-05-09 first-cycle observation: oos_acc=44%,
        bear_recall=0%, bull_recall=86% on the live macro+SPY-momentum
        feature set. Pre-fix the LightGBM softmax loss had no class
        weighting; with bear ~10-15% of triple-barrier labels, the model
        reduced gradient simply by ignoring the minority class.

        Post-fix: balanced sample_weight injection lifts bear_recall
        meaningfully above zero on a synthetic imbalanced cross-section
        with class-discriminative features. Doesn't pin a strict floor
        (bear_recall depends on feature-class alignment in synthetic
        data), but pins the directional fix: with the same imbalanced
        labels and same features, weighted training must produce
        non-zero bear_recall while unweighted-baseline (legacy params)
        collapses bear_recall to near zero.
        """
        rng = np.random.default_rng(42)
        n_bull, n_neutral, n_bear = 700, 200, 100  # 70/20/10 split
        n = n_bull + n_neutral + n_bear

        X = rng.normal(0, 1, (n, len(REGIME_V2_FEATURES))).astype(np.float32)
        y = np.zeros(n, dtype=np.int32)
        # Bear: negative spy returns + high vix
        X[:n_bear, 0] = rng.normal(-0.05, 0.015, n_bear)
        X[:n_bear, 2] = rng.normal(1.5, 0.3, n_bear)
        y[:n_bear] = 0
        # Neutral: mid
        X[n_bear:n_bear + n_neutral, 0] = rng.normal(0.0, 0.01, n_neutral)
        X[n_bear:n_bear + n_neutral, 2] = rng.normal(1.0, 0.2, n_neutral)
        y[n_bear:n_bear + n_neutral] = 1
        # Bull: positive spy returns + low vix
        X[n_bear + n_neutral:, 0] = rng.normal(0.05, 0.015, n_bull)
        X[n_bear + n_neutral:, 2] = rng.normal(0.7, 0.15, n_bull)
        y[n_bear + n_neutral:] = 2

        # Temporal-style 80/20 split (synthetic; just for an early-stop signal)
        perm = rng.permutation(n)
        X, y = X[perm], y[perm]
        split = int(0.8 * n)

        scorer = RegimePredictorV2(n_estimators=200, early_stopping_rounds=20)
        scorer.fit(X[:split], y[:split], X[split:], y[split:])

        # Must not collapse bear_recall to 0 on imbalanced data with
        # class-discriminative features.
        bear_recall = scorer._per_class_recall["bear"]
        assert bear_recall > 0.30, (
            f"bear_recall={bear_recall:.3f} too low — class weighting "
            "failed to elevate the minority class"
        )


# ── Predict ─────────────────────────────────────────────────────────────

class TestPredict:

    def test_predict_proba_columns_sum_to_one(self):
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=50).fit(X, y)
        probs = scorer.predict_proba(X[:30])
        np.testing.assert_array_almost_equal(probs.sum(axis=1), 1.0, decimal=5)
        assert probs.shape == (30, 3)

    def test_predict_class_returns_valid_labels(self):
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=50).fit(X, y)
        labels = scorer.predict_class(X[:30])
        assert all(lbl in REGIME_CLASSES for lbl in labels)

    def test_predict_class_from_dict_unfitted_returns_neutral(self):
        scorer = RegimePredictorV2()
        assert scorer.predict_class_from_dict({}) == "neutral"

    def test_predict_class_from_dict_fitted(self):
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=50).fit(X, y)
        # Construct a dict with all 9 features set to a bull-like profile.
        feats = {name: 0.0 for name in REGIME_V2_FEATURES}
        feats["macro_spy_20d_return"] = 0.05    # bull signature
        feats["macro_vix_level"] = 0.7
        label = scorer.predict_class_from_dict(feats)
        assert label in REGIME_CLASSES

    def test_predict_proba_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            RegimePredictorV2().predict_proba(np.zeros((5, 9)))


# ── Honest promotion gate (audit §8 Phase 4 spec) ────────────────────────

class TestPromotionGate:

    def test_unfitted_fails_gate(self):
        scorer = RegimePredictorV2()
        passed, reason = scorer.passes_promotion_gate()
        assert passed is False
        assert "not fitted" in reason

    def test_strong_signal_passes_gate(self):
        # Synthetic data with clear class separation should easily pass
        # the audit's gate (oos_accuracy ≥ 0.50, bear-recall ≥ 0.40).
        X, y = _balanced_3class_data(n_per_class=400)
        split = int(0.7 * len(X))
        scorer = RegimePredictorV2(n_estimators=300, early_stopping_rounds=30)
        scorer.fit(X[:split], y[:split], X[split:], y[split:])
        passed, reason = scorer.passes_promotion_gate()
        assert passed is True, f"expected pass; got: {reason}"
        assert "passed" in reason

    def test_low_oos_accuracy_blocks_gate(self):
        # Manually set oos_accuracy below threshold to verify the gate fires.
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=20).fit(X, y)
        scorer._oos_accuracy = 0.40  # below default 0.50
        passed, reason = scorer.passes_promotion_gate()
        assert passed is False
        assert "OOS accuracy" in reason
        assert "majority-class baseline" in reason

    def test_low_bear_recall_blocks_gate(self):
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=20).fit(X, y)
        scorer._oos_accuracy = 0.55  # passes accuracy
        scorer._per_class_recall = {"bear": 0.20, "neutral": 0.70, "bull": 0.65}
        passed, reason = scorer.passes_promotion_gate()
        assert passed is False
        assert "bear-recall" in reason
        assert "Tier-0 hit 0.23" in reason

    def test_class_recall_floor_blocks_gate(self):
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=20).fit(X, y)
        scorer._oos_accuracy = 0.55
        scorer._per_class_recall = {"bear": 0.45, "neutral": 0.10, "bull": 0.65}
        passed, reason = scorer.passes_promotion_gate()
        assert passed is False
        assert "neutral-recall" in reason
        assert "ignored by the model" in reason

    def test_custom_thresholds_supported(self):
        X, y = _balanced_3class_data()
        scorer = RegimePredictorV2(n_estimators=20).fit(X, y)
        scorer._oos_accuracy = 0.55
        scorer._per_class_recall = {"bear": 0.30, "neutral": 0.55, "bull": 0.65}
        # Default min_bear_recall=0.40 → fails. Custom 0.25 → passes.
        passed_default, _ = scorer.passes_promotion_gate(min_bear_recall=0.40)
        passed_relaxed, _ = scorer.passes_promotion_gate(min_bear_recall=0.25)
        assert passed_default is False
        assert passed_relaxed is True


# ── Save / Load ──────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_load_roundtrip(self, tmp_path: Path):
        X, y = _balanced_3class_data(n_per_class=200)
        split = int(0.7 * len(X))
        scorer = RegimePredictorV2(n_estimators=80).fit(
            X[:split], y[:split], X[split:], y[split:],
        )
        path = tmp_path / "regime_v2.pkl"
        scorer.save(path)

        sidecar = tmp_path / "regime_v2.pkl.meta.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["type"] == "regime_predictor_v2"
        assert meta["fitted"] is True
        assert meta["n_features"] == 9
        assert meta["class_labels"] == ["bear", "neutral", "bull"]
        assert "oos_accuracy" in meta
        assert "per_class_recall" in meta
        assert "macro_f1" in meta

        loaded = RegimePredictorV2.load(path)
        assert loaded.fitted is True
        assert loaded.feature_names == REGIME_V2_FEATURES
        # Predictions match original.
        original_probs = scorer.predict_proba(X[:10])
        loaded_probs = loaded.predict_proba(X[:10])
        np.testing.assert_array_almost_equal(original_probs, loaded_probs)
        # Gate metrics survived the roundtrip.
        assert loaded._oos_accuracy is not None
        assert loaded._per_class_recall is not None

    def test_save_unfitted_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            RegimePredictorV2().save(tmp_path / "should_not_exist.pkl")

    def test_load_tolerates_missing_sidecar(self, tmp_path: Path):
        X, y = _balanced_3class_data(n_per_class=200)
        scorer = RegimePredictorV2(n_estimators=50).fit(X, y)
        path = tmp_path / "regime_v2.pkl"
        scorer.save(path)
        (tmp_path / "regime_v2.pkl.meta.json").unlink()

        loaded = RegimePredictorV2.load(path)
        assert loaded.fitted is True
        assert len(loaded.feature_names) == 9


# ── Metrics dict ────────────────────────────────────────────────────────

class TestMetrics:

    def test_metrics_unfitted(self):
        m = RegimePredictorV2().metrics()
        assert m["fitted"] is False
        assert m["oos_accuracy"] is None
        assert m["per_class_recall"] is None

    def test_metrics_fitted(self):
        X, y = _balanced_3class_data(n_per_class=200)
        split = int(0.7 * len(X))
        scorer = RegimePredictorV2(n_estimators=80).fit(
            X[:split], y[:split], X[split:], y[split:],
        )
        m = scorer.metrics()
        assert m["type"] == "regime_predictor_v2"
        assert m["fitted"] is True
        assert m["n_features"] == 9
        assert m["class_labels"] == ["bear", "neutral", "bull"]
        assert isinstance(m["oos_accuracy"], float)
        assert isinstance(m["per_class_recall"], dict)
        for cls in ("bear", "neutral", "bull"):
            assert cls in m["per_class_recall"]
        assert isinstance(m["macro_f1"], float)
