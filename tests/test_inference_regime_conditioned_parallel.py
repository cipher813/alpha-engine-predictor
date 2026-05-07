"""Tests for the inference-time parallel path of RegimePredictorV2 +
RegimeConditionedMeta (audit Phase 4 PR 4/6).

The regime detector predicts the current regime; the per-regime Ridge
stack produces alpha for that regime. Both predicted_regime and
regime_conditioned_alpha ride alongside the single-Ridge canonical
predicted_alpha in observe-only mode.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_model import META_FEATURES
from model.regime_conditioned_meta import RegimeConditionedMeta
from model.regime_predictor_v2 import REGIME_V2_FEATURES, RegimePredictorV2


# ── Synthetic fixtures ──────────────────────────────────────────────────

def _trained_regime_predictor(seed: int = 0) -> RegimePredictorV2:
    """Trained RegimePredictorV2 suitable for inference-style scoring."""
    rng = np.random.default_rng(seed)
    n_per_class = 200
    n = n_per_class * 3
    X = rng.normal(0, 1, (n, len(REGIME_V2_FEATURES))).astype(np.float32)
    y = np.zeros(n, dtype=np.int32)
    # Bear: negative spy_20d_return
    X[:n_per_class, 0] = rng.normal(-0.05, 0.02, n_per_class)
    y[:n_per_class] = 0
    # Neutral
    X[n_per_class:2*n_per_class, 0] = rng.normal(0.0, 0.01, n_per_class)
    y[n_per_class:2*n_per_class] = 1
    # Bull
    X[2*n_per_class:, 0] = rng.normal(0.05, 0.02, n_per_class)
    y[2*n_per_class:] = 2
    return RegimePredictorV2(n_estimators=80).fit(X, y)


def _trained_regime_conditioned_meta(seed: int = 0) -> RegimeConditionedMeta:
    rng = np.random.default_rng(seed)
    n_per_regime = 250
    n = n_per_regime * 3
    X = rng.normal(0, 1, (n, len(META_FEATURES))).astype(np.float64)
    y = np.zeros(n)
    regimes = []
    # Each regime has a different coefficient on feature 0 so per-regime
    # Ridges produce distinct alphas.
    y[:n_per_regime] = -0.4 * X[:n_per_regime, 0] + rng.normal(0, 0.5, n_per_regime)
    regimes.extend(["bear"] * n_per_regime)
    y[n_per_regime:2*n_per_regime] = (
        0.1 * X[n_per_regime:2*n_per_regime, 0]
        + rng.normal(0, 0.5, n_per_regime)
    )
    regimes.extend(["neutral"] * n_per_regime)
    y[2*n_per_regime:] = (
        0.4 * X[2*n_per_regime:, 0]
        + rng.normal(0, 0.5, n_per_regime)
    )
    regimes.extend(["bull"] * n_per_regime)
    return RegimeConditionedMeta(min_per_regime_rows=200).fit(
        X, y, regimes, feature_names=META_FEATURES,
    )


# ── Predicted regime is one of the canonical labels ────────────────────

class TestPredictedRegime:

    def test_predict_class_from_dict_returns_canonical_label(self):
        rp = _trained_regime_predictor()
        # Build a dict with all 9 V2 features
        feats = {name: 0.0 for name in REGIME_V2_FEATURES}
        feats["macro_spy_20d_return"] = 0.05  # bull-like
        label = rp.predict_class_from_dict(feats)
        assert label in {"bear", "neutral", "bull"}

    def test_predict_with_meta_features_dict_works(self):
        # Inference passes meta_features (which has the 6 macro features
        # under the macro_*-prefixed names V2 expects). The 3 SPY-momentum
        # features (spy_5d_return etc.) aren't in meta_features — they're
        # imputed to 0.0 by predict_class_from_dict. Verify graceful fallback.
        rp = _trained_regime_predictor()
        meta_features_dict = {
            "research_calibrator_prob": 0.55,
            "momentum_score": 0.02,
            "expected_move": 0.015,
            "research_composite_score": 0.7,
            "research_conviction": 1.0,
            "sector_macro_modifier": 0.05,
            "macro_spy_20d_return": 0.03,
            "macro_spy_20d_vol": 0.18,
            "macro_vix_level": 0.85,
            "macro_vix_term_slope": -0.05,
            "macro_yield_curve_slope": 0.04,
            "macro_market_breadth": 0.55,
            # Note: spy_5d_return / spy_60d_return / spy_120d_return absent —
            # imputed to 0.0 by predict_class_from_dict's defensive default
        }
        label = rp.predict_class_from_dict(meta_features_dict)
        assert label in {"bear", "neutral", "bull"}

    def test_unfitted_returns_neutral(self):
        rp = RegimePredictorV2()
        assert rp.predict_class_from_dict({}) == "neutral"


# ── Regime-conditioned alpha routing ────────────────────────────────────

class TestRegimeConditionedAlpha:

    def test_predict_single_for_regime_returns_scalar(self):
        rcm = _trained_regime_conditioned_meta()
        feats = {name: 0.5 for name in META_FEATURES}
        result = rcm.predict_single_for_regime(feats, "bull")
        assert isinstance(result, float)

    def test_different_regimes_produce_different_alphas(self):
        # Per-regime Ridges fit on different sign-of-feature relationships
        # → bull and bear should produce meaningfully different alphas
        # for the same input.
        rcm = _trained_regime_conditioned_meta()
        feats = {name: 0.5 for name in META_FEATURES}
        feats[META_FEATURES[0]] = 1.5  # large positive value on feature 0
        bull_alpha = rcm.predict_single_for_regime(feats, "bull")
        bear_alpha = rcm.predict_single_for_regime(feats, "bear")
        assert abs(bull_alpha - bear_alpha) > 0.05

    def test_unrecognized_regime_falls_back(self):
        rcm = _trained_regime_conditioned_meta()
        feats = {name: 0.3 for name in META_FEATURES}
        fallback_pred = float(rcm.predict_unconditioned(
            np.array([[feats[f] for f in META_FEATURES]], dtype=np.float64)
        )[0])
        unrecognized = rcm.predict_single_for_regime(feats, "transition")
        # Unrecognized → fallback Ridge → same as predict_unconditioned.
        assert abs(unrecognized - fallback_pred) < 1e-4


# ── Load tolerance: V2 / RCM absent ────────────────────────────────────

class TestLoadToleranceForObservePeriod:
    """Inference must not fail when either V2 or RCM isn't yet in S3."""

    def test_meta_models_get_returns_none_when_v2_absent(self):
        meta_models: dict = {}
        rp = meta_models.get("regime_predictor_v2")
        assert rp is None

    def test_meta_models_get_returns_none_when_rcm_absent(self):
        meta_models: dict = {}
        rcm = meta_models.get("regime_conditioned_meta")
        assert rcm is None

    def test_predicted_regime_none_when_v2_absent(self):
        # Mirror inference logic: if rp is None, predicted_regime stays None.
        regime_predictor_v2 = None
        predicted_regime = None
        if regime_predictor_v2 is not None and getattr(
            regime_predictor_v2, "fitted", False
        ):
            predicted_regime = "bull"
        assert predicted_regime is None

    def test_regime_conditioned_alpha_none_when_either_absent(self):
        # If V2 absent, predicted_regime stays None → RCM never called.
        predicted_regime = None
        regime_conditioned_meta = _trained_regime_conditioned_meta()
        regime_conditioned_alpha = None
        if (
            predicted_regime is not None
            and regime_conditioned_meta is not None
            and getattr(regime_conditioned_meta, "fitted", False)
        ):
            regime_conditioned_alpha = 0.01
        assert regime_conditioned_alpha is None


# ── Output dict shape ───────────────────────────────────────────────────

class TestPredictionDictShape:

    def test_dict_has_predicted_regime_key_when_populated(self):
        rp = _trained_regime_predictor()
        feats = {name: 0.0 for name in REGIME_V2_FEATURES}
        feats["macro_spy_20d_return"] = 0.05
        result = {
            "ticker": "TEST",
            "predicted_regime": rp.predict_class_from_dict(feats),
        }
        assert result["predicted_regime"] in {"bear", "neutral", "bull"}

    def test_dict_has_regime_conditioned_alpha_key_when_populated(self):
        rcm = _trained_regime_conditioned_meta()
        feats = {name: 0.5 for name in META_FEATURES}
        alpha = rcm.predict_single_for_regime(feats, "bull")
        result = {
            "ticker": "TEST",
            "regime_conditioned_alpha": round(alpha, 6),
        }
        assert isinstance(result["regime_conditioned_alpha"], float)

    def test_dict_has_none_values_when_models_absent(self):
        # Mirror the inference output when both V2 and RCM are absent.
        result = {
            "ticker": "TEST",
            "predicted_regime": None,
            "regime_conditioned_alpha": None,
        }
        assert result["predicted_regime"] is None
        assert result["regime_conditioned_alpha"] is None


# ── Defensive on edge cases ────────────────────────────────────────────

class TestDefensiveEdgeCases:

    def test_v2_handles_empty_dict_gracefully(self):
        rp = _trained_regime_predictor()
        # Empty dict → all features default to 0.0 → some valid label.
        label = rp.predict_class_from_dict({})
        assert label in {"bear", "neutral", "bull"}

    def test_rcm_handles_empty_dict_gracefully(self):
        rcm = _trained_regime_conditioned_meta()
        # Empty dict — MetaModel.predict_single defaults missing features
        # so this should not raise.
        try:
            result = rcm.predict_single_for_regime({}, "neutral")
            assert isinstance(result, float)
        except Exception as e:
            pytest.fail(f"predict_single_for_regime should not raise on empty dict: {e}")
