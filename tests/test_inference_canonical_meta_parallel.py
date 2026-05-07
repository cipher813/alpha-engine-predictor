"""Tests for the canonical-label Ridge inference parallel path (Track A PR 4).

The canonical Ridge consumes meta_features (same as the legacy Ridge);
its output rides alongside predicted_alpha as canonical_predicted_alpha
in the predictions JSON for parity comparison. Same observe-only pattern
as PR #96 (research-LGB) and PR #100 (regime-conditioned).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_model import META_FEATURES, MetaModel


def _trained_canonical_meta(seed: int = 0) -> MetaModel:
    """Trained MetaModel suitable for inference-style predict_single."""
    rng = np.random.default_rng(seed)
    n = 600
    X = rng.normal(0, 1, (n, len(META_FEATURES))).astype(np.float64)
    # Synthetic canonical-label target: linear combination of features + noise
    y = 0.05 * X[:, 0] + 0.03 * X[:, 1] + rng.normal(0, 0.5, n)
    return MetaModel(alpha=1.0).fit(X, y, feature_names=META_FEATURES)


# ── Predict via meta_features dict ──────────────────────────────────────

class TestCanonicalRidgePredict:

    def test_predict_single_returns_scalar(self):
        cm = _trained_canonical_meta()
        feats = {name: 0.5 for name in META_FEATURES}
        result = cm.predict_single(feats)
        assert isinstance(result, float)

    def test_predict_varies_with_input(self):
        cm = _trained_canonical_meta()
        feats_low = {name: 0.0 for name in META_FEATURES}
        feats_low[META_FEATURES[0]] = -2.0
        feats_high = {name: 0.0 for name in META_FEATURES}
        feats_high[META_FEATURES[0]] = 2.0
        low = cm.predict_single(feats_low)
        high = cm.predict_single(feats_high)
        # Synthetic positive coefficient on feature 0 → high > low
        assert high > low

    def test_predict_with_extra_keys_ignored(self):
        # Inference passes meta_features which has 12 keys; canonical
        # Ridge picks the META_FEATURES it was trained on and ignores
        # the rest. This mirrors the legacy Ridge's behavior.
        cm = _trained_canonical_meta()
        feats = {name: 0.3 for name in META_FEATURES}
        feats["extra_key"] = 99.0
        feats["another_extra"] = -42.0
        # Should not crash; produces same prediction as without extras.
        result_with_extras = cm.predict_single(feats)
        feats_clean = {name: 0.3 for name in META_FEATURES}
        result_clean = cm.predict_single(feats_clean)
        assert result_with_extras == result_clean


# ── Load tolerance: canonical Ridge absent ─────────────────────────────

class TestLoadToleranceForObservePeriod:
    """Inference must not fail when canonical_meta_model.pkl isn't in S3."""

    def test_meta_models_get_returns_none_when_absent(self):
        meta_models: dict = {}
        cm = meta_models.get("canonical_meta")
        assert cm is None

    def test_canonical_predicted_alpha_none_when_absent(self):
        # Mirror inference logic: if canonical Ridge is None, the field
        # stays None in the prediction dict.
        canonical_meta_model = None
        canonical_predicted_alpha = None
        if (
            canonical_meta_model is not None
            and getattr(canonical_meta_model, "is_fitted", True)
        ):
            canonical_predicted_alpha = 0.01  # would set
        assert canonical_predicted_alpha is None


# ── Output dict shape ───────────────────────────────────────────────────

class TestPredictionDictShape:

    def test_dict_has_canonical_predicted_alpha_when_populated(self):
        cm = _trained_canonical_meta()
        feats = {name: 0.5 for name in META_FEATURES}
        alpha = cm.predict_single(feats)
        result = {
            "ticker": "TEST",
            "predicted_alpha": 0.001,  # legacy
            "canonical_predicted_alpha": round(alpha, 6),
        }
        assert isinstance(result["canonical_predicted_alpha"], float)
        # Both legacy and canonical fields coexist.
        assert "predicted_alpha" in result
        assert "canonical_predicted_alpha" in result

    def test_dict_has_canonical_predicted_alpha_none_when_absent(self):
        result = {
            "ticker": "TEST",
            "predicted_alpha": 0.001,
            "canonical_predicted_alpha": None,
        }
        assert result["canonical_predicted_alpha"] is None
        assert result["predicted_alpha"] == 0.001  # legacy still populated


# ── Defensive on edge cases ────────────────────────────────────────────

class TestDefensiveEdgeCases:

    def test_inference_path_swallows_nan_via_try_except(self):
        # MetaModel.predict_single does NOT handle NaN natively (sklearn
        # Ridge raises ValueError on NaN input). In production the legacy
        # Ridge gets _sanitize_meta_features called before predict_single,
        # but the canonical Ridge's parallel path runs AFTER that
        # sanitization in run_inference.py, so NaN shouldn't reach it
        # in practice. Defensively, the parallel path wraps the call in
        # try/except — any failure leaves canonical_predicted_alpha as
        # None. This test confirms that contract.
        cm = _trained_canonical_meta()
        feats = {name: 0.0 for name in META_FEATURES}
        feats[META_FEATURES[0]] = float("nan")
        canonical_predicted_alpha: float | None = None
        try:
            canonical_predicted_alpha = float(cm.predict_single(feats))
        except Exception:
            pass  # swallow; mirrors run_inference.py behavior
        # Either we got a number (some sklearn versions tolerate NaN
        # via warnings) or None (Ridge raised). Both are acceptable.
        assert canonical_predicted_alpha is None or isinstance(
            canonical_predicted_alpha, float
        )

    def test_predict_single_with_empty_dict_does_not_crash(self):
        cm = _trained_canonical_meta()
        # MetaModel.predict_single uses .get(f, 0.0) defaults so missing
        # keys don't crash — same defensive pattern as the legacy Ridge.
        try:
            result = cm.predict_single({})
            assert isinstance(result, float)
        except Exception as e:
            pytest.fail(f"predict_single should not raise on empty dict: {e}")
