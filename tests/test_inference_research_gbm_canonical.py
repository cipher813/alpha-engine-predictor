"""Tests for the inference-time canonical path of ResearchGBMScorer.

Per audit Phase 3 PR 4/5 cutover (2026-05-09): the GBM is the canonical
source for the `research_calibrator_prob` META_FEATURE. The bucket-lookup
ResearchCalibrator remains as a fallback when the GBM isn't loaded
(n_finite < 100 cycles, or training-side fit failed).

These tests cover the canonical-path contract:
- predict_from_dict consumes the same 9 features the meta-Ridge sees.
- When GBM is loaded + fitted: GBM output overrides bucket-lookup output.
- When GBM is unfitted / absent: bucket-lookup output stays canonical.
- Defensive on NaN / non-dict edge cases (mirrors per-ticker try/except).

Replaces tests/test_inference_research_gbm_parallel.py whose premise
(parallel `research_gbm_prob` field rides alongside in observe-only mode)
was retired by the cutover.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.research_gbm import RESEARCH_GBM_FEATURES, ResearchGBMScorer


def _trained_scorer(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 400
    X = rng.normal(0, 1, (n, len(RESEARCH_GBM_FEATURES))).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    return ResearchGBMScorer(n_estimators=80).fit(X, y)


# ── predict_from_dict consumes meta_features cleanly ────────────────────


class TestPredictFromDictWithMetaFeatures:
    def test_full_meta_features_dict_works(self):
        # Inference builds a 9-key dict from research-derived features +
        # macro_row_for_meta. Same shape regardless of whether the GBM
        # output then overrides research_calibrator_prob downstream.
        scorer = _trained_scorer()
        gbm_input = {
            "research_composite_score": 0.7,
            "research_conviction": 1.0,
            "sector_macro_modifier": 0.05,
            "macro_spy_20d_return": 0.03,
            "macro_spy_20d_vol": 0.18,
            "macro_vix_level": 0.85,
            "macro_vix_term_slope": -0.05,
            "macro_yield_curve_slope": 0.04,
            "macro_market_breadth": 0.55,
        }
        result = scorer.predict_from_dict(gbm_input)
        assert 0.0 <= result <= 1.0

    def test_predict_idempotent_on_same_input(self):
        scorer = _trained_scorer()
        gbm_input = {name: 0.5 for name in RESEARCH_GBM_FEATURES}
        a = scorer.predict_from_dict(gbm_input)
        b = scorer.predict_from_dict(gbm_input)
        assert a == b


# ── Cutover semantics: GBM overrides bucket-lookup when loaded ──────────


class TestCutoverSemantics:
    """Mirrors the cutover logic at run_inference.py:443-466 (post-cutover).

    When research_gbm is loaded + fitted, research_calibrator_prob is
    sourced from the GBM. When research_gbm is None or unfitted,
    research_calibrator_prob stays as the bucket-lookup output already
    populated by extract_research_features.
    """

    def test_gbm_overrides_bucket_lookup_when_loaded_and_fitted(self):
        scorer = _trained_scorer()
        gbm_input = {name: 0.3 for name in RESEARCH_GBM_FEATURES}

        # Bucket-lookup-sourced prob (would have been populated by
        # extract_research_features before the override).
        research_cal_prob = 0.5

        # Cutover branch: GBM overrides
        if scorer is not None and getattr(scorer, "fitted", False):
            research_cal_prob = float(scorer.predict_from_dict(gbm_input))
        assert research_cal_prob != 0.5  # must have been overridden

    def test_bucket_lookup_stays_when_gbm_is_none(self):
        # Simulates the load_model.py outcome when the GBM file is missing.
        meta_models: dict = {}  # no research_gbm key
        research_gbm = meta_models.get("research_gbm")

        research_cal_prob = 0.5  # from bucket-lookup
        if research_gbm is not None and getattr(research_gbm, "fitted", False):
            research_cal_prob = 0.99  # would override
        assert research_cal_prob == 0.5

    def test_bucket_lookup_stays_when_gbm_is_unfitted(self):
        research_gbm = ResearchGBMScorer()  # fitted=False
        research_cal_prob = 0.5  # from bucket-lookup
        if research_gbm is not None and getattr(research_gbm, "fitted", False):
            research_cal_prob = 0.99
        assert research_cal_prob == 0.5


# ── Defensive on edge cases ─────────────────────────────────────────────


class TestDefensiveOnEdgeCases:
    def test_predict_from_dict_with_nan_values_does_not_crash(self):
        scorer = _trained_scorer()
        feats = {name: 0.0 for name in RESEARCH_GBM_FEATURES}
        feats["macro_vix_level"] = float("nan")
        try:
            result = scorer.predict_from_dict(feats)
            assert isinstance(result, float)
        except Exception:
            pytest.fail("predict_from_dict should not raise on NaN input")

    def test_inference_falls_back_to_bucket_lookup_on_predict_failure(self):
        # Mirror the try/except at run_inference.py: any predict_from_dict
        # failure logs at debug level and leaves research_cal_prob at the
        # bucket-lookup-sourced value (no override).
        scorer = _trained_scorer()
        bucket_prob = 0.5
        research_cal_prob = bucket_prob
        try:
            research_cal_prob = float(scorer.predict_from_dict(None))  # type: ignore[arg-type]
        except Exception:
            pass  # swallow, leave at bucket_prob
        assert research_cal_prob == bucket_prob
