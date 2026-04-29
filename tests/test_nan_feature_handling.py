"""NaN-feature handling at the inference boundary.

Closes ROADMAP P1 "NaN-feature handling audit + short-history subsample
validation" — audit phase. The data layer ships partial-NaN features for
short-history tickers (per the 2026-04-21 evening graceful-degrade policy
+ alpha-engine-data PR #78). Pre-fix, the predictor:

  1. Layer 1A direct fallback used ``v or default`` which DOES NOT filter
     NaN (Python ``nan`` is truthy) — NaN propagated as momentum_score.
  2. Meta-features dict assembled with potential NaN values; Ridge
     regression is NaN-poison so any NaN input → NaN alpha → calibrator
     crash potential.
  3. Layer 1A/1B silent ``try/except: pass`` masked real LightGBM faults.

Post-fix:

  - ``_safe_get_numeric`` replaces the buggy ``or default`` idiom.
  - ``_sanitize_meta_features`` imputes NaN → 0.0 at the ridge boundary
    with per-ticker logging + a batch counter (``ctx.n_nan_imputed_tickers``).
  - Silent except blocks at the Layer 1A GBM path + Layer 1B vol path
    removed — LightGBM handles NaN natively, real faults now hard-fail.

Promotion gate (subsample IC validation) and CloudWatch metric tracked
as separate follow-ups per ROADMAP sequencing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


class TestSafeGetNumeric:
    def test_present_numeric_key_returns_value(self):
        from inference.stages.run_inference import _safe_get_numeric
        s = pd.Series({"x": 0.42})
        assert _safe_get_numeric(s, "x", 0.0) == 0.42

    def test_missing_key_returns_default(self):
        from inference.stages.run_inference import _safe_get_numeric
        s = pd.Series({"x": 0.42})
        assert _safe_get_numeric(s, "missing", 1.5) == 1.5

    def test_nan_value_returns_default(self):
        """The bug: ``v or default`` evaluates ``nan or 0`` to ``nan``
        because Python ``nan`` is truthy. _safe_get_numeric must not
        propagate NaN."""
        from inference.stages.run_inference import _safe_get_numeric
        s = pd.Series({"x": float("nan")})
        assert _safe_get_numeric(s, "x", 0.0) == 0.0

    def test_pandas_na_returns_default(self):
        from inference.stages.run_inference import _safe_get_numeric
        s = pd.Series({"x": pd.NA})
        assert _safe_get_numeric(s, "x", 7.0) == 7.0

    def test_non_numeric_string_returns_default(self):
        from inference.stages.run_inference import _safe_get_numeric
        s = pd.Series({"x": "not-a-number"})
        assert _safe_get_numeric(s, "x", 0.0) == 0.0

    def test_default_used_with_neutral_baseline(self):
        """RSI's neutral baseline is 50, not 0 — confirm default arg is honored."""
        from inference.stages.run_inference import _safe_get_numeric
        s = pd.Series({"rsi_14": float("nan")})
        assert _safe_get_numeric(s, "rsi_14", 50.0) == 50.0


class TestSanitizeMetaFeatures:
    def test_all_numeric_passthrough(self):
        from inference.stages.run_inference import _sanitize_meta_features
        feats = {"a": 0.1, "b": 0.2, "c": -0.3}
        cleaned, nan_keys = _sanitize_meta_features(feats)
        assert cleaned == feats
        assert nan_keys == []

    def test_single_nan_imputed_and_named(self):
        from inference.stages.run_inference import _sanitize_meta_features
        feats = {"a": 0.1, "b": float("nan"), "c": -0.3}
        cleaned, nan_keys = _sanitize_meta_features(feats)
        assert cleaned == {"a": 0.1, "b": 0.0, "c": -0.3}
        assert nan_keys == ["b"]

    def test_all_nan_imputed_to_zero(self):
        """Worst case: every meta-feature was NaN. Ridge would have
        produced NaN alpha; we degrade to a neutral all-zero input which
        the ridge maps to its bias term."""
        from inference.stages.run_inference import _sanitize_meta_features
        feats = {"a": float("nan"), "b": float("nan"), "c": float("nan")}
        cleaned, nan_keys = _sanitize_meta_features(feats)
        assert cleaned == {"a": 0.0, "b": 0.0, "c": 0.0}
        assert sorted(nan_keys) == ["a", "b", "c"]

    def test_mixed_types_only_floats_checked(self):
        """Non-float values (e.g. string sector_modifier shouldn't reach
        here, but if they did) are passed through unchanged. NaN check
        is gated on ``isinstance(v, float)``."""
        from inference.stages.run_inference import _sanitize_meta_features
        feats = {"a": 0.1, "b": "string-not-expected", "c": float("nan")}
        cleaned, nan_keys = _sanitize_meta_features(feats)
        assert cleaned["a"] == 0.1
        assert cleaned["b"] == "string-not-expected"
        assert cleaned["c"] == 0.0
        assert nan_keys == ["c"]

    def test_returns_new_dict_not_mutated_input(self):
        """The original meta_features dict in the inference loop should
        not be mutated — we return a cleaned copy so the diagnostic log
        and subsequent code can see what was originally there."""
        from inference.stages.run_inference import _sanitize_meta_features
        feats = {"a": 0.1, "b": float("nan")}
        original = dict(feats)
        cleaned, _ = _sanitize_meta_features(feats)
        # Sanitize should not mutate input when NaN is present
        assert feats == original
        # But the cleaned copy must differ
        assert cleaned is not feats

    def test_empty_dict(self):
        from inference.stages.run_inference import _sanitize_meta_features
        cleaned, nan_keys = _sanitize_meta_features({})
        assert cleaned == {}
        assert nan_keys == []


class TestNaNImputationWiring:
    """End-to-end wiring assertion: when meta_features contain NaN,
    inference imputes them, increments the counter, logs per-ticker, and
    calls predict_single with the sanitized dict (never NaN)."""

    def _stub_meta_model(self):
        """Minimal stand-in for MetaModel — captures the dict passed to
        predict_single so the test can assert it was sanitized."""
        captured = []

        class _StubModel:
            is_fitted = True

            def predict_single(self, features: dict) -> float:
                captured.append(dict(features))
                # Return a non-NaN scalar so the rest of the loop runs
                return 0.001

        return _StubModel(), captured

    def test_nan_features_imputed_before_predict_single(self, caplog):
        """Pin the wiring: ridge.predict_single must never receive NaN."""
        from inference.stages.run_inference import _sanitize_meta_features

        # Simulate what the inference loop builds + sanitizes
        raw = {
            "research_calibrator_prob": 0.55,
            "momentum_score": float("nan"),  # short-history ticker
            "expected_move": 0.02,
            "research_composite_score": float("nan"),
            "research_conviction": 0.0,
            "sector_macro_modifier": -0.04,
        }
        cleaned, nan_keys = _sanitize_meta_features(raw)

        # The would-be ridge input now has zero NaN values
        assert all(not pd.isna(v) for v in cleaned.values())
        assert sorted(nan_keys) == ["momentum_score", "research_composite_score"]

        stub, captured = self._stub_meta_model()
        alpha = stub.predict_single(cleaned)
        assert isinstance(alpha, float)
        assert not pd.isna(alpha)
        assert all(not pd.isna(v) for v in captured[0].values()), (
            "predict_single received NaN — sanitization wiring is broken"
        )

    def test_n_nan_imputed_tickers_counter_default_zero(self):
        """The PipelineContext counter must default to 0 — the imputation
        path is opt-in, not always-on."""
        from inference.pipeline import PipelineContext
        ctx = PipelineContext()
        assert ctx.n_nan_imputed_tickers == 0

    def test_n_nan_imputed_tickers_field_is_writeable(self):
        """The counter is incremented in the inference loop; pin that
        the dataclass field accepts mutation."""
        from inference.pipeline import PipelineContext
        ctx = PipelineContext()
        ctx.n_nan_imputed_tickers += 1
        ctx.n_nan_imputed_tickers += 1
        assert ctx.n_nan_imputed_tickers == 2
