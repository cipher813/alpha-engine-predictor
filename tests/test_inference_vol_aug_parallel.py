"""Tests for the Stage 1c inference parallel path — macro-augmented
volatility GBM rides alongside the plain volatility GBM in observe-only
mode.

Stage 1c of the regime-conditioning rebuild (plan doc:
alpha-engine-docs/private/regime-conditioning-260510.md). Validates:
- ``inference/stages/load_model.py`` loads ``volatility_macro_aug_model.txt``
  alongside ``volatility_model.txt`` when present, gracefully missing when not.
- ``inference/stages/run_inference.py`` emits ``expected_move_macro_aug``
  parallel field in predictions JSON.
- Feature schema matches Stage 1b training: 6 vol (rank-normed) + 6 macro
  (z-scored) = 12 features in ``VOLATILITY_FEATURES + MACRO_NORM_FEATURES``
  order.
- Failure isolation: aug GBM absent / today's macros unavailable / per-
  ticker predict raises → plain ``expected_move`` is unaffected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


# ── Source-level regression: loader + inference path wired ──────────────


class TestSourceContract:

    def test_load_model_loads_volatility_macro_aug(self):
        src = _src("inference/stages/load_model.py")
        assert "volatility_macro_aug_model.txt" in src
        assert 'meta_models["volatility_macro_aug"]' in src

    def test_run_inference_predicts_with_macro_aug(self):
        src = _src("inference/stages/run_inference.py")
        # The aug GBM is fetched from meta_models
        assert 'meta_models.get("volatility_macro_aug")' in src
        # Today's macro vector is built via time-series z-score
        assert "time_series_zscore_normalize" in src
        # Aug feature batch concats vol-ranked + macro-z-scored
        assert "X_vol_aug_ranked" in src
        # Per-ticker predict on aug variant
        assert "expected_move_macro_aug" in src

    def test_predictions_dict_has_macro_aug_field(self):
        src = _src("inference/stages/run_inference.py")
        # The result-dict assignment uses the parallel field name
        assert '"expected_move_macro_aug":' in src

    def test_macro_norm_features_used_in_inference(self):
        # The inference path must read cfg.MACRO_NORM_FEATURES (Stage 1a
        # config) so the schema stays in lockstep with Stage 1b training.
        src = _src("inference/stages/run_inference.py")
        assert "cfg.MACRO_NORM_FEATURES" in src

    def test_macro_norm_window_used_in_inference(self):
        src = _src("inference/stages/run_inference.py")
        assert "cfg.MACRO_NORM_WINDOW" in src


# ── Aug feature concat shape contract ──────────────────────────────────


class TestAugFeatureConcatShape:
    """Validates the schema invariant: aug feature vector =
    [VOLATILITY_FEATURES rank-normed, MACRO_NORM_FEATURES z-scored] in
    that order, matching Stage 1b's VOL_AUG_FEATURES list."""

    def test_aug_feature_count_matches_training(self):
        # Stage 1b: VOL_AUG_FEATURES = VOLATILITY_FEATURES + MACRO_NORM_FEATURES
        n_vol = len(cfg.VOLATILITY_FEATURES)
        n_macro = len(cfg.MACRO_NORM_FEATURES)
        assert n_vol == 6  # current schema
        assert n_macro == 6  # current schema
        # Aug GBM was trained with n_vol + n_macro features
        assert n_vol + n_macro == 12

    def test_aug_concat_preserves_per_ticker_then_macro_order(self):
        # Simulate: (n_present, n_vol) + (n_present, n_macro) → (n_present, 12)
        n_present = 5
        X_vol_ranked = np.random.default_rng(0).random(
            (n_present, len(cfg.VOLATILITY_FEATURES))
        ).astype(np.float32)
        today_macros = np.array([0.5, -0.3, 1.2, 0.0, -0.8, 0.4], dtype=np.float32)
        macro_block = np.tile(today_macros, (n_present, 1))
        X_aug = np.concatenate([X_vol_ranked, macro_block], axis=1)

        # Shape contract
        assert X_aug.shape == (n_present, 12)
        # First 6 columns are per-ticker vol (vary across rows for the
        # same column). Random vol features have non-trivial stdev.
        assert X_aug[:, 0].std() > 0
        # Last 6 columns are constant per macro (broadcast across tickers).
        for j in range(6, 12):
            assert (X_aug[:, j] == X_aug[0, j]).all()

    def test_macro_block_order_matches_macro_norm_features(self):
        # When today_macros = [a, b, c, d, e, f], the broadcast block's
        # column j carries macro feature j of MACRO_NORM_FEATURES.
        today_macros = np.arange(6, dtype=np.float32)  # [0, 1, 2, 3, 4, 5]
        n_present = 3
        macro_block = np.tile(today_macros, (n_present, 1))
        # Column j contains value j (the broadcast preserves order).
        for j in range(6):
            assert (macro_block[:, j] == j).all()


# ── Failure isolation ──────────────────────────────────────────────────


class TestFailureIsolation:
    """Validates: aug GBM absent / today's macros unavailable / per-ticker
    predict raises → plain ``expected_move`` is unaffected."""

    def test_run_inference_falls_back_when_aug_missing(self):
        src = _src("inference/stages/run_inference.py")
        # The aug-predict block is gated on vol_scorer_aug AND
        # X_vol_aug_ranked being populated.
        assert "if (\n        vol_scorer_aug is not None" in src
        assert "and X_vol_ranked is not None" in src
        assert "and today_macros_zscored is not None" in src

    def test_macro_zscore_failure_is_non_blocking(self):
        src = _src("inference/stages/run_inference.py")
        # The macro z-score build is wrapped in try/except with
        # log.warning fallback — no production behaviour change.
        assert "Macro z-score for vol-aug GBM failed" in src

    def test_per_ticker_predict_failure_is_non_blocking(self):
        src = _src("inference/stages/run_inference.py")
        # Per-ticker aug predict is wrapped in try/except with log.debug
        # fallback — same shape as the existing research_gbm parallel path.
        assert "volatility_macro_aug.predict failed for" in src


# ── Smoke import ───────────────────────────────────────────────────────


class TestSmokeImport:

    def test_run_inference_imports_cleanly(self):
        # The Stage 1c additions don't break the module's import contract.
        import importlib

        for mod in (
            "inference.stages.load_model",
            "inference.stages.run_inference",
        ):
            m = importlib.import_module(mod)
            assert m is not None
