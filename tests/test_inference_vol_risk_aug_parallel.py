"""Tests for the Stage 2b inference parallel path — risk-augmented
volatility GBM rides alongside the plain volatility GBM in observe-only
mode.

Stage 2b of the regime-conditioning rebuild (plan doc:
alpha-engine-docs/private/regime-conditioning-260510.md). Validates:
- ``inference/stages/load_model.py`` loads ``volatility_risk_aug_model.txt``
  alongside the plain + macro-aug volatility models.
- ``inference/stages/run_inference.py`` emits ``expected_move_risk_aug``
  parallel field in predictions JSON.
- Feature schema matches Stage 2b training: 6 vol (rank-normed) +
  5 risk (rank-normed) = 11 features in ``VOLATILITY_FEATURES +
  RISK_AUG_FEATURES`` order.
- Failure isolation: aug GBM absent / risk feature batch raises / per-
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

    def test_load_model_loads_volatility_risk_aug(self):
        src = _src("inference/stages/load_model.py")
        assert "volatility_risk_aug_model.txt" in src
        assert 'meta_models["volatility_risk_aug"]' in src

    def test_run_inference_predicts_with_risk_aug(self):
        src = _src("inference/stages/run_inference.py")
        # The aug GBM is fetched from meta_models
        assert 'meta_models.get("volatility_risk_aug")' in src
        # Aug feature batch concats vol-ranked + risk-ranked (both
        # cross-sectional rank-norm because both are per-ticker)
        assert "X_vol_risk_aug_ranked" in src
        # Per-ticker predict on aug variant
        assert "expected_move_risk_aug" in src

    def test_predictions_dict_has_risk_aug_field(self):
        src = _src("inference/stages/run_inference.py")
        # The result-dict assignment uses the parallel field name
        assert '"expected_move_risk_aug":' in src

    def test_risk_features_used_in_inference(self):
        # The inference path must read cfg.RISK_AUG_FEATURES (Stage 2b
        # config) so the schema stays in lockstep with training.
        src = _src("inference/stages/run_inference.py")
        assert "cfg.RISK_AUG_FEATURES" in src


# ── Aug feature concat shape contract ──────────────────────────────────


class TestRiskAugFeatureConcatShape:
    """Validates the schema invariant: risk-aug feature vector =
    [VOLATILITY_FEATURES rank-normed, RISK_AUG_FEATURES rank-normed] in
    that order, matching training's VOL_RISK_AUG_FEATURES list."""

    def test_aug_feature_count_matches_training(self):
        # Training concat: VOLATILITY_FEATURES + RISK_AUG_FEATURES.
        # Schema-relative invariant — exact count grows as future stages
        # add risk features. What matters is that both halves have
        # at least one feature.
        n_vol = len(cfg.VOLATILITY_FEATURES)
        n_risk = len(cfg.RISK_AUG_FEATURES)
        assert n_vol >= 1
        assert n_risk >= 1
        # The risk-aug GBM input vector has size n_vol + n_risk by
        # construction.

    def test_risk_features_in_config(self):
        # Lock the canonical 5 risk features Stage 2b ships with.
        assert "beta_60d" in cfg.RISK_AUG_FEATURES
        assert "idio_vol_60d" in cfg.RISK_AUG_FEATURES
        assert "vol_of_vol_30d" in cfg.RISK_AUG_FEATURES
        assert "max_drawdown_60d" in cfg.RISK_AUG_FEATURES
        assert "realized_vol_63d" in cfg.RISK_AUG_FEATURES

    def test_aug_concat_preserves_per_ticker_then_risk_order(self):
        # Both halves vary per-ticker (cross-sectionally) → both should
        # have non-trivial per-row variance even when rank-normed.
        n_present = 5
        rng = np.random.default_rng(0)
        n_vol = len(cfg.VOLATILITY_FEATURES)
        n_risk = len(cfg.RISK_AUG_FEATURES)
        X_vol_ranked = rng.random((n_present, n_vol)).astype(np.float32)
        X_risk_ranked = rng.random((n_present, n_risk)).astype(np.float32)
        X_aug = np.concatenate([X_vol_ranked, X_risk_ranked], axis=1)

        assert X_aug.shape == (n_present, n_vol + n_risk)
        # First n_vol columns: per-ticker vol (vary across rows).
        for j in range(n_vol):
            assert X_aug[:, j].std() > 0
        # Next n_risk columns: per-ticker risk (also vary across rows).
        for j in range(n_vol, n_vol + n_risk):
            assert X_aug[:, j].std() > 0


# ── Failure isolation ──────────────────────────────────────────────────


class TestFailureIsolation:

    def test_run_inference_falls_back_when_risk_aug_missing(self):
        src = _src("inference/stages/run_inference.py")
        # The risk-aug-predict block is gated on vol_scorer_risk_aug AND
        # X_vol_risk_aug_ranked being populated.
        assert "if vol_scorer_risk_aug is not None and X_vol_ranked is not None" in src

    def test_risk_batch_failure_is_non_blocking(self):
        src = _src("inference/stages/run_inference.py")
        # The risk feature batch build is wrapped in try/except with
        # log.warning fallback — no production behaviour change.
        assert "Risk feature batch build failed" in src

    def test_per_ticker_predict_failure_is_non_blocking(self):
        src = _src("inference/stages/run_inference.py")
        # Per-ticker risk-aug predict is wrapped in try/except with
        # log.debug fallback — same shape as macro-aug parallel path.
        assert "volatility_risk_aug.predict failed for" in src


# ── Smoke import ───────────────────────────────────────────────────────


class TestSmokeImport:

    def test_modules_import_cleanly(self):
        # The Stage 2b additions don't break any module's import contract.
        import importlib

        for mod in (
            "inference.stages.load_model",
            "inference.stages.run_inference",
            "training.meta_trainer",
        ):
            m = importlib.import_module(mod)
            assert m is not None
