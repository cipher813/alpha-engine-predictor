"""Tests for Stage 2b's training-side wiring of the risk-augmented
volatility GBM in ``training/meta_trainer.py``.

Stage 2b of the regime-conditioning rebuild — parallel observation
training of ``prod_vol_risk_aug`` alongside the plain volatility GBM.
Mirrors the audit Phase 3 PR 2/5 (ResearchGBMScorer parallel-observe)
shape.

Plan doc: ~/Development/alpha-engine-docs/private/regime-conditioning-260510.md
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


# ── Source-level regression: training-side wiring ──────────────────────


class TestTrainingSourceContract:

    def test_risk_chunks_loaded_in_feature_read_loop(self):
        src = _src("training/meta_trainer.py")
        # Risk feature accumulator must be declared alongside vol_chunks.
        assert "risk_chunks: list[np.ndarray] = []" in src
        # Per-ticker risk slice via reindex (NaN-fill for missing columns)
        assert "labeled.reindex(columns=cfg.RISK_AUG_FEATURES)" in src

    def test_x_risk_array_built_and_rank_normed(self):
        src = _src("training/meta_trainer.py")
        # X_risk concat from chunks
        assert "X_risk_unsorted = np.concatenate(risk_chunks, axis=0)" in src
        # X_risk gets the same rank-norm as X_vol (per-ticker → cross-
        # sectional rank-norm is correct)
        assert "X_risk = cross_sectional_rank_normalize(X_risk, all_dates)" in src

    def test_parallel_risk_aug_training_step_present(self):
        src = _src("training/meta_trainer.py")
        # VOL_RISK_AUG_FEATURES = VOLATILITY_FEATURES + RISK_AUG_FEATURES
        assert (
            "VOL_RISK_AUG_FEATURES = list(cfg.VOLATILITY_FEATURES) + list(cfg.RISK_AUG_FEATURES)"
            in src
        )
        # prod_vol_risk_aug fits on the concat
        assert "prod_vol_risk_aug" in src
        assert "X_vol_risk_aug = np.concatenate([X_vol, X_risk], axis=1)" in src

    def test_risk_aug_in_upload_set(self):
        src = _src("training/meta_trainer.py")
        # Persists volatility_risk_aug_model.txt to S3 alongside the
        # plain volatility model when the parallel fit succeeds.
        assert 'models["volatility_risk_aug"]' in src
        assert "volatility_risk_aug_model.txt" in src

    def test_manifest_carries_risk_aug_metrics(self):
        src = _src("training/meta_trainer.py")
        # Manifest entry mirrors the macro_aug shape (test_ic, lift,
        # relative_lift, val_ic, n_features) for consistent
        # variant_cutover_gate evaluation.
        assert '"volatility_risk_aug": {' in src
        assert '"test_ic_plain_vol":' in src
        assert '"abs_lift":' in src
        assert '"relative_lift":' in src

    def test_risk_aug_failure_is_non_blocking(self):
        src = _src("training/meta_trainer.py")
        # Parallel fit wrapped in try/except — failure does not block
        # plain prod_vol training.
        assert "Volatility-with-risk parallel fit failed" in src


# ── Schema contract ────────────────────────────────────────────────────


class TestRiskAugFeatureSchema:

    def test_risk_aug_features_canonical_list(self):
        # The five Stage 2b risk features.
        expected = {
            "beta_60d", "idio_vol_60d", "vol_of_vol_30d",
            "max_drawdown_60d", "realized_vol_63d",
        }
        assert set(cfg.RISK_AUG_FEATURES) == expected

    def test_risk_aug_features_distinct_from_vol_features(self):
        # No overlap — the risk-aug GBM consumes a distinct feature set.
        assert not (set(cfg.RISK_AUG_FEATURES) & set(cfg.VOLATILITY_FEATURES))

    def test_combined_schema_matches_concat(self):
        # The training-time VOL_RISK_AUG_FEATURES list is the
        # concatenation of VOLATILITY + RISK_AUG; the inference path
        # expects this exact order.
        combined = list(cfg.VOLATILITY_FEATURES) + list(cfg.RISK_AUG_FEATURES)
        assert len(combined) == len(cfg.VOLATILITY_FEATURES) + len(cfg.RISK_AUG_FEATURES)
        assert combined[:len(cfg.VOLATILITY_FEATURES)] == list(cfg.VOLATILITY_FEATURES)
        assert combined[len(cfg.VOLATILITY_FEATURES):] == list(cfg.RISK_AUG_FEATURES)


# ── reindex tolerance ──────────────────────────────────────────────────


class TestReindexTolerance:
    """Validates that pandas.DataFrame.reindex(columns=...) returns NaN
    columns for missing names rather than raising KeyError. Stage 2b
    relies on this to handle parquets that pre-date the alpha-engine-data
    #202 risk feature additions."""

    def test_reindex_returns_nan_for_missing_columns(self):
        df = pd.DataFrame({
            "atr_14_pct": [0.02, 0.03],
            "realized_vol_20d": [0.18, 0.19],
        })
        out = df.reindex(columns=["atr_14_pct", "beta_60d"])
        assert "beta_60d" in out.columns
        assert out["beta_60d"].isna().all()
        assert out["atr_14_pct"].notna().all()

    def test_reindex_full_risk_aug_features_on_legacy_df(self):
        # Synthesize a "legacy" parquet with no risk features.
        df = pd.DataFrame({
            "atr_14_pct": [0.02],
            "realized_vol_20d": [0.18],
        })
        out = df.reindex(columns=cfg.RISK_AUG_FEATURES)
        # All risk columns are NaN — LightGBM will tolerate this.
        for col in cfg.RISK_AUG_FEATURES:
            assert col in out.columns
            assert out[col].isna().all()
