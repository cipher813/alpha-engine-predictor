"""Model-zoo challenger reliability fixes (2026-06-19).

The weekly rotation trains 4 specs but only 1 (residual-momentum) survived —
the other 3 crashed:

  * ``sota-directional-combine`` → ``IndexError`` in ``MetaModel._compute_importance``:
    ``meta_model_tb``/``meta_label_clf`` were fit with the FULL ``META_FEATURES``
    name list (13) while their X had 12 columns (``EXPECTED_MOVE_IN_META=False``
    drops one). Fix: pass the filtered ``TRAIN_META_FEATURES`` + a loud
    length guard in ``MetaModel.fit``.
  * ``horizon-60d`` / ``horizon-90d`` → ``RuntimeError: 0 rows survived the
    research-signal join``: a 60/90d label window predates the short
    signals.json history, so every row dropped. Fix: ``RESEARCH_FEATURES_IN_META``
    flag drops the research-derived meta features AND skips the join, so the
    spec trains on the years-deep price/macro features (validate-only —
    ``select_winner``'s horizon filter keeps it out of the 21d serving slot).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from model.meta_model import META_FEATURES, RESEARCH_META_FEATURES, MetaModel
from training.meta_trainer import build_train_meta_features
from training import model_zoo


# ── 2b: research-free meta-feature filtering ────────────────────────────


def test_research_meta_features_subset_of_meta_features():
    assert RESEARCH_META_FEATURES
    assert all(f in META_FEATURES for f in RESEARCH_META_FEATURES)


def test_build_meta_features_default_keeps_research():
    feats = build_train_meta_features(resid_mom_enabled=False)
    assert all(f in feats for f in RESEARCH_META_FEATURES)


def test_build_meta_features_research_free_drops_research():
    feats = build_train_meta_features(
        resid_mom_enabled=False, research_features_in_meta=False
    )
    assert not any(f in feats for f in RESEARCH_META_FEATURES)
    # Non-research columns (macro / model outputs) survive.
    assert "macro_spy_20d_return" in feats
    assert "momentum_score" in feats


def test_research_features_in_meta_config_default_true():
    assert hasattr(cfg, "RESEARCH_FEATURES_IN_META")
    assert cfg.RESEARCH_FEATURES_IN_META is True


def test_research_features_in_meta_is_zoo_override():
    assert "RESEARCH_FEATURES_IN_META" in model_zoo._ALLOWED_OVERRIDES


# ── 2a: MetaModel.fit length guard (the sota IndexError class) ───────────


def test_metamodel_fit_raises_on_feature_name_length_mismatch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    mm = MetaModel(alpha=1.0)
    # 4 names for a 3-column X — the exact mismatch sota hit (13 vs 12).
    with pytest.raises(ValueError, match="feature_names"):
        mm.fit(X, y, feature_names=["a", "b", "c", "d"])


def test_metamodel_fit_ok_when_names_match_columns():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] * 0.5 + rng.normal(size=50) * 0.1
    mm = MetaModel(alpha=1.0)
    mm.fit(X, y, feature_names=["a", "b", "c"])  # must not raise
    assert mm._feature_names == ["a", "b", "c"]


# ── 2c: promo label no longer reads as "the switch is off" ──────────────


def test_promo_label_does_not_say_auto_promote_off():
    src = (
        open(os.path.join(os.path.dirname(__file__), "..", "training", "train_handler.py"))
        .read()
    )
    assert "auto-promote off" not in src
    assert "model-zoo select_winner decides promotion" in src
