"""Guard: GBMScorer.load must accept both `feature_names` (current save
format) and `feature_list` (pre-2026-04-13 v2 save format).

Background: the backtester's predictor-backtest mode (`synthetic/
predictor_backtest.py:226`) loads `predictor/weights/gbm_latest.txt`
which is a v2 artifact written 2026-03-28, pre-dating the v2→v3
meta-model rip (2026-04-13). The v2 metadata JSON stored the trained
feature list under the key `feature_list`; the current save path
(lines 336 of gbm_scorer.py) writes `feature_names`. Without backwards
compatibility, every Saturday SF backtester step aborts with:

    RuntimeError: Loaded model has no feature_names metadata —
    cannot align input features.

This broke the 2026-04-20 Saturday SF dry-run (predictor_stats.json
status=error). Fix is a two-key fallback in `GBMScorer.load` that
prefers `feature_names` but falls through to `feature_list` when the
former is absent.

When the v2 gbm_latest.txt artifact is finally retired from S3 (see
ROADMAP P2 "v2 legacy artifact cleanup"), this backwards-compat can
be removed along with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _make_saved_gbm(tmp_path: Path, meta_payload: dict) -> Path:
    """Train a tiny LightGBM booster and save it with a custom meta.json."""
    try:
        import lightgbm as lgb
    except ImportError:
        pytest.skip("lightgbm not installed")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((256, 3)).astype(np.float32)
    y = rng.standard_normal(256).astype(np.float32)
    train = lgb.Dataset(X, label=y, feature_name=["a", "b", "c"])
    booster = lgb.train(
        {"objective": "regression", "verbose": -1, "num_leaves": 7},
        train,
        num_boost_round=5,
    )
    model_path = tmp_path / "model.txt"
    booster.save_model(str(model_path))
    (tmp_path / "model.txt.meta.json").write_text(json.dumps(meta_payload))
    return model_path


def test_load_reads_feature_names_key(tmp_path):
    """Current save format stores the list under feature_names."""
    from model.gbm_scorer import GBMScorer
    model_path = _make_saved_gbm(tmp_path, {
        "gbm_version": "test",
        "feature_names": ["a", "b", "c"],
        "best_iteration": 5,
        "val_ic": 0.02,
    })
    scorer = GBMScorer.load(model_path)
    assert scorer.feature_names == ["a", "b", "c"]


def test_load_falls_back_to_feature_list_key(tmp_path):
    """Pre-2026-04-13 v2 format (gbm_latest.txt 2026-03-28) stored
    the list under feature_list. Must still load."""
    from model.gbm_scorer import GBMScorer
    model_path = _make_saved_gbm(tmp_path, {
        "gbm_version": "GBM-v29",
        "feature_list": ["a", "b", "c"],
        "best_iteration": 29,
        "val_ic": 0.017474,
    })
    scorer = GBMScorer.load(model_path)
    assert scorer.feature_names == ["a", "b", "c"]


def test_load_prefers_feature_names_when_both_present(tmp_path):
    """If both keys exist (shouldn't happen, but be deterministic),
    the current key wins so a migrated artifact is preferred."""
    from model.gbm_scorer import GBMScorer
    model_path = _make_saved_gbm(tmp_path, {
        "gbm_version": "test",
        "feature_names": ["x", "y"],
        "feature_list": ["a", "b", "c"],  # stale, should be ignored
        "best_iteration": 5,
        "val_ic": 0.02,
    })
    scorer = GBMScorer.load(model_path)
    assert scorer.feature_names == ["x", "y"]


def test_load_returns_empty_when_neither_key_present(tmp_path):
    """Missing both keys — current behavior stays the same: empty list.
    The backtester's predictor-backtest still raises a clear error on
    empty feature_names; this test guards against the fallback
    accidentally populating with a default list."""
    from model.gbm_scorer import GBMScorer
    model_path = _make_saved_gbm(tmp_path, {
        "gbm_version": "test",
        "best_iteration": 5,
        "val_ic": 0.02,
    })
    scorer = GBMScorer.load(model_path)
    assert scorer.feature_names == []
