"""Regression: lock that the v2 single-GBM + multi-horizon + CatBoost
inference paths stay deleted.

Background:
    v2 inference (``_load_gbm`` in ``inference/stages/load_model.py``) was
    the dispatcher's else-branch when ``META_MODEL_ENABLED`` was False or
    ``model_type`` wasn't ``"gbm"``. Production has had ``META_MODEL_ENABLED=
    True`` since 2026-04-01 (training switch) and 2026-04-13 (training
    dispatcher collapse), so the v2 inference path was unreachable. The v2
    cleanup PR (2026-04-27) deleted it along with three siblings:
      - ``load_gbm_local`` / ``load_gbm_s3`` helpers (only used by ``_load_gbm``)
      - ``model/catboost_scorer.py`` (only imported by ``_load_gbm``)
      - 9 dead config vars (only consumed by ``_load_gbm``)

These tests prevent the dead surface from coming back via a half-aware
revert or a "quick add it back" patch.
"""

from __future__ import annotations

from pathlib import Path

import config as cfg


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


# ── Module + symbol absences ──────────────────────────────────────────────────


def test_load_model_no_v2_helpers():
    """``inference/stages/load_model.py`` must not redefine ``_load_gbm`` /
    ``load_gbm_local`` / ``load_gbm_s3``."""
    src = _src("inference/stages/load_model.py")
    assert "def _load_gbm(" not in src, (
        "v2 _load_gbm dispatcher resurrected. Production routes "
        "everything through _load_meta_models — the else-branch was dead."
    )
    assert "def load_gbm_local(" not in src, (
        "v2 load_gbm_local helper resurrected. Only consumer was _load_gbm "
        "which is gone — _load_meta_models has its own _dl helper."
    )
    assert "def load_gbm_s3(" not in src, (
        "v2 load_gbm_s3 helper resurrected. Only consumer was _load_gbm "
        "which is gone — _load_meta_models has its own _dl helper."
    )


def test_run_dispatcher_is_unconditional():
    """``run()`` must call ``_load_meta_models`` unconditionally — the
    META_MODEL_ENABLED gate was the dead-code surface."""
    src = _src("inference/stages/load_model.py")
    # Look for the gating pattern, not the symbol name (which legitimately
    # appears in the module docstring as historical context).
    assert 'getattr(cfg, "META_MODEL_ENABLED"' not in src, (
        "META_MODEL_ENABLED gate resurrected in load_model.py. The flag was "
        "the v2-vs-v3 toggle; v3 is the only path now and the gate is dead "
        "branching on a config value that is hardcoded True in production."
    )
    assert 'cfg.META_MODEL_ENABLED' not in src, (
        "Direct cfg.META_MODEL_ENABLED read resurrected in load_model.py — "
        "same dead-code class as the getattr form above."
    )


def test_catboost_module_deleted():
    """``model/catboost_scorer.py`` must not exist — its only importer was
    the deleted v2 ``_load_gbm`` block."""
    assert not (_REPO / "model" / "catboost_scorer.py").exists(), (
        "model/catboost_scorer.py resurrected. v2 cleanup deleted it after "
        "confirming its only importer (inference/stages/load_model.py "
        "_load_gbm CatBoost optional block) was gone."
    )


# ── Config-var absences ───────────────────────────────────────────────────────


def test_dead_config_vars_gone():
    """Config vars consumed only by deleted v2 paths must stay gone."""
    dead = [
        "GBM_ENSEMBLE_LAMBDARANK",
        "GBM_MSE_WEIGHTS_KEY",
        "GBM_MSE_WEIGHTS_META_KEY",
        "GBM_RANK_WEIGHTS_KEY",
        "GBM_RANK_WEIGHTS_META_KEY",
        "GBM_MODE_KEY",
        "GBM_WEIGHTS_KEY",  # Note: GBM_WEIGHTS_META_KEY stays — used by write_output._load_gbm_meta
        "MULTI_HORIZON_ENABLED",
        "MULTI_HORIZON_LIST",
        "CATBOOST_ENABLED",
        "CATBOOST_PARAMS",
        "CATBOOST_WEIGHTS_KEY",
        "CATBOOST_WEIGHTS_META_KEY",
    ]
    for name in dead:
        assert not hasattr(cfg, name), (
            f"cfg.{name} resurrected. v2 cleanup deleted it after confirming "
            f"its only consumer was the deleted ``_load_gbm`` path."
        )


def test_live_config_vars_kept():
    """Confirm the v3 meta path's config surface is intact (sanity check
    that the cleanup didn't over-delete)."""
    live = [
        "META_MODEL_ENABLED",
        "META_WEIGHTS_PREFIX",
        "GBM_TUNED_PARAMS",
        "MOMENTUM_GBM_TUNED_PARAMS",
        "CALIBRATION_ENABLED",
        "CALIBRATOR_WEIGHTS_KEY",
        "CALIBRATOR_WEIGHTS_META_KEY",
        # Used by write_output._load_gbm_meta (best-effort, separate cleanup)
        "GBM_WEIGHTS_META_KEY",
    ]
    for name in live:
        assert hasattr(cfg, name), (
            f"cfg.{name} missing after v2 cleanup — over-deletion. v3 meta "
            f"path needs this."
        )


# ── daily_predict.py re-export contract ───────────────────────────────────────


def test_daily_predict_no_v2_reexport():
    """``inference/daily_predict.py`` must not re-export the deleted v2
    helpers — that would re-introduce ImportError at module import time."""
    src = _src("inference/daily_predict.py")
    assert "load_gbm_local, load_gbm_s3" not in src, (
        "daily_predict.py still re-exports load_gbm_local/load_gbm_s3 from "
        "load_model — those helpers are gone, so the import would fail."
    )
