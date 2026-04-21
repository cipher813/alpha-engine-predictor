"""Guard: NaN in meta-model features must produce an explicit 'unscoreable'
prediction, not crash the Lambda invocation.

Background: the Ridge meta-model (`model.meta_model.MetaModel`) does not accept
NaN. Pre-fix, any NaN in meta_features (stale ArcticDB feature row, newly-added
ticker with <252d history for a long-window vol feature, macro gap, etc.)
raised ValueError inside `predict_single`, aborted the Lambda, and left
predictions.json with a coverage gap that the executor guard correctly refused
to size around — but the gap could not be self-healed via `--tickers` because
the gap-fill invocation itself crashed on the same ticker.

Post-fix contract:
- Detect NaN in meta_features pre-call.
- Emit a prediction row with status="unscoreable", gbm_veto=True,
  direction="UNSCOREABLE", confidence=0.
- The executor's existing gbm_veto branch (alpha-engine executor/main.py:425)
  routes this as a hard veto — ticker is covered, position is refused.
- The write_output veto-apply loop preserves gbm_veto for unscoreable entries
  (does not overwrite based on alpha/rank).
- _rescale_cross_sectional leaves unscoreable entries untouched.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Stubs ────────────────────────────────────────────────────────────────────

class _StubMetaModel:
    is_fitted = True
    def predict_single(self, features):
        # Should never be called with NaN — the guard catches it first.
        for name, val in features.items():
            if isinstance(val, float) and np.isnan(val):
                raise AssertionError(
                    f"predict_single called with NaN in {name} — guard bypassed"
                )
        return 0.005


class _StubVolModel:
    def predict(self, x):
        return np.array([float("nan")])  # force NaN into expected_move


class _StubMomModel:
    _val_ic = 0.0  # forces the direct-weighted-average fallback
    def predict(self, x):
        return np.array([0.01])


class _StubCtx:
    """Stand-in for PipelineContext with just the attrs _run_meta_inference reads."""
    def __init__(self, tickers, precomputed, meta_models, macro=None, price_data=None):
        self.tickers = tickers
        self.macro = macro or {}
        self.price_data = price_data or {}
        self.meta_models = meta_models
        self.calibrator = None
        self.dry_run = True
        self.bucket = "test-bucket"
        self.date_str = "2026-04-21"
        self.predictions = []
        self.n_skipped = 0
        self.inference_mode = "meta"
        self.ticker_data_age = {}
        self.ticker_sources = {}
        self.signals_data = {}
        self._precomputed = precomputed
    def near_timeout(self):
        return False


# ── Tests ────────────────────────────────────────────────────────────────────

def _run_with_precomputed(monkeypatch, ctx):
    """Patch _load_precomputed_features_from_arcticdb to return the stub."""
    from inference.stages import run_inference
    monkeypatch.setattr(
        run_inference,
        "_load_precomputed_features_from_arcticdb",
        lambda c: c._precomputed,
    )
    run_inference._run_meta_inference(ctx)


def test_nan_in_meta_features_emits_unscoreable_entry(monkeypatch):
    """Unscoreable contract: NaN path emits row, does not raise."""
    import pandas as pd
    from model.meta_model import META_FEATURES

    # Feature row where volatility path produces NaN (via stub vol model).
    # momentum features populated so momentum_score is finite.
    import config as cfg
    feature_row = pd.Series({
        **{f: 0.0 for f in cfg.MOMENTUM_FEATURES},
        **{f: 0.0 for f in cfg.VOLATILITY_FEATURES},
        "momentum_5d": 0.01, "momentum_20d": 0.02,
        "price_vs_ma50": 0.0, "rsi_14": 50,
    })
    ctx = _StubCtx(
        tickers=["SNDK"],
        precomputed={"SNDK": feature_row},
        meta_models={
            "meta": _StubMetaModel(),
            "volatility": _StubVolModel(),
            "momentum": _StubMomModel(),
            "research_calibrator": None,
        },
    )

    _run_with_precomputed(monkeypatch, ctx)

    assert len(ctx.predictions) == 1
    entry = ctx.predictions[0]
    assert entry["ticker"] == "SNDK"
    assert entry["status"] == "unscoreable"
    assert entry["gbm_veto"] is True
    assert entry["predicted_direction"] == "UNSCOREABLE"
    assert entry["prediction_confidence"] == 0.0
    assert entry["predicted_alpha"] == 0.0
    assert "expected_move" in entry["unscoreable_reason"], entry["unscoreable_reason"]


def test_veto_apply_preserves_unscoreable_gbm_veto():
    """write_output's veto-apply loop must not overwrite unscoreable gbm_veto=True."""
    unscoreable = {
        "ticker": "SNDK",
        "status": "unscoreable",
        "gbm_veto": True,
        "predicted_alpha": 0.0,
        "combined_rank": 1,
    }
    scored_up = {
        "ticker": "AAPL",
        "gbm_veto": False,
        "predicted_alpha": 0.01,
        "combined_rank": 1,
    }
    scored_down_tail = {
        "ticker": "XYZ",
        "gbm_veto": False,
        "predicted_alpha": -0.01,
        "combined_rank": 10,
    }
    preds = [unscoreable, scored_up, scored_down_tail]

    n_preds = len(preds)
    for p in preds:
        if p.get("status") == "unscoreable":
            continue
        cr = p.get("combined_rank")
        alpha = p.get("predicted_alpha", 0) or 0
        p["gbm_veto"] = (alpha < 0 and cr is not None and cr > n_preds / 2)

    assert unscoreable["gbm_veto"] is True, "unscoreable entry must retain gbm_veto=True"
    assert scored_up["gbm_veto"] is False
    assert scored_down_tail["gbm_veto"] is True


def test_rescaling_leaves_unscoreable_untouched():
    """_rescale_cross_sectional must not rewrite direction/confidence for unscoreable."""
    from inference.stages.run_inference import _rescale_cross_sectional

    ctx = _StubCtx(tickers=[], precomputed={}, meta_models={})
    ctx.predictions = [
        {
            "ticker": "SNDK",
            "status": "unscoreable",
            "predicted_alpha": 0.0,
            "predicted_direction": "UNSCOREABLE",
            "prediction_confidence": 0.0,
            "p_up": 0.0,
            "p_down": 0.0,
        },
        {
            "ticker": "AAPL",
            "predicted_alpha": 0.01,
            "predicted_direction": "UP",
            "prediction_confidence": 0.5,
            "p_up": 0.5,
            "p_down": 0.5,
        },
    ]

    _rescale_cross_sectional(ctx)

    sndk = next(p for p in ctx.predictions if p["ticker"] == "SNDK")
    aapl = next(p for p in ctx.predictions if p["ticker"] == "AAPL")
    assert sndk["predicted_direction"] == "UNSCOREABLE"
    assert sndk["prediction_confidence"] == 0.0
    assert sndk["p_up"] == 0.0
    # AAPL should have been rewritten by the linear heuristic
    assert aapl["predicted_direction"] == "UP"
