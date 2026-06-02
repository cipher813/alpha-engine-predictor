"""
inference/pipeline.py — Staged prediction pipeline orchestrator.

Breaks the monolithic main() into discrete stages that share state via
PipelineContext. Critical stages abort the pipeline; data stages degrade
gracefully and log warnings.

The pipeline is the primary execution path. daily_predict.main() delegates
to run_pipeline() and remains the stable entry point for handler.py.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg

log = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared state passed through pipeline stages."""

    # ── Inputs (set before pipeline starts) ──────────────────────────────────
    date_str: str = ""
    bucket: str = ""
    dry_run: bool = False
    local: bool = False
    model_type: str = "gbm"
    watchlist_path: Optional[str] = None
    fd: object = None  # file descriptor for dry-run output

    # Supplemental scoring: when non-empty, score ONLY these tickers and merge
    # the result into the existing predictions/{date}.json (rather than overwriting).
    # Used by the Step Function's coverage-gap re-invocation path when Research
    # produces buy_candidates after the first PredictorInference has already run.
    explicit_tickers: list = field(default_factory=list)

    # ── Timing ───────────────────────────────────────────────────────────────
    start_ts: float = 0.0
    soft_timeout_s: int = 780

    # ── Model state (set by load_model) ──────────────────────────────────────
    scorer: object = None           # primary GBM scorer
    mse_scorer: object = None       # MSE model
    rank_scorer: object = None      # Lambdarank model
    model: object = None            # MLP model
    checkpoint: dict = field(default_factory=dict)
    inference_mode: str = "mse"
    model_version: str = "unknown"
    val_loss: float = float("nan")
    calibrator: object = None       # PlattCalibrator (Platt scaling / isotonic)
    cat_scorer: object = None       # CatBoostScorer (for LGB-Cat ensemble)
    blend_weights: dict = None      # {"lgb": 0.5, "cat": 0.5}
    horizon_scorers: dict = field(default_factory=dict)  # {1: GBMScorer, 10: GBMScorer, 20: GBMScorer}
    meta_models: dict = field(default_factory=dict)  # v3.0: {"volatility": GBMScorer, "research_calibrator": ResearchCalibrator, "meta": MetaModel, ...} — momentum is deterministic (model/momentum_scorer.py), not loaded here

    # ── Universe (set by load_universe) ──────────────────────────────────────
    tickers: list = field(default_factory=list)
    ticker_sources: dict = field(default_factory=dict)
    signals_data: dict = field(default_factory=dict)
    sector_map: dict = field(default_factory=dict)

    # ── Prices (set by load_prices) ──────────────────────────────────────────
    price_data: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    ticker_data_age: dict = field(default_factory=dict)

    # ── Alternative data (set by fetch_alt_data) ─────────────────────────────
    earnings_all: dict = field(default_factory=dict)
    revision_all: dict = field(default_factory=dict)
    options_all: dict = field(default_factory=dict)
    fundamental_all: dict = field(default_factory=dict)

    # ── Results (set by run_inference) ───────────────────────────────────────
    predictions: list = field(default_factory=list)
    n_skipped: int = 0
    n_nan_imputed_tickers: int = 0
    store_rows: list = field(default_factory=list)  # feature store rows

    # ── GBM feature columns (set by run_inference) ───────────────────────────
    gbm_feature_cols: list = field(default_factory=list)

    # ── Regime substrate (set by run_inference, consumed by write_output) ────
    # Single market-wide composite intensity_z computed in run_inference's
    # regime_features_df step. Stamped on ctx so write_output's veto path
    # (Stage D' Wire 4) can apply a regime-conditional confidence threshold
    # without re-running the feature panel.
    regime_intensity_z: Optional[float] = None

    # ── Regime fast signal (set by regime_fast_signal stage, F2) ─────────────
    # Discrete forced-bear latch from the daily BOCPD circuit-breaker
    # (regime-fast-signal-260515.md). Consumed in-process by write_output's
    # veto clamp. Defaults False so a skipped/failed fast-signal stage
    # degrades the veto to its legacy (Wire 4 / discrete) behavior.
    regime_forced_bear: bool = False

    # ── Drawdown regime leg (set by regime_fast_signal stage; observe-only) ──
    # Composed effective regime from the deterministic drawdown leg
    # (regime-drawdown-hysteresis-260518.md). PR 2 = producer/observe
    # only: NO consumer reads this yet (the executor/predictor-veto
    # consumer PRs, gated `drawdown_regime_enabled` default-off, will).
    # Stamped from the drawdown artifact so the consumer PRs read it
    # in-process, mirroring `regime_forced_bear`.
    #
    # Type-system separation (v0.42.0 Phase 2A —
    # caution-regime-retirement-260528.md): drawdown_effective_regime is
    # the macro-axis projection (3-class) kept for backward-compat;
    # drawdown_protective_severity is the canonical drawdown-axis ordinal
    # (0=risk_on, 1=caution, 2=risk_off/alpha_bleed) consumed by veto
    # / sizing. New consumers should read the severity; legacy callsites
    # may still read drawdown_effective_regime during the transition.
    drawdown_effective_regime: Optional[str] = None
    drawdown_protective_severity: Optional[int] = None

    # ── Cross-sectional level-neutralization (set by run_inference; L4487) ───
    # Observe block from inference.level_neutralization: {enabled, applied,
    # xsec_mean_removed, n_direction_flips, direction_skew_raw/_centered}.
    # Emitted into metrics.json + predictions.json by write_output. None until
    # the neutralization pass runs. Observe-gated by cfg.XSEC_DEMEAN_ALPHA_
    # ENABLED — when False, live fields are byte-identical and this just
    # records what centering WOULD do.
    level_neutralization: Optional[dict] = None

    def near_timeout(self) -> bool:
        """Check if we're nearing the Lambda soft timeout."""
        elapsed = _time.monotonic() - self.start_ts
        if elapsed > self.soft_timeout_s:
            log.warning("Soft timeout reached (%.0fs / %ds)", elapsed, self.soft_timeout_s)
            return True
        return False

    def elapsed_seconds(self) -> float:
        return _time.monotonic() - self.start_ts


class PipelineAbort(Exception):
    """Raised by a stage to cleanly abort the pipeline (not an error)."""
    pass


class PipelineHardFail(Exception):
    """Raised by a stage when the pipeline must fail unconditionally — even
    from a non-critical stage. Propagates to Lambda/CLI as a real exception so
    Step Function Catch fires and downstream (executor) never sees a
    partially-written predictions.json.

    Distinguished from PipelineAbort (clean skip) and from regular exceptions
    (which non-critical stages log-and-continue on).
    """
    pass


# ── Stage registry ───────────────────────────────────────────────────────────

# Each stage: (name, module_path, critical)
# Critical stages abort the pipeline on failure. Non-critical stages log and continue.
STAGES = [
    ("load_model",     "inference.stages.load_model",     True),
    ("load_universe",  "inference.stages.load_universe",  True),
    ("load_prices",    "inference.stages.load_prices",    True),
    ("fetch_alt_data", "inference.stages.fetch_alt_data", False),
    ("run_inference",  "inference.stages.run_inference",  True),
    # regime-fast-signal-260515.md. Runs AFTER run_inference (which
    # stamps ctx.regime_intensity_z) and BEFORE write_output so the
    # F2 veto clamp can read ctx.regime_forced_bear in-process — no
    # extra S3 read, no T-1 staleness. F1 originally placed this LAST
    # (observe-only didn't care about order); F2's in-process veto
    # consumption requires it upstream of write_output. Non-critical:
    # a failure here defaults ctx.regime_forced_bear=False and never
    # affects predictions.
    ("regime_fast_signal", "inference.stages.regime_fast_signal", False),
    ("write_output",   "inference.stages.write_output",   False),
]


def run_pipeline(ctx: PipelineContext) -> None:
    """Execute all pipeline stages in sequence.

    Critical stages abort the pipeline on failure. Non-critical stages
    log a warning and continue with degraded output.
    """
    for stage_name, module_path, critical in STAGES:
        log.info("── Stage: %s ──", stage_name)
        try:
            import importlib
            mod = importlib.import_module(module_path)
            mod.run(ctx)
        except PipelineAbort as abort:
            log.info("Pipeline aborted cleanly by %s: %s", stage_name, abort)
            return
        except PipelineHardFail as hf:
            # Propagate unconditionally — do NOT honor the critical flag.
            log.error("HARD FAIL from %s: %s", stage_name, hf, exc_info=True)
            raise
        except Exception as exc:
            if critical:
                log.error("CRITICAL stage %s failed: %s", stage_name, exc, exc_info=True)
                raise
            else:
                log.warning("Non-critical stage %s failed: %s — continuing", stage_name, exc)

    log.info("Pipeline complete for %s (%.1fs)", ctx.date_str, ctx.elapsed_seconds())
