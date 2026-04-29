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
    meta_models: dict = field(default_factory=dict)  # v3.0: {"momentum": GBMScorer, "volatility": GBMScorer, "regime": RegimePredictor, "meta": MetaModel, ...}

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
