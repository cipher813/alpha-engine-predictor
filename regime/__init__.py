"""
regime/ — Quantitative regime substrate (v3).

Standalone regime model submodule, independent of the L1/L2 meta
ensemble in ``model/``. Runs as a Saturday SF Lambda **before** Research
and writes an S3 artifact the macro economist agent consumes as a
strong prior. The macro agent remains the final regime authority; this
substrate is informational scaffolding.

Architectural placement: model components (HMM fit, BOCPD, composite,
substrate writer) live here in alpha-engine-predictor. Feature inputs
(macros — VIX, HY OAS, yield curve slope, breadth, SPY returns) are
sourced from alpha-engine-data. This is the institutional
producer/consumer split: features in data, models in predictor.

Three independent quant components ensemble into the substrate JSON:

- ``hmm``       — 3-state Gaussian HMM (Hamilton 1989). Filter-only at
                  inference (no look-ahead). Smoother reserved for T1
                  retrospective ground-truth labeling with explicit lag.
- ``composite`` — AQR-style risk-on/risk-off macro z-score intensity.
                  Pure rule-based, zero estimation risk.
- ``bocpd``     — Adams & MacKay 2007 Bayesian online change-point
                  detection. Surfaces regime transitions.

The substrate orchestrator (``substrate.py``) assembles all three plus
guardrail flags (mirroring the macro agent's ``_validate_regime`` rules)
and writes the canonical-shape eval_artifacts JSON to S3.

Distinct from ``model/regime_predictor.py`` (v1, retired 2026-04-16)
and the parked v2 work — v3 is the third-attempt continuous-conditioning
architecture; v1/v2 were discrete classifiers killed by accuracy gates.

Reference: ``alpha-engine-docs/private/regime-v3-260514.md``.
"""

from .composite import compute_composite_intensity
from .hmm import HMMRegimeClassifier, REGIME_STATES
from .bocpd import BOCPDDetector
from .substrate import (
    REGIME_FEATURES,
    SCHEMA_VERSION,
    build_regime_substrate,
    write_regime_substrate,
)

__all__ = [
    "compute_composite_intensity",
    "HMMRegimeClassifier",
    "REGIME_STATES",
    "BOCPDDetector",
    "REGIME_FEATURES",
    "SCHEMA_VERSION",
    "build_regime_substrate",
    "write_regime_substrate",
]
