"""risk_model — Barra-style structural factor risk model (C.2a).

**Re-export shim.** The pure-math implementation was lifted to the shared
``nousergon_lib.quant.factor_risk_xs`` (LV1-AE leverage arc, 2026-06-03) so the
predictor and robodashboard consume one Σ=B·F·Bᵀ+D engine instead of parallel
reimplementations. This shim preserves the predictor's ``risk_model`` import
surface — ``training/risk_model_persist.py`` and any other consumer keep working
unchanged; the implementation + its unit tests now live in the lib.

The model produces the (K×K) factor-return covariance **F** and the (N,)
idiosyncratic variance **D** that, with the (N×K) factor-loading matrix **B** from
alpha-engine-data C.1, give Σ = B·F·Bᵀ + D for the constrained MVO solver
(workstream C.3, alpha-engine). See the lib module for the full math + references.
"""

from __future__ import annotations

from nousergon_lib.quant.factor_risk_xs import (
    _MIN_OBS_OVER_K,
    build_factor_returns_series,
    build_factor_risk_model,
    cross_sectional_factor_returns,
    estimate_factor_covariance,
    estimate_idiosyncratic_variance,
)

__all__ = [
    "cross_sectional_factor_returns",
    "build_factor_returns_series",
    "estimate_factor_covariance",
    "estimate_idiosyncratic_variance",
    "build_factor_risk_model",
    "_MIN_OBS_OVER_K",
]
