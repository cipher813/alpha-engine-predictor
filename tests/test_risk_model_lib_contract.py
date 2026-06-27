"""Consumer contract: predictor's risk_model shim ⇄ nousergon_lib.quant.factor_risk_xs.

The factor-risk math (C.2a) was lifted to the shared lib (LV1-AE leverage arc,
2026-06-03); ``risk_model/__init__.py`` is now a re-export shim over
``nousergon_lib.quant.factor_risk_xs``. This pins the contract the predictor
depends on so a lib version that drops/renames a consumed symbol fails here rather
than at Saturday training time. The exhaustive math tests live in the lib
(``test_quant_factor_risk_xs.py`` there); ``test_risk_model_persist.py`` exercises
the real ``build_factor_risk_model`` through the shim end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import risk_model


def test_shim_re_exports_public_surface():
    for name in (
        "cross_sectional_factor_returns",
        "build_factor_returns_series",
        "estimate_factor_covariance",
        "estimate_idiosyncratic_variance",
        "build_factor_risk_model",
    ):
        assert hasattr(risk_model, name), f"risk_model shim missing {name}"


def test_shim_is_identity_with_lib():
    from nousergon_lib.quant import factor_risk_xs

    assert risk_model.build_factor_risk_model is factor_risk_xs.build_factor_risk_model


def test_build_factor_risk_model_smoke():
    rng = np.random.RandomState(0)
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    tickers = [f"T{i}" for i in range(20)]
    returns_panel = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(tickers))), index=dates, columns=tickers)
    loadings = pd.DataFrame(rng.normal(0, 1, (len(tickers), 3)), index=tickers, columns=["mom", "val", "qual"])
    loadings_by_date = {dates[0]: loadings}

    out = risk_model.build_factor_risk_model(returns_panel, loadings_by_date, min_cov_obs=10, min_idio_obs=10)
    assert set(out) == {"factor_returns", "residuals", "F", "D", "metadata"}
    assert out["F"].shape == (4, 4)  # market intercept + 3 factors
    assert out["metadata"]["n_tickers"] == 20
