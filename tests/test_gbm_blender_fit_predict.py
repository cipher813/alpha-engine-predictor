"""Tests for the W4.1 (L4469) nonlinear-L2 shadow blender factory.

`gbm_blender_fit_predict` returns a LightGBM fit_predict_fn that drops into
`leakfree_meta_oos_ic` exactly like the BayesianRidge one — the shadow used to
measure whether a nonlinear meta recovers more leak-free IC from the same L1s.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.leakfree_meta_ic import gbm_blender_fit_predict, leakfree_meta_oos_ic


def test_factory_returns_callable_with_correct_shape():
    rng = np.random.default_rng(0)
    fp = gbm_blender_fit_predict(n_estimators=20)
    X_tr, y_tr = rng.normal(0, 1, (300, 4)), rng.normal(0, 1, 300)
    X_te = rng.normal(0, 1, (50, 4))
    out = fp(X_tr, y_tr, X_te)
    assert out.shape == (50,)
    assert np.all(np.isfinite(out))


def test_nonlinear_blender_recovers_interaction_signal_via_leakfree():
    # Label has a genuine NONLINEAR (interaction/threshold) dependence on the
    # features that a linear meta cannot fully capture; the GBM blender, run
    # through the leak-free WF, should recover a positive cross-sectional IC.
    rng = np.random.default_rng(3)
    n_dates, n_names = 140, 30
    dates = np.repeat(np.arange(n_dates), n_names)
    X = rng.normal(0, 1, (n_dates * n_names, 3))
    # XOR-ish interaction + a threshold term — nonlinear in X.
    y = np.sign(X[:, 0]) * np.abs(X[:, 1]) + (X[:, 2] > 0.5) * 1.5 + rng.normal(0, 0.3, X.shape[0])
    res = leakfree_meta_oos_ic(
        X, y, dates,
        fit_predict_fn=gbm_blender_fit_predict(n_estimators=80),
        forward_days=21, embargo_days=0,
    )
    assert res["status"] == "ok"
    assert res["xsec_ic"] > 0.2  # nonlinear meta recovers the interaction signal
