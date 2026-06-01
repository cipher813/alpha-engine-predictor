"""Tests for the W3.2 (L4469) leak-free per-horizon IC curve helper.

`leakfree_horizon_ic_curve` answers the operator's 5/21/60/90d question with the
honest (purged + embargoed, cross-sectional) IC at each horizon. Verifies the
contract (one entry per horizon, purge = h) and that a horizon whose realized
label is genuinely driven by the meta features scores a higher leak-free IC than
a pure-noise horizon.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.leakfree_meta_ic import leakfree_horizon_ic_curve


def _ridge_fit_predict(X_tr, y_tr, X_te):
    # Minimal OLS-ish fit/predict (the injected L2 in production is BayesianRidge).
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0)
    m.fit(X_tr, y_tr)
    return m.predict(X_te).ravel()


def _panel(n_dates=80, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates), n_names)
    X = rng.normal(0, 1, (n_dates * n_names, 3))
    return X, dates, rng


class TestContract:
    def test_one_entry_per_horizon_with_purge(self):
        X, dates, rng = _panel()
        labels = {h: rng.normal(0, 1, X.shape[0]) for h in (5, 21, 60)}
        curve = leakfree_horizon_ic_curve(
            X, labels, dates, fit_predict_fn=_ridge_fit_predict, embargo_days=0,
        )
        assert set(curve.keys()) == {"5d", "21d", "60d"}
        for h, key in ((5, "5d"), (21, "21d"), (60, "60d")):
            assert curve[key]["forward_days"] == h  # purge = the horizon

    def test_nonfinite_labels_handled(self):
        X, dates, rng = _panel()
        y = rng.normal(0, 1, X.shape[0])
        y[: X.shape[0] // 3] = np.nan  # tail-of-history style NaN
        curve = leakfree_horizon_ic_curve(
            X, {21: y}, dates, fit_predict_fn=_ridge_fit_predict,
        )
        assert "21d" in curve
        assert curve["21d"]["status"] in ("ok", "insufficient_folds")


class TestSignalVsNoise:
    def test_signal_horizon_beats_noise_horizon(self):
        # Horizon A's label is a strong linear function of the features (real,
        # leak-free predictable signal); horizon B's label is pure noise.
        X, dates, rng = _panel(n_dates=120, n_names=30, seed=3)
        beta = np.array([1.5, -1.0, 0.5])
        y_signal = X @ beta + rng.normal(0, 0.5, X.shape[0])  # high SNR
        y_noise = rng.normal(0, 1, X.shape[0])
        curve = leakfree_horizon_ic_curve(
            X, {21: y_signal, 60: y_noise}, dates,
            fit_predict_fn=_ridge_fit_predict, embargo_days=0,
        )
        ic_signal = curve["21d"]["xsec_ic"]
        ic_noise = curve["60d"]["xsec_ic"]
        assert curve["21d"]["status"] == "ok"
        assert ic_signal > 0.3          # strong leak-free signal recovered
        assert ic_signal > ic_noise + 0.2  # clearly beats the noise horizon
