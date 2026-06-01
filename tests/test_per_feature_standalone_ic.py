"""Tests for per_feature_standalone_ic (W4 watch-item, L4469).

Standalone cross-sectional alpha-IC of each meta-feature column vs the realized
label. Surfaces whether a high-coefficient input (e.g. expected_move, a vol L1)
actually predicts cross-sectional ALPHA or is in-sample dominance only.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.leakfree_meta_ic import per_feature_standalone_ic


def test_signal_column_high_ic_noise_column_zero():
    rng = np.random.default_rng(0)
    n_dates, n_names = 60, 25
    dates = np.repeat(np.arange(n_dates), n_names)
    y = rng.normal(0, 1, n_dates * n_names)
    signal = y + rng.normal(0, 0.1, y.shape)   # near-perfect predictor of y
    noise = rng.normal(0, 1, y.shape)           # unrelated to y
    X = np.column_stack([signal, noise])
    out = per_feature_standalone_ic(X, y, dates, ["signal", "noise"])
    assert set(out.keys()) == {"signal", "noise"}
    assert out["signal"]["xsec_ic"] > 0.8
    assert abs(out["noise"]["xsec_ic"]) < 0.2
    assert out["signal"]["n_dates"] == n_dates


def test_constant_column_yields_no_ic():
    rng = np.random.default_rng(1)
    n_dates, n_names = 40, 20
    dates = np.repeat(np.arange(n_dates), n_names)
    y = rng.normal(0, 1, n_dates * n_names)
    const = np.ones(y.shape)  # zero cross-sectional variance → no rank IC
    out = per_feature_standalone_ic(const[:, None], y, dates, ["const"])
    assert out["const"]["xsec_ic"] is None
    assert out["const"]["n_dates"] == 0
