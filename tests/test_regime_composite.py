"""Tests for regime/composite.py — composite z-score intensity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.composite import (
    DEFAULT_WEIGHTS,
    compute_composite_intensity,
)


def _build_history(n: int = 260, seed: int = 0) -> pd.DataFrame:
    """Synthetic 5y weekly macro history with realistic ranges."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "spy_20d_return": rng.normal(0.01, 0.04, n),
        "vix_level": rng.normal(18.0, 5.0, n).clip(min=9.0),
        "vix_term_slope": rng.normal(1.0, 0.3, n),
        "hy_oas_bps": rng.normal(400.0, 100.0, n).clip(min=200.0),
        "yield_curve_slope": rng.normal(0.5, 0.4, n),
        "market_breadth": rng.normal(0.55, 0.15, n).clip(0.0, 1.0),
    })


def test_composite_returns_required_keys() -> None:
    history = _build_history()
    current = {f: history[f].iloc[-1] for f in DEFAULT_WEIGHTS}
    result = compute_composite_intensity(current=current, history=history)
    assert {"intensity_z", "per_feature_z", "features_used"} <= set(result)
    assert isinstance(result["intensity_z"], float)
    assert isinstance(result["per_feature_z"], dict)
    assert isinstance(result["features_used"], list)


def test_composite_intensity_sign_under_stress() -> None:
    """High VIX + wide HY OAS + negative SPY return → positive intensity (risk-off)."""
    history = _build_history()
    current = {
        "spy_20d_return": -0.10,    # extreme drawdown
        "vix_level": 40.0,          # extreme stress
        "vix_term_slope": -0.5,     # backwardation
        "hy_oas_bps": 800.0,        # wide credit spreads
        "yield_curve_slope": -0.5,  # inverted
        "market_breadth": 0.20,     # narrow breadth
    }
    result = compute_composite_intensity(current=current, history=history)
    assert result["intensity_z"] > 1.0, (
        f"expected strong risk-off signal, got {result['intensity_z']}"
    )


def test_composite_intensity_sign_under_calm() -> None:
    """Low VIX + tight HY OAS + positive SPY return → negative intensity (risk-on)."""
    history = _build_history()
    current = {
        "spy_20d_return": 0.08,
        "vix_level": 11.0,
        "vix_term_slope": 1.5,
        "hy_oas_bps": 250.0,
        "yield_curve_slope": 1.2,
        "market_breadth": 0.85,
    }
    result = compute_composite_intensity(current=current, history=history)
    assert result["intensity_z"] < -1.0, (
        f"expected strong risk-on signal, got {result['intensity_z']}"
    )


def test_composite_handles_missing_features_gracefully() -> None:
    """Subset of features still returns a sensible composite — graceful degradation."""
    history = _build_history()
    current = {"vix_level": 30.0, "spy_20d_return": -0.05}
    result = compute_composite_intensity(current=current, history=history)
    assert set(result["features_used"]) == {"vix_level", "spy_20d_return"}
    assert np.isfinite(result["intensity_z"])


def test_composite_zero_when_no_history() -> None:
    """Empty history → intensity_z falls back to 0 (no signal to extract)."""
    history = pd.DataFrame(columns=list(DEFAULT_WEIGHTS))
    current = {f: 1.0 for f in DEFAULT_WEIGHTS}
    result = compute_composite_intensity(current=current, history=history)
    assert result["intensity_z"] == 0.0


def test_composite_per_feature_z_signs() -> None:
    """Each per-feature z-score has the correct sign relative to history."""
    history = _build_history()
    # Pick a current row that's at the high end of every feature
    current = {f: history[f].quantile(0.99) for f in DEFAULT_WEIGHTS}
    result = compute_composite_intensity(current=current, history=history)
    # All raw z-scores should be positive (high-end input vs historical
    # distribution); per-feature signs are pre-weight.
    for feat, z in result["per_feature_z"].items():
        assert z > 0, f"expected positive z for {feat} at q99, got {z}"


def test_composite_rolling_window_caps_history() -> None:
    """rolling_window_weeks=N truncates history to the most recent N rows."""
    history = _build_history(n=500)
    current = {f: history[f].iloc[-1] for f in DEFAULT_WEIGHTS}
    full = compute_composite_intensity(current=current, history=history)
    windowed = compute_composite_intensity(current=current, history=history, rolling_window_weeks=52)
    # The two intensities should differ because the normalization
    # mean/std change with the window.
    assert full["intensity_z"] != windowed["intensity_z"]
