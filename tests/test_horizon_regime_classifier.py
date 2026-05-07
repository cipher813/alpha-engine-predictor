"""Tests for the regime classifier used by the Track B per-regime IC breakdown.

Per the 2026-05-07 predictor audit Track B (PR 3/N): bull/neutral/bear
classification via macro_spy_20d_return thresholds at ±3%. Single-feature
heuristic chosen for transparency until the Tier-1 regime classifier is
built (audit §6.2 + §6.3 ROADMAP).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.meta_trainer import _classify_regime


class TestClassifyRegime:

    def test_strong_bull_above_threshold(self):
        assert _classify_regime({"macro_spy_20d_return": 0.10}) == "bull"

    def test_just_above_bull_threshold(self):
        # Exactly 0.03 is NOT above (strict >); just-above is bull.
        assert _classify_regime({"macro_spy_20d_return": 0.0301}) == "bull"

    def test_at_bull_threshold_is_neutral(self):
        # Exactly 0.03 should be neutral (not strictly > 0.03).
        assert _classify_regime({"macro_spy_20d_return": 0.03}) == "neutral"

    def test_strong_bear_below_threshold(self):
        assert _classify_regime({"macro_spy_20d_return": -0.10}) == "bear"

    def test_just_below_bear_threshold(self):
        # Exactly -0.03 is NOT below (strict <); just-below is bear.
        assert _classify_regime({"macro_spy_20d_return": -0.0301}) == "bear"

    def test_at_bear_threshold_is_neutral(self):
        # Exactly -0.03 should be neutral (not strictly < -0.03).
        assert _classify_regime({"macro_spy_20d_return": -0.03}) == "neutral"

    def test_zero_is_neutral(self):
        assert _classify_regime({"macro_spy_20d_return": 0.0}) == "neutral"

    def test_small_positive_is_neutral(self):
        # Within ±3% band — neutral.
        assert _classify_regime({"macro_spy_20d_return": 0.015}) == "neutral"

    def test_small_negative_is_neutral(self):
        assert _classify_regime({"macro_spy_20d_return": -0.015}) == "neutral"

    def test_missing_field_returns_neutral(self):
        # Defensive default when the macro feature isn't on the row at all.
        assert _classify_regime({}) == "neutral"

    def test_none_value_returns_neutral(self):
        assert _classify_regime({"macro_spy_20d_return": None}) == "neutral"

    def test_nan_value_returns_neutral(self):
        assert _classify_regime({"macro_spy_20d_return": float("nan")}) == "neutral"

    def test_non_numeric_value_returns_neutral(self):
        # Defensive against schema drift — string in a numeric field shouldn't crash.
        assert _classify_regime({"macro_spy_20d_return": "0.05"}) == "neutral"

    def test_only_uses_spy_20d_return(self):
        # Other macro features should NOT influence classification — the
        # heuristic is intentionally single-axis. Verify a bull SPY return
        # plus extreme VIX still classifies as bull.
        assert _classify_regime({
            "macro_spy_20d_return": 0.05,
            "macro_vix_level": 80.0,  # extreme VIX shouldn't matter
            "macro_market_breadth": 0.10,
        }) == "bull"

    def test_realistic_distribution_sample(self):
        # Sanity check on a realistic spread of values.
        rows = [
            {"macro_spy_20d_return": 0.08},   # bull
            {"macro_spy_20d_return": 0.04},   # bull
            {"macro_spy_20d_return": 0.02},   # neutral
            {"macro_spy_20d_return": -0.01},  # neutral
            {"macro_spy_20d_return": -0.04},  # bear
            {"macro_spy_20d_return": -0.10},  # bear
        ]
        regimes = [_classify_regime(r) for r in rows]
        assert regimes == ["bull", "bull", "neutral", "neutral", "bear", "bear"]
