"""Tests for the stratified-per-regime variant of the output-distribution gate.

Per the 2026-05-07 predictor audit Phase 2a-PROMOTE Option C: runs the
four invariants check independently on each regime's slice. Catches
regime-conditional degeneracy where the model collapses in only one
regime — a class the synthetic-sweep + live-batch variants both miss
(both can pass with the model healthy on average but degenerate in,
say, bear regime alone).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.output_distribution_gate import validate_stratified_per_regime


def _diverse_predictions(n: int = 30):
    """Spread of p_ups across [0.10, 0.85] — passes all four invariants.

    Critically, interleaves UP and DOWN so any contiguous slice has a
    balanced direction distribution (avoids triggering direction_skew
    when slicing for per-regime tests).
    """
    preds = []
    # Alternate above-and-below 0.5 by interleaving two ramps.
    half = n // 2
    down_ramp = [0.10 + (i / max(1, half - 1)) * 0.39 for i in range(half)]  # [0.10, 0.49]
    up_ramp = [0.51 + (i / max(1, n - half - 1)) * 0.34 for i in range(n - half)]  # [0.51, 0.85]
    # Interleave so slices are balanced.
    for i in range(n):
        if i % 2 == 0 and (i // 2) < len(down_ramp):
            p_up = down_ramp[i // 2]
        elif (i // 2) < len(up_ramp):
            p_up = up_ramp[i // 2]
        else:
            p_up = down_ramp[(i // 2) % len(down_ramp)]
        preds.append({
            "p_up": round(p_up, 4),
            "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
        })
    return preds


def _plateau_predictions(n: int = 30, plateau_value: float = 0.458):
    """All predictions clamped to plateau_value (uniqueness=1)."""
    return [
        {"p_up": plateau_value, "predicted_direction": "UP" if plateau_value >= 0.5 else "DOWN"}
        for _ in range(n)
    ]


# ── All-clean: every regime healthy ──────────────────────────────────────

class TestAllRegimesHealthy:

    def test_all_three_regimes_healthy_passes(self):
        preds = _diverse_predictions(90)  # 30 per regime
        regimes = ["bull"] * 30 + ["neutral"] * 30 + ["bear"] * 30
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is True
        assert result.failed_check is None
        assert result.metrics["n_regimes_evaluated"] == 3

    def test_per_regime_metrics_populated(self):
        preds = _diverse_predictions(90)
        regimes = ["bull"] * 30 + ["neutral"] * 30 + ["bear"] * 30
        result = validate_stratified_per_regime(preds, regimes)
        for regime in ("bull", "neutral", "bear"):
            assert regime in result.metrics["per_regime"]
            assert result.metrics["per_regime"][regime]["evaluated"] is True
            assert result.metrics["per_regime"][regime]["passed"] is True


# ── One regime collapses ─────────────────────────────────────────────────

class TestSingleRegimeCollapse:

    def test_bear_regime_plateau_blocks_promotion(self):
        # Bull + neutral healthy; bear is the 5/07-class plateau (all
        # tickers at p_up=0.458). Aggregate-pool gates would pass since
        # 60 of 90 tickers look fine. Per-regime gate catches the bear
        # collapse.
        preds = _diverse_predictions(60) + _plateau_predictions(30, 0.458)
        regimes = ["bull"] * 30 + ["neutral"] * 30 + ["bear"] * 30
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is False
        assert result.failed_check == "bear/unique_p_up"
        assert "bear" in result.reason
        assert "(n=30)" in result.reason

    def test_bull_regime_collapse_blocks(self):
        # Same shape but with bull in the plateau slot.
        preds = _plateau_predictions(30, 0.458) + _diverse_predictions(60)
        regimes = ["bull"] * 30 + ["neutral"] * 30 + ["bear"] * 30
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is False
        assert result.failed_check == "bull/unique_p_up"

    def test_neutral_regime_compressed_blocks(self):
        # Neutral regime has tightly compressed output (stdev fail).
        # Construct: n_unique > 8 to bypass uniqueness, but stdev < 0.005.
        preds_bull = _diverse_predictions(30)
        preds_neutral = []
        for i in range(30):
            p_up = 0.5000 + (i - 15) * 0.0001  # tight cluster, varied
            preds_neutral.append({
                "p_up": round(p_up, 6),
                "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
            })
        preds_bear = _diverse_predictions(30)
        preds = preds_bull + preds_neutral + preds_bear
        regimes = ["bull"] * 30 + ["neutral"] * 30 + ["bear"] * 30
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is False
        # Could be uniqueness OR stdev — both fail on the tight cluster.
        assert result.failed_check.startswith("neutral/")


# ── Insufficient sample handling ─────────────────────────────────────────

class TestInsufficientSample:

    def test_below_min_per_regime_size_skipped(self):
        # Bull = 30 (evaluated), neutral = 10 (skipped), bear = 30 (evaluated).
        # Result depends only on bull + bear.
        preds = (
            _diverse_predictions(30)
            + _diverse_predictions(10)
            + _diverse_predictions(30)
        )
        regimes = ["bull"] * 30 + ["neutral"] * 10 + ["bear"] * 30
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is True
        assert result.metrics["n_regimes_evaluated"] == 2
        assert result.metrics["per_regime"]["neutral"]["evaluated"] is False
        assert result.metrics["per_regime"]["bull"]["evaluated"] is True
        assert result.metrics["per_regime"]["bear"]["evaluated"] is True

    def test_no_regime_meets_min_size_passes_with_note(self):
        # All regimes too small.
        preds = _diverse_predictions(15)
        regimes = ["bull"] * 5 + ["neutral"] * 5 + ["bear"] * 5
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is True
        assert "stratified gate does not fire" in result.reason
        assert result.metrics["n_regimes_evaluated"] == 0

    def test_skipped_regime_does_not_block_pass(self):
        # Bull is the only regime with enough rows; it's healthy. Even
        # though bear and neutral are too small to evaluate, the gate
        # passes because the only evaluated regime is clean.
        preds = _diverse_predictions(30) + _diverse_predictions(5) + _diverse_predictions(5)
        regimes = ["bull"] * 30 + ["neutral"] * 5 + ["bear"] * 5
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is True
        assert result.metrics["n_regimes_evaluated"] == 1


# ── Other / unknown regime labels ────────────────────────────────────────

class TestUnknownRegimeLabels:

    def test_unknown_regime_labels_excluded(self):
        # 30 unknown-label rows + 30 bull (healthy) + 30 bear (healthy).
        # Unknown rows are excluded from any regime bucket.
        preds = (
            _diverse_predictions(30)  # "garbage" label
            + _diverse_predictions(30)  # bull
            + _diverse_predictions(30)  # bear
        )
        regimes = ["unknown"] * 30 + ["bull"] * 30 + ["bear"] * 30
        result = validate_stratified_per_regime(preds, regimes)
        assert result.passed is True
        # Only bull + bear evaluated.
        assert result.metrics["n_regimes_evaluated"] == 2


# ── Input validation ────────────────────────────────────────────────────

class TestInputValidation:

    def test_predictions_regimes_length_mismatch(self):
        result = validate_stratified_per_regime(
            _diverse_predictions(10),
            ["bull"] * 5,  # length disagrees
        )
        assert result.passed is False
        assert result.failed_check == "input_mismatch"


# ── Custom min_per_regime_size ──────────────────────────────────────────

class TestCustomThresholds:

    def test_relaxed_min_size_allows_smaller_regimes(self):
        preds = _diverse_predictions(45)
        regimes = ["bull"] * 15 + ["neutral"] * 15 + ["bear"] * 15
        # Default is 25 → all skipped. With min=10, all evaluated.
        result_default = validate_stratified_per_regime(preds, regimes)
        result_relaxed = validate_stratified_per_regime(preds, regimes, min_per_regime_size=10)
        assert result_default.metrics["n_regimes_evaluated"] == 0
        assert result_relaxed.metrics["n_regimes_evaluated"] == 3
