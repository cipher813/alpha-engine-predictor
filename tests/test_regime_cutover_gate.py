"""Tests for the Phase 4 regime-conditioned cutover gate validator (PR 5a).

Per audit §8 Phase 4: cutover gate is "regime-conditioned ensemble IC >
single-Ridge IC by ≥ 15% relative across the validation period." This
module computes that verdict from prediction history paired with realized
5d alpha.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.regime_cutover_gate import (
    CutoverGateResult,
    compute_parity_ic,
    compute_per_regime_ic,
    validate_cutover_gate,
)


def _synthetic_pairs(
    n: int = 200,
    sr_signal_strength: float = 0.10,
    rcm_signal_strength: float = 0.20,
    noise: float = 0.50,
    seed: int = 0,
):
    """Build (date, ticker)-style pairs with both alphas + realized.

    Each signal-strength parameter directly controls the resulting IC
    (Pearson correlation) of that alpha stream against realized — the
    noise term is large enough that ICs land in the realistic 0.05-0.30
    range that the audit operates in, NOT pinned at ~1.0.

    Set rcm > sr to test pass paths, sr > rcm for fail paths. The actual
    IC is approximately ``signal_strength / sqrt(signal_strength**2 + noise**2)``
    so default signals 0.10 / 0.20 with noise 0.50 yield ICs ~0.20 / ~0.37
    → ~85% relative lift (clears 15% threshold cleanly).
    """
    rng = np.random.default_rng(seed)
    realized = rng.normal(0, 1.0, n)  # standardized realized
    pairs = []
    regimes = ["bear", "neutral", "bull"]
    for i in range(n):
        sr = sr_signal_strength * realized[i] + noise * rng.normal()
        rcm = rcm_signal_strength * realized[i] + noise * rng.normal()
        pairs.append({
            "date": f"2026-05-{(i % 28) + 1:02d}",
            "ticker": f"T{i:03d}",
            "predicted_alpha": float(sr),
            "regime_conditioned_alpha": float(rcm),
            "predicted_regime": regimes[i % 3],
            "realized_alpha": float(realized[i]),
        })
    return pairs


# ── compute_parity_ic ───────────────────────────────────────────────────

class TestComputeParityIC:

    def test_basic_parity(self):
        pairs = _synthetic_pairs()
        sr_ic, rcm_ic, n = compute_parity_ic(pairs)
        assert n == 200
        assert sr_ic > 0  # synthetic positive correlation
        assert rcm_ic > 0
        # Stronger signal → higher IC (loose tolerance for stochasticity)
        assert rcm_ic > sr_ic

    def test_empty_input_returns_nan(self):
        sr, rcm, n = compute_parity_ic([])
        assert np.isnan(sr) and np.isnan(rcm) and n == 0

    def test_below_min_pairs_returns_nan(self):
        # Only 5 pairs — below the 10-pair internal threshold for a meaningful IC.
        pairs = _synthetic_pairs(n=5)
        sr, rcm, n = compute_parity_ic(pairs)
        assert np.isnan(sr) and np.isnan(rcm)
        assert n == 5

    def test_filters_none_realized(self):
        # Mix of finite + None realized — only finite contribute.
        pairs = _synthetic_pairs(n=50)
        for i in range(0, 50, 2):
            pairs[i]["realized_alpha"] = None
        sr, rcm, n = compute_parity_ic(pairs)
        # 25 finite pairs → above 10 threshold.
        assert n == 25
        assert np.isfinite(sr)

    def test_filters_none_alphas(self):
        # When predicted_alpha or regime_conditioned_alpha is None, that
        # row is excluded.
        pairs = _synthetic_pairs(n=50)
        for i in range(10):
            pairs[i]["regime_conditioned_alpha"] = None
        sr, rcm, n = compute_parity_ic(pairs)
        assert n == 40


# ── compute_per_regime_ic ───────────────────────────────────────────────

class TestComputePerRegimeIC:

    def test_per_regime_split(self):
        pairs = _synthetic_pairs(n=300)
        result = compute_per_regime_ic(pairs)
        assert set(result.keys()) == {"bull", "neutral", "bear"}
        for regime in ("bull", "neutral", "bear"):
            assert result[regime]["n"] == 100  # 300 / 3
            assert result[regime]["spearman"] is not None

    def test_per_regime_below_min_returns_none(self):
        pairs = _synthetic_pairs(n=15)  # 5 per regime — below 10 threshold
        result = compute_per_regime_ic(pairs)
        for regime in ("bull", "neutral", "bear"):
            assert result[regime]["spearman"] is None

    def test_unknown_regime_excluded(self):
        pairs = _synthetic_pairs(n=60)
        for p in pairs[:30]:
            p["predicted_regime"] = "unknown"
        result = compute_per_regime_ic(pairs)
        # Only 30 valid pairs across the 3 known regimes.
        total = sum(result[r]["n"] for r in ("bull", "neutral", "bear"))
        assert total == 30


# ── validate_cutover_gate ──────────────────────────────────────────────

class TestValidateCutoverGate:

    def test_pass_with_meaningful_lift(self):
        # Strong synthetic signal in regime-conditioned (rcm 2x sr)
        # → lift well above 1.15 threshold.
        pairs = _synthetic_pairs(
            n=600, sr_signal_strength=0.10, rcm_signal_strength=0.30, noise=0.50,
        )
        result = validate_cutover_gate(pairs)
        assert isinstance(result, CutoverGateResult)
        assert result.passed is True
        assert result.relative_lift > 1.15
        assert "PASS" in result.reason

    def test_fail_when_lift_below_threshold(self):
        # rcm = sr → lift ≈ 1.0; well below 15% threshold.
        pairs = _synthetic_pairs(
            n=600, sr_signal_strength=0.20, rcm_signal_strength=0.20, noise=0.50,
        )
        result = validate_cutover_gate(pairs)
        assert result.passed is False
        assert "FAIL" in result.reason

    def test_neutral_when_below_min_pairs(self):
        pairs = _synthetic_pairs(n=50)
        result = validate_cutover_gate(pairs, min_pairs=100)
        assert result.passed is False
        assert "insufficient data" in result.reason
        assert result.relative_lift is None

    def test_neutral_when_single_ridge_ic_too_weak(self):
        # Build pairs where both alphas correlate ~zero with realized.
        rng = np.random.default_rng(42)
        pairs = []
        for i in range(200):
            pairs.append({
                "date": f"d{i}", "ticker": f"T{i}",
                "predicted_alpha": float(rng.normal(0, 0.01)),
                "regime_conditioned_alpha": float(rng.normal(0, 0.01)),
                "predicted_regime": "neutral",
                "realized_alpha": float(rng.normal(0, 0.02)),
            })
        result = validate_cutover_gate(pairs)
        assert result.passed is False
        assert "single-Ridge IC" in result.reason
        assert "below floor" in result.reason
        assert result.relative_lift is None

    def test_per_regime_diagnostic_always_populated(self):
        pairs = _synthetic_pairs(n=300)
        result = validate_cutover_gate(pairs)
        # per_regime_ic populated for all three regimes.
        for regime in ("bull", "neutral", "bear"):
            assert regime in result.per_regime_ic
            assert "n" in result.per_regime_ic[regime]
            assert "spearman" in result.per_regime_ic[regime]

    def test_custom_threshold(self):
        # Equal signal strengths → lift ≈ 1.0; fails default 1.15
        # threshold and equally fails 1.05 threshold. Test that varying
        # the threshold actually changes acceptance for a borderline case:
        # rcm 1.5x sr should pass at 1.15 but fail at 1.50 threshold.
        pairs = _synthetic_pairs(
            n=600, sr_signal_strength=0.10, rcm_signal_strength=0.20, noise=0.50,
        )
        result_default = validate_cutover_gate(pairs, relative_lift_threshold=1.15)
        result_strict = validate_cutover_gate(pairs, relative_lift_threshold=2.50)
        assert result_default.passed is True   # ~2x lift clears 1.15
        assert result_strict.passed is False    # but not 2.50

    def test_result_dataclass_is_frozen(self):
        pairs = _synthetic_pairs(n=200)
        result = validate_cutover_gate(pairs)
        with pytest.raises(Exception):
            result.passed = not result.passed  # frozen → raise


# ── Defensive on edge cases ────────────────────────────────────────────

class TestDefensiveEdgeCases:

    def test_empty_pairs_returns_neutral(self):
        result = validate_cutover_gate([])
        assert result.passed is False
        assert result.n_pairs == 0
        assert "insufficient data" in result.reason

    def test_all_none_alphas_returns_neutral(self):
        pairs = [{
            "date": f"d{i}", "ticker": f"T{i}",
            "predicted_alpha": None,
            "regime_conditioned_alpha": None,
            "predicted_regime": "neutral",
            "realized_alpha": float(np.random.normal()),
        } for i in range(200)]
        result = validate_cutover_gate(pairs)
        assert result.passed is False
        assert "insufficient data" in result.reason

    def test_handles_nan_realized_alphas(self):
        pairs = _synthetic_pairs(n=300)
        for i in range(50):
            pairs[i]["realized_alpha"] = float("nan")
        result = validate_cutover_gate(pairs)
        # Filtering NaN realized leaves 250 finite pairs — above min_pairs.
        assert result.n_pairs == 250
