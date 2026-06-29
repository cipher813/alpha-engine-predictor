"""Tests for the variant-agnostic cutover gate validator (Stage 0b).

Generic version of the audit §8 Phase 4 cutover criterion: variant IC >
baseline IC by ≥ relative_lift_threshold (default 15%). Same gate
framework, parametrized field names so it works for any future cutover
under the regime-conditioning rebuild plan.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.variant_cutover_gate import (
    VariantCutoverGateResult,
    compute_parity_ic,
    compute_per_slice_ic,
    validate_cutover_gate,
)


def _synthetic_pairs(
    n: int = 200,
    baseline_signal_strength: float = 0.10,
    variant_signal_strength: float = 0.20,
    noise: float = 0.50,
    seed: int = 0,
    baseline_field: str = "predicted_alpha",
    variant_field: str = "regime_conditioned_alpha",
    slice_field: str = "predicted_regime",
):
    """Build (date, ticker)-style pairs with two parametrized alpha streams.

    Each signal-strength parameter directly controls the resulting IC
    (Pearson correlation) of that alpha stream against realized — the
    noise term is large enough that ICs land in the realistic 0.05-0.30
    range. Default signals 0.10 / 0.20 with noise 0.50 yield ICs ~0.20 /
    ~0.37 → ~85% relative lift (clears 15% threshold cleanly).
    """
    rng = np.random.default_rng(seed)
    realized = rng.normal(0, 1.0, n)
    pairs = []
    slice_values = ["bear", "neutral", "bull"]
    for i in range(n):
        baseline_val = baseline_signal_strength * realized[i] + noise * rng.normal()
        variant_val = variant_signal_strength * realized[i] + noise * rng.normal()
        pairs.append({
            "date": f"2026-05-{(i % 28) + 1:02d}",
            "ticker": f"T{i:03d}",
            baseline_field: float(baseline_val),
            variant_field: float(variant_val),
            slice_field: slice_values[i % 3],
            "realized_alpha": float(realized[i]),
        })
    return pairs


# ── compute_parity_ic ───────────────────────────────────────────────────


class TestComputeParityIC:

    def test_basic_parity(self):
        pairs = _synthetic_pairs()
        bl_ic, var_ic, n = compute_parity_ic(
            pairs, "predicted_alpha", "regime_conditioned_alpha",
        )
        assert n == 200
        assert bl_ic > 0
        assert var_ic > 0
        assert var_ic > bl_ic

    def test_empty_input_returns_nan(self):
        bl, var, n = compute_parity_ic([], "predicted_alpha", "variant_alpha")
        assert np.isnan(bl) and np.isnan(var) and n == 0

    def test_below_min_pairs_returns_nan(self):
        pairs = _synthetic_pairs(n=5)
        bl, var, n = compute_parity_ic(
            pairs, "predicted_alpha", "regime_conditioned_alpha",
        )
        assert np.isnan(bl) and np.isnan(var)
        assert n == 5

    def test_filters_none_realized(self):
        pairs = _synthetic_pairs(n=50)
        for i in range(0, 50, 2):
            pairs[i]["realized_alpha"] = None
        bl, var, n = compute_parity_ic(
            pairs, "predicted_alpha", "regime_conditioned_alpha",
        )
        assert n == 25
        assert np.isfinite(bl)

    def test_filters_none_variant(self):
        pairs = _synthetic_pairs(n=50)
        for i in range(10):
            pairs[i]["regime_conditioned_alpha"] = None
        bl, var, n = compute_parity_ic(
            pairs, "predicted_alpha", "regime_conditioned_alpha",
        )
        assert n == 40

    def test_arbitrary_field_names(self):
        # The gate doesn't care about specific field names — Stage 1+ will
        # use names like "predicted_alpha_macro_aug" or "predicted_alpha_tb".
        pairs = _synthetic_pairs(
            baseline_field="alpha_v1",
            variant_field="alpha_v2",
        )
        bl, var, n = compute_parity_ic(pairs, "alpha_v1", "alpha_v2")
        assert n == 200
        assert bl > 0 and var > 0

    def test_custom_realized_field(self):
        pairs = _synthetic_pairs(n=50)
        for p in pairs:
            p["actual_log_alpha"] = p.pop("realized_alpha")
        bl, var, n = compute_parity_ic(
            pairs, "predicted_alpha", "regime_conditioned_alpha",
            realized_field="actual_log_alpha",
        )
        assert n == 50


# ── compute_per_slice_ic ───────────────────────────────────────────────


class TestComputePerSliceIC:

    def test_per_slice_split(self):
        pairs = _synthetic_pairs(n=300)
        result = compute_per_slice_ic(
            pairs, "regime_conditioned_alpha", "predicted_regime",
        )
        assert set(result.keys()) == {"bull", "neutral", "bear"}
        for slice_val in ("bull", "neutral", "bear"):
            assert result[slice_val]["n"] == 100
            assert result[slice_val]["spearman"] is not None

    def test_per_slice_below_min_returns_none(self):
        pairs = _synthetic_pairs(n=15)
        result = compute_per_slice_ic(
            pairs, "regime_conditioned_alpha", "predicted_regime",
        )
        for slice_val in ("bull", "neutral", "bear"):
            assert result[slice_val]["spearman"] is None

    def test_none_slice_value_excluded(self):
        pairs = _synthetic_pairs(n=60)
        for p in pairs[:30]:
            p["predicted_regime"] = None
        result = compute_per_slice_ic(
            pairs, "regime_conditioned_alpha", "predicted_regime",
        )
        # Only 30 pairs have valid slice values across 3 regimes.
        total = sum(v["n"] for v in result.values())
        assert total == 30

    def test_arbitrary_slice_field(self):
        # Per-sector IC slicing — same machinery, different field.
        pairs = _synthetic_pairs(
            n=300,
            slice_field="sector",
        )
        result = compute_per_slice_ic(
            pairs, "regime_conditioned_alpha", "sector",
        )
        # Slice values were "bear"/"neutral"/"bull" but the field name
        # changed — the slicer is field-name-agnostic.
        assert set(result.keys()) == {"bull", "neutral", "bear"}


# ── validate_cutover_gate ──────────────────────────────────────────────


class TestValidateCutoverGate:

    def test_pass_with_meaningful_lift(self):
        pairs = _synthetic_pairs(
            n=600,
            baseline_signal_strength=0.10,
            variant_signal_strength=0.30,
            noise=0.50,
        )
        result = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
        )
        assert isinstance(result, VariantCutoverGateResult)
        assert result.passed is True
        assert result.relative_lift > 1.15
        assert "PASS" in result.reason
        assert result.baseline_field == "predicted_alpha"
        assert result.variant_field == "regime_conditioned_alpha"

    def test_fail_when_lift_below_threshold(self):
        pairs = _synthetic_pairs(
            n=600,
            baseline_signal_strength=0.20,
            variant_signal_strength=0.20,
            noise=0.50,
        )
        result = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
        )
        assert result.passed is False
        assert "FAIL" in result.reason

    def test_neutral_when_below_min_pairs(self):
        pairs = _synthetic_pairs(n=50)
        result = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
        )
        assert result.passed is False
        assert result.relative_lift is None
        assert "insufficient data" in result.reason

    def test_neutral_when_baseline_ic_below_floor(self):
        # Baseline signal essentially zero — gate stays neutral
        # (won't claim PASS just because variant is non-zero). Using an
        # explicit high floor (0.10) so the test is deterministic against
        # finite-sample noise that can push a zero-signal baseline IC above
        # the default 0.005 floor.
        pairs = _synthetic_pairs(
            n=600,
            baseline_signal_strength=0.0,
            variant_signal_strength=0.20,
            noise=0.50,
        )
        result = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
            min_baseline_ic_floor=0.10,
        )
        assert result.passed is False
        assert result.relative_lift is None
        assert "below floor" in result.reason

    def test_per_slice_populated_when_slice_field_set(self):
        pairs = _synthetic_pairs(
            n=600,
            baseline_signal_strength=0.10,
            variant_signal_strength=0.30,
            noise=0.50,
        )
        result = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
            slice_field="predicted_regime",
        )
        assert result.passed is True
        assert result.slice_field == "predicted_regime"
        assert set(result.per_slice_ic.keys()) == {"bull", "neutral", "bear"}

    def test_per_slice_empty_when_slice_field_unset(self):
        pairs = _synthetic_pairs(n=200)
        result = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
        )
        assert result.slice_field is None
        assert result.per_slice_ic == {}

    def test_custom_threshold(self):
        # Tightening threshold from 1.15 to 1.50 should flip a borderline
        # pass into a fail.
        pairs = _synthetic_pairs(
            n=600,
            baseline_signal_strength=0.10,
            variant_signal_strength=0.13,  # ~30% relative lift
            noise=0.50,
        )
        result_15 = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
            relative_lift_threshold=1.15,
        )
        result_50 = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
            relative_lift_threshold=1.50,
        )
        assert result_15.passed is True
        assert result_50.passed is False

    def test_arbitrary_field_names_for_future_stages(self):
        # Stage 1 example: macro-augmented L1 vs current.
        pairs = _synthetic_pairs(
            n=200,
            baseline_field="alpha_l1_baseline",
            variant_field="alpha_l1_macro_aug",
        )
        result = validate_cutover_gate(
            pairs,
            baseline_field="alpha_l1_baseline",
            variant_field="alpha_l1_macro_aug",
        )
        assert result.baseline_field == "alpha_l1_baseline"
        assert result.variant_field == "alpha_l1_macro_aug"
        assert variant_field_in_reason(result.reason, "alpha_l1_macro_aug")

    def test_dataclass_serializable_to_dict(self):
        # asdict() must work for CLI JSON output + manifest persistence.
        from dataclasses import asdict

        pairs = _synthetic_pairs(n=200)
        result = validate_cutover_gate(
            pairs,
            baseline_field="predicted_alpha",
            variant_field="regime_conditioned_alpha",
            slice_field="predicted_regime",
        )
        d = asdict(result)
        # Required keys present.
        for key in (
            "passed", "baseline_ic", "variant_ic", "relative_lift", "n_pairs",
            "reason", "relative_lift_threshold", "baseline_field",
            "variant_field", "slice_field", "per_slice_ic",
        ):
            assert key in d


def variant_field_in_reason(reason: str, variant_field: str) -> bool:
    """Helper — the reason string should mention the variant field name."""
    return variant_field in reason


class TestCliMainFillsRealizedAlpha:
    """The generic CLI ``main()`` must compute realized_alpha from S3 price
    data + horizon alignment (regression: it previously emitted a TODO
    warning and always returned an insufficient-data verdict because
    realized_alpha was never filled, so the Stage 1d / 2d cutover gates
    could not be evaluated ad-hoc). Refs config#805.
    """

    @staticmethod
    def _price_series(n_days=80, start=100.0, drift=0.0):
        import numpy as np
        import pandas as pd
        idx = pd.bdate_range("2026-01-01", periods=n_days)
        vals = start * np.exp(np.cumsum(np.full(n_days, drift)))
        return pd.Series(vals, index=idx)

    def _patch_cli(self, monkeypatch, pairs, prices, sector_map):
        import analysis.variant_cutover_gate as gate
        import analysis.triple_barrier_cutover_runner as runner

        monkeypatch.setattr(
            gate, "_load_prediction_history",
            lambda *a, **k: [dict(p) for p in pairs],
        )
        monkeypatch.setattr(
            runner, "_load_sector_map_from_s3",
            lambda *a, **k: dict(sector_map),
        )
        monkeypatch.setattr(
            runner, "_load_prices_from_s3",
            lambda *a, **k: dict(prices),
        )

    def _build_pairs(self):
        # Variant carries strictly more signal than baseline against the
        # realized forward return so the gate has a real (non-neutral)
        # verdict to render. 150 (date, ticker) rows on one prediction date
        # with closed forward windows.
        import numpy as np
        rng = np.random.default_rng(7)
        n = 150
        pairs = []
        prices = {"SPY": self._price_series(drift=0.0)}
        sector_map = {}
        # Realized forward alpha is encoded by giving each ticker a price
        # series whose forward return is proportional to a latent signal;
        # the variant tracks that signal more tightly than the baseline.
        for i in range(n):
            t = f"T{i:03d}"
            signal = rng.normal()
            # ticker price with a forward drift proportional to signal so
            # realized_alpha (vs flat SPY) correlates with `signal`.
            drift = 0.001 * signal
            prices[t] = self._price_series(drift=drift)
            pairs.append({
                "date": "2026-01-02",
                "ticker": t,
                "predicted_alpha": 0.2 * signal + 0.9 * rng.normal(),
                "expected_move_macro_aug": 0.8 * signal + 0.2 * rng.normal(),
                "realized_alpha": None,
            })
        return pairs, prices, sector_map

    def test_main_fills_realized_and_returns_real_verdict(self, monkeypatch, capsys):
        import json
        import pytest
        import analysis.variant_cutover_gate as gate

        pairs, prices, sector_map = self._build_pairs()
        self._patch_cli(monkeypatch, pairs, prices, sector_map)
        monkeypatch.setattr(sys, "argv", [
            "variant_cutover_gate",
            "--baseline", "predicted_alpha",
            "--variant", "expected_move_macro_aug",
            "--window", "5", "--horizon", "21",
        ])
        with pytest.raises(SystemExit) as ei:
            gate.main()
        # Exit code is 0 (pass) or 2 (fail) — NOT 1 (no-history) and the
        # verdict must be a real IC comparison, not the old TODO stub.
        assert ei.value.code in (0, 2)
        out = json.loads(capsys.readouterr().out)
        # realized_alpha was actually computed → enough pairs cleared the
        # min_pairs floor, so the reason is a PASS/FAIL verdict, never
        # "insufficient data".
        assert out["n_pairs"] >= 100
        assert "insufficient data" not in out["reason"]
        assert out["baseline_field"] == "predicted_alpha"
        assert out["variant_field"] == "expected_move_macro_aug"

    def test_main_honors_horizon_arg(self, monkeypatch):
        import pytest
        import analysis.variant_cutover_gate as gate

        captured = {}
        pairs, prices, sector_map = self._build_pairs()
        self._patch_cli(monkeypatch, pairs, prices, sector_map)

        real_fn = gate.compute_realized_alpha_for_pairs

        def _spy(pairs_arg, horizon_days, **k):
            captured["horizon"] = horizon_days
            return real_fn(pairs_arg, horizon_days=horizon_days, **k)

        monkeypatch.setattr(gate, "compute_realized_alpha_for_pairs", _spy)
        monkeypatch.setattr(sys, "argv", [
            "variant_cutover_gate",
            "--baseline", "predicted_alpha",
            "--variant", "expected_move_macro_aug",
            "--window", "5", "--horizon", "5",
        ])
        with pytest.raises(SystemExit):
            gate.main()
        assert captured["horizon"] == 5
