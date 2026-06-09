"""L4565c — the meta-model promotion gates must compute a real verdict.

Root cause (verified against the live 6/07 manifest): the W1.3 robustness
battery (downside-Sortino + overfit-DSR) that ``model_zoo.select_winner``
gates on was fed the EXPANDING-WALK-FORWARD leak-free ``ic_series``, which is
structurally ``insufficient_folds`` at the 21d horizon (only ~45 OOS meta-dates;
a fold needs ~42 purged dates of history before its test block). Empty series →
``status=insufficient`` → ``passes_*_gate=False`` for EVERY candidate →
``select_winner._gate_pass`` rejected all → nothing could ever promote, even
though the CPCV mean IC (0.058) and ranking worked fine.

Fix: feed the battery the CPCV per-combo ``ics`` distribution (status=ok, ~15
purged combos) — the institutionally-correct DSR/Sortino input — falling back to
the expanding-WF series only when CPCV is empty. Plus harden ``_cpcv_mean`` so a
NaN CPCV collapses to None (a date-starved future run is labeled ``no_cpcv``,
not the misleading ``below_champion_plus_margin``).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.deflated_sharpe import deflated_sharpe_ratio, downside_ic_stats
from training.meta_trainer import _select_promotion_ic_series


# ── producer: source selection (_select_promotion_ic_series) ───────────────


class TestSelectPromotionICSeries:
    def test_prefers_cpcv_ics_when_present(self):
        cpcv = {"status": "ok", "ics": [0.2, 0.1, 0.3]}
        wf = {"status": "ok", "ic_series": [0.05, 0.06]}
        series, source = _select_promotion_ic_series(cpcv, wf)
        assert source == "cpcv"
        assert series == [0.2, 0.1, 0.3]

    def test_falls_back_to_wf_when_cpcv_empty(self):
        # The live failure shape: CPCV empty/absent AND expanding-WF empty.
        cpcv = {"status": "no_valid_combos", "ics": []}
        wf = {"status": "ok", "ic_series": [0.05, 0.06]}
        series, source = _select_promotion_ic_series(cpcv, wf)
        assert source == "expanding_wf"
        assert series == [0.05, 0.06]

    def test_both_empty_returns_empty(self):
        series, source = _select_promotion_ic_series(
            {"status": "no_valid_combos", "ics": []},
            {"status": "insufficient_folds", "ic_series": []},
        )
        assert series == []
        assert source == "expanding_wf"

    def test_non_dict_inputs_safe(self):
        series, source = _select_promotion_ic_series(None, None)
        assert series == []
        assert source == "expanding_wf"

    def test_returns_a_copy_not_the_manifest_list(self):
        ics = [0.2, 0.1]
        series, _ = _select_promotion_ic_series({"ics": ics}, {})
        series.append(0.9)
        assert ics == [0.2, 0.1]  # caller can't mutate the manifest dict


# ── the gates now compute a verdict from a CPCV distribution ───────────────


class TestBatteryComputesVerdictFromCPCV:
    def test_empty_series_is_the_degenerate_freeze(self):
        # Documents the pre-fix failure: empty series → insufficient → False.
        assert downside_ic_stats([]).get("status") == "insufficient"
        assert deflated_sharpe_ratio([], n_trials=10).get("passes_overfit_gate") is False

    def test_strong_cpcv_distribution_passes(self):
        cpcv = {"status": "ok", "n_combos": 15,
                "ics": [0.23, 0.19, 0.19, 0.17, 0.10, 0.20, 0.06, 0.03, 0.11,
                        0.02, 0.24, 0.26, 0.42, 0.31, 0.53]}
        series, source = _select_promotion_ic_series(cpcv, {"ic_series": []})
        assert source == "cpcv"
        ds = downside_ic_stats(series, sortino_threshold=0.0)
        ov = deflated_sharpe_ratio(series, n_trials=max(cpcv["n_combos"], 10),
                                   threshold=0.95)
        assert ds.get("status") == "ok"
        assert ov.get("status") == "ok"
        assert ds.get("passes_downside_gate") is True
        assert ov.get("passes_overfit_gate") is True

    def test_weak_noisy_distribution_fails(self):
        # A mean~0, half-negative distribution must NOT clear the overfit gate —
        # the fix makes the gate HONEST, not auto-pass.
        weak = [0.05, -0.04, 0.03, -0.06, 0.02, -0.03, 0.04, -0.05, 0.01,
                -0.02, 0.06, -0.07, 0.02, -0.01, 0.03]
        ov = deflated_sharpe_ratio(weak, n_trials=15, threshold=0.95)
        assert ov.get("passes_overfit_gate") is False


# ── consumer: _cpcv_mean NaN hardening ─────────────────────────────────────


class TestCpcvMeanNanHardening:
    def test_nan_collapses_to_none(self):
        from training.model_zoo import _cpcv_mean
        assert _cpcv_mean({"meta_model_oos_ic_cpcv": {"mean_ic": float("nan")}}) is None

    def test_valid_value_passes_through(self):
        from training.model_zoo import _cpcv_mean
        assert _cpcv_mean({"meta_model_oos_ic_cpcv": {"mean_ic": 0.058}}) == 0.058

    def test_absent_or_error_is_none(self):
        from training.model_zoo import _cpcv_mean
        assert _cpcv_mean({"meta_model_oos_ic_cpcv": {"status": "no_valid_combos"}}) is None
        assert _cpcv_mean(None) is None


# ── CPCV density: n_groups=10 clears the n≥5 battery floor at ~45 dates ─────


class TestCpcvDensityAtScarceDates:
    """L4565c part 2: at the 21d horizon with ~45 OOS meta-dates, the 21d purge +
    21d overlapping-label embargo consume ~42 dates per combo, so n_groups=6
    leaves too few valid (purged+embargoed) combos to clear the downside/overfit
    battery's n≥5 floor. Raising n_groups (leak protection UNCHANGED — purge +
    embargo applied identically per combo) recovers enough combinations. This
    locks the methodological choice against the REAL fold generator."""

    @staticmethod
    def _shape(n_dates=45, per=20, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        dates = np.array([d for d in range(n_dates) for _ in range(per)])
        X = rng.normal(0, 1, (len(dates), 13))
        y = X @ rng.normal(0, 1, 13) * 0.1 + rng.normal(0, 1, len(dates))

        def fitpred(Xtr, ytr, Xte):
            b, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            return Xte @ b
        return X, y, dates, fitpred

    def test_default_six_groups_is_too_thin(self):
        from training.leakfree_meta_ic import cpcv_meta_oos_ic
        X, y, dates, fp = self._shape()
        r = cpcv_meta_oos_ic(X, y, dates, fit_predict_fn=fp,
                             forward_days=21, embargo_days=None, n_groups=6, k_test=2)
        # n_groups=6 leaves < 5 valid combos at this date count (the freeze).
        assert r["status"] == "ok"
        assert r["n_combos"] < 5

    def test_ten_groups_clears_the_floor(self):
        from training.leakfree_meta_ic import cpcv_meta_oos_ic
        X, y, dates, fp = self._shape()
        r = cpcv_meta_oos_ic(X, y, dates, fit_predict_fn=fp,
                             forward_days=21, embargo_days=None, n_groups=10, k_test=2)
        assert r["status"] == "ok"
        assert r["n_combos"] >= 5  # enough for the downside/overfit battery

    def test_config_default_is_ten(self):
        import config as cfg
        assert cfg.WF_CPCV_N_GROUPS == 10  # L4565c raised 6→10
