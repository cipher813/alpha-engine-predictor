"""Tests for analysis/horizon_battery.py — standalone offline harness.

Per the 2026-05-07 predictor audit Track B (PR 4/N): the analysis
module reads OOS rows persisted by training and recomputes the full
overlap/non-overlap/per-regime/bootstrap-CI battery without requiring
a training rerun. Tests target the pure logic (compute_horizon_battery
+ format_report) using synthetic in-memory rows.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.horizon_battery import (
    _fmt,
    _fmt_ci,
    _round_or_none,
    compute_horizon_battery,
    format_report,
)


def _synthetic_oos_rows(n_dates: int = 60, tickers_per_date: int = 20, seed: int = 0):
    """Build a DataFrame matching meta_trainer's OOS row schema.

    Includes ALL META_FEATURES + actual_fwd + actual_fwd_{h}d for each
    diagnostic horizon. Synthetic strong signal: actual is a noisy linear
    function of the META_FEATURES so the fitted Ridge will produce a
    meaningful IC.
    """
    from model.meta_model import META_FEATURES

    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2025-01-02")
    bdays = pd.bdate_range(start=base, periods=n_dates)
    for d in bdays:
        for t in range(tickers_per_date):
            row = {
                f: float(rng.normal(0, 1)) for f in META_FEATURES
            }
            # Construct an actual signal correlated with the features.
            actual = sum(row[f] for f in META_FEATURES) * 0.05 + rng.normal(0, 1.5)
            row["actual_fwd"] = float(actual)
            # Multi-horizon labels — noisier at long horizons (more drift).
            for h in [5, 10, 15, 21, 40, 60, 90]:
                row[f"actual_fwd_{h}d"] = float(actual + rng.normal(0, 0.3 * h / 5))
            row["date"] = d.strftime("%Y-%m-%d")
            # Override macro_spy_20d_return for regime classification:
            # cycle through bull/neutral/bear so all three are populated.
            phase = (rng.integers(0, 3))
            row["macro_spy_20d_return"] = float(
                {0: 0.06, 1: 0.0, 2: -0.06}[int(phase)]
            )
            rows.append(row)
    return pd.DataFrame(rows)


# ── _round_or_none / formatting helpers ──────────────────────────────────

class TestFormattingHelpers:

    def test_round_or_none_finite(self):
        assert _round_or_none(0.123456789, ndigits=4) == 0.1235

    def test_round_or_none_nan(self):
        assert _round_or_none(float("nan")) is None

    def test_round_or_none_none(self):
        assert _round_or_none(None) is None

    def test_fmt_finite(self):
        assert _fmt(0.0123) == "+0.0123"

    def test_fmt_negative(self):
        assert _fmt(-0.0456) == "-0.0456"

    def test_fmt_none(self):
        assert _fmt(None) == "—"

    def test_fmt_ci_finite(self):
        assert _fmt_ci(0.01, 0.05) == "[+0.010,+0.050]"

    def test_fmt_ci_none_lo(self):
        assert _fmt_ci(None, 0.05) == "[—]"


# ── compute_horizon_battery ──────────────────────────────────────────────

class TestComputeHorizonBattery:

    def test_basic_shape(self):
        df = _synthetic_oos_rows(n_dates=40, tickers_per_date=15)
        report = compute_horizon_battery(df, bootstrap_iter=50)
        assert "horizons" in report
        assert "regime_distribution" in report
        assert "curve" in report
        assert "n_rows" in report
        assert report["n_rows"] == 40 * 15
        # All diagnostic horizons present in curve (if columns exist).
        for h in ["5d", "10d", "15d", "21d", "40d", "60d", "90d"]:
            assert h in report["curve"]

    def test_curve_per_horizon_fields(self):
        df = _synthetic_oos_rows(n_dates=60, tickers_per_date=20)
        report = compute_horizon_battery(df, bootstrap_iter=100)
        h5 = report["curve"]["5d"]
        # All required fields present.
        for field in (
            "spearman", "n", "spearman_ci_lo", "spearman_ci_hi",
            "spearman_nonoverlap", "n_nonoverlap",
            "spearman_nonoverlap_ci_lo", "spearman_nonoverlap_ci_hi",
            "by_regime",
        ):
            assert field in h5, f"missing field: {field}"
        # by_regime contains all three regimes.
        for regime in ("bull", "neutral", "bear"):
            assert regime in h5["by_regime"]
            assert "spearman" in h5["by_regime"][regime]
            assert "n" in h5["by_regime"][regime]

    def test_regime_distribution_sums_to_n_rows(self):
        df = _synthetic_oos_rows(n_dates=50, tickers_per_date=12)
        report = compute_horizon_battery(df, bootstrap_iter=50)
        dist = report["regime_distribution"]
        assert sum(dist.values()) == report["n_rows"]

    def test_horizons_argument_filters_curve(self):
        df = _synthetic_oos_rows(n_dates=30, tickers_per_date=10)
        report = compute_horizon_battery(
            df, horizons=[5, 21], bootstrap_iter=50,
        )
        assert set(report["curve"].keys()) == {"5d", "21d"}

    def test_bootstrap_iter_propagates_to_report(self):
        df = _synthetic_oos_rows(n_dates=30, tickers_per_date=10)
        report = compute_horizon_battery(df, bootstrap_iter=250)
        assert report["bootstrap_iter"] == 250

    def test_strong_signal_yields_finite_ic(self):
        # Synthetic signal should produce a finite IC (sign depends on
        # Ridge fit, but should not be NaN with enough rows).
        df = _synthetic_oos_rows(n_dates=80, tickers_per_date=20, seed=7)
        report = compute_horizon_battery(df, bootstrap_iter=100)
        # Overall 5d IC should be finite.
        assert report["curve"]["5d"]["spearman"] is not None

    def test_missing_horizon_column_skipped_gracefully(self):
        # Drop one of the horizon columns; the helper should log a
        # warning and skip that horizon, not crash.
        df = _synthetic_oos_rows(n_dates=20, tickers_per_date=8)
        df = df.drop(columns=["actual_fwd_60d"])
        report = compute_horizon_battery(df, bootstrap_iter=50)
        assert "60d" not in report["curve"]
        assert "5d" in report["curve"]  # other horizons still computed


# ── format_report ────────────────────────────────────────────────────────

class TestFormatReport:

    def test_includes_header(self):
        df = _synthetic_oos_rows(n_dates=20, tickers_per_date=8)
        report = compute_horizon_battery(df, bootstrap_iter=50)
        rendered = format_report(report)
        assert "Horizon battery" in rendered
        assert "Regime distribution" in rendered
        # Column headers present.
        assert "Horizon" in rendered
        assert "IC bull" in rendered

    def test_per_horizon_row_present(self):
        df = _synthetic_oos_rows(n_dates=25, tickers_per_date=10)
        report = compute_horizon_battery(df, horizons=[5, 21], bootstrap_iter=50)
        rendered = format_report(report)
        assert "5d" in rendered
        assert "21d" in rendered

    def test_handles_nan_ic_gracefully(self):
        # Force a horizon with too few finite labels by zeroing them.
        df = _synthetic_oos_rows(n_dates=20, tickers_per_date=5)
        df["actual_fwd_90d"] = float("nan")
        report = compute_horizon_battery(df, bootstrap_iter=50)
        rendered = format_report(report)
        # Should render without crashing; "—" or similar for NaN cells.
        assert "90d" in rendered
