"""Tests for the bootstrap-by-date IC confidence interval helper.

Per the 2026-05-07 predictor audit Track B (PR 2/N): 1000-iter bootstrap
resampling unique dates with replacement, not rows. Resampling rows would
treat overlapping forward windows as independent observations — same
autocorrelation problem the non-overlapping subsample addresses for the
point estimate. The right unit of independence is the date.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.meta_trainer import _bootstrap_ic_ci_by_date


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


class TestBootstrapICConfidenceInterval:

    def test_returns_nan_on_empty_input(self):
        lo, hi = _bootstrap_ic_ci_by_date([], [], [])
        assert np.isnan(lo) and np.isnan(hi)

    def test_returns_nan_on_length_mismatch(self):
        # predictions, actuals, dates lengths must agree
        lo, hi = _bootstrap_ic_ci_by_date([1.0, 2.0], [1.0], [_ts("2026-01-01")])
        assert np.isnan(lo) and np.isnan(hi)

    def test_returns_nan_when_fewer_than_three_unique_dates(self):
        # CI is meaningless on < 3 dates. Only 2 unique dates → NaN.
        preds = [0.1, 0.2, 0.3, 0.4]
        actuals = [0.05, 0.10, 0.15, 0.20]
        dates = [_ts("2026-01-01"), _ts("2026-01-01"),
                 _ts("2026-01-02"), _ts("2026-01-02")]
        lo, hi = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=100)
        assert np.isnan(lo) and np.isnan(hi)

    def test_returns_finite_ci_on_strong_signal(self):
        # Synthetic strong signal: actuals = predictions + small noise.
        # Bootstrap CI should land tightly around 1.0 (perfect rank corr).
        rng = np.random.default_rng(0)
        preds = rng.normal(0, 1, 200)
        actuals = preds + rng.normal(0, 0.1, 200)  # near-perfect correlation
        dates = [_ts(f"2026-01-{(i % 28) + 1:02d}") for i in range(200)]
        lo, hi = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=200, seed=42)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo > 0.9, f"expected lo > 0.9 on near-perfect correlation, got {lo}"
        assert hi <= 1.0
        assert lo <= hi

    def test_ci_brackets_zero_on_random_data(self):
        # No signal: predictions and actuals are independent.
        # 95% CI should bracket 0 with high probability.
        rng = np.random.default_rng(1)
        n = 300
        preds = rng.normal(0, 1, n)
        actuals = rng.normal(0, 1, n)
        dates = [_ts(f"2026-{(i // 28) % 12 + 1:02d}-{(i % 28) + 1:02d}")
                 for i in range(n)]
        lo, hi = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=500, seed=42)
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < 0 < hi, f"expected CI to bracket 0, got [{lo}, {hi}]"

    def test_deterministic_under_same_seed(self):
        # Same seed → same CI exactly.
        rng = np.random.default_rng(7)
        preds = rng.normal(0, 1, 100)
        actuals = preds + rng.normal(0, 0.5, 100)
        dates = [_ts(f"2026-01-{(i % 28) + 1:02d}") for i in range(100)]

        lo1, hi1 = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=100, seed=42)
        lo2, hi2 = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=100, seed=42)
        assert lo1 == lo2 and hi1 == hi2

    def test_different_seeds_yield_similar_but_not_identical_cis(self):
        # Seed change should perturb but not drastically alter the CI.
        rng = np.random.default_rng(11)
        preds = rng.normal(0, 1, 100)
        actuals = preds + rng.normal(0, 0.5, 100)
        dates = [_ts(f"2026-01-{(i % 28) + 1:02d}") for i in range(100)]

        lo_a, hi_a = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=200, seed=1)
        lo_b, hi_b = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=200, seed=99)
        # CIs should overlap substantially but not be byte-identical.
        assert (lo_a, hi_a) != (lo_b, hi_b)
        # Width should be of similar order.
        width_a, width_b = hi_a - lo_a, hi_b - lo_b
        assert 0.5 <= (width_a / width_b) <= 2.0

    def test_handles_nan_actuals(self):
        # NaN actuals are excluded from each bootstrap iteration.
        rng = np.random.default_rng(3)
        n = 200
        preds = rng.normal(0, 1, n)
        actuals = preds + rng.normal(0, 0.2, n)
        actuals[::5] = np.nan  # 20% NaN
        dates = [_ts(f"2026-01-{(i % 28) + 1:02d}") for i in range(n)]
        lo, hi = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=200, seed=42)
        assert np.isfinite(lo) and np.isfinite(hi)
        # Signal is still strong despite NaN holes.
        assert lo > 0.7

    def test_lo_le_hi_invariant(self):
        # CI lower bound must always be <= upper bound.
        rng = np.random.default_rng(17)
        for trial_seed in range(5):
            preds = rng.normal(0, 1, 100)
            actuals = rng.normal(0, 1, 100)
            dates = [_ts(f"2026-01-{(i % 28) + 1:02d}") for i in range(100)]
            lo, hi = _bootstrap_ic_ci_by_date(
                preds, actuals, dates, n_iter=100, seed=trial_seed
            )
            if np.isfinite(lo) and np.isfinite(hi):
                assert lo <= hi, f"lo={lo} > hi={hi} (seed={trial_seed})"

    def test_accepts_string_dates(self):
        # Dates as strings should coerce via pd.Timestamp.
        rng = np.random.default_rng(19)
        n = 100
        preds = rng.normal(0, 1, n)
        actuals = preds + rng.normal(0, 0.2, n)
        dates = [f"2026-01-{(i % 28) + 1:02d}" for i in range(n)]
        lo, hi = _bootstrap_ic_ci_by_date(preds, actuals, dates, n_iter=100, seed=42)
        assert np.isfinite(lo) and np.isfinite(hi)
