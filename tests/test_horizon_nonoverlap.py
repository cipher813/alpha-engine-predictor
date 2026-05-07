"""Tests for the non-overlapping date-mask helper used by Track B horizon battery.

Per the 2026-05-07 predictor audit: the existing horizon IC diagnostic in
``training/meta_trainer.py`` samples (date, ticker) rows with a forward
window that overlaps heavily across consecutive dates. The
``_nonoverlapping_date_mask`` helper subsamples one date per non-overlapping
h-day window so IC computed on the subsample reflects independent
observations.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.meta_trainer import _nonoverlapping_date_mask


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


class TestNonoverlappingDateMask:

    def test_empty_input_returns_empty(self):
        assert _nonoverlapping_date_mask([], 21) == []

    def test_single_date_kept(self):
        assert _nonoverlapping_date_mask([_ts("2026-01-01")], 21) == [True]

    def test_consecutive_days_5d_horizon_thins_to_one_per_5d(self):
        # 10 consecutive trading days; horizon=5d should keep ~2 (day 0 and day 5).
        dates = [_ts(f"2026-01-{d:02d}") for d in range(1, 11)]
        mask = _nonoverlapping_date_mask(dates, 5)
        # The first date is always kept; subsequent only if >= 5 days past.
        assert mask[0] is True
        # Date 1 (idx 0) kept; next kept must be >= 2026-01-06 (idx 5).
        kept_indices = [i for i, m in enumerate(mask) if m]
        # At minimum: idx 0 + idx 5 — could also include idx 9 (2026-01-10
        # is 4 days past idx 5, so NOT kept; idx 10 would be 5 days past).
        assert kept_indices == [0, 5]

    def test_non_sorted_input_yields_sorted_greedy_selection(self):
        # Input out of order — helper sorts internally and returns mask
        # aligned to original positions. With horizon=21, dates sorted are
        # [2026-01-01, 2026-02-01, 2026-03-01]. From sorted: keep all three
        # (each >= 21 days past previous). Mask in original order:
        dates = [_ts("2026-03-01"), _ts("2026-01-01"), _ts("2026-02-01")]
        mask = _nonoverlapping_date_mask(dates, 21)
        assert mask == [True, True, True]

    def test_horizon_exceeds_span_keeps_only_first(self):
        # 3 dates spanning 60 days; horizon=90 means only the first sticks.
        dates = [_ts("2026-01-01"), _ts("2026-01-31"), _ts("2026-02-28")]
        mask = _nonoverlapping_date_mask(dates, 90)
        # Sorted ascending: 1/1, 1/31, 2/28. First kept; second is 30d past
        # → < 90d → drop. Third is 58d past first → < 90d → drop.
        kept = [i for i, m in enumerate(mask) if m]
        assert len(kept) == 1
        # The single kept date should be the earliest one (2026-01-01),
        # which sits at original index 0.
        assert kept == [0]

    def test_exact_horizon_boundary_kept(self):
        # Two dates exactly h=21 days apart — second should be kept (>= boundary).
        dates = [_ts("2026-01-01"), _ts("2026-01-22")]  # 21-day gap
        mask = _nonoverlapping_date_mask(dates, 21)
        assert mask == [True, True]

    def test_one_day_inside_boundary_dropped(self):
        # Two dates 20 days apart, horizon=21 — second dropped.
        dates = [_ts("2026-01-01"), _ts("2026-01-21")]  # 20-day gap
        mask = _nonoverlapping_date_mask(dates, 21)
        assert mask == [True, False]

    def test_long_horizon_realistic_case(self):
        # 24 months of daily dates (~504 trading days), horizon=90d.
        # Expect ~8 kept windows (504 / 90 ≈ 5.6 calendar months → 8 windows).
        # Use calendar days (more conservative spacing) so 24 months × 30 ≈ 720d.
        # 720d / 90d horizon → ~8 windows.
        import datetime
        start = datetime.date(2024, 1, 1)
        dates = [pd.Timestamp(start + datetime.timedelta(days=i)) for i in range(720)]
        mask = _nonoverlapping_date_mask(dates, 90)
        n_kept = sum(mask)
        assert 7 <= n_kept <= 9, f"expected ~8 kept dates, got {n_kept}"

    def test_short_horizon_realistic_case(self):
        # 24 months daily, horizon=5d → ~720/5 = ~144 kept.
        import datetime
        start = datetime.date(2024, 1, 1)
        dates = [pd.Timestamp(start + datetime.timedelta(days=i)) for i in range(720)]
        mask = _nonoverlapping_date_mask(dates, 5)
        n_kept = sum(mask)
        assert 140 <= n_kept <= 145, f"expected ~144 kept dates, got {n_kept}"

    def test_accepts_string_dates(self):
        # Helper coerces via pd.Timestamp; string input should work.
        dates = ["2026-01-01", "2026-01-22", "2026-02-12"]
        mask = _nonoverlapping_date_mask(dates, 21)
        # 0d, 21d, 42d → all kept (each >= 21d past previous).
        assert mask == [True, True, True]

    def test_duplicate_dates_keep_one(self):
        # Two rows with the same date — only one kept (the second is 0
        # days past the first, < horizon).
        dates = [_ts("2026-01-01"), _ts("2026-01-01"), _ts("2026-01-22")]
        mask = _nonoverlapping_date_mask(dates, 21)
        # First idx of any duplicate-date set is kept (greedy walk in
        # sorted order); second copy of 2026-01-01 has zero gap → drop.
        # 2026-01-22 is 21d past → keep.
        kept = [i for i, m in enumerate(mask) if m]
        assert len(kept) == 2
        # Original-index of the kept Jan-01 is the smaller index (sort-stable).
        assert 0 in kept and 2 in kept
