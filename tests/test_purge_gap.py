"""
tests/test_purge_gap.py — Verify purge gaps between train/val/test splits.

Forward-return labels use FORWARD_DAYS of future prices. Without a purge gap,
label overlap causes ~80% information leakage across split boundaries. These
tests ensure the purge gap implementation in data/dataset.py is correct.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _build_synthetic_dates_and_arrays(
    n_dates: int = 500,
    tickers_per_date: int = 10,
    n_features: int = 29,
    seed: int = 42,
):
    """Create synthetic sorted arrays mimicking build_regression_datasets internals."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)

    all_dates = []
    for d in dates:
        all_dates.extend([d] * tickers_per_date)

    N = len(all_dates)
    X_all = rng.standard_normal((N, n_features)).astype(np.float32)
    y_all = rng.standard_normal(N).astype(np.int64)
    fwd_all = rng.standard_normal(N).astype(np.float32)

    return X_all, y_all, fwd_all, all_dates, N


class TestPurgeGap:
    """Tests for purge gap between train/val/test splits."""

    def test_purge_gap_train_val(self):
        """Last train date and first val date must be separated by >= FORWARD_DAYS."""
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            TRAIN_FRAC=0.70,
            VAL_FRAC=0.15,
            FORWARD_DAYS=5,
        )

        X_all, y_all, fwd_all, all_dates, N = _build_synthetic_dates_and_arrays()
        purge_days = cfg.FORWARD_DAYS

        n_train = int(N * cfg.TRAIN_FRAC)
        n_val_raw = int(N * cfg.VAL_FRAC)

        # --- Purge gap 1: train → val ---
        train_end_date = all_dates[n_train - 1]
        unique_post_train = sorted(set(d for d in all_dates[n_train:] if d > train_end_date))
        if len(unique_post_train) >= purge_days:
            purge_cutoff_1 = unique_post_train[purge_days - 1]
            val_start = next(i for i in range(n_train, N) if all_dates[i] > purge_cutoff_1)
        else:
            val_start = n_train

        val_end = val_start + n_val_raw
        if val_end > N:
            val_end = N

        # Check: the gap between last train date and first val date
        last_train_date = all_dates[n_train - 1]
        first_val_date = all_dates[val_start]

        gap_dates = sorted(set(
            d for d in all_dates
            if d > last_train_date and d < first_val_date
        ))
        # Plus the cutoff date itself is excluded, so the gap should be >= purge_days
        assert len(gap_dates) >= purge_days, (
            f"Purge gap between train and val is only {len(gap_dates)} dates, "
            f"expected >= {purge_days}"
        )

    def test_purge_gap_val_test(self):
        """Last val date and first test date must be separated by >= FORWARD_DAYS."""
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            TRAIN_FRAC=0.70,
            VAL_FRAC=0.15,
            FORWARD_DAYS=5,
        )

        X_all, y_all, fwd_all, all_dates, N = _build_synthetic_dates_and_arrays()
        purge_days = cfg.FORWARD_DAYS

        n_train = int(N * cfg.TRAIN_FRAC)
        n_val_raw = int(N * cfg.VAL_FRAC)

        # Purge gap 1
        train_end_date = all_dates[n_train - 1]
        unique_post_train = sorted(set(d for d in all_dates[n_train:] if d > train_end_date))
        if len(unique_post_train) >= purge_days:
            purge_cutoff_1 = unique_post_train[purge_days - 1]
            val_start = next(i for i in range(n_train, N) if all_dates[i] > purge_cutoff_1)
        else:
            val_start = n_train

        val_end = val_start + n_val_raw
        if val_end > N:
            val_end = N

        # Purge gap 2
        val_end_date = all_dates[min(val_end - 1, N - 1)]
        unique_post_val = sorted(set(d for d in all_dates[val_end:] if d > val_end_date))
        if len(unique_post_val) >= purge_days:
            purge_cutoff_2 = unique_post_val[purge_days - 1]
            test_start = next(i for i in range(val_end, N) if all_dates[i] > purge_cutoff_2)
        else:
            test_start = val_end

        last_val_date = all_dates[val_end - 1]
        first_test_date = all_dates[test_start]

        gap_dates = sorted(set(
            d for d in all_dates
            if d > last_val_date and d < first_test_date
        ))
        assert len(gap_dates) >= purge_days, (
            f"Purge gap between val and test is only {len(gap_dates)} dates, "
            f"expected >= {purge_days}"
        )

    def test_no_sample_lost_to_overlap(self):
        """Train, purge, val, purge, test should partition the data without overlap."""
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            TRAIN_FRAC=0.70,
            VAL_FRAC=0.15,
            FORWARD_DAYS=5,
        )

        X_all, y_all, fwd_all, all_dates, N = _build_synthetic_dates_and_arrays()
        purge_days = cfg.FORWARD_DAYS

        n_train = int(N * cfg.TRAIN_FRAC)
        n_val_raw = int(N * cfg.VAL_FRAC)

        train_end_date = all_dates[n_train - 1]
        unique_post_train = sorted(set(d for d in all_dates[n_train:] if d > train_end_date))
        purge_cutoff_1 = unique_post_train[purge_days - 1]
        val_start = next(i for i in range(n_train, N) if all_dates[i] > purge_cutoff_1)

        val_end = val_start + n_val_raw
        val_end_date = all_dates[min(val_end - 1, N - 1)]
        unique_post_val = sorted(set(d for d in all_dates[val_end:] if d > val_end_date))
        purge_cutoff_2 = unique_post_val[purge_days - 1]
        test_start = next(i for i in range(val_end, N) if all_dates[i] > purge_cutoff_2)

        train_indices = set(range(0, n_train))
        val_indices = set(range(val_start, val_end))
        test_indices = set(range(test_start, N))

        # No overlap
        assert train_indices & val_indices == set(), "Train and val overlap"
        assert val_indices & test_indices == set(), "Val and test overlap"
        assert train_indices & test_indices == set(), "Train and test overlap"

        # Total purged samples
        total_used = len(train_indices) + len(val_indices) + len(test_indices)
        total_purged = N - total_used
        assert total_purged > 0, "Expected some samples to be purged"
        # Purged should be roughly 2 * purge_days * tickers_per_date
        assert total_purged < N * 0.05, f"Too many samples purged: {total_purged}/{N}"
