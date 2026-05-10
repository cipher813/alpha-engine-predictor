"""Tests for ``labeling.sample_weights.average_uniqueness_weights``.

Validates LdP Ch. 4.4 algorithm on hand-checkable cases:
- Non-overlapping windows yield weight 1.0 (max uniqueness)
- Fully overlapping identical windows yield weight 1/n (n co-occurring labels)
- Group separation prevents cross-group concurrency pollution
- API edge cases: empty input, scalar/array length param, validation
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeling.sample_weights import (
    _avg_uniqueness_single_group,
    average_uniqueness_weights,
)


class TestAverageUniquenessSingleGroup:

    def test_non_overlapping_windows_yield_weight_one(self):
        # Three windows at [0,5), [10,15), [20,25) — no overlap.
        # Each position has concurrency 1; mean(1/1) = 1.0 per label.
        starts = np.array([0, 10, 20])
        lengths = np.array([5, 5, 5])
        weights = _avg_uniqueness_single_group(starts, lengths)
        assert np.allclose(weights, 1.0)

    def test_three_identical_windows_yield_one_third(self):
        # Three identical windows at [0,5) — concurrency=3 everywhere
        # in the window. mean(1/3) = 1/3 per label.
        starts = np.array([0, 0, 0])
        lengths = np.array([5, 5, 5])
        weights = _avg_uniqueness_single_group(starts, lengths)
        assert np.allclose(weights, 1.0 / 3.0)

    def test_partial_overlap_intermediate_weight(self):
        # Two windows [0,10) and [5,15). Overlap region [5,10) has
        # concurrency 2; outside-overlap regions have concurrency 1.
        # Each window: 5 positions with c=1 and 5 positions with c=2.
        # Average uniqueness per label = mean(1.0, 1.0, 1.0, 1.0, 1.0,
        # 0.5, 0.5, 0.5, 0.5, 0.5) = 0.75.
        starts = np.array([0, 5])
        lengths = np.array([10, 10])
        weights = _avg_uniqueness_single_group(starts, lengths)
        assert np.allclose(weights, 0.75)

    def test_chain_overlap_decays_smoothly(self):
        # Windows starting every step at same length — concurrency grows
        # then plateaus then decays. Inner windows have c≈window_length;
        # edge windows have lower mean concurrency. Weights should be
        # monotonically smallest in the middle.
        starts = np.arange(10)
        lengths = np.full(10, 5)
        weights = _avg_uniqueness_single_group(starts, lengths)
        # Sanity: all weights are in (0, 1]; smallest is around the middle
        # of the chain (most overlap concurrency).
        assert (weights > 0).all()
        assert (weights <= 1.0).all()
        middle_weight = weights[5]
        edge_weight = weights[0]
        assert middle_weight < edge_weight


class TestAverageUniquenessWeights:

    def test_scalar_length_broadcasts(self):
        starts = np.array([0, 10, 20])
        weights = average_uniqueness_weights(starts, window_length=5)
        # Same as the non-overlap test above
        assert np.allclose(weights, 1.0)

    def test_array_length_per_row(self):
        # Three labels with different individual horizons but no overlap.
        starts = np.array([0, 100, 200])
        lengths = np.array([5, 10, 20])
        weights = average_uniqueness_weights(starts, window_length=lengths)
        assert np.allclose(weights, 1.0)

    def test_groups_isolate_concurrency(self):
        # Two tickers, both with three identical windows at [0, 5).
        # WITHIN each ticker → concurrency=3, weight=1/3.
        # WITHOUT grouping → concurrency=6, weight=1/6.
        starts = np.array([0, 0, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        grouped = average_uniqueness_weights(starts, window_length=5, group_ids=groups)
        ungrouped = average_uniqueness_weights(starts, window_length=5)
        assert np.allclose(grouped, 1.0 / 3.0)
        assert np.allclose(ungrouped, 1.0 / 6.0)

    def test_realistic_per_ticker_walk(self):
        # 60 trading days, two tickers, daily labels with 21d window.
        # Mid-corpus rows of each ticker should have low weights (max
        # within-ticker overlap); first/last rows should have higher
        # weights (less overlap).
        n_per_ticker = 60
        forward = 21
        starts = np.tile(np.arange(n_per_ticker), 2)
        groups = np.array(["A"] * n_per_ticker + ["B"] * n_per_ticker)
        weights = average_uniqueness_weights(
            starts, window_length=forward, group_ids=groups,
        )
        # Within ticker A: weight at row 30 (mid) < weight at row 0 (edge)
        assert weights[30] < weights[0]
        # Cross-ticker independence: the corresponding row of ticker B
        # should have the same weight as ticker A's row at same position.
        assert np.allclose(weights[30], weights[n_per_ticker + 30])

    def test_empty_input(self):
        weights = average_uniqueness_weights(np.array([], dtype=np.int64))
        assert weights.shape == (0,)

    def test_mismatched_array_length_raises(self):
        starts = np.array([0, 5, 10])
        bad_lengths = np.array([5, 5])  # wrong shape
        try:
            average_uniqueness_weights(starts, window_length=bad_lengths)
        except ValueError:
            return
        raise AssertionError("expected ValueError on length mismatch")

    def test_mismatched_group_length_raises(self):
        starts = np.array([0, 5, 10])
        bad_groups = np.array(["A", "B"])  # wrong shape
        try:
            average_uniqueness_weights(starts, window_length=5, group_ids=bad_groups)
        except ValueError:
            return
        raise AssertionError("expected ValueError on group length mismatch")
