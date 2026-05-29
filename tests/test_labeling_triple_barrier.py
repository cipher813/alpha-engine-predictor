"""Tests for ``labeling.triple_barrier`` — generic triple-barrier labels.

Covers the two label flavors:

- ``triple_barrier_class_labels`` — 3-class direction labels. Parity
  with the v2 module's existing tests, plus generic-naming validation.

- ``triple_barrier_alpha_labels`` — continuous regression target.
  Validates barrier capping, time-out passthrough, and tail sentinel.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from labeling.triple_barrier import (
    TRIPLE_BARRIER_SENTINEL,
    triple_barrier_alpha_labels,
    triple_barrier_class_labels,
    triple_barrier_touch_order,
)


# ── Class-label tests ───────────────────────────────────────────────────


class TestTripleBarrierClassLabels:

    def test_all_zero_returns_yields_neutral(self):
        log_returns = np.zeros(100)
        labels = triple_barrier_class_labels(log_returns, forward_window=21)
        finite = labels[labels != TRIPLE_BARRIER_SENTINEL]
        assert (finite == 1).all()

    def test_steady_uptrend_yields_up_class(self):
        log_returns = np.full(100, 0.01)
        labels = triple_barrier_class_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.05,
            down_barrier_pct=0.05,
        )
        finite = labels[labels != TRIPLE_BARRIER_SENTINEL]
        assert (finite == 2).all()

    def test_steady_downtrend_yields_down_class(self):
        log_returns = np.full(100, -0.01)
        labels = triple_barrier_class_labels(log_returns, forward_window=21)
        finite = labels[labels != TRIPLE_BARRIER_SENTINEL]
        assert (finite == 0).all()

    def test_tail_rows_get_sentinel(self):
        log_returns = np.full(50, 0.005)
        labels = triple_barrier_class_labels(log_returns, forward_window=21)
        assert (labels[-21:] == TRIPLE_BARRIER_SENTINEL).all()
        assert (labels[:-21] != TRIPLE_BARRIER_SENTINEL).all()

    def test_window_end_drift_above_half_band_yields_up_class(self):
        # Each step is small enough not to cross the full barrier, but
        # the cumulative drift over the window crosses the half-band.
        # At step 0.003 over 21 days, cum = 0.063 > up/2 = 0.025 → up-class.
        log_returns = np.full(50, 0.003)
        labels = triple_barrier_class_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.05,
            down_barrier_pct=0.05,
        )
        finite = labels[labels != TRIPLE_BARRIER_SENTINEL]
        assert (finite == 2).all()

    def test_label_distribution_on_mixed_returns(self):
        rng = np.random.default_rng(0)
        log_returns = rng.normal(0, 0.005, 200)
        labels = triple_barrier_class_labels(log_returns, forward_window=21)
        finite = labels[labels != TRIPLE_BARRIER_SENTINEL]
        assert len(finite) == 200 - 21
        unique = set(finite.tolist())
        # Neutral should appear at this volatility scale.
        assert 1 in unique


# ── Alpha-label tests ───────────────────────────────────────────────────


class TestTripleBarrierAlphaLabels:

    def test_all_zero_returns_yields_zero_label(self):
        log_returns = np.zeros(100)
        labels = triple_barrier_alpha_labels(log_returns, forward_window=21)
        finite = labels[~np.isnan(labels)]
        assert np.allclose(finite, 0.0)

    def test_steady_uptrend_caps_at_up_barrier(self):
        # +1% per day — crosses +5% barrier on day 5; label = +0.05 (capped).
        log_returns = np.full(100, 0.01)
        labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.05,
            down_barrier_pct=0.05,
        )
        finite = labels[~np.isnan(labels)]
        assert np.allclose(finite, 0.05)

    def test_steady_downtrend_caps_at_down_barrier(self):
        log_returns = np.full(100, -0.01)
        labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.05,
            down_barrier_pct=0.05,
        )
        finite = labels[~np.isnan(labels)]
        assert np.allclose(finite, -0.05)

    def test_no_barrier_hit_returns_window_end_cumulative(self):
        # +0.001/day over 21 days — cum = 0.021 < up barrier 0.05; label
        # should be approximately the window-end cumulative return.
        log_returns = np.full(50, 0.001)
        labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.05,
            down_barrier_pct=0.05,
        )
        finite = labels[~np.isnan(labels)]
        # 21 steps of 0.001 each = 0.021 cumulative
        assert np.allclose(finite, 0.021, atol=1e-9)

    def test_tail_rows_get_nan_sentinel(self):
        log_returns = np.full(50, 0.005)
        labels = triple_barrier_alpha_labels(log_returns, forward_window=21)
        assert np.isnan(labels[-21:]).all()
        assert (~np.isnan(labels[:-21])).all()

    def test_label_within_barrier_envelope(self):
        # Mixed series: every label must lie in [-down_barrier, +up_barrier].
        rng = np.random.default_rng(0)
        log_returns = rng.normal(0, 0.005, 300)
        labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.04,
            down_barrier_pct=0.03,
        )
        finite = labels[~np.isnan(labels)]
        # Time-out paths can have window-end cum return outside the
        # barriers' midline (since we only check first-touch); but for
        # a no-hit path the cum return is bounded by NOT having crossed
        # either barrier — so it lies in (-down_barrier, +up_barrier).
        # Hit paths are capped exactly at the touched barrier.
        # Therefore: every label ∈ [-down_barrier, +up_barrier].
        assert (finite >= -0.03 - 1e-9).all()
        assert (finite <= 0.04 + 1e-9).all()

    def test_asymmetric_barriers_respected(self):
        # Up barrier 0.10, down barrier 0.02 — uptrend touches up first.
        log_returns = np.full(50, 0.01)
        labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.10,
            down_barrier_pct=0.02,
        )
        finite = labels[~np.isnan(labels)]
        # 10 days at 0.01 = 0.10 → first touches up barrier on day 10.
        assert np.allclose(finite, 0.10)


# ── Per-row (ndarray) barrier tests — Stage 3 vol-scaling support ──────


class TestTripleBarrierAlphaPerRowBarriers:
    """LdP Ch. 3.4 vol-scaled barriers — per-row barrier widths broadcast
    via ndarray. PR 1 of the Stage 3 arc adds this signature widening.
    """

    def test_scalar_and_array_barriers_agree_when_uniform(self):
        # Scalar 0.05 vs ndarray of 0.05 should produce identical labels.
        log_returns = np.full(60, 0.005)
        scalar_labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.05,
            down_barrier_pct=0.05,
        )
        array_labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=np.full(60, 0.05),
            down_barrier_pct=np.full(60, 0.05),
        )
        # Use array_equal w/ NaN-equal semantics
        assert np.array_equal(scalar_labels, array_labels, equal_nan=True)

    def test_per_row_barriers_apply_row_wise(self):
        # Two distinct regimes encoded into the barrier array:
        # rows 0-29: tight barrier 0.02 (steep cap)
        # rows 30-59: wide barrier 0.20 (effectively no barrier hit at this scale)
        # Underlying returns are 0.005/day → 21-day cum = 0.105.
        # Tight regime: 4 days × 0.005 = 0.02 → caps at 0.02.
        # Wide regime: 21 days × 0.005 = 0.105 < 0.20 → time-out at 0.105.
        log_returns = np.full(60, 0.005)
        up_arr = np.concatenate([np.full(30, 0.02), np.full(30, 0.20)])
        down_arr = np.concatenate([np.full(30, 0.02), np.full(30, 0.20)])
        labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=up_arr,
            down_barrier_pct=down_arr,
        )
        # Rows 0-29 cap at 0.02
        assert np.allclose(labels[:30], 0.02)
        # Rows 30-38 time out at 0.105 (last 21 rows are NaN tail; 39-59 → tail)
        time_out_region = labels[30:60 - 21]
        assert np.allclose(time_out_region, 0.105, atol=1e-9)

    def test_nan_barrier_at_row_yields_nan_label(self):
        # Insufficient vol history → NaN barrier → NaN label at that row.
        log_returns = np.full(60, 0.005)
        up_arr = np.full(60, 0.05)
        up_arr[:10] = np.nan  # simulate front-of-history vol gap
        down_arr = np.full(60, 0.05)
        down_arr[:10] = np.nan
        labels = triple_barrier_alpha_labels(
            log_returns,
            forward_window=21,
            up_barrier_pct=up_arr,
            down_barrier_pct=down_arr,
        )
        # Rows 0-9 NaN due to NaN barrier
        assert np.isnan(labels[:10]).all()
        # Rows 10-38 finite (39-59 are tail NaN)
        assert (~np.isnan(labels[10:60 - 21])).all()

    def test_mismatched_barrier_length_raises(self):
        # Per-row arrays must match log_returns length — broadcast_to raises.
        log_returns = np.full(60, 0.005)
        bad_up = np.full(50, 0.05)  # wrong length
        try:
            triple_barrier_alpha_labels(
                log_returns,
                forward_window=21,
                up_barrier_pct=bad_up,
                down_barrier_pct=0.05,
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError on length mismatch")


# ── Cross-flavor sanity ────────────────────────────────────────────────


class TestCrossFlavor:
    """Class and alpha labels should agree on direction for unambiguous paths."""

    def test_uptrend_class_up_alpha_positive(self):
        log_returns = np.full(50, 0.01)
        class_labels = triple_barrier_class_labels(log_returns, forward_window=21)
        alpha_labels = triple_barrier_alpha_labels(log_returns, forward_window=21)
        finite_class = class_labels[class_labels != TRIPLE_BARRIER_SENTINEL]
        finite_alpha = alpha_labels[~np.isnan(alpha_labels)]
        assert (finite_class == 2).all()
        assert (finite_alpha > 0).all()

    def test_downtrend_class_down_alpha_negative(self):
        log_returns = np.full(50, -0.01)
        class_labels = triple_barrier_class_labels(log_returns, forward_window=21)
        alpha_labels = triple_barrier_alpha_labels(log_returns, forward_window=21)
        finite_class = class_labels[class_labels != TRIPLE_BARRIER_SENTINEL]
        finite_alpha = alpha_labels[~np.isnan(alpha_labels)]
        assert (finite_class == 0).all()
        assert (finite_alpha < 0).all()

    def test_flat_class_neutral_alpha_zero(self):
        log_returns = np.zeros(50)
        class_labels = triple_barrier_class_labels(log_returns, forward_window=21)
        alpha_labels = triple_barrier_alpha_labels(log_returns, forward_window=21)
        finite_class = class_labels[class_labels != TRIPLE_BARRIER_SENTINEL]
        finite_alpha = alpha_labels[~np.isnan(alpha_labels)]
        assert (finite_class == 1).all()
        assert np.allclose(finite_alpha, 0.0)


# ── Touch-order meta-label tests (Task B) ───────────────────────────────


class TestTripleBarrierTouchOrder:
    """López de Prado meta-label: which barrier (up/down) is touched first.

    Supervision target for the Task B meta-label classifier whose calibrated
    P(up before down) feeds executor position sizing.
    """

    def test_steady_uptrend_yields_one(self):
        # +1%/day crosses the up barrier first → label 1.0.
        log_returns = np.full(100, 0.01)
        labels = triple_barrier_touch_order(
            log_returns, forward_window=21, up_barrier_pct=0.05, down_barrier_pct=0.05
        )
        finite = labels[~np.isnan(labels)]
        assert (finite == 1.0).all()

    def test_steady_downtrend_yields_zero(self):
        log_returns = np.full(100, -0.01)
        labels = triple_barrier_touch_order(
            log_returns, forward_window=21, up_barrier_pct=0.05, down_barrier_pct=0.05
        )
        finite = labels[~np.isnan(labels)]
        assert (finite == 0.0).all()

    def test_timeout_nan_policy_excludes_no_touch_rows(self):
        # Flat returns never touch either barrier → time-out → NaN (default).
        log_returns = np.zeros(60)
        labels = triple_barrier_touch_order(log_returns, forward_window=21)
        # every non-tail row is a time-out → all NaN under "nan" policy
        assert np.isnan(labels).all()

    def test_timeout_sign_policy_labels_by_window_end(self):
        # Small positive drift never touches the barrier but ends positive
        # → "sign" policy labels it 1.0.
        log_returns = np.full(60, 0.001)
        labels = triple_barrier_touch_order(
            log_returns,
            forward_window=21,
            up_barrier_pct=0.05,
            down_barrier_pct=0.05,
            timeout_policy="sign",
        )
        finite = labels[~np.isnan(labels)]
        assert (finite == 1.0).all()

    def test_timeout_sign_policy_negative_drift_zero(self):
        log_returns = np.full(60, -0.001)
        labels = triple_barrier_touch_order(
            log_returns, forward_window=21, timeout_policy="sign"
        )
        finite = labels[~np.isnan(labels)]
        assert (finite == 0.0).all()

    def test_tail_rows_get_nan(self):
        log_returns = np.full(50, 0.01)
        labels = triple_barrier_touch_order(log_returns, forward_window=21)
        assert np.isnan(labels[-21:]).all()

    def test_asymmetric_barriers_change_touch_order(self):
        # Mild uptrend: with a tight up barrier it touches up; with a tight
        # down barrier (and wide up) the same path would not touch up first.
        log_returns = np.full(60, 0.004)  # +0.4%/day
        up_first = triple_barrier_touch_order(
            log_returns, forward_window=21, up_barrier_pct=0.02, down_barrier_pct=0.50
        )
        assert (up_first[~np.isnan(up_first)] == 1.0).all()

    def test_per_row_barrier_nan_propagates(self):
        log_returns = np.full(60, 0.01)
        up_arr = np.full(60, 0.05)
        up_arr[:10] = np.nan
        labels = triple_barrier_touch_order(
            log_returns, forward_window=21, up_barrier_pct=up_arr, down_barrier_pct=0.05
        )
        assert np.isnan(labels[:10]).all()
        assert (~np.isnan(labels[10:60 - 21])).all()

    def test_invalid_timeout_policy_raises(self):
        with pytest.raises(ValueError):
            triple_barrier_touch_order(np.zeros(50), timeout_policy="bogus")
