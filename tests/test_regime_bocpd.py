"""Tests for regime/bocpd.py — Bayesian online change-point detection."""
from __future__ import annotations

import numpy as np
import pytest

from regime.bocpd import BOCPDDetector, BOCPDState


pytest.importorskip("scipy")


def test_bocpd_initial_state_well_formed() -> None:
    detector = BOCPDDetector()
    state = detector.initial_state()
    assert isinstance(state, BOCPDState)
    assert len(state.runlength_probs) == 1
    assert state.runlength_probs[0] == pytest.approx(1.0)


def test_bocpd_run_length_grows_by_one_per_update() -> None:
    detector = BOCPDDetector()
    state = detector.initial_state()
    for n in range(1, 6):
        state = detector.update(state, observation=0.5 * n)
        assert len(state.runlength_probs) == n + 1


def test_bocpd_run_length_probs_sum_to_one() -> None:
    detector = BOCPDDetector()
    state = detector.initial_state()
    for x in np.random.default_rng(0).normal(0.0, 1.0, 30):
        state = detector.update(state, observation=float(x))
        assert state.runlength_probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_bocpd_detects_step_change_in_observation_series() -> None:
    """Synthetic step-change: 50 obs around mean=0, then 50 obs around
    mean=3 — change confidence should spike shortly after the shift."""
    detector = BOCPDDetector(hazard=1.0 / 50.0)
    rng = np.random.default_rng(0)
    pre_change = rng.normal(0.0, 0.5, 50)
    post_change = rng.normal(3.0, 0.5, 50)
    observations = np.concatenate([pre_change, post_change])

    state = detector.initial_state()
    confidences = []
    for x in observations:
        state = detector.update(state, observation=float(x))
        sig = detector.change_signal(state)
        confidences.append(sig["change_confidence"])
    confidences = np.array(confidences)

    # Pre-change confidence (latter half of the calm regime) should be low
    pre_change_conf = confidences[40:50].mean()
    # Post-change confidence (immediately after the shift) should be high
    post_change_conf = confidences[50:55].max()
    assert post_change_conf > pre_change_conf + 0.3, (
        f"pre-change conf {pre_change_conf:.3f} → post-change conf {post_change_conf:.3f}; "
        f"expected a sharp jump after the regime shift."
    )


def test_bocpd_no_false_alarm_on_stationary_series() -> None:
    """50 obs from a single Gaussian — change_signal should not fire
    after the algorithm's warm-up period.

    Early observations naturally produce inflated change_confidence
    because the run-length posterior has little mass to spread across
    long run lengths. Skip the first 15 obs (warm-up) and require zero
    false alarms in the stationary tail.
    """
    detector = BOCPDDetector()
    rng = np.random.default_rng(0)
    observations = rng.normal(0.0, 1.0, 50)

    state = detector.initial_state()
    fired_after_warmup = 0
    for i, x in enumerate(observations):
        state = detector.update(state, observation=float(x))
        sig = detector.change_signal(state, change_threshold=0.5)
        if i >= 15 and sig["change_signal"]:
            fired_after_warmup += 1
    assert fired_after_warmup == 0, (
        f"expected zero false alarms on stationary tail after warm-up, got {fired_after_warmup}"
    )


def test_bocpd_tail_collapse_bounds_memory() -> None:
    """After many observations the run-length array should be bounded
    by ``max_runlength_horizon``."""
    detector = BOCPDDetector(max_runlength_horizon=20)
    state = detector.initial_state()
    for _ in range(100):
        state = detector.update(state, observation=0.5)
    assert len(state.runlength_probs) <= 20
    assert state.runlength_probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_bocpd_run_offline_returns_signal_per_step() -> None:
    detector = BOCPDDetector()
    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 1.0, 25)
    final_state, signals = detector.run_offline(obs)
    assert len(signals) == 25
    assert isinstance(final_state, BOCPDState)
    for sig in signals:
        assert set(sig.keys()) == {
            "change_signal",
            "max_runlength_prob",
            "change_confidence",
        }
