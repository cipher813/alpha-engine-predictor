"""Stage F1 — daily BOCPD fast-signal tests (regime-fast-signal-260515.md).

Covers: daily-hazard params (weekly path untouched), state (de)serialization
round-trip + corruption, the confirmation/hysteresis state machine
(warmup suppression, fast latch, slow asymmetric release), and
trading-day idempotency (no double BOCPD advance on coverage re-invoke).
"""
from __future__ import annotations

import numpy as np
import pytest

from regime.bocpd import (
    DAILY_HAZARD,
    DAILY_MAX_RUNLENGTH_HORIZON,
    WEEKLY_HAZARD,
    BOCPDDetector,
)
from regime.fast_signal import (
    FAST_SIGNAL_SCHEMA_VERSION,
    FastSignalState,
    FastSignalTunables,
    deserialize_bocpd_state,
    dump_state,
    load_state,
    serialize_bocpd_state,
    step,
)


# ── Cadence-specific priors ──────────────────────────────────────────────

def test_weekly_defaults_untouched():
    """The bare constructor (used by the weekly substrate handler) must
    still carry the weekly prior — F1 must not regress the weekly path."""
    d = BOCPDDetector()
    assert d.hazard == WEEKLY_HAZARD == 1.0 / 100.0
    assert d.max_runlength_horizon == 520


def test_for_daily_uses_daily_prior_and_overrides():
    d = BOCPDDetector.for_daily()
    assert d.hazard == DAILY_HAZARD == 1.0 / 500.0
    assert d.max_runlength_horizon == DAILY_MAX_RUNLENGTH_HORIZON
    swept = BOCPDDetector.for_daily(hazard=1 / 750.0)
    assert swept.hazard == 1 / 750.0
    assert swept.max_runlength_horizon == DAILY_MAX_RUNLENGTH_HORIZON


# ── Serialization ────────────────────────────────────────────────────────

def test_bocpd_state_roundtrip():
    det = BOCPDDetector.for_daily()
    st = det.initial_state()
    for x in (0.1, -0.4, 2.0, -3.0):
        st = det.update(st, x)
    back = deserialize_bocpd_state(serialize_bocpd_state(st))
    for attr in ("runlength_probs", "mu_t", "kappa_t", "alpha_t", "beta_t"):
        np.testing.assert_allclose(getattr(back, attr), getattr(st, attr))


def test_full_state_roundtrip():
    det = BOCPDDetector.for_daily()
    st = FastSignalState(
        bocpd=det.update(det.initial_state(), -2.0),
        forced_bear=True,
        forced_bear_since="2026-05-14",
        consecutive_change_days=3,
        consecutive_clear_days=0,
        last_update_trading_day="2026-05-15",
        last_intensity_z=-2.0,
        observations_seen=42,
    )
    back = load_state(dump_state(st))
    assert back.forced_bear is True
    assert back.forced_bear_since == "2026-05-14"
    assert back.consecutive_change_days == 3
    assert back.last_update_trading_day == "2026-05-15"
    assert back.observations_seen == 42
    np.testing.assert_allclose(
        back.bocpd.runlength_probs, st.bocpd.runlength_probs
    )


def test_load_state_rejects_schema_mismatch():
    det = BOCPDDetector.for_daily()
    d = dump_state(FastSignalState(bocpd=det.initial_state()))
    d["schema_version"] = FAST_SIGNAL_SCHEMA_VERSION + 1
    with pytest.raises(ValueError):
        load_state(d)


def test_deserialize_rejects_corrupt_arrays():
    with pytest.raises(ValueError):
        deserialize_bocpd_state({
            "runlength_probs": [1.0, 0.0],
            "mu_t": [0.0],          # inconsistent length
            "kappa_t": [1.0],
            "alpha_t": [1.0],
            "beta_t": [1.0],
        })


# ── State machine ────────────────────────────────────────────────────────

def _drive(values, *, tun, det=None):
    """Feed a list of intensity_z values one trading day at a time."""
    det = det or BOCPDDetector.for_daily()
    state = None
    arts = []
    for i, v in enumerate(values):
        d = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        state, art = step(
            state, intensity_z=float(v), trading_day=d, calendar_date=d,
            run_id="t", detector=det, tunables=tun,
        )
        arts.append(art)
    return state, arts


def test_warmup_suppresses_forced_bear():
    """A cold start that immediately looks bearish must NOT latch while
    still in the warmup window."""
    tun = FastSignalTunables(min_runlength_for_change=8, min_persist_days=2)
    _, arts = _drive([-5.0] * 5, tun=tun)
    assert all(a["warmup"] for a in arts)
    assert not any(a["forced_bear"] for a in arts)
    assert arts[0]["cold_start"] is True


def test_latch_then_asymmetric_release():
    """Stable regime → sharp sustained risk-off → forced_bear latches
    after min_persist_days; recovery releases only after min_clear_days
    AND intensity_z back above the (higher) exit band."""
    tun = FastSignalTunables(
        min_runlength_for_change=3, min_persist_days=2,
        min_clear_days=4, intensity_floor=-1.0, exit_band=-0.3,
    )
    # 40 calm days (escape warmup, settle the posterior), then a strong
    # sustained negative shift, then a sustained recovery above exit band.
    calm = [0.0] * 40
    shock = [-4.0] * 6
    recover = [0.5] * 8
    state, arts = _drive(calm + shock + recover, tun=tun)

    shock_arts = arts[40:46]
    assert any(a["forced_bear"] for a in shock_arts), (
        "forced_bear should latch during a sustained confirmed break"
    )
    # Latch is not instantaneous — needs persistence.
    assert shock_arts[0]["forced_bear"] is False
    # By end of recovery it has released.
    assert arts[-1]["forced_bear"] is False
    assert state.forced_bear is False


def test_idempotent_same_trading_day():
    """Re-invocation for an already-processed trading_day must not
    advance the BOCPD posterior (CheckPredictorCoverage re-invoke)."""
    tun = FastSignalTunables(min_runlength_for_change=3)
    det = BOCPDDetector.for_daily()
    s1, a1 = step(None, intensity_z=-0.5, trading_day="2026-05-15",
                   calendar_date="2026-05-15", run_id="r1", detector=det,
                   tunables=tun)
    assert a1["observed"] is True
    obs_after_first = s1.observations_seen
    rl_after_first = len(s1.bocpd.runlength_probs)

    s2, a2 = step(s1, intensity_z=-9.9, trading_day="2026-05-15",
                   calendar_date="2026-05-15", run_id="r2", detector=det,
                   tunables=tun)
    assert a2["observed"] is False
    assert s2 is s1                                  # unchanged state object
    assert s2.observations_seen == obs_after_first   # no double-advance
    assert len(s2.bocpd.runlength_probs) == rl_after_first


def test_new_trading_day_advances():
    tun = FastSignalTunables(min_runlength_for_change=3)
    det = BOCPDDetector.for_daily()
    s1, _ = step(None, intensity_z=0.1, trading_day="2026-05-15",
                  calendar_date="2026-05-15", run_id="r", detector=det,
                  tunables=tun)
    s2, a2 = step(s1, intensity_z=0.2, trading_day="2026-05-16",
                  calendar_date="2026-05-16", run_id="r", detector=det,
                  tunables=tun)
    assert a2["observed"] is True
    assert s2.observations_seen == s1.observations_seen + 1
    assert s2.last_update_trading_day == "2026-05-16"
