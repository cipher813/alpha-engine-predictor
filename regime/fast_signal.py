"""
regime/fast_signal.py — daily fast regime-break circuit-breaker.

Stage F1 of regime-fast-signal-260515.md. Pure-logic core (no S3, no
boto3) so it is testable in isolation; the inference stage
(inference/stages/regime_fast_signal.py) wires the S3 I/O.

What this module owns
---------------------
1. (De)serialization of the persisted online state
   (``regime/bocpd_state.json``) — the BOCPD sufficient statistics plus
   the fast-signal state-machine bookkeeping.
2. The confirmation / asymmetric-hysteresis state machine that turns a
   noisy daily BOCPD change signal into a stable ``forced_bear`` latch.
3. ``step()`` — advance the state by one trading day, idempotently.

Why a state machine and not the raw BOCPD signal
------------------------------------------------
A single-day intensity_z spike that mean-reverts must NOT latch
forced-bear (the classic fast-overlay failure: de-risking into the
hole, missing the snap-back). Per regime-fast-signal-260515.md §2 the
three load-bearing guardrails are encoded here:

1. **Two-of-three confirmation.** A "break day" requires BOTH a BOCPD
   change (``change_confidence > change_threshold``) AND intensity_z
   corroboration (``intensity_z < intensity_floor``). The third leg —
   persistence — is the ``min_persist_days`` counter below.
2. **Asymmetric hysteresis.** Latch in fast (``min_persist_days``,
   default 2), release slow (``min_clear_days``, default 4) AND only
   once intensity_z recovers above a *higher* band (``exit_band``,
   default −0.3) than the entry band (``intensity_floor``, default
   −1.0). Asymmetry is correct because the loss function is asymmetric
   (drawdown ≫ opportunity cost under the skilled-risk gate).
3. **No double-counting** is enforced downstream (executor/veto take the
   max-protection of {continuous path, bear floor}); this module only
   emits the discrete ``forced_bear`` boolean.

Sign convention
----------------
``intensity_z`` here is the *inference-path* convention that
``run_inference`` stamps on ``ctx.regime_intensity_z``: **negative =
risk-off / bearish, positive = risk-on / bullish** (the downstream
contract; the composite is inverted at the substrate layer). Hence the
risk-off corroboration test is ``intensity_z < intensity_floor`` with a
negative floor. The BOCPD detector observes this same scalar — it keys
on the *change* in the series, so the sign convention only matters for
the corroboration test, not the detector itself.

Idempotency
-----------
``step()`` is keyed on ``trading_day``. Predictor inference can be
re-invoked the same day by the Step Function's ``CheckPredictorCoverage``
path; advancing the run-length posterior twice for one trading day would
corrupt the detector. A re-invocation for an already-processed
``trading_day`` returns the prior state unchanged and re-emits the same
artifact with ``observed=False``.

Cold start / corruption
-----------------------
This module never silently fabricates a settled "no bear". On a cold
start (no persisted state) or a corrupt/incompatible state file the
caller passes ``prev=None``; ``step()`` initializes a fresh detector and
flags the artifact ``warmup=True`` + ``cold_start=True``. The stage logs
this loudly and emits a CloudWatch metric (per feedback_no_silent_fails).
Rigorous historical calibration is the F1 backfill script's job, not the
live warm-up path.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .bocpd import (
    DAILY_MIN_RUNLENGTH_FOR_CHANGE,
    BOCPDDetector,
    BOCPDState,
)

logger = logging.getLogger(__name__)

# Bump on any breaking change to the persisted-state or artifact schema.
FAST_SIGNAL_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class FastSignalTunables:
    """Confirmation + hysteresis knobs. Defaults are F1 starting points;
    the backfill recalibrates them before F2 wires any consumer."""

    change_threshold: float = 0.5      # BOCPD change-confidence gate (leg A)
    intensity_floor: float = -1.0      # risk-off corroboration band (leg B)
    exit_band: float = -0.3            # hysteresis release band (> floor)
    min_persist_days: int = 2          # fast latch
    min_clear_days: int = 4            # slow release
    min_runlength_for_change: int = DAILY_MIN_RUNLENGTH_FOR_CHANGE

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class FastSignalState:
    """Full persisted state: BOCPD sufficient stats + machine bookkeeping.

    Serialized to ``regime/bocpd_state.json``. The five BOCPD arrays are
    stored as plain lists; everything else is JSON scalars.
    """

    bocpd: BOCPDState
    forced_bear: bool = False
    forced_bear_since: str | None = None
    consecutive_change_days: int = 0
    consecutive_clear_days: int = 0
    last_update_trading_day: str | None = None
    last_intensity_z: float | None = None
    observations_seen: int = 0
    schema_version: int = FAST_SIGNAL_SCHEMA_VERSION


# ── Serialization ────────────────────────────────────────────────────────

def serialize_bocpd_state(s: BOCPDState) -> dict[str, list[float]]:
    return {
        "runlength_probs": s.runlength_probs.astype(float).tolist(),
        "mu_t": s.mu_t.astype(float).tolist(),
        "kappa_t": s.kappa_t.astype(float).tolist(),
        "alpha_t": s.alpha_t.astype(float).tolist(),
        "beta_t": s.beta_t.astype(float).tolist(),
    }


def deserialize_bocpd_state(d: dict[str, Any]) -> BOCPDState:
    keys = ("runlength_probs", "mu_t", "kappa_t", "alpha_t", "beta_t")
    arrs = {k: np.asarray(d[k], dtype=float) for k in keys}
    lengths = {len(v) for v in arrs.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(
            f"corrupt BOCPD state: inconsistent/empty array lengths {lengths}"
        )
    return BOCPDState(**arrs)


def dump_state(st: FastSignalState) -> dict[str, Any]:
    return {
        "schema_version": st.schema_version,
        "bocpd": serialize_bocpd_state(st.bocpd),
        "forced_bear": bool(st.forced_bear),
        "forced_bear_since": st.forced_bear_since,
        "consecutive_change_days": int(st.consecutive_change_days),
        "consecutive_clear_days": int(st.consecutive_clear_days),
        "last_update_trading_day": st.last_update_trading_day,
        "last_intensity_z": (
            None if st.last_intensity_z is None else float(st.last_intensity_z)
        ),
        "observations_seen": int(st.observations_seen),
    }


def load_state(d: dict[str, Any]) -> FastSignalState:
    """Rehydrate a persisted state dict. Raises ``ValueError`` on schema
    mismatch or corruption — the caller treats that as a cold start."""
    ver = d.get("schema_version")
    if ver != FAST_SIGNAL_SCHEMA_VERSION:
        raise ValueError(
            f"fast-signal state schema {ver} != {FAST_SIGNAL_SCHEMA_VERSION}"
        )
    return FastSignalState(
        bocpd=deserialize_bocpd_state(d["bocpd"]),
        forced_bear=bool(d.get("forced_bear", False)),
        forced_bear_since=d.get("forced_bear_since"),
        consecutive_change_days=int(d.get("consecutive_change_days", 0)),
        consecutive_clear_days=int(d.get("consecutive_clear_days", 0)),
        last_update_trading_day=d.get("last_update_trading_day"),
        last_intensity_z=d.get("last_intensity_z"),
        observations_seen=int(d.get("observations_seen", 0)),
        schema_version=ver,
    )


# ── Core step ────────────────────────────────────────────────────────────

def step(
    prev: FastSignalState | None,
    *,
    intensity_z: float,
    trading_day: str,
    calendar_date: str,
    run_id: str,
    detector: BOCPDDetector,
    tunables: FastSignalTunables | None = None,
) -> tuple[FastSignalState, dict[str, Any]]:
    """Advance the fast signal by one trading day.

    Returns ``(new_state, artifact)``. ``new_state`` is what the stage
    persists to ``regime/bocpd_state.json``; ``artifact`` is the
    observe-only payload written to ``regime/fast_signal/``.

    ``prev=None`` ⇒ cold start (fresh detector, ``warmup``/``cold_start``
    flagged). Re-invocation for an already-processed ``trading_day`` is a
    no-op on the detector (idempotent; ``observed=False``).
    """
    tun = tunables or FastSignalTunables()
    cold_start = prev is None

    # ── Idempotent re-emit ───────────────────────────────────────────────
    if prev is not None and prev.last_update_trading_day == trading_day:
        logger.info(
            "fast-signal: trading_day %s already processed — idempotent "
            "re-emit (no BOCPD advance)", trading_day,
        )
        sig = detector.change_signal(
            prev.bocpd,
            min_runlength_for_change=tun.min_runlength_for_change,
            change_threshold=tun.change_threshold,
        )
        return prev, _artifact(
            prev, sig, intensity_z=prev.last_intensity_z if prev.last_intensity_z
            is not None else intensity_z,
            trading_day=trading_day, calendar_date=calendar_date,
            run_id=run_id, detector=detector, tun=tun,
            observed=False, warmup=False, cold_start=False,
            cold_start_reason=None,
        )

    # ── Advance the detector ─────────────────────────────────────────────
    state = (
        FastSignalState(bocpd=detector.initial_state())
        if cold_start else prev
    )
    new_bocpd = detector.update(state.bocpd, float(intensity_z))
    sig = detector.change_signal(
        new_bocpd,
        min_runlength_for_change=tun.min_runlength_for_change,
        change_threshold=tun.change_threshold,
    )

    # ── Two-of-three confirmation: break-day test ────────────────────────
    leg_a = bool(sig["change_signal"])                       # BOCPD change
    leg_b = float(intensity_z) < tun.intensity_floor         # risk-off corrob.
    break_day = leg_a and leg_b

    consec_change = state.consecutive_change_days
    consec_clear = state.consecutive_clear_days
    if break_day:
        consec_change += 1
        consec_clear = 0
    else:
        consec_clear += 1
        consec_change = 0

    forced_bear = state.forced_bear
    forced_bear_since = state.forced_bear_since

    # Latch (fast): not yet bear and persistence met.
    if not forced_bear and consec_change >= tun.min_persist_days:
        forced_bear = True
        forced_bear_since = trading_day
        logger.warning(
            "fast-signal: FORCED_BEAR latched on %s "
            "(intensity_z=%.3f, change_conf=%.3f, persisted %d days)",
            trading_day, intensity_z, sig["change_confidence"], consec_change,
        )
    # Release (slow + hysteresis band): clear days met AND recovered
    # above the *higher* exit band.
    elif (
        forced_bear
        and consec_clear >= tun.min_clear_days
        and float(intensity_z) > tun.exit_band
    ):
        forced_bear = False
        forced_bear_since = None
        logger.warning(
            "fast-signal: FORCED_BEAR released on %s "
            "(intensity_z=%.3f > exit_band=%.2f, cleared %d days)",
            trading_day, intensity_z, tun.exit_band, consec_clear,
        )

    new_state = FastSignalState(
        bocpd=new_bocpd,
        forced_bear=forced_bear,
        forced_bear_since=forced_bear_since,
        consecutive_change_days=consec_change,
        consecutive_clear_days=consec_clear,
        last_update_trading_day=trading_day,
        last_intensity_z=float(intensity_z),
        observations_seen=state.observations_seen + 1,
    )

    # Warmup = the detector has not yet seen enough observations for the
    # run-length posterior to be meaningful (a fresh NIG prior is
    # uninformative). Gate forced_bear OFF while warming so a cold start
    # cannot fire on its first informative-looking step.
    warmup = new_state.observations_seen < tun.min_runlength_for_change
    if warmup and new_state.forced_bear:
        logger.warning(
            "fast-signal: suppressing forced_bear during warmup "
            "(%d/%d obs)", new_state.observations_seen,
            tun.min_runlength_for_change,
        )
        new_state.forced_bear = False
        new_state.forced_bear_since = None

    art = _artifact(
        new_state, sig, intensity_z=float(intensity_z),
        trading_day=trading_day, calendar_date=calendar_date,
        run_id=run_id, detector=detector, tun=tun,
        observed=True, warmup=warmup, cold_start=cold_start,
        cold_start_reason=("no_prior_state" if cold_start else None),
    )
    return new_state, art


def _artifact(
    st: FastSignalState,
    sig: dict[str, Any],
    *,
    intensity_z: float,
    trading_day: str,
    calendar_date: str,
    run_id: str,
    detector: BOCPDDetector,
    tun: FastSignalTunables,
    observed: bool,
    warmup: bool,
    cold_start: bool,
    cold_start_reason: str | None,
) -> dict[str, Any]:
    return {
        "trading_day": trading_day,
        "calendar_date": calendar_date,
        "run_id": run_id,
        "schema_version": FAST_SIGNAL_SCHEMA_VERSION,
        "forced_bear": bool(st.forced_bear),
        "forced_bear_since": st.forced_bear_since,
        "change_signal": bool(sig["change_signal"]),
        "change_confidence": float(sig["change_confidence"]),
        "max_runlength_prob": float(sig["max_runlength_prob"]),
        "intensity_z": float(intensity_z),
        "consecutive_change_days": int(st.consecutive_change_days),
        "consecutive_clear_days": int(st.consecutive_clear_days),
        "observations_seen": int(st.observations_seen),
        "hazard": float(detector.hazard),
        "min_runlength_for_change": int(tun.min_runlength_for_change),
        "observed": bool(observed),
        "warmup": bool(warmup),
        "cold_start": bool(cold_start),
        "cold_start_reason": cold_start_reason,
        "tunables": tun.to_dict(),
    }
