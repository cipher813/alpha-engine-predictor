"""
regime/drawdown.py — deterministic drawdown regime leg (3rd ensemble leg).

Plan: ``regime-drawdown-hysteresis-260518.md``. Pure-logic core (no S3,
no boto3 for the classifier/composition) mirroring ``regime/fast_signal.py``
so it is testable in isolation; the inference stage / handler wire any
S3 I/O.

The regime ensemble
-------------------
Institutional regime overlay = an ensemble with conservative
aggregation, not a single model:

1. **Statistical** — the weekly HMM (``regime/substrate.py``).
2. **Changepoint** — the daily BOCPD ``forced_bear`` (``regime/fast_signal.py``).
3. **Deterministic rule** — drawdown de-risking (THIS module): the
   missing, lowest-estimation-risk leg.

This module owns leg 3 + the **most-protective composition** helper
(promised in ``substrate.py``'s module docstring — "executor/veto take
the max-protection of {continuous path, bear floor}" — but never
implemented until now).

Two sub-signals, **pure-level asymmetric hysteresis**
-----------------------------------------------------
* **SPY market drawdown** — tiered ``risk_on`` / ``caution`` / ``risk_off``.
* **book-vs-market excess drawdown** — ``risk_on`` / ``alpha_bleed``
  (book bleeding materially worse than SPY ⇒ alpha-negative by the
  literal objective ``alpha = portfolio - SPY``).

Debounce design (settled 2026-05-18, plan §3): the asymmetric
enter/exit **band is the sole debounce** — pure-level, NOT Stage-F's
persistence-day confirmation. Rationale: (1) the F1 backfill ran this
exact pure-level −10/−5 machine on the realized SPY series and it
passed whipsaw decisively; (2) the loss function is asymmetric
(drawdown ≫ opportunity cost) so persistence-days *delay protection*,
the wrong direction; (3) **single-source-of-truth invariant** — the
extracted ``bear_stretches`` here is *both* the production leg and the
L95 / F1-F2 eval reference, so pure-level keeps
``production == reference == one validated implementation``.
``min_persist_days`` is exposed but **defaults to 1 (≡ pure-level)**;
raising it above 1 re-opens the F1 calibration AND must be applied
identically on the eval-reference path or the measurement decouples
from production.

Sign convention
---------------
Drawdown ``dd = price / running_peak - 1`` is ``<= 0``. Thresholds are
expressed as positive depth magnitudes (``0.10`` = 10% below peak); the
classifier compares against ``magnitude = -dd``.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Bump on any breaking change to the persisted-state or artifact schema.
DRAWDOWN_SCHEMA_VERSION: int = 1


# ── Tunables ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DrawdownTunables:
    """Hysteresis bands (positive depth magnitudes, fraction of peak).

    Defaults are the institutional convention; the −10/−5 SPY ``risk_off``
    pair is the F1-backfill-validated reference (predictor #167) — do not
    change without re-running that calibration.
    """

    # SPY market-drawdown tiers
    spy_caution_enter: float = 0.05    # ≥5% below peak ⇒ caution
    spy_caution_exit: float = 0.03     # recover to within 3% ⇒ leave caution
    spy_risk_off_enter: float = 0.10   # ≥10% below peak ⇒ risk_off
    spy_risk_off_exit: float = 0.07    # recover to within 7% ⇒ risk_off→caution

    # Book-vs-market excess drawdown (portfolio_dd − spy_dd, depth magnitude)
    excess_enter: float = 0.05         # book ≥5pp deeper than SPY ⇒ alpha_bleed
    excess_exit: float = 0.02          # gap back within 2pp ⇒ clear

    # Pure-level by default. >1 re-opens F1 calibration and MUST match the
    # eval-reference path (single-source-of-truth invariant, plan §3).
    min_persist_days: int = 1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ── Canonical regime vocabulary + most-protective composition ────────────

# Protection ordering. Higher = more protective (de-risked).
_PROTECTION_ORDER: dict[str, int] = {
    "bull": 0,
    "neutral": 1,
    "caution": 2,
    "bear": 3,
}


def most_protective(*labels: str | None) -> str:
    """Conservative aggregation: the most-protective non-None label.

    The canonical helper the ``substrate.py`` docstring promised. Capital
    protection needs no consensus — any leg can escalate. ``None`` legs
    (a leg that does not escalate, e.g. drawdown ``risk_on``) are ignored;
    all-None ⇒ ``"neutral"`` (no information, no escalation).
    """
    present = [str(l) for l in labels if l]
    if not present:
        return "neutral"
    return max(present, key=lambda l: _PROTECTION_ORDER.get(l, 1))


def spy_tier_to_regime(tier: str) -> str | None:
    """Map an SPY drawdown tier onto the regime vocabulary (None = no
    escalation contribution)."""
    return {"risk_on": None, "caution": "caution", "risk_off": "bear"}.get(tier)


def excess_tier_to_regime(tier: str) -> str | None:
    """Map the book-vs-market excess tier onto the regime vocabulary."""
    return {"risk_on": None, "alpha_bleed": "bear"}.get(tier)


def compose_effective_regime(
    *,
    hmm_argmax: str | None = None,
    spy_tier: str | None = None,
    excess_tier: str | None = None,
    forced_bear: bool = False,
) -> dict[str, Any]:
    """Compose the effective regime = most-protective over the ensemble.

    Returns ``{"effective_regime": str, "drivers": {leg: contribution}}``.
    The raw legs are preserved so the composed value is always
    explainable in audit. ``hmm_argmax`` is bull/neutral/bear (already
    the regime vocabulary); the drawdown tiers map via the helpers; a
    Stage-F ``forced_bear`` contributes ``"bear"``.
    """
    drivers: dict[str, str | None] = {
        "hmm": hmm_argmax,
        "drawdown_spy": spy_tier_to_regime(spy_tier) if spy_tier else None,
        "drawdown_excess": (
            excess_tier_to_regime(excess_tier) if excess_tier else None
        ),
        "forced_bear": "bear" if forced_bear else None,
    }
    effective = most_protective(*drivers.values())
    return {"effective_regime": effective, "drivers": drivers}


# ── Vectorized bear stretches (THE single source of truth) ───────────────

def bear_stretches(
    close: pd.Series,
    *,
    enter: float = 0.10,
    exit: float = 0.05,
) -> list[tuple[Any, Any]]:
    """SPY peak-to-trough drawdown bear stretches — pure-level hysteresis.

    Extracted verbatim (semantics-preserving) from the script-local
    ``scripts/backfill_regime_fast_signal.py::_bear_stretches`` so the
    production leg and the L95 / F1-F2 eval reference are the *same*
    code. Onset = close ≥ ``enter`` below the trailing all-time high;
    the stretch ends when it recovers to within ``exit`` of the running
    peak. The ``enter``/``exit`` gap is the sole debounce.

    Returns a list of ``(start_ts, end_ts)`` tuples; an unterminated
    final stretch is closed at the last index.
    """
    s = close.dropna()
    if s.empty:
        return []
    peak = s.cummax()
    dd = s / peak - 1.0  # ≤ 0
    stretches: list[tuple[Any, Any]] = []
    in_bear = False
    start: Any | None = None
    for ts, d in dd.items():
        if not in_bear and d <= -enter:
            in_bear, start = True, ts
        elif in_bear and d >= -exit:
            stretches.append((start, ts))
            in_bear = False
    if in_bear and start is not None:
        stretches.append((start, dd.index[-1]))
    return stretches


# ── Online tiered hysteresis ─────────────────────────────────────────────

def _advance_spy_tier(prev: str, magnitude: float, t: DrawdownTunables) -> str:
    """One-step SPY tier transition. ``magnitude`` = -dd ≥ 0 (depth).

    Ladder with asymmetric bands: risk_on ⇄ caution ⇄ risk_off. A jump
    straight from risk_on to risk_off is allowed on entry; de-escalation
    is one rung at a time (risk_off→caution→risk_on) so the exit bands
    are honoured.
    """
    if prev == "risk_off":
        if magnitude <= t.spy_risk_off_exit:
            # leave risk_off; may land in caution or, if fully recovered,
            # risk_on (still gated by the caution exit band).
            return "risk_on" if magnitude <= t.spy_caution_exit else "caution"
        return "risk_off"
    if prev == "caution":
        if magnitude >= t.spy_risk_off_enter:
            return "risk_off"
        if magnitude <= t.spy_caution_exit:
            return "risk_on"
        return "caution"
    # prev == "risk_on"
    if magnitude >= t.spy_risk_off_enter:
        return "risk_off"
    if magnitude >= t.spy_caution_enter:
        return "caution"
    return "risk_on"


def _advance_excess_tier(prev: str, magnitude: float, t: DrawdownTunables) -> str:
    """One-step excess (book-vs-market) tier transition. ``magnitude`` =
    max(spy_dd − portfolio_dd, 0) depth: how much deeper the book is."""
    if prev == "alpha_bleed":
        return "risk_on" if magnitude <= t.excess_exit else "alpha_bleed"
    return "alpha_bleed" if magnitude >= t.excess_enter else "risk_on"


@dataclass
class DrawdownState:
    """Persisted online state. Serialized to ``regime/drawdown_state.json``."""

    spy_tier: str = "risk_on"
    excess_tier: str = "risk_on"
    spy_peak: float | None = None
    nav_peak: float | None = None
    # Pending-transition bookkeeping for min_persist_days > 1 (no-op at 1).
    pending_spy_tier: str | None = None
    pending_spy_days: int = 0
    pending_excess_tier: str | None = None
    pending_excess_days: int = 0
    last_update_trading_day: str | None = None
    observations_seen: int = 0
    schema_version: int = DRAWDOWN_SCHEMA_VERSION


def dump_state(st: DrawdownState) -> dict[str, Any]:
    d = asdict(st)
    # Floats may be numpy — coerce for clean JSON.
    for k in ("spy_peak", "nav_peak"):
        d[k] = None if d[k] is None else float(d[k])
    return d


def load_state(d: dict[str, Any]) -> DrawdownState:
    """Rehydrate persisted state. Raises ``ValueError`` on schema
    mismatch — the caller treats that as a cold start (per
    ``feedback_no_silent_fails``: never silently fabricate a settled
    'no drawdown')."""
    ver = d.get("schema_version")
    if ver != DRAWDOWN_SCHEMA_VERSION:
        raise ValueError(
            f"drawdown state schema {ver} != {DRAWDOWN_SCHEMA_VERSION}"
        )
    return DrawdownState(
        spy_tier=d.get("spy_tier", "risk_on"),
        excess_tier=d.get("excess_tier", "risk_on"),
        spy_peak=d.get("spy_peak"),
        nav_peak=d.get("nav_peak"),
        pending_spy_tier=d.get("pending_spy_tier"),
        pending_spy_days=int(d.get("pending_spy_days", 0)),
        pending_excess_tier=d.get("pending_excess_tier"),
        pending_excess_days=int(d.get("pending_excess_days", 0)),
        last_update_trading_day=d.get("last_update_trading_day"),
        observations_seen=int(d.get("observations_seen", 0)),
        schema_version=ver,
    )


def _persisted_transition(
    current: str,
    proposed: str,
    pending_tier: str | None,
    pending_days: int,
    min_persist_days: int,
) -> tuple[str, str | None, int]:
    """Apply the optional persistence gate. Default min_persist_days=1 ⇒
    immediate (pure-level). Returns (new_tier, new_pending, new_days)."""
    if proposed == current:
        return current, None, 0
    if min_persist_days <= 1:
        return proposed, None, 0
    if pending_tier == proposed:
        pending_days += 1
    else:
        pending_tier, pending_days = proposed, 1
    if pending_days >= min_persist_days:
        return proposed, None, 0
    return current, pending_tier, pending_days


def step(
    prev: DrawdownState | None,
    *,
    spy_close: float,
    trading_day: str,
    calendar_date: str,
    run_id: str,
    nav: float | None = None,
    tunables: DrawdownTunables | None = None,
) -> tuple[DrawdownState, dict[str, Any]]:
    """Advance the drawdown leg by one observation (weekly or daily).

    Mirrors ``fast_signal.step`` contract: returns ``(new_state,
    artifact)``; idempotent on ``trading_day`` (re-invocation re-emits
    with ``observed=False`` and no peak/tier advance). ``prev=None`` ⇒
    cold start (peaks seed from the first observation). ``nav=None`` ⇒
    the excess leg is unavailable (paper NAV short/gappy / not wired):
    the SPY leg still acts; ``excess_tier`` holds ``risk_on`` and
    ``excess_available=False``.
    """
    t = tunables or DrawdownTunables()
    cold_start = prev is None

    if prev is not None and prev.last_update_trading_day == trading_day:
        return prev, _artifact(
            prev, t, trading_day=trading_day, calendar_date=calendar_date,
            run_id=run_id, spy_close=spy_close, nav=nav,
            excess_available=nav is not None and prev.nav_peak is not None,
            observed=False, cold_start=False,
        )

    st = prev if prev is not None else DrawdownState()

    spy_peak = spy_close if st.spy_peak is None else max(st.spy_peak, spy_close)
    spy_dd = spy_close / spy_peak - 1.0 if spy_peak else 0.0
    spy_mag = -spy_dd

    proposed_spy = _advance_spy_tier(st.spy_tier, spy_mag, t)
    new_spy_tier, pend_spy, pend_spy_days = _persisted_transition(
        st.spy_tier, proposed_spy, st.pending_spy_tier,
        st.pending_spy_days, t.min_persist_days,
    )

    # ── Excess (book-vs-market) leg — graceful-degrade when NAV absent ──
    excess_available = nav is not None
    if excess_available:
        nav_peak = nav if st.nav_peak is None else max(st.nav_peak, nav)
        nav_dd = nav / nav_peak - 1.0 if nav_peak else 0.0
        # How much DEEPER the book is than the market (positive = worse).
        excess_mag = max(spy_dd - nav_dd, 0.0)
        proposed_excess = _advance_excess_tier(st.excess_tier, excess_mag, t)
        new_excess_tier, pend_ex, pend_ex_days = _persisted_transition(
            st.excess_tier, proposed_excess, st.pending_excess_tier,
            st.pending_excess_days, t.min_persist_days,
        )
    else:
        nav_peak = st.nav_peak
        nav_dd = None
        excess_mag = None
        new_excess_tier, pend_ex, pend_ex_days = "risk_on", None, 0

    new_state = DrawdownState(
        spy_tier=new_spy_tier,
        excess_tier=new_excess_tier,
        spy_peak=spy_peak,
        nav_peak=nav_peak,
        pending_spy_tier=pend_spy,
        pending_spy_days=pend_spy_days,
        pending_excess_tier=pend_ex,
        pending_excess_days=pend_ex_days,
        last_update_trading_day=trading_day,
        observations_seen=st.observations_seen + 1,
    )

    art = _artifact(
        new_state, t, trading_day=trading_day, calendar_date=calendar_date,
        run_id=run_id, spy_close=spy_close, nav=nav,
        excess_available=excess_available,
        observed=True, cold_start=cold_start,
        spy_dd=spy_dd, nav_dd=nav_dd, excess_mag=excess_mag,
    )
    return new_state, art


def _artifact(
    st: DrawdownState,
    t: DrawdownTunables,
    *,
    trading_day: str,
    calendar_date: str,
    run_id: str,
    spy_close: float,
    nav: float | None,
    excess_available: bool,
    observed: bool,
    cold_start: bool,
    spy_dd: float | None = None,
    nav_dd: float | None = None,
    excess_mag: float | None = None,
) -> dict[str, Any]:
    if spy_dd is None and st.spy_peak:
        spy_dd = spy_close / st.spy_peak - 1.0
    return {
        "trading_day": trading_day,
        "calendar_date": calendar_date,
        "run_id": run_id,
        "schema_version": DRAWDOWN_SCHEMA_VERSION,
        "spy": {
            "tier": st.spy_tier,
            "drawdown": None if spy_dd is None else float(spy_dd),
            "peak": None if st.spy_peak is None else float(st.spy_peak),
            "regime_contribution": spy_tier_to_regime(st.spy_tier),
        },
        "excess": {
            "available": bool(excess_available),
            "tier": st.excess_tier,
            "nav_drawdown": None if nav_dd is None else float(nav_dd),
            "excess_depth": None if excess_mag is None else float(excess_mag),
            "regime_contribution": (
                excess_tier_to_regime(st.excess_tier)
                if excess_available else None
            ),
        },
        "observations_seen": int(st.observations_seen),
        "observed": bool(observed),
        "cold_start": bool(cold_start),
        "tunables": t.to_dict(),
    }


# ── eod_pnl NAV reader (graceful-degrade) ────────────────────────────────

def read_eod_pnl_nav(
    s3_client: Any,
    *,
    bucket: str,
    key: str = "trades/eod_pnl.csv",
) -> "pd.Series | None":
    """Best-effort read of the executor-produced ``trades/eod_pnl.csv``
    NAV series (paper NAV). Graceful-degrade is load-bearing: any
    failure (missing / empty / no NAV column / < 2 rows) returns
    ``None`` so the excess leg goes unavailable while the SPY leg still
    acts — never raises, never blocks the regime path.

    The NAV column is matched case-insensitively (any column whose name
    contains 'nav'); a date-like column, if present, becomes the index.
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(BytesIO(obj["Body"].read()))
    except Exception as exc:  # noqa: BLE001 — best-effort, never block
        logger.warning(
            "drawdown: eod_pnl read failed at s3://%s/%s (%s) — excess "
            "leg unavailable; SPY leg unaffected.", bucket, key,
            type(exc).__name__,
        )
        return None
    if df is None or df.empty:
        logger.warning("drawdown: eod_pnl empty — excess leg unavailable.")
        return None
    nav_cols = [c for c in df.columns if "nav" in str(c).lower()]
    if not nav_cols:
        logger.warning(
            "drawdown: eod_pnl has no NAV-like column (cols=%s) — excess "
            "leg unavailable.", list(df.columns),
        )
        return None
    series = pd.to_numeric(df[nav_cols[0]], errors="coerce").dropna()
    date_cols = [c for c in df.columns if "date" in str(c).lower()]
    if date_cols:
        try:
            idx = pd.to_datetime(df.loc[series.index, date_cols[0]])
            series.index = idx
        except Exception:  # noqa: BLE001 — index is a nicety, not required
            pass
    if len(series) < 2:
        logger.warning(
            "drawdown: eod_pnl NAV series too short (%d rows) — excess "
            "leg unavailable.", len(series),
        )
        return None
    return series


# ── Cold-start seeding ───────────────────────────────────────────────────

def seed_state(
    spy_close_history: "pd.Series",
    *,
    nav_history: "pd.Series | None" = None,
    tunables: DrawdownTunables | None = None,
) -> DrawdownState:
    """Derive a cold-start state from price/NAV history.

    Why this exists: ``step()`` accumulates the running peak online, so a
    naive cold start (peak = today's close) would report ~0% drawdown
    even if the market is mid-crash — silently fabricating a settled
    'no drawdown' (the exact failure ``feedback_no_silent_fails``
    warns about). Seeding the **true trailing peak** (history cummax) +
    a conservative initial tier (the deepest tier the *current* depth
    warrants, via the ENTER thresholds — pure-level needs no prior-tier
    history) makes the very first online ``step()`` report the real
    drawdown.

    Returns a ``DrawdownState`` with ``observations_seen=0`` and
    ``last_update_trading_day=None`` so the first real ``step()`` advances
    (not idempotent-skipped).
    """
    t = tunables or DrawdownTunables()
    s = spy_close_history.dropna()
    if s.empty:
        return DrawdownState()
    spy_peak = float(s.cummax().iloc[-1])
    spy_close = float(s.iloc[-1])
    spy_dd = spy_close / spy_peak - 1.0 if spy_peak else 0.0
    spy_mag = -spy_dd
    if spy_mag >= t.spy_risk_off_enter:
        spy_tier = "risk_off"
    elif spy_mag >= t.spy_caution_enter:
        spy_tier = "caution"
    else:
        spy_tier = "risk_on"

    nav_peak: float | None = None
    excess_tier = "risk_on"
    if nav_history is not None:
        nv = nav_history.dropna()
        if not nv.empty:
            nav_peak = float(nv.cummax().iloc[-1])
            nav_close = float(nv.iloc[-1])
            nav_dd = nav_close / nav_peak - 1.0 if nav_peak else 0.0
            if max(spy_dd - nav_dd, 0.0) >= t.excess_enter:
                excess_tier = "alpha_bleed"

    return DrawdownState(
        spy_tier=spy_tier,
        excess_tier=excess_tier,
        spy_peak=spy_peak,
        nav_peak=nav_peak,
        observations_seen=0,
    )
