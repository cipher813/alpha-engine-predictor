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
from typing import Any, Mapping

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

# Macro regime protection ordering. 3-class Ang-Bekaert taxonomy
# (v0.42.0 / 2026-05-28). Legacy 4th "caution" tier retired per
# caution-regime-retirement-260528.md — macro-side stress signals
# (VIX, HY OAS, SPY 30d return) flow continuously through
# regime_intensity_z (predictor META_FEATURE 13). The drawdown leg's
# protective state is now a SEPARATE axis (see
# DRAWDOWN_PROTECTIVE_SEVERITY below) and no longer aliases into the
# macro regime vocabulary at compose time.
_PROTECTION_ORDER: dict[str, int] = {
    "bull": 0,
    "neutral": 1,
    "bear": 2,
}

# Drawdown-leg protective-severity ordinal. 3-state hysteresis preserved
# as the institutional Bridgewater All-Weather / AQR risk-parity tiered
# de-risking pattern. This axis is ORTHOGONAL to macro regime;
# downstream consumers compose via most-protective override at decision
# time (executor _rank, predictor veto threshold).
#   0 = no protection (risk_on)
#   1 = caution (half-step protective; veto base - cap/2)
#   2 = full protective (risk_off / excess alpha_bleed; veto base - cap)
DRAWDOWN_PROTECTIVE_SEVERITY: dict[str, int] = {
    "risk_on": 0,
    "caution": 1,
    "risk_off": 2,
    "alpha_bleed": 2,  # excess tier escalates immediately
}


def most_protective(*labels: str | None) -> str:
    """Conservative aggregation over MACRO regime labels: the
    most-protective non-None label in the 3-class vocabulary.

    ``None`` legs (a leg that does not escalate, e.g. drawdown ``risk_on``)
    are ignored; all-None ⇒ ``"neutral"`` (no information, no escalation).

    Note: post-v0.42.0 the drawdown leg's protective state is a separate
    axis (see ``DRAWDOWN_PROTECTIVE_SEVERITY``) and only contributes
    ``"bear"`` to the macro composition when its tier reaches ``risk_off``
    (severity 2). Caution-tier drawdown (severity 1) is observable via
    ``drawdown_protective_severity`` but does NOT alias into the macro
    regime vocabulary.
    """
    present = [str(l) for l in labels if l]
    if not present:
        return "neutral"
    return max(present, key=lambda l: _PROTECTION_ORDER.get(l, 1))


def spy_tier_to_regime(tier: str) -> str | None:
    """Map an SPY drawdown tier onto the MACRO regime vocabulary.

    Post-v0.42.0 (caution-regime-retirement-260528.md): the drawdown
    leg's ``caution`` tier no longer aliases into the macro regime — it
    sits on the orthogonal drawdown axis (see
    ``spy_tier_to_protective_severity``). Only ``risk_off`` escalates
    the macro composition to bear. Use
    ``spy_tier_to_protective_severity`` for the drawdown-axis ordinal
    used by veto / sizing consumers.
    """
    return {"risk_on": None, "caution": None, "risk_off": "bear"}.get(tier)


def excess_tier_to_regime(tier: str) -> str | None:
    """Map the book-vs-market excess tier onto the macro regime
    vocabulary. ``alpha_bleed`` still escalates the composition to
    bear directly (the drawdown leg's most-conservative state)."""
    return {"risk_on": None, "alpha_bleed": "bear"}.get(tier)


def spy_tier_to_protective_severity(tier: str | None) -> int:
    """Map an SPY drawdown tier to its protective-severity ordinal
    (0=risk_on, 1=caution, 2=risk_off). Unknown tiers and ``None``
    return 0 (no escalation)."""
    if tier is None:
        return 0
    return DRAWDOWN_PROTECTIVE_SEVERITY.get(tier, 0)


def excess_tier_to_protective_severity(tier: str | None) -> int:
    """Map the book-vs-market excess tier to protective severity."""
    if tier is None:
        return 0
    return DRAWDOWN_PROTECTIVE_SEVERITY.get(tier, 0)


# ── Signed continuous regime score (issue #1299) ─────────────────────────
#
# Convention (frozen fleet-wide, issue #1299): +1 = bull / risk-on,
# 0 = neutral, -1 = bear / risk-off.
#
# The categorical `effective_regime` above is a *max-severity* merge whose
# only bull/neutral-capable leg is the HMM argmax; every other leg is a
# one-directional bear ratchet, and the natively-signed continuous head
# (intensity_z) plus the HMM's own conviction (the posterior) never enter
# the composition at all. `compose_regime_score` fixes the composition:
# each head is defined as a signed contribution on [-1, 1], they are blended
# with documented weights, intensity_z is folded in, and the HMM enters as a
# posterior EXPECTATION (E = P(bull) - P(bear)) — keeping conviction while
# demoting the stuck argmax.

# Blend weights (v0 baseline — issue #1299 §2 stipulates these are a stated
# 1/N-style baseline to be re-fit to forward 21d market-relative return by
# the backtest decision artifact; they are down-weighted vs equal to avoid
# triple-counting SPY, which the HMM / intensity_z / drawdown legs all load
# on). Weights are renormalized over the heads that are actually present, so
# a missing leg never silently biases the score toward 0.
REGIME_SCORE_WEIGHTS: dict[str, float] = {
    # Natively-signed, most market-grounded continuous head. Highest weight:
    # it is the only leg that was previously EXCLUDED yet is the cleanest
    # signed read of macro stress.
    "intensity_z": 0.40,
    # HMM posterior expectation (conviction-weighted, argmax demoted).
    "hmm": 0.35,
    # Downside-only drawdown legs — can only push toward bear ([-1, 0]),
    # honest about what a drawdown measures. Down-weighted because they
    # correlate with the SPY-driven heads above.
    "drawdown_spy": 0.15,
    "drawdown_excess": 0.10,
}

# tanh squash scale for intensity_z: intensity_z is a z-score in ~[-3, 3];
# dividing by 2 puts a 1σ move at tanh(0.5)≈0.46 and a 2σ move at
# tanh(1.0)≈0.76, so typical readings span most of [-1, 1] without
# saturating on noise.
INTENSITY_Z_TANH_SCALE: float = 2.0

# HMM staleness demotion (issue #1299: "demote the stuck HMM argmax").
# The audited cohort had a 56-week filter run-length — a documented
# label-stability artifact, not 56 weeks of genuine signal. Beyond mapping
# argmax→posterior expectation (which already drops the hard argmax), a HMM
# stuck in the same state for many weeks has its conviction multiplicatively
# discounted toward 0 (neutral) so it cannot single-handedly force the
# blend. The discount is a soft ramp: full weight up to ``_grace`` weeks,
# decaying to a ``_floor`` multiplier as it sticks. This keeps a freshly
# transitioned, high-conviction HMM authoritative while neutering a stuck
# one — exactly the defect the issue calls out.
HMM_STALENESS_GRACE_WEEKS: int = 8
HMM_STALENESS_HALFLIFE_WEEKS: float = 12.0
HMM_STALENESS_FLOOR: float = 0.25

# Categorical-projection thresholds on the signed score (issue #1299 §"Why
# [-1,1]": symmetric gating). |score| < t_neutral ⇒ neutral; the band is
# symmetric so the projection is sign-honest.
REGIME_SCORE_NEUTRAL_BAND: float = 0.20

# Downside-only tier ladders: SPY caution/risk_off and excess alpha_bleed
# map into [-1, 0] (they can never vote positive).
_SPY_TIER_SCORE: dict[str, float] = {
    "risk_on": 0.0,
    "caution": -0.5,
    "risk_off": -1.0,
}
_EXCESS_TIER_SCORE: dict[str, float] = {
    "risk_on": 0.0,
    "alpha_bleed": -1.0,
}


def _tanh(x: float) -> float:
    import math

    return math.tanh(x)


def hmm_staleness_discount(weeks_in_current_state: int | None) -> float:
    """Multiplier in ``[HMM_STALENESS_FLOOR, 1.0]`` that demotes a STUCK
    HMM argmax (issue #1299). Full weight (1.0) up to the grace window, then
    an exponential decay toward ``HMM_STALENESS_FLOOR`` as the state sticks
    (the audited cohort's 56-week run-length is the label-stability artifact
    this neutralizes). ``None``/unknown ⇒ no discount (1.0)."""
    if weeks_in_current_state is None:
        return 1.0
    try:
        wk = int(weeks_in_current_state)
    except (TypeError, ValueError):
        return 1.0
    if wk <= HMM_STALENESS_GRACE_WEEKS:
        return 1.0
    import math

    over = wk - HMM_STALENESS_GRACE_WEEKS
    decay = 0.5 ** (over / HMM_STALENESS_HALFLIFE_WEEKS)
    return HMM_STALENESS_FLOOR + (1.0 - HMM_STALENESS_FLOOR) * decay


def hmm_posterior_expectation(
    probs: Mapping[str, float] | None,
    *,
    weeks_in_current_state: int | None = None,
) -> float | None:
    """HMM head as a signed posterior EXPECTATION on [-1, 1]:

        E = (+1)·P(bull) + 0·P(neutral) + (-1)·P(bear)

    Keeps the HMM's conviction while *demoting the stuck argmax* (issue
    #1299): a stuck ``P(bear)=1.00`` argmax that previously dominated the
    categorical max-severity merge now contributes only its posterior
    expectation, and that expectation is further multiplicatively discounted
    by ``hmm_staleness_discount(weeks_in_current_state)`` so a HMM stuck in
    one state for many weeks (a documented label-stability artifact) decays
    toward neutral instead of dominating the blend. Returns ``None`` when no
    usable posterior is supplied (the leg drops out of the renormalized
    blend rather than voting a fabricated 0).
    """
    if not probs:
        return None
    try:
        p_bull = float(probs.get("bull", 0.0))
        p_bear = float(probs.get("bear", 0.0))
    except (TypeError, ValueError):
        return None
    expectation = max(-1.0, min(1.0, p_bull - p_bear))
    return expectation * hmm_staleness_discount(weeks_in_current_state)


def spy_tier_to_score(tier: str | None) -> float | None:
    """SPY drawdown tier → downside-only signed contribution in [-1, 0].
    Unknown/None ⇒ ``None`` (leg absent, drops out of the blend)."""
    if tier is None:
        return None
    return _SPY_TIER_SCORE.get(tier)


def excess_tier_to_score(tier: str | None) -> float | None:
    """Excess (book-vs-market) tier → downside-only contribution in [-1, 0]."""
    if tier is None:
        return None
    return _EXCESS_TIER_SCORE.get(tier)


def intensity_z_to_score(intensity_z: float | None) -> float | None:
    """Composite intensity z-score → signed [-1, 1] via tanh.

    ``intensity_z`` MUST already be in the standardized fleet convention
    (positive = risk-on / bull), i.e. the value carried in the substrate
    payload's ``composite.intensity_z`` (sign standardized at the
    substrate chokepoint — issue #1299 footgun note). ``tanh(z / s)`` is
    natively zero-centered so neutral z ⇒ 0 score.
    """
    if intensity_z is None:
        return None
    try:
        z = float(intensity_z)
    except (TypeError, ValueError):
        return None
    import math

    if not math.isfinite(z):
        return None
    return _tanh(z / INTENSITY_Z_TANH_SCALE)


def regime_score_to_categorical(
    score: float, *, neutral_band: float = REGIME_SCORE_NEUTRAL_BAND
) -> str:
    """Project the signed [-1, 1] score onto the legacy 3-class macro
    vocabulary at documented symmetric thresholds, so existing categorical
    consumers keep working unchanged:

        score >  +neutral_band ⇒ "bull"
        |score| ≤ neutral_band ⇒ "neutral"
        score <  -neutral_band ⇒ "bear"
    """
    if score > neutral_band:
        return "bull"
    if score < -neutral_band:
        return "bear"
    return "neutral"


def compose_regime_score(
    *,
    hmm_probs: Mapping[str, float] | None = None,
    hmm_weeks_in_state: int | None = None,
    intensity_z: float | None = None,
    spy_tier: str | None = None,
    excess_tier: str | None = None,
    forced_bear: bool = False,
    bocpd_change_signal: bool = False,
    bocpd_confidence: float | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Signed continuous regime score on [-1, 1] (issue #1299).

    Each head is a signed contribution on [-1, 1]; they are blended with
    ``REGIME_SCORE_WEIGHTS`` renormalized over the *present* heads. The HMM
    enters as a posterior expectation (argmax demoted); ``intensity_z`` is
    folded in (it was previously excluded); the two drawdown legs are
    downside-only ([-1, 0]). BOCPD is NOT an axis vote — a detected change
    signal is surfaced as an uncertainty flag for the hysteresis/gate layer.

    The asymmetric protective floor is preserved: ``forced_bear`` clamps the
    score to ``-1.0`` so a confirmed multi-signal bear (the substrate
    ``vix_bear AND spy_30d_bear`` tail override) can never be washed out by
    averaging.

    Returns:
        {
          "regime_score": <float in [-1, 1]>,
          "regime_score_categorical": <bull/neutral/bear projection>,
          "head_scores": {head: signed contribution, present heads only},
          "weights_used": {head: renormalized weight},
          "forced_bear_floor_applied": <bool>,
          "bocpd_change_signal": <bool>,
        }
    """
    w = dict(weights or REGIME_SCORE_WEIGHTS)

    raw_heads: dict[str, float | None] = {
        "intensity_z": intensity_z_to_score(intensity_z),
        "hmm": hmm_posterior_expectation(
            hmm_probs, weeks_in_current_state=hmm_weeks_in_state
        ),
        "drawdown_spy": spy_tier_to_score(spy_tier),
        "drawdown_excess": excess_tier_to_score(excess_tier),
    }
    head_scores = {k: v for k, v in raw_heads.items() if v is not None}

    present_weight = sum(w.get(k, 0.0) for k in head_scores)
    if present_weight > 0:
        weights_used = {k: w.get(k, 0.0) / present_weight for k in head_scores}
        score = sum(weights_used[k] * head_scores[k] for k in head_scores)
    else:
        weights_used = {}
        score = 0.0

    floor_applied = False
    if forced_bear:
        # Asymmetric protective floor — clamp defensive regardless of blend.
        score = -1.0
        floor_applied = True

    score = max(-1.0, min(1.0, score))

    return {
        "regime_score": float(score),
        "regime_score_categorical": regime_score_to_categorical(score),
        "head_scores": head_scores,
        "weights_used": weights_used,
        "forced_bear_floor_applied": floor_applied,
        "hmm_staleness_discount": hmm_staleness_discount(hmm_weeks_in_state),
        "bocpd_change_signal": bool(bocpd_change_signal),
        "bocpd_confidence": (
            None if bocpd_confidence is None else float(bocpd_confidence)
        ),
    }


def compose_effective_regime(
    *,
    hmm_argmax: str | None = None,
    spy_tier: str | None = None,
    excess_tier: str | None = None,
    forced_bear: bool = False,
    hmm_probs: Mapping[str, float] | None = None,
    hmm_weeks_in_state: int | None = None,
    intensity_z: float | None = None,
    bocpd_change_signal: bool = False,
    bocpd_confidence: float | None = None,
) -> dict[str, Any]:
    """Compose effective regime (macro axis) + drawdown protective
    state (separate axis) + the signed continuous regime score (issue #1299).

    Returns:
        {
          "effective_regime": <3-class macro regime: bull/neutral/bear>,
          "drivers": {leg: macro-regime contribution or None},
          "drawdown_tier": <raw SPY hysteresis state: risk_on/caution/risk_off, or None>,
          "drawdown_protective_severity": <0 | 1 | 2 ordinal across the drawdown axis>,
          "forced_bear": <bool>,
          # Additive (issue #1299, S3-contract ADD-don't-rename):
          "regime_score": <float in [-1, 1]>,
          "regime_score_categorical": <bull/neutral/bear projection of the score>,
          "regime_score_detail": {head_scores, weights_used, ...},
        }

    Backward-compat contract (issue #1299 rollout step 1 — observe-first):
    the legacy categorical ``effective_regime`` (most-protective over the
    HMM argmax + bear-ratchet legs) is UNCHANGED — existing categorical
    consumers see identical behaviour. The signed ``regime_score`` is added
    ADDITIVELY alongside it (no rename, no behaviour change) so it can be
    logged + backtested before any consumer cuts over.

    Type-system separation (v0.42.0 / 2026-05-28 —
    caution-regime-retirement-260528.md):

    * ``effective_regime`` is the 3-class macro axis (Ang-Bekaert).
      Drawdown ``risk_off`` and excess ``alpha_bleed`` still alias to
      ``bear`` here (their most-protective tier IS macro bear); drawdown
      ``caution`` no longer aliases at all (its protective semantic lives
      on the drawdown axis below).
    * ``drawdown_protective_severity`` is the canonical drawdown axis
      ordinal that veto / sizing consumers should read. Composed as
      ``max(spy_severity, excess_severity)``.

    Consumers compose the two axes via most-protective override at
    decision time (executor _rank table, predictor veto threshold).
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

    drawdown_severity = max(
        spy_tier_to_protective_severity(spy_tier),
        excess_tier_to_protective_severity(excess_tier),
    )

    score_block = compose_regime_score(
        hmm_probs=hmm_probs,
        hmm_weeks_in_state=hmm_weeks_in_state,
        intensity_z=intensity_z,
        spy_tier=spy_tier,
        excess_tier=excess_tier,
        forced_bear=forced_bear,
        bocpd_change_signal=bocpd_change_signal,
        bocpd_confidence=bocpd_confidence,
    )

    return {
        "effective_regime": effective,
        "drivers": drivers,
        "drawdown_tier": spy_tier,
        "drawdown_protective_severity": drawdown_severity,
        "forced_bear": forced_bear,
        # Additive signed-continuous outputs (issue #1299).
        "regime_score": score_block["regime_score"],
        "regime_score_categorical": score_block["regime_score_categorical"],
        "regime_score_detail": {
            "head_scores": score_block["head_scores"],
            "weights_used": score_block["weights_used"],
            "forced_bear_floor_applied": score_block["forced_bear_floor_applied"],
            "hmm_staleness_discount": score_block["hmm_staleness_discount"],
            "bocpd_change_signal": score_block["bocpd_change_signal"],
            "bocpd_confidence": score_block["bocpd_confidence"],
        },
    }


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


# ── Forensic batch view (weekly substrate block) ─────────────────────────

def block_from_history(
    spy_close: "pd.Series",
    *,
    nav_history: "pd.Series | None" = None,
    tunables: DrawdownTunables | None = None,
) -> dict[str, Any] | None:
    """Hysteresis-correct *current* drawdown state, replayed from price
    history — the forensic weekly-substrate view.

    The weekly substrate block must be reproducible from the price cache
    alone (same philosophy as the HMM refitting from history each week),
    so this **replays the full SPY series through ``step()``** to get
    the hysteresis-correct current tier (NOT ``seed_state``'s
    conservative cold-start shortcut). The excess (book-vs-market) leg
    is a **point-in-time** read of the latest NAV-vs-SPY gap — the NAV
    series is executor-produced + short, so its hysteresis history is
    not reproducible here; a point-in-time forensic read is the honest
    surface.

    Returns the same ``spy``/``excess`` sub-block shape as the daily
    artifact (so the substrate ``drawdown`` key is uniform weekly vs
    daily), or ``None`` when there is no SPY history (caller omits the
    block — S3-contract-safe, additive).
    """
    t = tunables or DrawdownTunables()
    s = spy_close.dropna()
    if s.empty:
        return None
    st: DrawdownState | None = None
    for i, px in enumerate(s.to_list()):
        st, _ = step(
            st, spy_close=float(px), trading_day=str(i),
            calendar_date=str(i), run_id="hist", tunables=t,
        )
    assert st is not None  # s is non-empty ⇒ at least one step ran
    spy_dd = (
        float(s.iloc[-1]) / st.spy_peak - 1.0 if st.spy_peak else 0.0
    )
    spy_block = {
        "tier": st.spy_tier,
        "drawdown": float(spy_dd),
        "peak": None if st.spy_peak is None else float(st.spy_peak),
        "regime_contribution": spy_tier_to_regime(st.spy_tier),
    }
    excess_block: dict[str, Any] = {
        "available": False, "tier": "risk_on",
        "nav_drawdown": None, "excess_depth": None,
        "regime_contribution": None,
    }
    if nav_history is not None:
        nv = nav_history.dropna()
        if len(nv) >= 2:
            nav_peak = float(nv.cummax().iloc[-1])
            nav_dd = (
                float(nv.iloc[-1]) / nav_peak - 1.0 if nav_peak else 0.0
            )
            excess_mag = max(spy_dd - nav_dd, 0.0)
            ex_tier = _advance_excess_tier("risk_on", excess_mag, t)
            excess_block = {
                "available": True,
                "tier": ex_tier,
                "nav_drawdown": float(nav_dd),
                "excess_depth": float(excess_mag),
                "regime_contribution": excess_tier_to_regime(ex_tier),
            }
    return {"spy": spy_block, "excess": excess_block}
