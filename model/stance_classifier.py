"""Heuristic stance classifier — derives per-pick stance loadings from
precomputed features and emits BOTH continuous loadings AND the
dominant discrete label.

Stance taxonomy: 4-element closed vocabulary (``momentum`` / ``value`` /
``quality`` / ``catalyst``) defined in
``alpha_engine_lib.agent_schemas.StanceLiteral``. Loadings shape lives
in ``StanceLoadings``. This module's job: assign loadings + dominant
stance to each ticker the predictor scores so the executor can route
to stance-appropriate gates without the LLM agents needing to self-tag.

**Why continuous + dominant (institutional pattern)**: factor models at
Barra / AQR / Fama-French / BlackRock Aladdin derive CONTINUOUS factor
loadings from data — they don't force a single-stance assignment. Most
picks have mixed exposure (e.g., 0.65 momentum + 0.20 quality +
0.10 value + 0.05 catalyst). Discrete labels alone force an artificial
single-choice when reality is mixed. We emit both:
  - ``stance_loadings`` (continuous) for institutional-grade consumers
  - ``stance`` (discrete, argmax) for simple v1 consumers

**Why heuristic v1 (not ML)**:
  * No labeled training data yet (first time stance gets tagged).
  * Heuristic rules → smooth functions → transparent calibration.
  * ML upgrade (multi-class classifier with calibrated softmax) is
    roadmapped as P3 in alpha-engine-config ROADMAP. The continuous
    shape this v1 emits is a drop-in compatible target for the ML
    model — no future schema migration required.

Threshold / scale parameters below are ad hoc cold-start values.
Backtester (stance-arc PR 4) adds them to the parameter search space
once 4+ weeks of stance-tagged history accumulates; values auto-tune
weekly from then.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alpha_engine_lib.agent_schemas import STANCE_NAMES, StanceLiteral

# Cold-start scale parameters. Each parameter sets the "width" of the
# corresponding smooth function so a feature value at the half-point
# yields a raw score of ~0.5. Backtester-tunable post-launch.
MOMENTUM_SCALE = 0.05    # half-strength at momentum_20d ≈ ±5%
VALUE_SCALE = 0.05       # half-strength at momentum_20d ≈ ∓5%
CATALYST_PEAK_DAYS = 10  # bell-curve peaks for events ~10 days out
CATALYST_WIDTH_DAYS = 12  # std-dev of the bell curve (in days)
QUALITY_VOL_SCALE = 0.30  # half-strength at realized_vol_20d ≈ 30%/yr

# Softmax temperature multiplier — applied to raw scores before
# softmax. Lower T (we use T < 1 by multiplying by 1/T) → sharper
# distribution (winner takes more). T = 0.33 (1/3) at this scale gives
# the dominant stance ~50-65% loading on a clear feature shape, vs
# the ~25-40% the unsharpened softmax produces. Tunable later.
SOFTMAX_TEMPERATURE_INVERSE = 3.0


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Read a numeric feature; return ``default`` on missing / NaN / non-numeric.

    Returns ``None`` by default so smooth-function callers can branch
    on absence rather than silently substituting a neutral value.
    """
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid. Maps ℝ → (0, 1)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax. Subtracts the max before exponentiating
    so large inputs don't overflow. Preserves dict key order."""
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _raw_scores(features: Any) -> dict[str, float]:
    """Compute per-stance RAW scores from features. Higher = stronger
    stance match. Returns a dict keyed by stance name.

    The smooth functions (sigmoid, gaussian) replace the prior
    discrete rule-based assignment so a ticker with momentum_20d = +6%
    gets ~0.55 momentum loading rather than discrete "momentum=YES"
    (which discards the gradient information that +20% momentum is
    stronger than +6%).

    Each raw score lives roughly in [0, 1] BEFORE softmax; softmax
    then converts to a proper probability distribution by amplifying
    differences. The pre-softmax structure lets you read per-stance
    "raw strength" if you want it; the post-softmax loadings are what
    consumers actually use.
    """
    get = features.get if hasattr(features, "get") else lambda k, d=None: d

    momentum_20d = _safe_float(get("momentum_20d"), 0.0) or 0.0
    price_vs_ma50 = _safe_float(get("price_vs_ma50"), 1.0) or 1.0
    realized_vol_20d = _safe_float(get("realized_vol_20d"), 0.20) or 0.20
    debt_to_equity = _safe_float(get("debt_to_equity"), 0.5) or 0.5
    eps_ttm = _safe_float(get("eps_ttm"), 0.0) or 0.0

    # momentum: sigmoid on 20d return, scaled. Bonus for price > MA50
    # (confirmation that the move isn't just a one-day spike). Returns
    # ~0.5 at zero, ~0.73 at +5%, ~0.95 at +10%.
    ma50_bonus = max(0.0, min(0.2, (price_vs_ma50 - 1.0) * 2.0))  # 0..0.2
    momentum_raw = _sigmoid(momentum_20d / MOMENTUM_SCALE) + ma50_bonus

    # value: sigmoid on NEGATIVE 20d return (oversold = high score),
    # gated by fundamental defensibility (debt low + EPS positive).
    # Without the fundamentals gate, distressed names get high value
    # loading — but the data is saying "broken business", not "value".
    oversold_strength = _sigmoid(-momentum_20d / VALUE_SCALE)
    debt_quality = _sigmoid((1.5 - debt_to_equity) * 2.0)  # 0..1, ~0.5 at D/E=1.5
    earnings_quality = _sigmoid(eps_ttm * 0.5)  # 0..1, ~0.5 at EPS=0
    fundamental_quality = (debt_quality + earnings_quality) / 2.0
    value_raw = oversold_strength * fundamental_quality * 2.0  # rescale to keep magnitudes comparable

    # quality: inverse-vol smoothness. Low realized vol → high quality
    # loading. Pairs with low debt for the conservative-business profile.
    vol_quality = _sigmoid((QUALITY_VOL_SCALE - realized_vol_20d) / QUALITY_VOL_SCALE)
    quality_raw = (vol_quality + debt_quality) / 2.0 * 1.5

    # catalyst: bell curve centered at CATALYST_PEAK_DAYS days out,
    # peak raw score boosted to 1.5 (vs ~1.1 max for the other
    # stances). Rationale: a scheduled event is the strongest signal
    # in the set — momentum / value / quality are all derived from
    # noisy backward-looking features, while a confirmed earnings
    # date is forward-looking ground truth. The amplification ensures
    # catalyst dominates the loadings when present, matching the
    # institutional pattern of overlaying event-driven sleeves on top
    # of factor models.
    #
    # No earnings field → 0 catalyst loading (event is unknown). The
    # peak shape captures that an event RIGHT NOW (days=0) is risky
    # in-flight, an event 10 days out is the sweet spot, and a far-
    # future event (60+ days) provides no actionable signal.
    days_to_earnings = _safe_float(get("days_to_earnings"))
    if days_to_earnings is None or days_to_earnings < 0 or days_to_earnings > 60:
        catalyst_raw = 0.0
    else:
        diff = days_to_earnings - CATALYST_PEAK_DAYS
        catalyst_raw = 1.5 * math.exp(-(diff * diff) / (2.0 * CATALYST_WIDTH_DAYS * CATALYST_WIDTH_DAYS))

    # Softmax temperature: raw scores are already in roughly 0-2 range.
    # No additional temperature multiplier (T=1) — could be tuned later
    # to sharpen (T<1, more concentration on top stance) or smooth
    # (T>1, more uniform) the distribution.
    return {
        "momentum": momentum_raw,
        "value": value_raw,
        "quality": quality_raw,
        "catalyst": catalyst_raw,
    }


def classify_stance(
    features: Any,
) -> tuple[StanceLiteral, dict[str, float], str | None]:
    """Return ``(dominant_stance, loadings_dict, catalyst_date_or_None)``.

    ``loadings_dict`` is a dict of 4 floats in [0, 1] summing to 1.0
    (the softmax of raw per-stance scores). It matches the field shape
    of ``alpha_engine_lib.agent_schemas.StanceLoadings``.

    ``dominant_stance`` is ``argmax(loadings_dict)`` — the simple-
    consumer routing label. Ties broken in canonical ``STANCE_NAMES``
    order.

    ``catalyst_date`` is sourced from the feature row's
    ``next_earnings_date`` field when present (FMP earnings calendar
    feed, populated upstream). Returns ``None`` when not provided; the
    executor's catalyst gate computes the hard-exit date from
    ``days_to_earnings`` directly in that case.

    Pure function — no I/O, no state. Takes a dict / pandas Series /
    anything with ``.get()``.
    """
    raw = _raw_scores(features)
    # Apply temperature sharpening to amplify the dominant stance
    # without going all the way to one-hot. T < 1 (we use 1/T = 3.0)
    # makes the dominant stance ~50-65% of mass while preserving the
    # mixed-exposure shape the continuous design exists to capture.
    sharpened = {k: v * SOFTMAX_TEMPERATURE_INVERSE for k, v in raw.items()}
    loadings = _softmax(sharpened)

    # argmax with canonical tie-break order
    dominant: StanceLiteral = STANCE_NAMES[0]  # type: ignore[assignment]
    best = -1.0
    for name in STANCE_NAMES:
        if loadings[name] > best:
            best = loadings[name]
            dominant = name  # type: ignore[assignment]

    get = features.get if hasattr(features, "get") else lambda k, d=None: d
    catalyst_date = get("next_earnings_date")
    if catalyst_date is not None and not isinstance(catalyst_date, str):
        catalyst_date = None

    # Round loadings to 6 decimals for serialization stability + sum-to-1
    # tolerance compatibility with StanceLoadings validator (±1e-3).
    loadings = {k: round(v, 6) for k, v in loadings.items()}

    return dominant, loadings, catalyst_date
