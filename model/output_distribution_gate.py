"""Output-distribution gate (audit Phase 2a).

Validates that a calibrator produces a non-degenerate output distribution
on a synthetic alpha sweep before the model is promoted to live or used
for inference. Catches the calibrator-plateau / saturation / direction-skew
class of failure that bit production on 2026-04-28, 2026-05-04, and
2026-05-07.

Two execution sites are supported by the same gate logic:

- **Promotion-time** (training/meta_trainer.py, audit Phase 2a-PROMOTE):
  blocks live promotion of weights when the calibrator output collapses
  on a synthetic input sweep. Same architectural class as the existing
  IC gate and per-component subsample gate at meta_trainer.py:1019.

- **Inference-time** (inference/handler.py, audit Phase 2a-INFER):
  fails closed if the calibrator output collapses on the live batch's
  predicted alphas. Replaces the post-hoc variance fallback rescue path
  with refuse-to-write — predictions.json isn't written; executor reads
  prior-day fallback. (PR-INFER follows this PR; this module exposes
  the helper that both sites call.)

This first PR ships the gate logic + the synthetic-sweep variant
(audit's §9.2 Option B). The stratified-historical-sample variant
(Option C) and the inference-time integration follow in subsequent PRs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputDistributionGateResult:
    """Result of an output-distribution gate check.

    ``passed=True`` means all four checks (uniqueness, saturation, stdev,
    direction skew) cleared their thresholds. ``passed=False`` means at
    least one failed; ``failed_check`` carries the first-failed name and
    ``reason`` carries a human-readable explanation suitable for log
    output and gate-block messages.

    ``metrics`` carries the four measured values regardless of pass/fail
    so the manifest can persist them for trend analysis (a borderline
    pass over multiple training cycles is itself a useful signal).
    """
    passed: bool
    failed_check: Optional[str]
    reason: str
    metrics: dict


def validate_calibrator_distribution(
    calibrator,
    *,
    alpha_range: tuple[float, float] = (-0.05, 0.05),
    n_synthetic: int = 25,
    min_unique_p_up: int = 8,
    max_saturation_rate: float = 0.25,
    min_stdev: float = 0.005,
    max_direction_skew: float = 0.85,
    floor: float = 0.011,
    ceiling: float = 0.989,
) -> OutputDistributionGateResult:
    """Run an evenly-spaced synthetic alpha sweep through the calibrator and
    check four distribution-shape invariants.

    The four checks (and what they catch):

    1. **Unique p_up values** — the calibrator should produce at least
       ``min_unique_p_up`` distinct outputs across the synthetic sweep.
       Catches the isotonic-plateau failure mode (16 of 27 tickers all
       at p_up=0.458, 2026-05-07): the calibrator learned a flat region
       in the raw-alpha space and maps any input in that region to the
       same output.

    2. **Saturation rate** — at most ``max_saturation_rate`` of outputs
       can sit at the calibrator's floor (0.01) or ceiling (0.99) clip
       boundaries. Catches the floor-clipping mode (5 of 27 tickers all
       at p_up=0.010, conf=0.99 on 2026-05-07): every input below the
       calibrator's learned minimum maps to the same saturated value,
       which is operationally equivalent to "this ticker is in the
       bottom-N alpha bucket" but reads as "high confidence DOWN".

    3. **Output stdev** — the calibrator's outputs across the sweep
       must have ``min_stdev`` standard deviation. Catches the
       compressed-output failure (all p_up close to 0.5119 on
       2026-04-28): when training data hardcoded constants for research
       features, the meta-Ridge produced a near-constant output and
       isotonic mapped everything to a tight cluster.

    4. **Direction skew** — the implied direction (p_up > 0.5 → UP,
       < 0.5 → DOWN) shouldn't be ``max_direction_skew`` or more on one
       side. Catches the false-collapse failure (0 UP / 26 DOWN on
       2026-05-04): when the upstream model produces a degenerate
       output, the calibrator faithfully maps it all to one direction.

    Returns OutputDistributionGateResult. Pass-on-fitted-calibrator-only:
    if the calibrator isn't fitted (linear fallback path), the gate
    returns ``passed=True`` with a metrics dict showing zero unique
    values from the sweep — the linear fallback's monotonic continuous
    output is the desired behavior in that case.

    Args:
        calibrator: An object exposing ``calibrate_prediction(raw_alpha)``
            returning a dict with ``"p_up"`` and ``"predicted_direction"``.
            Both PlattCalibrator and IsotonicCalibrator from model/calibrator.py
            satisfy this. Caller passes the calibrator under test.
        alpha_range: (low, high) bounds for the synthetic sweep. Default
            (-0.05, +0.05) covers the typical operating range; tighten
            to (-0.02, +0.02) for a more sensitive plateau check around
            the calibrator's high-density region.
        n_synthetic: Number of evenly-spaced alphas in the sweep. Default
            25 is enough to detect plateaus while keeping the gate cheap
            (~25 calibrator calls = milliseconds).
        min_unique_p_up: Lower bound on distinct p_up values across the
            sweep. Default 8 (out of 25) means at least ~30% of the input
            range produces distinct outputs.
        max_saturation_rate: Upper bound on the fraction of outputs that
            sit at floor or ceiling. Default 0.25 = at most 1/4 of the
            sweep can be saturated.
        min_stdev: Lower bound on the stdev of the sweep's output. Default
            0.005 = output must spread at least ~1pp on the p_up scale.
        max_direction_skew: Upper bound on max(n_up, n_down) /
            (n_up + n_down). Default 0.85 = at most 85% same direction.
        floor: Calibrator floor clip value. Default 0.011 (1pp tolerance
            above isotonic's hard 0.01 floor for round-tripping).
        ceiling: Calibrator ceiling clip value. Default 0.989 (similar
            tolerance below 0.99 hard ceiling).
    """
    import numpy as np

    # If the calibrator isn't fitted, the linear-fallback path produces
    # smooth monotonic output by construction. The gate has nothing to
    # validate — pass with metrics indicating "fallback path active."
    if not getattr(calibrator, "_fitted", True):
        return OutputDistributionGateResult(
            passed=True,
            failed_check=None,
            reason="calibrator not fitted (linear fallback active — by construction monotonic continuous)",
            metrics={"calibrator_fitted": False},
        )

    synthetic_alphas = np.linspace(alpha_range[0], alpha_range[1], n_synthetic)
    results = [calibrator.calibrate_prediction(float(a)) for a in synthetic_alphas]
    p_ups = np.array([r["p_up"] for r in results])
    directions = [r.get("predicted_direction") for r in results]

    # Check 1: unique p_up count (round to 6 decimals to handle FP noise)
    n_unique = int(len(np.unique(np.round(p_ups, 6))))

    # Check 2: saturation rate at floor/ceiling
    n_saturated = int(np.sum((p_ups <= floor) | (p_ups >= ceiling)))
    saturation_rate = float(n_saturated / len(p_ups))

    # Check 3: stdev across the sweep
    stdev = float(np.std(p_ups))

    # Check 4: direction skew (excluding any FLAT — though current
    # calibrator returns only UP/DOWN, this is forward-compatible).
    n_up = sum(1 for d in directions if d == "UP")
    n_down = sum(1 for d in directions if d == "DOWN")
    n_directional = n_up + n_down
    direction_skew = (
        float(max(n_up, n_down) / n_directional)
        if n_directional > 0 else 0.0
    )

    metrics = {
        "calibrator_fitted": True,
        "n_synthetic": n_synthetic,
        "alpha_range": list(alpha_range),
        "n_unique_p_up": n_unique,
        "saturation_rate": round(saturation_rate, 4),
        "stdev_p_up": round(stdev, 6),
        "direction_skew": round(direction_skew, 4),
        "n_up": n_up,
        "n_down": n_down,
    }

    # Run checks in priority order (most distinctive failure first).
    if n_unique < min_unique_p_up:
        reason = (
            f"only {n_unique} unique p_up values across {n_synthetic} "
            f"synthetic alphas (min: {min_unique_p_up}) — calibrator "
            f"likely has flat regions; the 2026-05-07-class plateau"
        )
        log.warning("Output-distribution gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="unique_p_up", reason=reason,
            metrics=metrics,
        )

    if saturation_rate > max_saturation_rate:
        reason = (
            f"saturation rate {saturation_rate:.2%} exceeds {max_saturation_rate:.0%} "
            f"({n_saturated}/{n_synthetic} outputs at floor or ceiling) — "
            f"calibrator clipping a wide alpha band to its bounds"
        )
        log.warning("Output-distribution gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="saturation_rate", reason=reason,
            metrics=metrics,
        )

    if stdev < min_stdev:
        reason = (
            f"p_up stdev {stdev:.4f} below threshold {min_stdev:.4f} — "
            f"output too compressed; the 2026-04-28-class collapse"
        )
        log.warning("Output-distribution gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="stdev", reason=reason,
            metrics=metrics,
        )

    if n_directional > 0 and direction_skew > max_direction_skew:
        reason = (
            f"direction skew {direction_skew:.2%} exceeds {max_direction_skew:.0%} "
            f"(n_up={n_up}, n_down={n_down}) — calibrator strongly biased"
        )
        log.warning("Output-distribution gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="direction_skew", reason=reason,
            metrics=metrics,
        )

    log.info(
        "Output-distribution gate PASSED — "
        "%d unique p_up, saturation %.1f%%, stdev %.4f, skew %.1f%%",
        n_unique, saturation_rate * 100, stdev, direction_skew * 100,
    )
    return OutputDistributionGateResult(
        passed=True, failed_check=None, reason="all checks passed",
        metrics=metrics,
    )
