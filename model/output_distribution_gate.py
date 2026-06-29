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
    max_modal_fraction: float = 0.5,
    max_saturation_rate: float = 0.25,
    min_stdev: float = 0.005,
    max_direction_skew: float = 0.85,
    floor: float = 0.011,
    ceiling: float = 0.989,
) -> OutputDistributionGateResult:
    """Run an evenly-spaced synthetic alpha sweep through the calibrator and
    check four distribution-shape invariants.

    The four checks (and what they catch):

    1. **Flat-region collapse** — fires only when the output has BOTH a
       low distinct-value count (< ``min_unique_p_up``) AND a single
       dominant modal value (>= ``max_modal_fraction`` of the batch piled
       on one p_up). Catches the isotonic-plateau failure mode (16 of 27
       tickers all at p_up=0.458, 2026-05-07 → modal 59%): the calibrator
       learned a flat region and maps any input in it to the same output.
       The modal-concentration condition is what stops this check from
       false-halting on benign isotonic-staircase coarseness — an isotonic
       calibrator quantizes alpha→p_up into monotonic steps, so a thin or
       low-dispersion live batch legitimately lands in few distinct values
       with NO dominant mode (2026-06-22: 26 post-holiday tickers → 7
       monotonic steps spanning 0.35→0.70, modal 8/26=31% — healthy spread,
       must pass). Raw distinct-count alone cannot separate these.

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

    extra_metrics = {
        "calibrator_fitted": True,
        "n_synthetic": n_synthetic,
        "alpha_range": list(alpha_range),
    }
    return _evaluate_distribution_invariants(
        p_ups=p_ups,
        directions=directions,
        source_label=f"synthetic alpha sweep ({n_synthetic} samples)",
        min_unique_p_up=min_unique_p_up,
        max_modal_fraction=max_modal_fraction,
        max_saturation_rate=max_saturation_rate,
        min_stdev=min_stdev,
        max_direction_skew=max_direction_skew,
        floor=floor,
        ceiling=ceiling,
        extra_metrics=extra_metrics,
    )


def validate_live_batch_distribution(
    predictions: list[dict],
    *,
    min_unique_p_up: int = 8,
    max_modal_fraction: float = 0.5,
    max_saturation_rate: float = 0.25,
    min_stdev: float = 0.005,
    max_direction_skew: float = 0.85,
    floor: float = 0.011,
    ceiling: float = 0.989,
    min_batch_size: int = 5,
) -> OutputDistributionGateResult:
    """Validate the LIVE inference batch's p_up distribution before write.

    Audit Phase 2a-INFER (2026-05-07): same four invariants as the
    promotion-time gate (``validate_calibrator_distribution``), but
    applied to today's actual inference output rather than a synthetic
    alpha sweep. Catches degenerate live-batch output even when the
    calibrator itself is healthy on synthetic input — for example,
    today's 2026-05-07 incident where 16 of 27 tickers all clamped to
    p_up=0.458: the calibrator's flat region happened to be where the
    Ridge's live output landed for most of the universe, so a synthetic
    sweep might still show diversity (hitting other regions of the
    calibrator) while the live batch collapses.

    Designed to be called from inference/stages/write_output.py at the
    top of ``write_predictions``, before any S3 write. On failure,
    raises ``RuntimeError`` so the Lambda handler returns an error and
    the SF's ``Catch [States.ALL]`` fires (per audit §9.5 fail-closed).

    Pass-on-empty-batch: if ``predictions`` is empty or below
    ``min_batch_size``, returns ``passed=True`` with metrics indicating
    insufficient data — the gate doesn't fire on early-cutover days
    where coverage is incomplete by design.

    Args:
        predictions: list of prediction dicts. Each must have ``p_up``;
            ``predicted_direction`` is optional but used for skew check.
        Other args: same thresholds as ``validate_calibrator_distribution``.
            Defaults intentionally identical so promotion-time pass and
            inference-time pass agree on what "degenerate" means.
    """
    import numpy as np

    if not predictions or len(predictions) < min_batch_size:
        return OutputDistributionGateResult(
            passed=True,
            failed_check=None,
            reason=(
                f"batch size {len(predictions) if predictions else 0} "
                f"below min_batch_size={min_batch_size} — gate does not fire"
            ),
            metrics={"batch_size": len(predictions) if predictions else 0},
        )

    p_ups = np.array([
        p.get("p_up") for p in predictions
        if p.get("p_up") is not None and isinstance(p.get("p_up"), (int, float))
    ])
    directions = [
        p.get("predicted_direction") for p in predictions
        if p.get("p_up") is not None and isinstance(p.get("p_up"), (int, float))
    ]

    if len(p_ups) < min_batch_size:
        return OutputDistributionGateResult(
            passed=True,
            failed_check=None,
            reason=(
                f"only {len(p_ups)} predictions had finite p_up (below "
                f"min_batch_size={min_batch_size}) — gate does not fire"
            ),
            metrics={"batch_size": len(p_ups)},
        )

    extra_metrics = {
        "batch_size": len(p_ups),
        "n_predictions_total": len(predictions),
    }
    return _evaluate_distribution_invariants(
        p_ups=p_ups,
        directions=directions,
        source_label=f"live inference batch ({len(p_ups)} tickers)",
        min_unique_p_up=min_unique_p_up,
        max_modal_fraction=max_modal_fraction,
        max_saturation_rate=max_saturation_rate,
        min_stdev=min_stdev,
        max_direction_skew=max_direction_skew,
        floor=floor,
        ceiling=ceiling,
        extra_metrics=extra_metrics,
    )


def validate_live_batch_invariant_health(
    predictions: list[dict],
    *,
    min_batch_size: int = 5,
    max_nonfinite_rate: float = 0.20,
    max_alpha_modal_fraction: float = 0.90,
    max_sign_skew: float = 0.97,
    max_saturation_rate: float = 0.25,
    floor: float = 0.011,
    ceiling: float = 0.989,
) -> OutputDistributionGateResult:
    """Inference-time circuit-breaker that halts ONLY on recalibration-invariant
    *broken-output* signals — never on benign isotonic-``p_up`` coarseness.

    Redesign per config#1373 (2026-06-29 GE-drop false-halt). The legacy
    inference gate (``validate_live_batch_distribution``) halts on isotonic
    ``p_up`` *uniqueness / modal-fraction*. That signal is an artifact of the
    calibrator's staircase quantization: on a low-dispersion-but-perfectly-
    healthy day the level-neutralized ``predicted_alpha`` the optimizer actually
    trades on is cleanly differentiated, yet the isotonic map collapses it onto
    a handful of ``p_up`` steps — tripping the gate and false-halting the book.
    The executor hold-book was re-grounded on ``predicted_alpha`` dispersion in
    crucible-executor#308; this re-grounds the *predictor-side* circuit-breaker
    on the same invariant so the predictor stops EMITTING a calibration-artifact
    halt verdict that any downstream consumer could trip on.

    Only **recalibration-invariant** brokenness halts here — failures that are
    real regardless of which calibrator (isotonic vs Platt, config#1176) maps
    alpha->p_up:

    1. **Non-finite tradable signal** — fraction of NaN / +/-inf
       ``predicted_alpha`` above ``max_nonfinite_rate``. A non-finite tradable
       signal is broken under any calibrator.
    2. **Tradable-signal collapse** — a single ``predicted_alpha`` value holds
       ``max_alpha_modal_fraction`` or more of the batch, i.e. the model emitted
       an essentially CONSTANT tradable signal (the 2026-04-28 compression
       class), measured on ``predicted_alpha`` directly rather than on its
       isotonic image. Note: a *low-dispersion-but-differentiated* book (many
       distinct small alphas — the 2026-06-29 GE case) has a tiny modal fraction
       and correctly PASSES; handling genuinely low-CONVICTION days by *sizing
       down* (not halting) is the separate conviction-scaling concern, owned by
       the sizing layer per config#1373 item-2, NOT this circuit-breaker.
    3. **Sign skew** — fraction of one ``sign(predicted_alpha)`` at or above
       ``max_sign_skew``. Level-neutralization demeans alpha cross-sectionally,
       so an all-one-sign tradable signal is degenerate by construction
       (the 2026-05-04 false-collapse class), invariant to the calibrator.
    4. **Calibrator saturation** — fraction of ``p_up`` clipped to the
       calibrator floor/ceiling above ``max_saturation_rate``. Retained from
       the legacy gate because hard floor/ceiling clipping reflects a wide
       alpha band crushed to the bounds — a genuine output-health failure, not
       a quantization artifact (config#1373 item-1 "NaN/saturation rate").

    The legacy isotonic-``p_up`` shape stats (``n_unique_p_up``,
    ``modal_fraction``, ``stdev_p_up``) are still COMPUTED and returned in
    ``metrics`` as **observe-only** telemetry (so the staircase coarseness
    remains visible for the calibration-method soak), but they NEVER drive the
    pass/fail verdict.

    Pass-on-thin-batch: empty / sub-``min_batch_size`` batches return
    ``passed=True`` (early-cutover coverage is incomplete by design), matching
    ``validate_live_batch_distribution``.

    Args:
        predictions: per-ticker dicts. Halt logic reads ``predicted_alpha``
            (the level-neutralized tradable signal); ``p_up`` is used for the
            saturation check and observe-only shape stats. Rows missing
            ``predicted_alpha`` degrade gracefully (excluded from the tradable-
            signal checks, not treated as broken).
    """
    import math

    import numpy as np

    if not predictions or len(predictions) < min_batch_size:
        return OutputDistributionGateResult(
            passed=True,
            failed_check=None,
            reason=(
                f"batch size {len(predictions) if predictions else 0} "
                f"below min_batch_size={min_batch_size} — gate does not fire"
            ),
            metrics={"batch_size": len(predictions) if predictions else 0},
        )

    # --- Tradable signal: predicted_alpha (recalibration-invariant) ----------
    raw_alphas = [p.get("predicted_alpha") for p in predictions]
    present = [a for a in raw_alphas if a is not None and isinstance(a, (int, float))]
    n_present = len(present)
    n_nonfinite = sum(1 for a in present if not math.isfinite(float(a)))
    finite_alphas = np.array(
        [float(a) for a in present if math.isfinite(float(a))], dtype=float
    )
    nonfinite_rate = float(n_nonfinite / n_present) if n_present else 0.0

    # Modal concentration of the tradable signal (round to absorb FP noise).
    if finite_alphas.size:
        rounded_alpha = np.round(finite_alphas, 9)
        _av, _ac = np.unique(rounded_alpha, return_counts=True)
        alpha_modal_fraction = float(_ac.max() / finite_alphas.size)
        n_unique_alpha = int(len(_av))
        alpha_stdev = float(np.std(finite_alphas))
        n_pos = int(np.sum(finite_alphas > 0))
        n_neg = int(np.sum(finite_alphas < 0))
        n_signed = n_pos + n_neg
        alpha_sign_skew = (
            float(max(n_pos, n_neg) / n_signed) if n_signed > 0 else 0.0
        )
    else:
        alpha_modal_fraction = 0.0
        n_unique_alpha = 0
        alpha_stdev = 0.0
        alpha_sign_skew = 0.0

    # --- Calibrator saturation on p_up (output-health, not shape coarseness) --
    p_ups = np.array([
        p.get("p_up") for p in predictions
        if p.get("p_up") is not None and isinstance(p.get("p_up"), (int, float))
    ], dtype=float)
    n_saturated = int(np.sum((p_ups <= floor) | (p_ups >= ceiling))) if p_ups.size else 0
    saturation_rate = float(n_saturated / p_ups.size) if p_ups.size else 0.0

    # --- Observe-only isotonic-p_up shape stats (NEVER drive the verdict) -----
    if p_ups.size:
        rounded_p = np.round(p_ups, 6)
        _uv, _uc = np.unique(rounded_p, return_counts=True)
        n_unique_p_up = int(len(_uv))
        p_up_modal_fraction = float(_uc.max() / p_ups.size)
        stdev_p_up = float(np.std(p_ups))
    else:
        n_unique_p_up = 0
        p_up_modal_fraction = 0.0
        stdev_p_up = 0.0

    metrics = {
        "batch_size": len(predictions),
        "n_predictions_total": len(predictions),
        # Recalibration-invariant tradable-signal health (drives the verdict):
        "n_alpha_present": n_present,
        "n_unique_alpha": n_unique_alpha,
        "alpha_nonfinite_rate": round(nonfinite_rate, 4),
        "alpha_modal_fraction": round(alpha_modal_fraction, 4),
        "alpha_stdev": round(alpha_stdev, 8),
        "alpha_sign_skew": round(alpha_sign_skew, 4),
        # Calibrator output-health (drives the verdict):
        "saturation_rate": round(saturation_rate, 4),
        # Observe-only isotonic shape stats (do NOT drive the verdict):
        "n_unique_p_up": n_unique_p_up,
        "modal_fraction": round(p_up_modal_fraction, 4),
        "stdev_p_up": round(stdev_p_up, 6),
    }

    source_label = f"live inference batch ({len(predictions)} tickers)"

    # Check 1: non-finite tradable signal.
    if n_present >= min_batch_size and nonfinite_rate > max_nonfinite_rate:
        reason = (
            f"{nonfinite_rate:.2%} of predicted_alpha non-finite "
            f"({n_nonfinite}/{n_present}) on {source_label} (max "
            f"{max_nonfinite_rate:.0%}) — tradable signal is broken"
        )
        log.warning("Invariant output-health gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="alpha_nonfinite_rate", reason=reason,
            metrics=metrics,
        )

    # Check 2: tradable-signal collapse (essentially constant predicted_alpha).
    if finite_alphas.size >= min_batch_size and alpha_modal_fraction >= max_alpha_modal_fraction:
        reason = (
            f"predicted_alpha collapsed — one value holds "
            f"{alpha_modal_fraction:.0%} of {finite_alphas.size} finite tickers "
            f"(>= {max_alpha_modal_fraction:.0%}) on {source_label}; the tradable "
            f"signal is essentially constant (2026-04-28-class compression)"
        )
        log.warning("Invariant output-health gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="alpha_collapse", reason=reason,
            metrics=metrics,
        )

    # Check 3: sign skew on the (demeaned) tradable signal.
    if finite_alphas.size >= min_batch_size and alpha_sign_skew >= max_sign_skew:
        reason = (
            f"predicted_alpha sign skew {alpha_sign_skew:.2%} >= "
            f"{max_sign_skew:.0%} on {source_label} — degenerate one-sided "
            f"tradable signal (2026-05-04-class false collapse)"
        )
        log.warning("Invariant output-health gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="alpha_sign_skew", reason=reason,
            metrics=metrics,
        )

    # Check 4: calibrator saturation (clip-to-bounds).
    if p_ups.size >= min_batch_size and saturation_rate > max_saturation_rate:
        reason = (
            f"saturation rate {saturation_rate:.2%} exceeds "
            f"{max_saturation_rate:.0%} ({n_saturated}/{p_ups.size} p_up at "
            f"floor/ceiling on {source_label}) — calibrator clipping a wide "
            f"alpha band to its bounds"
        )
        log.warning("Invariant output-health gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="saturation_rate", reason=reason,
            metrics=metrics,
        )

    log.info(
        "Invariant output-health gate PASSED — alpha: %d unique (modal %.0f%%, "
        "nonfinite %.1f%%, sign-skew %.0f%%), saturation %.1f%%  |  "
        "[observe-only] isotonic p_up: %d unique, modal %.0f%%, stdev %.4f",
        n_unique_alpha, alpha_modal_fraction * 100, nonfinite_rate * 100,
        alpha_sign_skew * 100, saturation_rate * 100,
        n_unique_p_up, p_up_modal_fraction * 100, stdev_p_up,
    )
    return OutputDistributionGateResult(
        passed=True, failed_check=None,
        reason=(
            "all recalibration-invariant output-health checks passed "
            "(isotonic p_up shape recorded observe-only)"
        ),
        metrics=metrics,
    )


def validate_stratified_per_regime(
    predictions: list[dict],
    regimes: list[str],
    *,
    min_per_regime_size: int = 25,
    min_unique_p_up: int = 8,
    max_modal_fraction: float = 0.5,
    max_saturation_rate: float = 0.25,
    min_stdev: float = 0.005,
    max_direction_skew: float = 0.85,
    floor: float = 0.011,
    ceiling: float = 0.989,
) -> OutputDistributionGateResult:
    """Run the four invariants check INDEPENDENTLY on each regime's slice
    of the prediction batch.

    Audit Phase 2a-PROMOTE Option C (2026-05-07): the synthetic-sweep
    variant catches calibrator-internal failures (isotonic plateaus,
    floor clipping); the live-batch variant catches degenerate live
    output. Both can pass while the model collapses in only one regime
    (e.g. healthy IC + non-degenerate output in bull/neutral, but in
    bear regime the Ridge output lands entirely in a calibrator flat
    region). Aggregate-pool gates miss this; a per-regime stratified
    gate catches it.

    Per audit §9.2: "C catches the cross-regime failure class that B
    (synthetic) can't see. **B + C together catch all four recent
    incidents at the gate.** A or D alone catches at most three."

    Designed to be called from training/meta_trainer.py after the
    composite IC + synthetic-sweep gates pass. Inputs are the same OOS
    rows the existing horizon-IC diagnostic uses, with predictions
    already calibrated and a regime label attached per row.

    Pass semantics:
      - All regimes with >= ``min_per_regime_size`` rows must individually
        pass all four invariants.
      - Regimes with < ``min_per_regime_size`` rows are SKIPPED with a
        note in the metrics — too few rows to make a meaningful
        per-regime call. This matches the audit's "don't confuse
        low-sample noise with signal" framing.
      - If every regime is below ``min_per_regime_size``, the gate
        returns ``passed=True`` with a metrics dict indicating
        "insufficient stratified sample" — operator decides whether
        the training corpus is too narrow.

    Fail semantics:
      - Returns ``passed=False`` with ``failed_check`` formatted as
        ``"{regime}/{check}"`` (e.g. ``"bear/unique_p_up"``).
      - ``reason`` includes the regime name explicitly so operators
        immediately know which regime collapsed.
      - ``metrics`` carries per-regime invariant results regardless
        of pass/fail, so a borderline-pass regime over multiple
        training cycles is itself a useful trend signal.

    Args:
        predictions: list of prediction dicts (same shape as
            ``validate_live_batch_distribution`` input — must have
            ``p_up`` and ``predicted_direction``).
        regimes: parallel list of regime labels (one per prediction).
            Each label must be one of ``"bull"`` / ``"neutral"`` /
            ``"bear"``; other labels are tolerated but bucketed into
            ``"other"`` and excluded from the per-regime check.
        min_per_regime_size: minimum rows required for a regime to
            participate. Default 25 — fewer rows than that and the
            within-regime stdev / uniqueness checks are too noisy.
        Other args: same thresholds as the other gate variants.
    """
    import numpy as np

    if len(predictions) != len(regimes):
        return OutputDistributionGateResult(
            passed=False,
            failed_check="input_mismatch",
            reason=(
                f"predictions ({len(predictions)}) and regimes "
                f"({len(regimes)}) length disagree"
            ),
            metrics={},
        )

    # Bucket predictions by regime.
    by_regime: dict[str, list[dict]] = {"bull": [], "neutral": [], "bear": []}
    for p, r in zip(predictions, regimes):
        if r in by_regime:
            by_regime[r].append(p)

    per_regime_results: dict[str, dict] = {}
    failed_regime: str | None = None
    failed_result: OutputDistributionGateResult | None = None
    n_regimes_evaluated = 0

    for regime in ("bull", "neutral", "bear"):
        regime_preds = by_regime[regime]
        if len(regime_preds) < min_per_regime_size:
            per_regime_results[regime] = {
                "evaluated": False,
                "n": len(regime_preds),
                "reason": f"below min_per_regime_size={min_per_regime_size}",
            }
            continue

        n_regimes_evaluated += 1
        # Reuse the live-batch helper — same four invariants, applied
        # to this regime's slice. We pass min_batch_size=1 because the
        # outer loop already enforces min_per_regime_size; we don't want
        # the inner helper to second-guess it.
        regime_result = validate_live_batch_distribution(
            regime_preds,
            min_unique_p_up=min_unique_p_up,
            max_modal_fraction=max_modal_fraction,
            max_saturation_rate=max_saturation_rate,
            min_stdev=min_stdev,
            max_direction_skew=max_direction_skew,
            floor=floor,
            ceiling=ceiling,
            min_batch_size=1,
        )
        per_regime_results[regime] = {
            "evaluated": True,
            "n": len(regime_preds),
            "passed": regime_result.passed,
            "failed_check": regime_result.failed_check,
            "reason": regime_result.reason,
            "metrics": regime_result.metrics,
        }
        if not regime_result.passed and failed_regime is None:
            failed_regime = regime
            failed_result = regime_result

    aggregate_metrics = {
        "n_regimes_evaluated": n_regimes_evaluated,
        "min_per_regime_size": min_per_regime_size,
        "per_regime": per_regime_results,
    }

    if n_regimes_evaluated == 0:
        return OutputDistributionGateResult(
            passed=True,
            failed_check=None,
            reason=(
                f"no regime met min_per_regime_size={min_per_regime_size} "
                f"— stratified gate does not fire"
            ),
            metrics=aggregate_metrics,
        )

    if failed_regime is not None and failed_result is not None:
        regime_count = per_regime_results[failed_regime]["n"]
        failure_label = f"{failed_regime}/{failed_result.failed_check}"
        compose_reason = (
            f"regime '{failed_regime}' (n={regime_count}) failed "
            f"{failed_result.failed_check}: {failed_result.reason}"
        )
        log.warning(
            "Stratified per-regime gate FAILED — %s", compose_reason,
        )
        return OutputDistributionGateResult(
            passed=False,
            failed_check=failure_label,
            reason=compose_reason,
            metrics=aggregate_metrics,
        )

    log.info(
        "Stratified per-regime gate PASSED — all %d evaluated regimes clean",
        n_regimes_evaluated,
    )
    return OutputDistributionGateResult(
        passed=True,
        failed_check=None,
        reason=f"all {n_regimes_evaluated} evaluated regimes passed",
        metrics=aggregate_metrics,
    )


def _evaluate_distribution_invariants(
    *,
    p_ups,
    directions: list,
    source_label: str,
    min_unique_p_up: int,
    max_modal_fraction: float,
    max_saturation_rate: float,
    min_stdev: float,
    max_direction_skew: float,
    floor: float,
    ceiling: float,
    extra_metrics: dict,
) -> OutputDistributionGateResult:
    """Shared invariants check for both synthetic-sweep and live-batch variants.

    Both ``validate_calibrator_distribution`` (promotion gate) and
    ``validate_live_batch_distribution`` (inference gate) call here
    after constructing their respective ``p_ups`` array. Single source
    of truth for the four checks (uniqueness / saturation / stdev /
    direction skew); extracted so promotion-time pass and inference-time
    pass never disagree on what "degenerate" means.
    """
    import numpy as np

    # Check 1: unique p_up count (round to 6 decimals to handle FP noise)
    rounded = np.round(p_ups, 6)
    uniq_vals, uniq_counts = np.unique(rounded, return_counts=True)
    n_unique = int(len(uniq_vals))
    # Modal concentration: the fraction of the batch sitting on the single
    # most common p_up value. This is what distinguishes a genuine flat-region
    # COLLAPSE (one dominant value swallowing most of the mass — the
    # 2026-05-07 pathology) from benign quantization coarseness (an isotonic
    # calibrator maps alpha→p_up as a STAIRCASE, so a thin / low-dispersion
    # live batch legitimately lands in only a handful of monotonic steps with
    # NO single value dominating). Raw unique-count alone can't tell these
    # apart and false-halts the book on low-dispersion days (2026-06-22: 26
    # post-holiday tickers → 7 monotonic steps, p_up 0.35→0.70, modal 8/26=31%
    # — gate fired on unique_p_up=7<8 despite a perfectly healthy spread).
    modal_fraction = float(uniq_counts.max() / len(rounded))

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
        **extra_metrics,
        "n_unique_p_up": n_unique,
        "modal_fraction": round(modal_fraction, 4),
        "saturation_rate": round(saturation_rate, 4),
        "stdev_p_up": round(stdev, 6),
        "direction_skew": round(direction_skew, 4),
        "n_up": n_up,
        "n_down": n_down,
    }

    # Run checks in priority order (most distinctive failure first).
    #
    # Check 1 fires only on a genuine flat-region COLLAPSE: a low distinct-value
    # count AND a single dominant modal value (>= max_modal_fraction of the
    # batch piled on one p_up). A low distinct-count WITHOUT a dominant mode is
    # benign isotonic-staircase coarseness on a thin/low-dispersion batch and
    # must NOT halt the book (saturation / stdev / skew checks below still guard
    # the true degeneracy modes). See modal_fraction comment above.
    if n_unique < min_unique_p_up and modal_fraction >= max_modal_fraction:
        reason = (
            f"only {n_unique} unique p_up values across {source_label} "
            f"(min: {min_unique_p_up}) AND a dominant mode holds "
            f"{modal_fraction:.0%} of the batch (>= {max_modal_fraction:.0%}) "
            f"— output collapsed onto a flat region; the 2026-05-07-class plateau"
        )
        log.warning("Output-distribution gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="unique_p_up", reason=reason,
            metrics=metrics,
        )

    if saturation_rate > max_saturation_rate:
        reason = (
            f"saturation rate {saturation_rate:.2%} exceeds {max_saturation_rate:.0%} "
            f"({n_saturated}/{len(p_ups)} outputs at floor or ceiling on {source_label}) — "
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
            f"(n_up={n_up}, n_down={n_down}) on {source_label} — strongly biased"
        )
        log.warning("Output-distribution gate FAILED — %s", reason)
        return OutputDistributionGateResult(
            passed=False, failed_check="direction_skew", reason=reason,
            metrics=metrics,
        )

    log.info(
        "Output-distribution gate PASSED — "
        "%d unique p_up (modal %.0f%%), saturation %.1f%%, stdev %.4f, skew %.1f%%",
        n_unique, modal_fraction * 100, saturation_rate * 100, stdev, direction_skew * 100,
    )
    return OutputDistributionGateResult(
        passed=True, failed_check=None, reason="all checks passed",
        metrics=metrics,
    )
