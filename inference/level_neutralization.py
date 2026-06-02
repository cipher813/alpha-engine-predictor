"""Cross-sectional level-neutralization of predicted alpha (ROADMAP L4487).

The meta-L2 puts real weight on macro features (vix_term_slope, market_breadth,
spy_20d_vol, …) that are **identical across every ticker on a given day**, so an
adverse-macro day shifts the ENTIRE predicted-alpha vector negative — a
common-mode level shift. Downstream:

  * the portfolio optimizer compares each name's alpha against the SPY=0 anchor
    (``executor/optimizer_shadow.py::_build_alpha_hat``), so a common-mode −X%
    makes every name lose to SPY → mass ``optimizer_target_zero`` flush;
  * ``gbm_veto`` (``write_output.py``: ``alpha < 0 AND bottom-half-rank AND
    confidence ≥ threshold``) and the ``output_distribution_gate`` direction
    skew degrade the same way.

Canonical alpha is **market-relative**, so the universe's cross-sectional mean
α̂ SHOULD be ≈0 — a large persistent common-mode is a bias artifact, not signal
(empirically: the 2026-06-01/06-02 batches went ~90% DOWN while realized
market-relative up-rate is ~40–56% even in bear regimes). The fix is to
subtract the cross-sectional mean of the per-name predicted_alpha so the SPY=0
anchor is the true neutral and the genuine cross-sectional skill (leak-free
CPCV meta IC ≈0.18, research-calibrator L1 IC ≈0.35) drives
direction/veto/allocation.

**Single source of truth (producer-side).** Centering here, before
``write_output`` derives ``gbm_veto`` + the gate and before the executor reads
``predicted_alpha``, means the optimizer, veto, sizing, dashboard and EOD all
inherit ONE corrected value — no per-consumer re-derivation.

**Observe-gated** behind ``cfg.XSEC_DEMEAN_ALPHA_ENABLED`` (default False):

  * **disabled** → live fields (``predicted_alpha`` / ``predicted_direction`` /
    ``prediction_confidence`` / ``p_up`` / ``p_down``) are **byte-identical** to
    today; this module only *adds* the observability fields
    (``predicted_alpha_raw``, ``predicted_alpha_centered``) and returns an
    observe block recording what centering WOULD do (mean removed, direction
    flips, skew before/after) so the effect can be inspected over 2–3 firings
    before cutover (per the observe-before-cutover discipline);
  * **enabled** → the centered alpha becomes the canonical ``predicted_alpha``
    and direction/confidence/p_up are re-derived from it via the SAME mapping
    the inference loop used (``calibrator.calibrate_prediction``); ``gbm_veto``
    + the optimizer inherit it downstream.

Limitation (documented): when the rare calibrator-collapse linear fallback in
``_rescale_cross_sectional`` engaged, the live direction came from a batch
``meta_clip`` rather than per-ticker ``calibrate_prediction``. The centered
re-derivation here always uses ``calibrate_prediction`` (a monotonic,
ranking-preserving mapping), so the two can differ slightly on that degraded
path. Harmless while disabled (no live mutation); validated before cutover.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def cross_sectional_mean(predictions: list[dict]) -> float:
    """Mean of the per-name ``predicted_alpha`` across the batch (the
    common-mode level to remove). All entries are universe names — SPY/cash are
    optimizer-side sentinels, not predictions — so this is the name mean."""
    vals = [float(p.get("predicted_alpha", 0.0) or 0.0) for p in predictions]
    return float(np.mean(vals)) if vals else 0.0


def _derive(alpha: float, calibrator, label_clip: float) -> dict:
    """Re-derive (p_up, p_down, predicted_direction, prediction_confidence) for
    a (possibly centered) alpha using the SAME mapping the inference loop used.

    ``calibrator.calibrate_prediction`` already linear-falls-back when unfitted,
    so we route through it whenever a calibrator object exists; the pure-linear
    branch only fires when there is no calibrator object at all.
    """
    lc = label_clip if (label_clip and label_clip > 0) else 0.15
    if calibrator is not None:
        return calibrator.calibrate_prediction(alpha, label_clip=lc)
    p_up = float(np.clip(0.5 + alpha / (2.0 * lc), 0.0, 1.0))
    p_down = 1.0 - p_up
    return {
        "p_up": round(p_up, 4),
        "p_down": round(p_down, 4),
        "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
        "prediction_confidence": round(abs(p_up - 0.5) * 2.0, 4),
    }


def _down_count(directions: list[str]) -> int:
    return sum(1 for d in directions if d == "DOWN")


def apply_cross_sectional_neutralization(
    predictions: list[dict],
    *,
    enabled: bool,
    calibrator,
    label_clip: float,
) -> dict:
    """Mutate ``predictions`` in place and return an observe block.

    ALWAYS stamps ``predicted_alpha_raw`` + ``predicted_alpha_centered`` on each
    prediction and computes the raw-vs-centered direction diff. ONLY overwrites
    the live fields (``predicted_alpha`` + direction/confidence/p_up) when
    ``enabled`` is True — the gate's ``gbm_veto`` and the executor optimizer
    inherit the centered value downstream.
    """
    n = len(predictions)
    if n < 2:
        # Degenerate cross-section: centering one name to its own mean zeroes
        # its alpha. Skip the transform; still stamp raw for forensics.
        for p in predictions:
            p.setdefault(
                "predicted_alpha_raw",
                round(float(p.get("predicted_alpha", 0.0) or 0.0), 6),
            )
        block = {
            "enabled": bool(enabled),
            "applied": False,
            "reason": "n<2",
            "n_predictions": n,
            "xsec_mean_removed": 0.0,
        }
        log.info("Level-neutralization skipped (n=%d < 2)", n)
        return block

    mean_raw = cross_sectional_mean(predictions)

    # Skew the gate currently sees (from the LIVE predicted_direction — raw when
    # disabled). Compared apples-to-apples against the centered re-derivation.
    raw_dirs = [str(p.get("predicted_direction") or "") for p in predictions]
    n_down_raw = _down_count(raw_dirs)

    n_flips_sign = 0
    centered_dirs: list[str] = []
    for p in predictions:
        raw = float(p.get("predicted_alpha", 0.0) or 0.0)
        centered = raw - mean_raw
        p["predicted_alpha_raw"] = round(raw, 6)
        p["predicted_alpha_centered"] = round(centered, 6)
        if (raw < 0.0) != (centered < 0.0):
            n_flips_sign += 1

        der = _derive(centered, calibrator, label_clip)
        centered_dirs.append(der["predicted_direction"])

        if enabled:
            p["predicted_alpha"] = round(centered, 6)
            p["p_up"] = der["p_up"]
            p["p_down"] = der["p_down"]
            p["predicted_direction"] = der["predicted_direction"]
            p["prediction_confidence"] = der["prediction_confidence"]

    n_down_centered = _down_count(centered_dirs)
    block = {
        "enabled": bool(enabled),
        "applied": bool(enabled),
        "n_predictions": n,
        "xsec_mean_removed": round(mean_raw, 6),
        "n_direction_flips": n_flips_sign,
        "direction_skew_raw": round(n_down_raw / n, 4),
        "direction_skew_centered": round(n_down_centered / n, 4),
    }
    log.info(
        "Level-neutralization %s: mean_removed=%.6f  flips=%d  "
        "skew %.3f → %.3f (n=%d)",
        "APPLIED" if enabled else "OBSERVE",
        mean_raw, n_flips_sign,
        block["direction_skew_raw"], block["direction_skew_centered"], n,
    )
    return block
