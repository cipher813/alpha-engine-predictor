"""
model/research_features.py — Shared per-ticker research-feature extraction.

The four meta-model research features (`research_calibrator_prob`,
`research_composite_score`, `research_conviction`, `sector_macro_modifier`)
are sourced from research's weekly ``signals/{date}/signals.json`` snapshot.
Both the training pipeline (``training/meta_trainer.py``) and the inference
pipeline (``inference/stages/run_inference.py``) need to extract them — the
training-side join was added 2026-04-28 (PR #55) to close the meta-model
collapse root cause; the inference-side call site was buggy in the same
way (``run_inference.py:293`` read ``sector_modifiers`` from the per-ticker
dict instead of the top-level signals payload, always returning 0.0).
This module is the single source of truth so both pipelines stay in sync.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


# Sector-name canonicalization. Research's per-ticker ``universe[].sector``
# uses GICS-style names ("Health Care", "Information Technology") while the
# top-level ``sector_modifiers`` dict is keyed on shorter labels
# ("Healthcare", "Technology"). The map is sourced from research's actual
# output; any new sector must be added here AND in research's
# sector_modifiers writer in tandem.
SECTOR_NAME_CANONICAL: dict[str, str] = {
    "Health Care": "Healthcare",
    "Information Technology": "Technology",
    "Financials": "Financial",
}


def extract_research_features(
    signals_payload: dict | None,
    ticker: str,
    research_cal,
) -> dict | None:
    """Extract the four per-ticker research features from a signals payload.

    Parameters
    ----------
    signals_payload
        The full ``signals.json`` dict (with top-level ``sector_modifiers``
        and ``universe`` list). ``None`` if no snapshot is available for
        the target date — callers must drop the row / fall back per their
        own contract (training drops; inference falls back to neutral
        defaults).
    ticker
        Ticker to look up inside ``signals_payload["universe"]``.
    research_cal
        A fitted ``ResearchCalibrator`` (or any object exposing
        ``is_fitted: bool`` and ``predict(score) -> float``). ``None`` or
        an unfitted instance triggers the smooth ``score / 100`` fallback
        for ``research_calibrator_prob`` so downstream meta-model inputs
        never collapse to a constant.

    Returns
    -------
    Either a dict of the four features, or ``None`` when the payload is
    absent, the ticker is missing from ``universe``, or the score field
    is absent. ``None`` is the explicit "no real signal available"
    signal — callers must handle it without silently substituting
    constants (per ``feedback_no_silent_fails``).

    The four features:

    - ``research_calibrator_prob``: ``research_cal.predict(score)`` if the
      calibrator is fitted, else ``score / 100`` as a smooth fallback.
    - ``research_composite_score``: ``score / 100`` (research scores are
      raw 0–100; meta-model expects 0–1).
    - ``research_conviction``: ``{"rising": 1, "stable": 0, "declining": -1}``.
    - ``sector_macro_modifier``: ``sector_modifiers[canonical_sector] - 1``
      so a neutral sector contributes 0 (matches the meta-model's
      training distribution).
    """
    if signals_payload is None:
        return None

    ticker_sig = next(
        (
            s for s in signals_payload.get("universe", [])
            if s.get("ticker") == ticker
        ),
        None,
    )
    if ticker_sig is None:
        return None

    raw_score = ticker_sig.get("score")
    if raw_score is None:
        return None
    score_norm = float(raw_score) / 100.0

    conv = ticker_sig.get("conviction", "stable")
    conviction = {
        "rising": 1.0, "stable": 0.0, "declining": -1.0,
    }.get(conv, 0.0)

    sector = ticker_sig.get("sector", "") or ""
    sector_canonical = SECTOR_NAME_CANONICAL.get(sector, sector)
    sector_modifier = float(
        signals_payload.get("sector_modifiers", {}).get(sector_canonical, 1.0)
    ) - 1.0

    if research_cal is not None and getattr(research_cal, "is_fitted", False):
        cal_prob = float(research_cal.predict(raw_score))
    else:
        cal_prob = score_norm

    return {
        "research_calibrator_prob": cal_prob,
        "research_composite_score": score_norm,
        "research_conviction": conviction,
        "sector_macro_modifier": sector_modifier,
    }
