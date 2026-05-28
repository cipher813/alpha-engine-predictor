"""
regime/substrate.py — regime substrate orchestrator + S3 writer.

Assembles HMM + composite + BOCPD + guardrail outputs into the canonical
``regime_substrate.json`` payload and writes it via the canonical
``alpha_engine_lib.eval_artifacts`` shape (YYMMDDHHMM run_id + latest.json
sidecar pointer).

The Lambda handler is a thin wrapper that:

1. Fetches the historical feature DataFrame + current row from upstream
   data sources (data/macro/, ArcticDB macro library, etc.).
2. Calls ``build_regime_substrate()`` to assemble the payload.
3. Calls ``write_regime_substrate()`` to publish to S3.

This separation keeps the orchestrator pure-pandas and testable in
isolation (no S3 / no fetcher dependencies).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .bocpd import BOCPDDetector, BOCPDState
from .composite import compute_composite_intensity
from .hmm import HMMRegimeClassifier, REGIME_STATES


logger = logging.getLogger(__name__)


# Schema version stamped onto every artifact. Bump on breaking changes.
SCHEMA_VERSION: int = 1


# The canonical six-feature input set. Mirrors the macro economist's
# guardrail features + the existing predictor META_FEATURES macros.
REGIME_FEATURES: tuple[str, ...] = (
    "spy_20d_return",
    "vix_level",
    "vix_term_slope",
    "hy_oas_bps",
    "yield_curve_slope",
    "market_breadth",
)


# Default S3 layout. Override at Lambda-handler layer if needed.
DEFAULT_S3_BUCKET: str = "alpha-engine-research"
DEFAULT_S3_PREFIX: str = "regime"


# Guardrail thresholds. Mirror the existing
# ``alpha-engine-research/agents/macro_agent.py:_validate_regime`` rules
# so the substrate exposes the same severity flags the macro agent's
# guardrail layer would compute. The macro agent still owns the final
# escalation; the substrate just surfaces the flags for observability.
GUARDRAIL_DEFAULTS = {
    "vix_caution_threshold": 22.0,
    "vix_bear_threshold": 30.0,
    "spy_30d_caution_threshold": -5.0,   # percent
    "spy_30d_bear_threshold": -10.0,     # percent
    "hy_oas_caution_threshold_bps": 500.0,
}


def _compute_guardrail_flags(
    current: Mapping[str, float],
    thresholds: Mapping[str, float] | None = None,
) -> dict:
    """Compute the same severity flags the macro agent's
    ``_validate_regime`` would compute. Surfaces them for observability;
    macro agent still owns the final regime call.

    Note: ``spy_30d_return`` is not in REGIME_FEATURES (the HMM uses
    20d). The guardrail check accepts either a ``spy_30d_return`` key
    in ``current`` or falls back to scaling 20d return.
    """
    t = dict(thresholds or GUARDRAIL_DEFAULTS)
    vix = float(current.get("vix_level", np.nan))
    hy_oas = float(current.get("hy_oas_bps", np.nan))
    spy_30d_raw = current.get("spy_30d_return")
    if spy_30d_raw is None:
        spy_20d = float(current.get("spy_20d_return", np.nan))
        # Best-effort 20d → 30d scaling for the guardrail check only;
        # macro agent reads true 30d return when computing its own
        # guardrails downstream.
        spy_30d = spy_20d * (30.0 / 20.0) if np.isfinite(spy_20d) else np.nan
    else:
        spy_30d = float(spy_30d_raw)
    # Convert to percent if it looks like a decimal
    if np.isfinite(spy_30d) and abs(spy_30d) < 1.0:
        spy_30d_pct = spy_30d * 100.0
    else:
        spy_30d_pct = spy_30d

    vix_caution = bool(np.isfinite(vix) and vix > t["vix_caution_threshold"])
    vix_bear = bool(np.isfinite(vix) and vix > t["vix_bear_threshold"])
    spy_30d_caution = bool(np.isfinite(spy_30d_pct) and spy_30d_pct < t["spy_30d_caution_threshold"])
    spy_30d_bear = bool(np.isfinite(spy_30d_pct) and spy_30d_pct < t["spy_30d_bear_threshold"])
    hy_oas_caution = bool(np.isfinite(hy_oas) and hy_oas > t["hy_oas_caution_threshold_bps"])

    # Macro severity-floor hint (v0.42.0 — caution-regime-retirement-260528.md):
    # only the "bear" hint survives. The soft-caution hints retired with
    # the macro-agent rule-based override (their driving signals already
    # flow continuously through regime_intensity_z). The threshold-
    # crossing boolean feature flags (vix/spy_30d/hy_oas_caution_breached)
    # remain for observability — they're feature flags, not regime
    # classifications.
    active_severity_floor: str | None = None
    if vix_bear and spy_30d_bear:
        active_severity_floor = "bear"

    return {
        "vix_caution_breached": vix_caution,
        "vix_bear_breached": vix_bear,
        "spy_30d_caution_breached": spy_30d_caution,
        "spy_30d_bear_breached": spy_30d_bear,
        "hy_oas_caution_breached": hy_oas_caution,
        "active_severity_floor": active_severity_floor,
    }


def build_regime_substrate(
    *,
    feature_history: pd.DataFrame,
    current_features: Mapping[str, float],
    hmm: HMMRegimeClassifier,
    bocpd_state: BOCPDState | None = None,
    bocpd_detector: BOCPDDetector | None = None,
    run_id: str | None = None,
    calendar_date: str | None = None,
    trading_day: str | None = None,
    fit_window_start: str | None = None,
    fit_window_end: str | None = None,
    hmm_weights_s3_key: str | None = None,
    drawdown_block: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the regime substrate payload.

    Parameters
    ----------
    feature_history
        Historical macro feature DataFrame. Must contain all of
        ``REGIME_FEATURES`` as columns plus an index per observation
        (typically weekly). The HMM may have been fit on a subset of
        this history; ``feature_history`` here is used for the composite
        z-score's rolling window normalization and for the HMM filter
        path (the last row drives the inference).
    current_features
        Mapping of feature name → scalar value for the inference
        timestamp. Must contain all of ``REGIME_FEATURES``.
    hmm
        Fitted ``HMMRegimeClassifier``.
    bocpd_state
        BOCPD state carried over from prior runs. Pass ``None`` to start
        from the prior — the first weeks of a new pipeline will not
        produce meaningful change signals until enough history has
        accumulated.
    bocpd_detector
        Optional pre-configured detector. Defaults to ``BOCPDDetector()``.
    run_id
        ``YYMMDDHHMM`` run identifier. When None, minted via
        ``new_eval_run_id()`` from the lib.
    calendar_date, trading_day
        Dual-tracked dates per ``DATE_CONVENTIONS.md``. When None,
        derived via ``alpha_engine_lib.dates.now_dual()``.
    fit_window_start, fit_window_end
        Diagnostic metadata about the HMM's training window. Optional.
    hmm_weights_s3_key
        Pointer to where the HMM weights live in S3, for replay /
        forensic recovery. Optional.

    Returns
    -------
    Substrate JSON payload as a dict. Serializable via ``json.dumps``.
    """
    from alpha_engine_lib.dates import now_dual
    from alpha_engine_lib.eval_artifacts import new_eval_run_id

    if run_id is None:
        run_id = new_eval_run_id()
    if calendar_date is None or trading_day is None:
        dual = now_dual()
        calendar_date = calendar_date or dual.calendar_date
        trading_day = trading_day or dual.trading_day

    # ─ HMM filter inference (forward-only, no look-ahead) ────────────
    # Use the full feature_history so the filter can warm up, then take
    # the last row's posterior as the current-week classification.
    hmm_input = feature_history[list(hmm._feature_names)].dropna()
    if len(hmm_input) == 0:
        raise ValueError("feature_history is empty after dropna on HMM features.")
    filter_probs = hmm.predict_proba_filter(hmm_input)
    current_probs = filter_probs[-1]
    hmm_argmax_idx = int(np.argmax(current_probs))
    hmm_argmax = REGIME_STATES[hmm_argmax_idx]
    log_likelihood = hmm.log_likelihood(hmm_input)

    # Weeks in current state — walk back through the filter argmax until
    # the state changes.
    argmax_path = np.argmax(filter_probs, axis=1)
    weeks_in_state = 1
    for i in range(len(argmax_path) - 2, -1, -1):
        if argmax_path[i] == hmm_argmax_idx:
            weeks_in_state += 1
        else:
            break

    hmm_block = {
        "probs": {
            "bear": float(current_probs[0]),
            "neutral": float(current_probs[1]),
            "bull": float(current_probs[2]),
        },
        "argmax": hmm_argmax,
        "weeks_in_current_state": int(weeks_in_state),
        "log_likelihood": float(log_likelihood),
    }

    # ─ Composite z-score intensity ───────────────────────────────────
    composite_result = compute_composite_intensity(
        current=current_features,
        history=feature_history,
    )
    # The composite's intensity_z is "stress orientation" (positive =
    # risk-off). Downstream consumers expect higher = risk-on, so we
    # invert here to align the substrate output with the contract.
    composite_block = {
        "intensity_z": -composite_result["intensity_z"],
        "per_feature_z": composite_result["per_feature_z"],
        "features_used": composite_result["features_used"],
        "implied_severity": _classify_intensity(-composite_result["intensity_z"]),
    }

    # ─ BOCPD (optional; first-run = uninformative) ───────────────────
    if bocpd_detector is None:
        bocpd_detector = BOCPDDetector()
    if bocpd_state is None:
        bocpd_state = bocpd_detector.initial_state()
        # Warm up on history so the first emitted artifact has a
        # meaningful change signal. Use the composite intensity series
        # as the observation projection.
        for _, row in feature_history.iterrows():
            row_intensity = compute_composite_intensity(
                current=row.to_dict(),
                history=feature_history,
            )
            bocpd_state = bocpd_detector.update(bocpd_state, -row_intensity["intensity_z"])
    else:
        # Incremental update with the current observation
        bocpd_state = bocpd_detector.update(bocpd_state, composite_block["intensity_z"])
    bocpd_block = bocpd_detector.change_signal(bocpd_state)

    # ─ Guardrail flags ───────────────────────────────────────────────
    guardrails = _compute_guardrail_flags(current_features)

    # ─ Assemble ──────────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "hmm": hmm_block,
        "composite": composite_block,
        "bocpd": bocpd_block,
        "guardrails": guardrails,
        "features": {f: float(current_features[f]) for f in REGIME_FEATURES if f in current_features},
        "model_metadata": {
            "hmm_weights_s3": hmm_weights_s3_key,
            "hmm_feature_columns": list(hmm._feature_names) if hmm._feature_names else None,
            "composite_weights_version": "v1.0",
            "fit_window_start": fit_window_start,
            "fit_window_end": fit_window_end,
            "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }

    # ─ Drawdown regime leg (additive, optional — S3-contract-safe) ────
    # The block is assembled by the caller (handler/stage) which has S3
    # access to the SPY price cache + eod_pnl; substrate.py stays pure.
    # When absent (None), the key is omitted entirely → zero change for
    # existing consumers / the macro brief's HMM-only path.
    if drawdown_block is not None:
        from .drawdown import compose_effective_regime

        payload["drawdown"] = dict(drawdown_block)
        spy_leg = drawdown_block.get("spy") or {}
        excess_leg = drawdown_block.get("excess") or {}
        payload["effective_regime"] = compose_effective_regime(
            hmm_argmax=hmm_argmax,
            spy_tier=spy_leg.get("tier"),
            excess_tier=(
                excess_leg.get("tier")
                if excess_leg.get("available") else None
            ),
        )

    return payload


def _classify_intensity(intensity_z: float) -> str:
    """Map the inverted composite intensity to a coarse severity label.

    Note: the substrate's authoritative label is computed by the macro
    agent downstream. This is a hint surfaced for observability.
    """
    if intensity_z > 1.0:
        return "risk_on"
    if intensity_z > 0.3:
        return "risk_on_tilted"
    if intensity_z > -0.3:
        return "neutral"
    if intensity_z > -1.0:
        return "risk_off_tilted"
    return "risk_off"


def write_regime_substrate(
    payload: Mapping[str, Any],
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_S3_PREFIX,
    write_latest: bool = True,
) -> dict[str, str]:
    """Publish substrate payload to S3 in canonical eval_artifacts shape.

    Writes:

    - ``{prefix}/{run_id}.json`` — forensic dated artifact (always)
    - ``{prefix}/latest.json``    — operator-UX sidecar pointer (when
                                    ``write_latest=True``, the default)

    Pass ``write_latest=False`` for backfill — the live ``latest.json``
    pointer must reflect the current week's substrate, never an older
    backfilled run.

    Returns dict with ``artifact_key`` and (when written)
    ``latest_key``.
    """
    from alpha_engine_lib.eval_artifacts import (
        eval_artifact_key,
        eval_latest_key,
    )

    run_id = payload["run_id"]
    artifact_key = eval_artifact_key(prefix, run_id)  # default basename → {run_id}.json

    body = json.dumps(payload, default=str).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=artifact_key,
        Body=body,
        ContentType="application/json",
    )

    result = {"artifact_key": artifact_key}

    if write_latest:
        latest_key = eval_latest_key(prefix)
        sidecar = {
            "run_id": run_id,
            "artifact_key": artifact_key,
            "calendar_date": payload["calendar_date"],
            "trading_day": payload["trading_day"],
            "schema_version": payload["schema_version"],
            "hmm_argmax": payload["hmm"]["argmax"],
            "composite_intensity_z": payload["composite"]["intensity_z"],
            "regime_change_signal": payload["bocpd"]["change_signal"],
            "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        # Additive: surface the composed effective regime when the
        # drawdown leg is present (consumers reading latest.json get the
        # composed value without resolving the full artifact).
        if "effective_regime" in payload:
            sidecar["effective_regime"] = payload["effective_regime"][
                "effective_regime"
            ]
        s3_client.put_object(
            Bucket=bucket,
            Key=latest_key,
            Body=json.dumps(sidecar).encode("utf-8"),
            ContentType="application/json",
        )
        result["latest_key"] = latest_key
        logger.info(
            "[regime_substrate] wrote run_id=%s → s3://%s/%s (latest=%s)",
            run_id, bucket, artifact_key, latest_key,
        )
    else:
        logger.info(
            "[regime_substrate] wrote run_id=%s → s3://%s/%s (latest skipped — backfill)",
            run_id, bucket, artifact_key,
        )

    return result


def read_regime_substrate(
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_S3_PREFIX,
) -> dict[str, Any] | None:
    """Consumer-side read — resolves ``latest.json`` → artifact.

    Returns the substrate payload dict, or ``None`` if no artifact has
    been written yet. Used by the macro economist agent (Stage C) and
    the predictor L2 feature loader (Stage D).
    """
    from alpha_engine_lib.eval_artifacts import eval_latest_key

    latest_key = eval_latest_key(prefix)
    try:
        sidecar_obj = s3_client.get_object(Bucket=bucket, Key=latest_key)
    except Exception as e:
        logger.info(
            "[regime_substrate] latest.json read failed at s3://%s/%s (%s) — "
            "no substrate available yet",
            bucket, latest_key, type(e).__name__,
        )
        return None

    sidecar = json.loads(sidecar_obj["Body"].read())
    artifact_key = sidecar.get("artifact_key")
    if not artifact_key:
        return None

    body_obj = s3_client.get_object(Bucket=bucket, Key=artifact_key)
    return json.loads(body_obj["Body"].read())
