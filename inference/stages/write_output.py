"""Stage: write_output — Build metrics, apply veto, write predictions, send email, write health."""

from __future__ import annotations

import json
import logging
import os
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from alpha_engine_lib.secrets import get_secret

import config as cfg
from inference.pipeline import PipelineContext
from inference.s3_io import _s3_put_json

log = logging.getLogger(__name__)


# ── S3-delivered predictor params (veto threshold) ────────────────────────────

_predictor_params_cache: dict | None = None
_predictor_params_loaded: bool = False
# Local cache persists last known optimal across Lambda cold-starts (via /tmp)
# and EC2 restarts (via project dir).
_PREDICTOR_PARAMS_CACHE_PATH = Path(
    os.environ.get("PREDICTOR_PARAMS_CACHE", "/tmp/predictor_params_cache.json")
)


def _load_predictor_params_from_s3(s3_bucket: str) -> dict | None:
    """Read config/predictor_params.json from S3. Cache per cold-start.

    Fallback chain: S3 → local cache file → None (hardcoded defaults).
    On successful S3 read, writes a local cache so the last known optimal
    params survive transient S3 failures.
    """
    global _predictor_params_cache, _predictor_params_loaded
    if _predictor_params_loaded:
        return _predictor_params_cache
    _predictor_params_loaded = True

    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key="config/predictor_params.json")
        data = json.loads(obj["Body"].read())
        if "veto_confidence" in data:
            _predictor_params_cache = data
            log.info("Loaded predictor params from S3: veto_confidence=%.2f", data["veto_confidence"])
            # Persist to local cache for fault tolerance
            try:
                _PREDICTOR_PARAMS_CACHE_PATH.write_text(json.dumps(data, indent=2))
            except Exception:
                pass  # best-effort
        return _predictor_params_cache
    except Exception as e:
        log.warning("Could not read predictor params from S3: %s", e)

    # Fallback: last known optimal from local cache
    try:
        if _PREDICTOR_PARAMS_CACHE_PATH.exists():
            data = json.loads(_PREDICTOR_PARAMS_CACHE_PATH.read_text())
            if "veto_confidence" in data:
                _predictor_params_cache = data
                log.info(
                    "Loaded predictor params from local cache (last known optimal): veto_confidence=%.2f",
                    data["veto_confidence"],
                )
                return _predictor_params_cache
    except Exception as e2:
        log.debug("Could not read local predictor params cache: %s", e2)

    return None


def regime_conditional_veto_adjustment(
    intensity_z: float | None,
    *,
    scale: float = 0.10,
    cap: float = 0.20,
) -> float:
    """Stage D' Wire 4 — continuous regime adjustment to the veto threshold.

    Maps the predictor regime substrate's composite intensity_z to an
    additive adjustment on the base veto confidence threshold:
      * Risk-off (intensity_z < 0): NEGATIVE adjustment → LOWER threshold
        → more aggressive vetoing (protects capital in stress).
      * Risk-on (intensity_z > 0): POSITIVE adjustment → HIGHER threshold
        → more permissive (avoids missing opportunities in tailwind).

    Direction matches the discrete bear/neutral/bull adjustment table
    in ``get_veto_threshold`` — Wire 4 is the continuous version of the
    same signal.

    Math: ``intensity_z * scale`` clamped to ``[-cap, +cap]``. Defaults
    (scale=0.10, cap=0.20) give:
      * z=-2 → -0.20 (matches the discrete ``bear`` adjustment)
      * z=+2 → +0.20 (twice the discrete ``bull`` adjustment, since the
        continuous gate doesn't need to be conservative on the upside —
        a strong risk-on reading is a stronger signal than a one-word
        macro classification).
      * z=±0 (neutral) → 0.00 (no adjustment).
      * None (substrate unavail) → 0.00 (legacy behavior preserved).
    """
    if intensity_z is None:
        return 0.0
    raw = float(intensity_z) * float(scale)
    return max(-float(cap), min(float(cap), raw))


def get_veto_threshold(
    s3_bucket: str,
    market_regime: str = "",
    regime_intensity_z: float | None = None,
    forced_bear: bool = False,
    drawdown_effective_regime: str | None = None,
    drawdown_protective_severity: int | None = None,
) -> float:
    """
    Return the active veto confidence threshold, adjusted by market regime.

    In bear regimes, the threshold is lowered (more aggressive vetoing)
    to protect capital. In bull regimes, the threshold is raised (more permissive)
    to avoid missing opportunities. Drawdown-protective state is a separate
    axis (see ``drawdown_protective_severity`` parameter below).

    Confidence semantics post-2026-05-12: ``|p_up - 0.5| * 2`` (ROADMAP L1615).
    A threshold of 0.30 corresponds to the prior 0.65 winner-probability gate
    via the linear map ``new = (old - 0.5) * 2``. Regime adjustments doubled
    to preserve the same relative pull on the new [0, 1] axis.

    Discrete regime adjustments (3-class Ang-Bekaert, post v0.42.0 —
    caution-regime-retirement-260528.md) when Stage D' Wire 4 is OFF:
      bear:    -0.20  (e.g., 0.30 → 0.10 — veto more aggressively)
      neutral:  0.00  (no adjustment)
      bullish: +0.10  (e.g., 0.30 → 0.40 — allow more entries)
    Legacy "caution" / "bearish" emissions coerce to the nearest 3-class
    neighbor (caution → neutral, bearish → bear) for grandfather safety;
    raw LLM caution is coerced upstream at macro-agent
    ``_validate_regime``.

    Continuous regime adjustment (Stage D' Wire 4, regime-v3-260514):
      When ``regime_veto_enabled=True`` in S3 predictor_params AND
      ``regime_intensity_z`` is non-None, the continuous adjustment
      replaces the discrete one (both encode the same regime signal —
      one numeric, one categorical). The continuous adjustment is
      ``intensity_z * scale`` clamped to ``[-cap, +cap]``, defaults
      scale=0.10 cap=0.20. Direction matches the discrete table.

      Default ``regime_veto_enabled=False`` ships Wire 4 dormant.

    Forced-bear clamp (regime-fast-signal-260515.md Stage F2):
      When ``regime_forced_bear_enabled=True`` in S3 predictor_params AND
      ``forced_bear=True`` (the daily BOCPD circuit-breaker latched), the
      threshold is clamped to its **bear floor** — the most-aggressive
      veto the continuous gate allows (``base - cap``) — taking the
      max-protection of {whatever Wire 4 / discrete produced, bear
      floor}. This is the predictor-side mirror of the executor's
      forced-bear posture; both gating surfaces flip together so a
      confirmed break is not half-honored.

      Default ``regime_forced_bear_enabled=False``: the clamp is dormant
      but, when ``forced_bear=True``, the counterfactual is logged
      (parallel-observe) so the F2 gate has data during the gated window.

    Drawdown-regime clamp (regime-drawdown-hysteresis-260518.md, PR 3):
      When ``drawdown_regime_enabled=True`` in S3 predictor_params AND
      the drawdown leg's protective severity ordinal
      (``ctx.drawdown_protective_severity``, stamped by the daily stage:
      0=risk_on, 1=caution, 2=risk_off/alpha_bleed) is non-zero, the
      threshold is clamped — severity 2 → bear floor (``base - cap``,
      same as forced-bear); severity 1 → milder caution floor
      (``base - cap/2``). Composed most-protective with the forced-bear
      clamp and the Wire-4/discrete result: all clamps accumulate via
      ``min`` (lower threshold = more aggressive veto); none is
      half-honored.

      Backward-compat: if ``drawdown_protective_severity`` is None but
      ``drawdown_effective_regime`` is provided (legacy callsites or
      historical artifacts), the severity is derived from the string
      via the type-system separation introduced in v0.42.0 (caution-
      regime-retirement-260528.md Phase 2A). Drop the string-arg path
      once all callers stamp the ordinal directly.

      Default ``drawdown_regime_enabled=False``: dormant; the
      counterfactual is logged (parallel-observe) for the promotion gate.
    """
    params = _load_predictor_params_from_s3(s3_bucket)
    if params and "veto_confidence" in params:
        base = float(params["veto_confidence"])
    else:
        base = cfg.MIN_CONFIDENCE

    cap = float(params.get("regime_veto_cap", 0.20) if params else 0.20)

    # Wire 4 path: continuous regime adjustment via intensity_z. Replaces
    # (does NOT layer on top of) the discrete adjustment when enabled.
    wire4_enabled = bool(params and params.get("regime_veto_enabled", False))
    if wire4_enabled and regime_intensity_z is not None:
        scale = float(
            params.get("regime_veto_scale", 0.10) if params else 0.10
        )
        adjustment = regime_conditional_veto_adjustment(
            regime_intensity_z, scale=scale, cap=cap,
        )
        threshold = max(0.0, min(0.80, base + adjustment))
        if adjustment != 0.0:
            log.info(
                "Veto threshold regime-adjusted (Wire 4 continuous): "
                "base=%.2f %+.3f (intensity_z=%.3f) → %.3f",
                base, adjustment, regime_intensity_z, threshold,
            )
    else:
        # Legacy path: discrete adjustment by qualitative regime string.
        # 3-class Ang-Bekaert (v0.42.0). Legacy 4-class strings ("caution",
        # "bearish") grandfathered for forward compat with historical
        # artifacts and pre-cutover predictor_params.json values; the
        # macro-agent _validate_regime upstream coerces raw LLM "caution"
        # to "neutral" so new emissions stay 3-class.
        regime = market_regime.lower().strip() if market_regime else ""
        regime_adjustments = {
            "bear": -0.20,
            "bearish": -0.20,    # legacy alias (caution-regime-retirement-260528)
            "caution": -0.10,    # legacy 4-class (grandfather; coerced upstream)
            "neutral": 0.0,
            "bull": 0.10,
            "bullish": 0.10,
        }
        adjustment = regime_adjustments.get(regime, 0.0)
        threshold = max(0.0, min(0.80, base + adjustment))
        if adjustment != 0.0:
            log.info(
                "Veto threshold regime-adjusted: base=%.2f %+.2f (%s) → %.2f",
                base, adjustment, regime, threshold,
            )

    # ── Max-protection clamps (forced-bear + drawdown) ───────────────────
    # Both accumulate via min() so the most-protective wins and neither
    # is half-honored. Each is independently gated default-OFF; flag-off
    # logs the counterfactual (parallel-observe) without changing
    # behavior. With both flags off (the only live state today) the
    # return value is exactly ``threshold`` — unchanged.
    effective = threshold

    if forced_bear:
        bear_floor = max(0.0, min(0.80, base - cap))
        clamped = min(threshold, bear_floor)  # lower = more aggressive veto
        if bool(params and params.get("regime_forced_bear_enabled", False)):
            if clamped < effective:
                log.warning(
                    "Veto threshold FORCED-BEAR clamp: %.3f → %.3f "
                    "(bear_floor=base-cap=%.2f-%.2f)",
                    effective, clamped, base, cap,
                )
            effective = min(effective, clamped)
        else:
            log.info(
                "regime_forced_bear OBSERVE (flag off): forced_bear=True, "
                "veto would clamp %.3f → %.3f (bear_floor=%.3f)",
                threshold, clamped, bear_floor,
            )

    # Drawdown-protective clamp — read the SOTA severity ordinal if
    # stamped; fall back to deriving it from the legacy string field
    # for grandfather safety. v0.42.0 Phase 2A type-system separation
    # (caution-regime-retirement-260528.md): drawdown leg is an axis
    # ORTHOGONAL to macro market_regime; severity ordinal is the
    # canonical input.
    dd_severity = drawdown_protective_severity
    if dd_severity is None and drawdown_effective_regime is not None:
        # Backward-compat: derive severity from the legacy macro-aliased
        # field. Mapping mirrors the pre-cutover string check.
        legacy = (drawdown_effective_regime or "").lower().strip()
        dd_severity = {"bear": 2, "caution": 1}.get(legacy, 0)

    if dd_severity and dd_severity > 0:
        dd_floor = max(0.0, min(0.80, base - (cap if dd_severity >= 2 else cap / 2.0)))
        dd_clamped = min(threshold, dd_floor)
        dd_label = "risk_off" if dd_severity >= 2 else "caution"
        if bool(params and params.get("drawdown_regime_enabled", False)):
            if dd_clamped < effective:
                log.warning(
                    "Veto threshold DRAWDOWN clamp (severity=%d, %s): %.3f → %.3f "
                    "(dd_floor=%.3f)", dd_severity, dd_label,
                    effective, dd_clamped, dd_floor,
                )
            effective = min(effective, dd_clamped)
        else:
            log.info(
                "drawdown_regime OBSERVE (flag off): severity=%d (%s), "
                "veto would clamp %.3f → %.3f (dd_floor=%.3f)",
                dd_severity, dd_label, threshold, dd_clamped, dd_floor,
            )

    return effective


# ── Output writing (migrated from daily_predict.py) ──────────────────────────

def _read_existing_predictions(s3_bucket: str, date_str: str) -> list[dict]:
    """Read existing predictions/{date}.json from S3.

    Returns the `predictions` list, or [] if the object is absent or unreadable.
    Used by supplemental-scoring mode to merge new predictions into the
    existing day's output without overwriting prior tickers.
    """
    import boto3
    from botocore.exceptions import ClientError
    dated_key = cfg.PREDICTIONS_KEY.format(date=date_str)
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key=dated_key)
        payload = json.loads(obj["Body"].read())
        return list(payload.get("predictions") or [])
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "AccessDenied", "404", "403"):
            log.info(
                "Supplemental mode: no existing predictions at s3://%s/%s (%s) — "
                "merge is a no-op union with empty",
                s3_bucket, dated_key, code,
            )
            return []
        raise


def _merge_predictions(
    new: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Union existing + new predictions by ticker (new wins on collision),
    then re-sort by predicted_alpha desc and reassign combined_rank.

    gbm_veto is NOT recomputed here — caller is responsible for running the
    veto block over the merged list (it depends on `n_preds` which changes
    after union).
    """
    by_ticker: dict[str, dict] = {}
    for p in existing:
        t = (p.get("ticker") or "").upper()
        if t:
            by_ticker[t] = dict(p)  # copy, will mutate combined_rank below
    for p in new:
        t = (p.get("ticker") or "").upper()
        if t:
            by_ticker[t] = dict(p)  # new overrides existing on collision

    merged = list(by_ticker.values())
    merged.sort(key=lambda p: -(p.get("predicted_alpha") or 0))
    for i, p in enumerate(merged):
        p["combined_rank"] = i + 1
    log.info(
        "Supplemental merge: %d new + %d existing → %d union (%d collisions)",
        len(new), len(existing), len(merged),
        len(new) + len(existing) - len(merged),
    )
    return merged


def write_predictions(
    predictions: list[dict],
    date_str: str,
    s3_bucket: str,
    metrics: dict,
    dry_run: bool = False,
    veto_threshold: float | None = None,
    fd=None,
    level_neutralization: dict | None = None,
    n_universe: int | None = None,
    n_universe_covered: int | None = None,
) -> None:
    """
    Write predictions JSON to S3 at both the dated key and latest.json.
    Also writes metrics/latest.json. All S3 operations are best-effort.

    Parameters
    ----------
    predictions : List of per-ticker prediction dicts.
    date_str :    Date string YYYY-MM-DD.
    s3_bucket :   S3 bucket name.
    metrics :     Metrics dict to write to predictor/metrics/latest.json.
    dry_run :     If True, print to stdout instead of writing to S3.
    veto_threshold : Confidence threshold for veto gate. Defaults to cfg.MIN_CONFIDENCE.
    """
    # Audit Phase 2a-INFER (2026-05-07): output-distribution gate at the
    # inference-time write site. Validates the live batch's p_up
    # distribution against the same four invariants the promotion gate
    # uses (uniqueness / saturation / stdev / direction skew). Behind a
    # feature flag for one observation cycle so we measure baseline pass
    # rate before promoting the gate to fail-closed.
    #
    # When the flag is True and the gate fails:
    #   - log ERROR with structured reason
    #   - record the failure in the metrics output for forensic trail
    #   - raise RuntimeError so the Lambda handler returns an error,
    #     SF Catch [States.ALL] fires, flow-doctor alerts, predictions.json
    #     is NOT written (executor reads prior-day fallback)
    #
    # The gate's metrics ride along with the standard metrics.json write
    # in observe-only mode, so we get a forensic record of "would this
    # have blocked today's invocation" without risk to live trading.
    from model.output_distribution_gate import validate_live_batch_distribution
    inference_gate_blocking = bool(
        getattr(cfg, "OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING", False)
    )
    inference_gate_result = validate_live_batch_distribution(predictions)
    if not inference_gate_result.passed:
        log.error(
            "Output-distribution gate FAILED at inference-time on %d-row batch: %s "
            "(blocking=%s)",
            len(predictions), inference_gate_result.reason, inference_gate_blocking,
        )
    else:
        log.info(
            "Output-distribution gate (inference-time) PASSED — %s",
            inference_gate_result.reason,
        )

    threshold = veto_threshold if veto_threshold is not None else cfg.MIN_CONFIDENCE
    # Build the predictions envelope
    n_high_confidence = sum(
        1 for p in predictions
        if p.get("prediction_confidence", 0) >= threshold
    )

    # Single gate-result block, carried in BOTH metrics.json (forensic trail)
    # AND predictions.json (so the executor's hold-book safeguard can read the
    # gate atomically with the batch it describes — the consumer-cutover for the
    # 2026-06-01 hold-book decision). Additive field; no predictor behavior
    # change here — the executor decides whether to act on it.
    gate_block = {
        "passed": inference_gate_result.passed,
        "failed_check": inference_gate_result.failed_check,
        "reason": inference_gate_result.reason,
        "metrics": inference_gate_result.metrics,
        "blocking": inference_gate_blocking,
        "would_have_blocked_if_blocking": not inference_gate_result.passed,
    }

    output = {
        "date": date_str,
        "model_version": metrics.get("model_version", "unknown"),
        "model_hit_rate_30d": metrics.get("hit_rate_30d_rolling", None),
        "n_predictions": len(predictions),
        "n_high_confidence": n_high_confidence,
        # Executor reads this for the hold-book safeguard (2026-06-01): a
        # strongly-biased batch (passed=False) must not drive an optimizer
        # rotation — the executor holds the current book instead.
        "output_distribution_gate": gate_block,
        # Cross-sectional level-neutralization observe block (L4487). Additive;
        # records the common-mode mean removed + the raw-vs-centered direction
        # diff. When its `applied` is True the predictions' predicted_alpha is
        # already centered (the optimizer + gbm_veto inherit it).
        "level_neutralization": level_neutralization,
        "predictions": predictions,
    }

    metrics_out = {
        **metrics,
        "n_predictions_today": len(predictions),
        "n_high_confidence": n_high_confidence,
        # config#1075: the tradable-universe coverage denominator + covered count
        # so the report card's inference_coverage grades a real value in [0,1]
        # (covered/universe) instead of a permanent N/A. Computed at the call site
        # against signals.json's universe ∩ the merged predictions. None on the
        # degraded write paths (soft-timeout / abort partials) — honest absence.
        "n_universe": n_universe,
        "n_universe_covered": n_universe_covered,
        "last_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        # Audit Phase 2a-INFER: persist the gate result alongside the
        # standard metrics for forensic trail. Carries pass/fail +
        # failed_check + the four measured invariants (n_unique_p_up,
        # saturation_rate, stdev_p_up, direction_skew) regardless of
        # whether the gate was in blocking mode this run.
        "output_distribution_gate": gate_block,
        # Cross-sectional level-neutralization observe block (L4487) — forensic
        # trail alongside the gate (the two share the same skew→flush root).
        "level_neutralization": level_neutralization,
    }

    predictions_json = json.dumps(output, indent=2)
    metrics_json = json.dumps(metrics_out, indent=2)

    # Audit Phase 2a-INFER: fail-closed at the boundary. If the gate
    # failed AND the blocking flag is on, raise BEFORE the dry_run
    # short-circuit and BEFORE any S3 write. The raise must fire in
    # dry-run drills too — operators want to catch degenerate batches
    # during a release simulation the same way as in production. Per
    # audit §9.5: the alert IS the record; writing stale metrics with
    # status=ok would mislead dashboards.
    if inference_gate_blocking and not inference_gate_result.passed:
        raise RuntimeError(
            f"Output-distribution gate refused write at inference time: "
            f"{inference_gate_result.reason}. predictions.json NOT written; "
            f"executor falls back to prior-day predictions. "
            f"Investigate the model output (manifest's output_distribution_gate "
            f"sub-dict from the most recent training run) before next "
            f"inference; if the model is genuinely unhealthy, roll back to "
            f"a known-good archive at predictor/weights/meta/archive/."
        )

    if dry_run:
        print("=== PREDICTIONS (dry-run) ===")
        print(predictions_json)
        print("\n=== METRICS (dry-run) ===")
        print(metrics_json)
        return

    dated_key = cfg.PREDICTIONS_KEY.format(date=date_str)
    latest_key = cfg.PREDICTIONS_LATEST_KEY
    metrics_key = cfg.METRICS_KEY

    import boto3
    s3 = boto3.client("s3")

    # Write each S3 object independently so partial failures don't block others
    writes = [
        (dated_key, predictions_json, "predictions (dated)"),
        (latest_key, predictions_json, "predictions (latest)"),
        (metrics_key, metrics_json, "metrics"),
    ]
    n_ok = 0
    for key, body, label in writes:
        try:
            _s3_put_json(s3, s3_bucket, key, body)
            log.info("Written s3://%s/%s", s3_bucket, key)
            n_ok += 1
        except Exception as exc:
            log.error("S3 write failed for %s: %s", label, exc)
    if n_ok < len(writes):
        log.error(
            "Partial S3 write: %d/%d succeeded. Check IAM permissions for s3://%s",
            n_ok, len(writes), s3_bucket,
        )


# ── Predictor email ────────────────────────────────────────────────────────────

def _load_executor_params_for_email(bucket: str) -> dict | None:
    """Best-effort fetch of `config/executor_params.json` from the shared
    research bucket — the auto-tuned params the executor reads at
    cold-start. Used by `_build_predictor_email` to surface effective
    optimizer params alongside the morning briefing (ROADMAP L234,
    morning-email side). Returns None on any failure — the email
    builder degrades gracefully (block is skipped, briefing still
    sends).

    Cross-module read: both predictor and executor reference the same
    artifact (`alpha-engine/executor/main.py::_load_executor_params_from_s3`
    is the consumer; backtester's `executor_optimizer.apply()` is the
    producer). This loader exists to surface the live state on the
    morning email, NOT to drive any predictor behaviour.
    """
    try:
        import boto3
        import json
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="config/executor_params.json")
        data = json.loads(obj["Body"].read())
        if isinstance(data, dict):
            return data
    except Exception as e:  # noqa: BLE001 — secondary observability; primary briefing path unaffected
        log.debug("L234 executor-params email block: fetch failed (%s) — block skipped", e)
    return None


def _build_predictor_email(
    predictions: list[dict],
    metrics: dict,
    date_str: str,
    signals_data: dict | None = None,
    veto_threshold: float | None = None,
    bucket: str | None = None,
) -> tuple[str, str, str]:
    """
    Build subject, HTML body, and plain-text body for the combined morning briefing.

    When signals_data is supplied (the raw signals.json payload from the research
    pipeline), a research section is prepended containing market regime, buy
    candidates, and sector ratings. The GBM predictions follow as the second half.

    When ``bucket`` is supplied, the email also includes an "Effective
    optimizer params" block sourced from ``config/executor_params.json``
    (ROADMAP L234 morning-email side). Best-effort — missing artifact
    silently skips the block.

    Returns
    -------
    (subject, html_body, plain_body)
    """
    import datetime as _dt

    _vt = veto_threshold if veto_threshold is not None else cfg.MIN_CONFIDENCE
    model_version = metrics.get("model_version", "unknown")
    val_ic        = metrics.get("ic_30d")        # 30-day information coefficient
    n_total       = len(predictions)
    is_meta       = metrics.get("inference_mode") == "meta" or "meta" in model_version.lower()

    # Single list sorted by combined_rank (best first)
    sorted_preds = sorted(predictions, key=lambda p: p.get("combined_rank") or 999)

    # Counts for subject line
    ups   = [p for p in predictions if p.get("predicted_direction") == "UP"]
    downs = [p for p in predictions if p.get("predicted_direction") == "DOWN"]

    # Vetoes: negative predicted alpha AND combined_rank in bottom half
    n_preds = len(predictions)
    vetoes = [
        p for p in predictions
        if (p.get("predicted_alpha", 0) or 0) < 0
        and p.get("combined_rank") is not None
        and p["combined_rank"] > n_preds / 2
    ]
    n_vetoed = len(vetoes)

    # ── Research data extraction ───────────────────────────────────────────────
    sd = signals_data or {}
    market_regime    = sd.get("market_regime", "")
    population       = sd.get("universe", []) or sd.get("population", [])
    buy_candidates   = sd.get("buy_candidates", []) or []
    sector_ratings   = sd.get("sector_ratings", {})
    sorted_sectors: list = []

    # Tickers the executor can act on that the GBM did NOT score — surface these
    # prominently so a stale/short predictions run can't silently hide buys.
    _pred_tickers     = {p.get("ticker") for p in predictions}
    unscored_buys     = [
        c for c in buy_candidates
        if isinstance(c, dict) and c.get("ticker") not in _pred_tickers
    ]
    unscored_tickers  = {c.get("ticker") for c in unscored_buys}

    # ── Subject ───────────────────────────────────────────────────────────────
    veto_str    = f" | {n_vetoed} veto{'es' if n_vetoed != 1 else ''}" if n_vetoed else ""
    regime_str  = f" | {market_regime.upper()}" if market_regime else ""
    cand_str    = f" | {len(population)} stocks" if population else ""
    subject = (
        f"Alpha Engine Brief | {date_str}{regime_str}{cand_str} | "
        f"{len(ups)} UP / {len(downs)} DOWN"
        f"{veto_str}"
    )

    # ── Helpers ───────────────────────────────────────────────────────────────
    _pt = _dt.timezone(_dt.timedelta(hours=-7))  # PDT
    run_time = _dt.datetime.now(_pt).strftime("%-I:%M %p PT")
    ic_str   = f"{val_ic:.4f}" if isinstance(val_ic, (int, float)) else "—"

    def _source_tag(p: dict) -> str:
        return ""  # buy_candidates merged into universe — no separate source tag

    def _alpha_str(p: dict) -> str:
        a = p.get("predicted_alpha")
        if a is None:
            return "—"
        return f"{'+' if a >= 0 else ''}{a * 100:.2f}%"

    def _conf_pct(p: dict) -> str:
        return f"{p.get('prediction_confidence', 0) * 100:.0f}%"

    # ── HTML ──────────────────────────────────────────────────────────────────
    TH = 'style="background:#f0f0f0; padding:4px 8px; text-align:left; border:1px solid #ccc;"'
    TD = 'style="padding:4px 8px; border:1px solid #ddd;"'
    TDR = 'style="padding:4px 8px; border:1px solid #ddd; text-align:right;"'
    TABLE = 'style="border-collapse:collapse; width:100%; font-family:monospace; font-size:12px;"'

    def _dir_badge(p: dict) -> str:
        d = p.get("predicted_direction", "")
        is_veto = (d == "DOWN" and p.get("prediction_confidence", 0) >= _vt)
        if is_veto:
            return '<span style="color:#c62828; font-weight:bold;">⚠ VETO</span>'
        colors = {"UP": "#2e7d32", "DOWN": "#c62828"}
        return f'<span style="color:{colors.get(d, "#888")}; font-weight:bold;">{d}</span>'

    def _meta_cols(p: dict) -> str:
        """Extra columns for meta-model predictions: momentum, vol, research.
        Regime column removed 2026-04-16 (Tier 0 classifier retired). The
        LLM-derived market_regime from research still renders in the header
        pill above the table — that's the regime signal the executor consumes."""
        mom = p.get("momentum_confirmation")
        vol = p.get("expected_move")
        rscore = p.get("research_calibrator_prob")
        return (
            f'<td {TDR}>{f"{mom:+.3f}" if mom is not None else "—"}</td>'
            f'<td {TDR}>{f"{vol:.3f}" if vol is not None else "—"}</td>'
            f'<td {TDR}>{f"{rscore:.0%}" if rscore is not None else "—"}</td>'
        )

    def _html_prediction_table(preds: list[dict]) -> str:
        if not preds:
            return '<p style="color:#888; font-style:italic;">No predictions available.</p>'
        n_preds = len(preds)
        rows = []
        for p in preds:
            cr = p.get("combined_rank")
            cr_str = f'{cr:.1f}' if cr is not None else "—"
            is_vetoed = (
                p.get("predicted_alpha", 0) is not None
                and (p.get("predicted_alpha", 0) or 0) < 0
                and cr is not None
                and cr > n_preds / 2
            )
            veto_tag = ' <span style="color:#d32f2f;">⚠ VETO</span>' if is_vetoed else ""
            rows.append(
                f'<tr>'
                f'<td {TD}><b>{p["ticker"]}{_source_tag(p)}</b></td>'
                f'<td {TDR}>{_alpha_str(p)}</td>'
                f'<td {TDR}>{cr_str}</td>'
                f'<td {TDR}>{_conf_pct(p)}</td>'
                f'<td {TD} style="text-align:center;">{_dir_badge(p)}{veto_tag}</td>'
                + (_meta_cols(p) if is_meta else "")
                + f'<td {TD}>{p.get("watchlist_source", "—")}</td>'
                f'</tr>'
            )
        meta_headers = (
            f'<th {TH}>Mom</th>'
            f'<th {TH}>Vol</th>'
            f'<th {TH}>Res.Cal</th>'
        ) if is_meta else ""
        return (
            f'<table {TABLE}>'
            f'<tr>'
            f'<th {TH}>Ticker</th>'
            f'<th {TH}>Alpha</th>'
            f'<th {TH}>Rank</th>'
            f'<th {TH}>Conf</th>'
            f'<th {TH}>Signal</th>'
            + meta_headers
            + f'<th {TH}>Source</th>'
            f'</tr>'
            + "\n".join(rows)
            + f'</table>'
        )

    veto_section_html = ""
    if vetoes:
        veto_tickers = ", ".join(p["ticker"] for p in vetoes)
        veto_section_html = (
            f'<hr style="border:1px solid #eee; margin:16px 0;">'
            f'<h3 style="color:#c62828;">⚠ Vetoes ({n_vetoed})</h3>'
            f'<p style="font-size:12px; margin:4px 0;">'
            f'Negative predicted α + bottom-half combined rank — executor will override ENTER → HOLD:</p>'
            f'<p style="font-family:monospace; font-size:13px;"><b>{veto_tickers}</b></p>'
        )

    # ── Research section HTML ─────────────────────────────────────────────────
    research_html = ""
    if sd:
        # Market regime pill
        regime_color = {"bullish": "#2e7d32", "bearish": "#c62828"}.get(
            market_regime.lower(), "#555"
        )
        regime_pill = (
            f'<span style="display:inline-block; background:{regime_color}; color:#fff; '
            f'font-size:11px; padding:2px 8px; border-radius:3px; font-weight:bold;">'
            f'{market_regime.upper() if market_regime else "NEUTRAL"}</span>'
        )

        def _render_research_row(c: dict) -> str:
            score      = c.get("score") or c.get("long_term_score") or "—"
            conviction = c.get("conviction", "—")
            signal     = c.get("signal") or c.get("long_term_rating") or "—"
            sector     = c.get("sector", "—")
            gbm_veto   = c.get("gbm_veto", False)
            score_str  = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            badges = ""
            if gbm_veto:
                badges += ' <span style="color:#c62828; font-weight:bold; font-size:10px;">GBM⚠</span>'
            if c.get("ticker") in unscored_tickers:
                badges += ' <span style="color:#d84315; font-weight:bold; font-size:10px;" title="Actionable but no GBM prediction">NO PRED</span>'
            return (
                f'<tr>'
                f'<td {TD}><b>{c.get("ticker","?")}</b>{badges}</td>'
                f'<td {TDR}>{score_str}</td>'
                f'<td {TD}>{conviction}</td>'
                f'<td {TD}>{signal}</td>'
                f'<td {TD}>{sector}</td>'
                f'</tr>'
            )

        def _render_research_table(rows_src: list) -> str:
            rows = "".join(_render_research_row(c) for c in rows_src if isinstance(c, dict))
            if not rows:
                rows = f'<tr><td colspan="5" style="padding:4px 8px; color:#888; font-style:italic;">none</td></tr>'
            return (
                f'<table {TABLE}>'
                f'<tr><th {TH}>Ticker</th><th {TH}>Score</th><th {TH}>Conviction</th>'
                f'<th {TH}>Signal</th><th {TH}>Sector</th></tr>'
                f'{rows}'
                f'</table>'
            )

        # Buy Candidates = the actionable set (signal == ENTER). This is what the
        # executor sizes positions on; render it explicitly so every tradeable
        # ticker is visible in the morning brief even if it isn't in predictions.
        buy_table = _render_research_table(buy_candidates) if buy_candidates else ""
        cand_table = _render_research_table(population)

        # Sector ratings (top sectors sorted by rating desc, skip empty)
        sector_rows = ""
        sorted_sectors = sorted(
            [(s, v) for s, v in sector_ratings.items() if isinstance(v, dict)],
            key=lambda x: x[1].get("rating", 0),
            reverse=True,
        )
        for sector, v in sorted_sectors[:8]:
            rating   = v.get("rating", "—")
            modifier = v.get("modifier", "—")
            rating_str   = f"{rating:.0f}" if isinstance(rating, (int, float)) else str(rating)
            modifier_str = f"{modifier:.2f}x" if isinstance(modifier, (int, float)) else str(modifier)
            sector_rows += f'<tr><td {TD}>{sector}</td><td {TDR}>{rating_str}</td><td {TDR}>{modifier_str}</td></tr>'
        sector_table = ""
        if sector_rows:
            sector_table = (
                f'<table {TABLE}>'
                f'<tr><th {TH}>Sector</th><th {TH}>Rating</th><th {TH}>Modifier</th></tr>'
                f'{sector_rows}'
                f'</table>'
            )

        buy_block = ""
        if buy_table:
            unscored_note = ""
            if unscored_buys:
                _names = ", ".join(sorted(t for t in unscored_tickers if t))
                unscored_note = (
                    f'<p style="margin:4px 0 0 0; font-size:11px; color:#d84315;">'
                    f'⚠ {len(unscored_buys)} buy candidate{"s" if len(unscored_buys) != 1 else ""} '
                    f'not scored by GBM (executor can still size them): <b>{_names}</b>'
                    f'</p>'
                )
            buy_block = (
                f'<h4 style="margin:8px 0 4px 0; font-size:12px; color:#2e7d32;">'
                f'Buy Candidates ({len(buy_candidates)}) — actionable</h4>'
                f'{buy_table}'
                f'{unscored_note}'
            )

        research_html = (
            f'<div style="background:#f8f9fa; border-left:3px solid #555; padding:12px 16px; margin-bottom:16px;">'
            f'<h3 style="margin:0 0 8px 0; font-size:14px; color:#333;">Research Brief</h3>'
            f'<p style="margin:0 0 8px 0;">Market Regime: {regime_pill}</p>'
            f'{buy_block}'
            f'<h4 style="margin:8px 0 4px 0; font-size:12px; color:#555;">Population ({len(population)})</h4>'
            f'{cand_table}'
            f'{"<h4 style=margin:8px 0 4px 0; font-size:12px; color:#555;>Sector Ratings</h4>" + sector_table if sector_table else ""}'
            f'</div>'
        )

    # ── Effective optimizer params (ROADMAP L234) ──────────────────────────
    # Surface the auto-tuned executor params alongside the briefing so
    # the operator sees effective min_score_to_enter / max_position_pct /
    # atr_multiplier (and the rest) without tailing executor.log. Best-
    # effort — missing artifact silently skips the block.
    executor_params_html = ""
    executor_params_plain = ""
    if bucket:
        _ep = _load_executor_params_for_email(bucket)
        if _ep:
            _ep_keys_to_surface = [
                ("min_score", "min_score_to_enter"),
                ("max_position_pct", "max_position_pct"),
                ("atr_multiplier", "atr_multiplier"),
                ("profit_take_pct", "profit_take_pct"),
                ("time_decay_reduce_days", "time_decay_reduce_days"),
                ("time_decay_exit_days", "time_decay_exit_days"),
            ]
            _surface_rows: list[tuple[str, str]] = []
            for key, label in _ep_keys_to_surface:
                if key not in _ep:
                    continue
                val = _ep[key]
                if isinstance(val, float):
                    display = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
                else:
                    display = str(val)
                _surface_rows.append((label, display))
            _updated = _ep.get("updated_at", "—")
            _best_sharpe = _ep.get("best_sharpe")
            _improvement = _ep.get("improvement_pct")
            _manual_override = bool(_ep.get("manual_override"))
            if _surface_rows:
                _meta_parts = [f"updated <b>{_updated}</b>"]
                if isinstance(_best_sharpe, (int, float)):
                    _meta_parts.append(f"best Sharpe <b>{_best_sharpe:.2f}</b>")
                if isinstance(_improvement, (int, float)):
                    _meta_parts.append(f"improvement <b>{_improvement:+.1%}</b>")
                if _manual_override:
                    _meta_parts.append('<span style="color:#b71c1c; font-weight:bold;">manual override</span>')
                _ep_rows_html = "".join(
                    f'<tr><td {TD}>{lbl}</td><td {TD} align="right">{v}</td>'
                    f'<td {TD} style="color:#888; font-size:11px;">S3 (auto-tuned)</td></tr>'
                    for lbl, v in _surface_rows
                )
                executor_params_html = (
                    f'<div style="background:#fffbe6; border-left:3px solid #b8860b; padding:12px 16px; margin-bottom:16px;">'
                    f'<h3 style="margin:0 0 8px 0; font-size:14px; color:#333;">Effective Optimizer Params</h3>'
                    f'<p style="margin:0 0 8px 0; font-size:11px; color:#555;">'
                    f'{" &middot; ".join(_meta_parts)}</p>'
                    f'<table style="border-collapse:collapse; font-size:12px;">'
                    f'<tr><th {TH}>Param</th><th {TH}>Live value</th><th {TH}>Source</th></tr>'
                    f'{_ep_rows_html}'
                    f'</table>'
                    f'<p style="font-size:10px; color:#888; margin:8px 0 0 0;">'
                    f'Keys absent here fall through to the executor\'s local risk.yaml.'
                    f'</p>'
                    f'</div>'
                )
                executor_params_plain = (
                    "\n--- EFFECTIVE OPTIMIZER PARAMS (auto-tuned) ---\n"
                    + " | ".join(p.replace("<b>", "").replace("</b>", "").replace('<span style="color:#b71c1c; font-weight:bold;">', "").replace("</span>", "") for p in _meta_parts) + "\n"
                    + "\n".join(f"  {lbl:30s} {v}" for lbl, v in _surface_rows) + "\n"
                    + "(Keys absent fall through to risk.yaml.)\n"
                )

    html_body = (
        f'<html><body style="font-family:sans-serif; font-size:13px; color:#222; max-width:700px;">'
        f'<h2 style="margin-bottom:4px;">Alpha Engine Brief — {date_str}</h2>'
        f'<p style="color:#555; font-size:12px; margin-top:0;">'
        f'Model: <b>{model_version}</b> &nbsp;|&nbsp;'
        f'IC (val): <b>{ic_str}</b> &nbsp;|&nbsp;'
        f'Mode: <b>{metrics.get("inference_mode", "mse")}</b> &nbsp;|&nbsp;'
        f'Universe: <b>{n_total}</b> tickers &nbsp;|&nbsp;'
        f'Run at <b>{run_time}</b></p>'
        f'{research_html}'
        f'{executor_params_html}'
        f'<h3 style="font-size:13px; color:#333; margin-bottom:4px;">{"Predictions" if is_meta else "GBM Predictions"}</h3>'
        f'{_html_prediction_table(sorted_preds)}'
        f'{veto_section_html}'
        f'<p style="font-size:11px; color:#aaa; margin-top:24px;">'
        f'⚠ VETO = negative α + bottom-half rank'
        + (' &nbsp;|&nbsp; Mom = momentum &nbsp;|&nbsp; Vol = expected move &nbsp;|&nbsp; Res.Cal = research calibrator P(correct)' if is_meta else '')
        + f'</p>'
        f'<p style="font-size:11px; color:#aaa; margin-top:8px;">'
        f'ⓘ α is <b>log-domain decimal</b> at the 21d horizon (post 2026-05-09 '
        f'canonical-alpha cutover; was 5d arithmetic %). Displayed as % for '
        f'readability — small magnitudes are equivalent, but tail values '
        f'diverge (e.g. -10% log ≈ -9.5% arithmetic). Direction + ranking '
        f'unchanged.'
        f'</p>'
        f'</body></html>'
    )

    # ── Plain text ────────────────────────────────────────────────────────────
    def _plain_prediction_list(preds: list[dict]) -> str:
        if not preds:
            return "  (none)\n"
        _n = len(preds)
        lines = []
        for p in preds:
            cr = p.get("combined_rank")
            cr_str = f"{cr:.1f}" if cr is not None else "—"
            is_vetoed = (
                (p.get("predicted_alpha", 0) or 0) < 0
                and cr is not None
                and cr > _n / 2
            )
            veto = " [VETO]" if is_vetoed else ""
            lines.append(
                f"  {p['ticker']:<6}  α={_alpha_str(p):>7}  Rank {cr_str:<5}"
                f"  {_conf_pct(p):>4}  {p.get('predicted_direction','—'):<4}"
                f"  {p.get('watchlist_source', '—')}{_source_tag(p)}{veto}"
            )
        return "\n".join(lines) + "\n"

    def _plain_research_row(c: dict) -> str:
        score = c.get("score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "—"
        flags = ""
        if c.get("gbm_veto"):
            flags += " [GBM VETO]"
        if c.get("ticker") in unscored_tickers:
            flags += " [NO PRED]"
        return (
            f"  {c.get('ticker','?'):<6}  score={score_str:>5}  "
            f"{c.get('conviction','—'):<10}  {c.get('signal','—'):<8}  "
            f"{c.get('sector','—')}{flags}\n"
        )

    # Research plain section
    research_plain = ""
    if sd:
        research_plain = (
            f"\n{'='*60}\n"
            f"RESEARCH BRIEF\n"
            f"{'='*60}\n"
            f"Market Regime: {market_regime.upper() if market_regime else 'NEUTRAL'}\n"
        )
        if buy_candidates:
            research_plain += f"\nBuy Candidates ({len(buy_candidates)}) — actionable:\n"
            for c in buy_candidates:
                if isinstance(c, dict):
                    research_plain += _plain_research_row(c)
            if unscored_buys:
                _names = ", ".join(sorted(t for t in unscored_tickers if t))
                research_plain += (
                    f"  ⚠ {len(unscored_buys)} not scored by GBM: {_names}\n"
                )
        if population:
            research_plain += f"\nPopulation ({len(population)}):\n"
            for c in population:
                if isinstance(c, dict):
                    research_plain += _plain_research_row(c)
        if sector_ratings:
            research_plain += "\nSector Ratings:\n"
            for sector, v in sorted_sectors[:8]:
                rating = v.get("rating", "—")
                modifier = v.get("modifier", "—")
                rating_str   = f"{rating:.0f}" if isinstance(rating, (int, float)) else str(rating)
                modifier_str = f"{modifier:.2f}x" if isinstance(modifier, (int, float)) else str(modifier)
                research_plain += f"  {sector:<20}  rating={rating_str:>3}  modifier={modifier_str}\n"

    plain_body = (
        f"Alpha Engine Brief — {date_str}\n"
        f"Model: {model_version}  IC(val): {ic_str}  Mode: {metrics.get('inference_mode', 'mse')}  Universe: {n_total}  Run: {run_time}\n"
        f"{research_plain}"
        f"{executor_params_plain}"
        f"\n{'='*60}\n"
        f"{'PREDICTIONS' if is_meta else 'GBM PREDICTIONS'}\n"
        f"{'='*60}\n"
        f"\nPredictions (sorted by combined rank, {len(ups)} UP / {len(downs)} DOWN)\n"
        f"{_plain_prediction_list(sorted_preds)}"
    )
    if vetoes:
        veto_tickers = ", ".join(p["ticker"] for p in vetoes)
        plain_body += (
            f"\nOPTION A VETOES ({n_vetoed}): {veto_tickers}\n"
            f"(DOWN + conf >= {int(_vt * 100)}% → executor HOLD override)\n"
        )

    plain_body += (
        "\nNote: α is log-domain decimal at the 21d horizon "
        "(post 2026-05-09 canonical-alpha cutover; was 5d arithmetic %). "
        "Displayed as % for readability — small magnitudes are equivalent, "
        "tail values diverge (e.g. -10% log ≈ -9.5% arithmetic).\n"
    )

    return subject, html_body, plain_body


def send_predictor_email(
    predictions: list[dict],
    metrics: dict,
    date_str: str,
    signals_data: dict | None = None,
    veto_threshold: float | None = None,
    bucket: str | None = None,
) -> bool:
    """
    Send combined morning briefing email via Gmail SMTP (primary) or SES (fallback).

    When signals_data is provided (research pipeline's signals.json payload),
    the email includes a research section (market regime, buy candidates, sector
    ratings) followed by the GBM predictions — one complete morning briefing.

    When ``bucket`` is provided, the email also includes an "Effective
    optimizer params" block sourced from ``config/executor_params.json``
    (ROADMAP L234 morning-email side). Best-effort.

    Reads from environment / config:
        EMAIL_SENDER        — from-address
        EMAIL_RECIPIENTS    — list of recipient addresses
        GMAIL_APP_PASSWORD  — enables Gmail SMTP path (recommended)
        AWS_REGION          — SES region fallback

    Returns True on success, False on any failure. Never raises.
    """
    sender     = cfg.EMAIL_SENDER
    recipients = cfg.EMAIL_RECIPIENTS

    if not sender or not recipients:
        log.info(
            "Predictor email skipped — set EMAIL_SENDER and EMAIL_RECIPIENTS "
            "env vars in the Lambda to enable"
        )
        return False

    try:
        subject, html_body, plain_body = _build_predictor_email(
            predictions, metrics, date_str, signals_data=signals_data,
            veto_threshold=veto_threshold,
            bucket=bucket,
        )
    except Exception as exc:
        log.warning("Failed to build predictor email body: %s", exc)
        return False

    # Morning briefing HTML archival removed — no consumers read it (email delivers the content).

    # SMTP/SES dispatch via the alpha_engine_lib.email_sender chokepoint
    # (L4356 — Gmail SMTP primary, SES fallback). Identical semantics to
    # the pre-consolidation inline path; never raises, returns False on
    # any failure mode.
    from alpha_engine_lib.email_sender import send_email as _send_email
    return _send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=cfg.AWS_REGION,
    )


# ── Stage entry point ────────────────────────────────────────────────────────

def run(ctx: PipelineContext) -> None:
    """Write predictions, metrics, email, and health status."""

    # ── Build metrics ────────────────────────────────────────────────────────
    # In meta mode, ``_load_gbm_meta`` reads predictor/weights/meta/manifest.json
    # and returns {trained_date, promoted, meta_val_ic, momentum_test_ic, ...}.
    # In legacy gbm mode, it reads predictor/weights/gbm_latest.txt.meta.json
    # and returns the v2 single-model shape ({trained_date, n_train, test_ic}).
    gbm_meta = _load_gbm_meta(ctx)

    if ctx.inference_mode == "meta":
        last_trained = gbm_meta.get("trained_date", "unknown")
    elif ctx.model_type == "gbm":
        last_trained = gbm_meta.get("trained_date", getattr(ctx.scorer, "_best_iteration", "unknown"))
    else:
        last_trained = ctx.checkpoint.get("epoch", "unknown")

    metrics = {
        "model_version": ctx.model_version,
        "model_type": ctx.model_type,
        "inference_mode": ctx.inference_mode,
        "last_trained": last_trained,
        "training_samples": gbm_meta.get("n_train") if ctx.model_type == "gbm" and ctx.inference_mode != "meta" else None,
        "val_loss": round(float(ctx.val_loss), 6) if isinstance(ctx.val_loss, (int, float)) else None,
        "ic_30d": gbm_meta.get("test_ic") if ctx.model_type == "gbm" and ctx.inference_mode != "meta" else None,
        "ic_ir_30d": gbm_meta.get("ic_ir") if ctx.model_type == "gbm" and ctx.inference_mode != "meta" else None,
        "hit_rate_30d_rolling": None,
        "price_freshness": {
            "max_age_days": max(ctx.ticker_data_age.values()) if ctx.ticker_data_age else -1,
            "n_stale": sum(1 for d in ctx.ticker_data_age.values() if d > 1),
        },
    }
    # Meta-only fields — surface manifest values directly so dashboard / ops
    # can see promotion state and per-component ICs without parsing manifest.
    # The l1_ic / l2_ic / confidence_calibration keys satisfy the
    # `predictor_decisions` row of the transparency inventory
    # (alpha_engine_lib/transparency_inventory.yaml).
    if ctx.inference_mode == "meta":
        metrics["promoted"] = gbm_meta.get("promoted")
        metrics["meta_val_ic"] = gbm_meta.get("meta_val_ic")
        metrics["momentum_test_ic"] = gbm_meta.get("momentum_test_ic")
        metrics["volatility_test_ic"] = gbm_meta.get("volatility_test_ic")
        metrics["research_calibrator_n_samples"] = gbm_meta.get("research_calibrator_n_samples")
        metrics["l1_ic"] = {
            "momentum": gbm_meta.get("momentum_test_ic"),
            "volatility": gbm_meta.get("volatility_test_ic"),
            "research_calibrator": None,
        }
        metrics["l2_ic"] = gbm_meta.get("meta_val_ic")
        metrics["confidence_calibration"] = {
            "method": "isotonic",
            "ece_before": gbm_meta.get("isotonic_ece_before"),
            "ece_after": gbm_meta.get("isotonic_ece_after"),
            "n_samples": gbm_meta.get("isotonic_n_samples"),
        }

    # ── Supplemental merge (re-invocation coverage-gap path) ────────────────
    # When invoked with explicit_tickers, merge these new predictions into the
    # existing predictions/{date}.json so the first-run's predictions are not
    # overwritten. combined_rank is recomputed across the merged union; veto
    # compute (below) then runs on the merged set.
    if ctx.explicit_tickers and not ctx.dry_run:
        existing = _read_existing_predictions(ctx.bucket, ctx.date_str)
        ctx.predictions = _merge_predictions(ctx.predictions, existing)

    # ── Coverage hard-fail (buy_candidates ⊆ predictions) ────────────────────
    # Defense in depth: the Step Function's coverage-gap Choice state is the
    # self-healing mechanism that ensures every buy_candidate gets scored. This
    # assertion catches any future regression of that orchestration: if the
    # write is about to complete with buy_candidates that have no prediction
    # row, raise so the Step Function Catch fires and downstream (executor)
    # never sees a partially-scored predictions.json. This is the hard-fail
    # side of the alpha-engine "no_silent_fails" posture.
    sd = ctx.signals_data or {}
    _buy = sd.get("buy_candidates") or []
    _buy_tickers = {
        (e.get("ticker") or "").upper()
        for e in _buy if isinstance(e, dict) and e.get("ticker")
    }
    _scored = {(p.get("ticker") or "").upper() for p in ctx.predictions}
    _missing = sorted(_buy_tickers - _scored)
    if _missing:
        from inference.pipeline import PipelineHardFail
        msg = (
            f"Coverage gap: {len(_missing)} buy_candidate(s) not scored by predictor — "
            f"refusing to write predictions.json. Missing: {', '.join(_missing)}. "
            "Step Function coverage-gap Choice state should have re-invoked "
            "with --tickers; escalate to that wiring if you're seeing this."
        )
        raise PipelineHardFail(msg)

    # ── Inference coverage denominator (config#1075) ─────────────────────────
    # Persist the tradable-universe count + how many of those names got a
    # prediction, so the report card's inference_coverage grades covered/universe
    # ∈ [0,1] (robust to the supplemental coverage-gap re-invocation, which keeps
    # the same signals.json universe and merges predictions). Universe entries
    # may be ticker strings or {ticker: …} dicts — normalize both.
    _universe = sd.get("universe") or []
    _universe_tickers = {
        ((e.get("ticker") if isinstance(e, dict) else e) or "").upper()
        for e in _universe
    }
    _universe_tickers.discard("")
    n_universe = len(_universe_tickers)
    n_universe_covered = len(_universe_tickers & _scored)

    # ── Veto logic ───────────────────────────────────────────────────────────
    market_regime = ctx.signals_data.get("market_regime", "") if ctx.signals_data else ""
    # Stage D' Wire 4: pass intensity_z (stamped on ctx by run_inference
    # when the regime feature panel was built). None when the panel
    # couldn't be built — falls back to the discrete adjustment via
    # ``market_regime``.
    veto_thresh = get_veto_threshold(
        ctx.bucket,
        market_regime=market_regime,
        regime_intensity_z=getattr(ctx, "regime_intensity_z", None),
        forced_bear=getattr(ctx, "regime_forced_bear", False),
        drawdown_effective_regime=getattr(
            ctx, "drawdown_effective_regime", None
        ),
        drawdown_protective_severity=getattr(
            ctx, "drawdown_protective_severity", None
        ),
    )

    # Veto fires only when the model is BOTH bearish AND confident.
    # The confidence floor restores the 3-class veto semantic on binary
    # output: with ``p_flat = 0`` (PR #N+inference/run_inference.py:676),
    # every prediction is forced into UP or DOWN even when the underlying
    # alpha estimate is near zero. Without the confidence gate, low-
    # confidence DOWN predictions (which a 3-class model would have left
    # in the FLAT mass) fire ``gbm_veto`` and over-block research's ENTER
    # candidates. 2026-05-11 incident: 15 of 30 vetos fired with mean
    # DOWN confidence 0.749; 8 of 28 research ENTERs blocked via predictor.
    #
    # The threshold is regime-adapted via ``get_veto_threshold`` (bull +0.05,
    # bear -0.10, …), sourced from ``config/predictor_params.json``'s
    # ``veto_confidence`` field with ``cfg.MIN_CONFIDENCE`` (0.65) fallback.
    # Prior to this change the ``veto_thresh`` value was computed but only
    # consumed for the ``n_high_confidence`` metric — dead code w.r.t. the
    # actual veto decision.
    n_preds = len(ctx.predictions)
    for p in ctx.predictions:
        cr = p.get("combined_rank")
        alpha = p.get("predicted_alpha", 0) or 0
        confidence = p.get("prediction_confidence", 0) or 0
        p["gbm_veto"] = (
            alpha < 0
            and cr is not None
            and cr > n_preds / 2
            and confidence >= veto_thresh
        )

    # ── Write predictions ────────────────────────────────────────────────────
    write_predictions(ctx.predictions, ctx.date_str, ctx.bucket, metrics,
                      dry_run=ctx.dry_run, veto_threshold=veto_thresh, fd=ctx.fd,
                      level_neutralization=getattr(ctx, "level_neutralization", None),
                      n_universe=n_universe, n_universe_covered=n_universe_covered)

    # ── Send email ───────────────────────────────────────────────────────────
    # Goal: exactly one morning briefing per day, reflecting the final merged
    # predictions.json (including any coverage-gap supplementals).
    #
    # The weekday Step Function invokes PredictorInference up to twice:
    #   1. First invocation (explicit_tickers=[]) scores the research population.
    #   2. `check_coverage` Choice state re-invokes with `tickers=[…]` when
    #      signals.json has buy_candidates not scored by run 1.
    #
    # Rule:
    #   - Supplemental invocation (explicit_tickers non-empty): always send.
    #     This is the terminal write; predictions.json is now complete.
    #   - First invocation: send only if no coverage-gap re-invoke will follow.
    #     Detect this by running the same delta `check_coverage` will run —
    #     if `has_gap`, defer and let the supplemental invocation send.
    if not ctx.dry_run:
        should_send = True
        if not ctx.explicit_tickers:
            try:
                from inference.coverage_check import compute_coverage_delta
                delta = compute_coverage_delta(ctx.bucket, ctx.date_str)
                if delta.get("has_gap"):
                    should_send = False
                    log.info(
                        "Deferring morning email — coverage-gap re-invoke "
                        "expected for %d ticker(s): %s",
                        delta.get("missing_count", 0),
                        ", ".join(delta.get("missing_tickers", [])[:10])
                        + ("…" if delta.get("missing_count", 0) > 10 else ""),
                    )
            except Exception as exc:
                # Fail-open on the gating check: if we can't determine whether
                # a re-invoke is coming, prefer sending over losing the email.
                log.warning(
                    "Coverage-delta check for email gating failed (%s) — "
                    "sending email from this invocation", exc,
                )

        if should_send:
            email_sent = send_predictor_email(
                ctx.predictions, metrics, ctx.date_str,
                signals_data=ctx.signals_data, veto_threshold=veto_thresh,
                bucket=ctx.bucket,
            )
            if not email_sent:
                log.warning("Predictor email failed to send (Gmail + SES both failed)")

    # ── Health status ────────────────────────────────────────────────────────
    try:
        from health_status import write_health
        n_up = sum(1 for p in ctx.predictions if p.get("predicted_direction") == "UP")
        n_down = sum(1 for p in ctx.predictions if p.get("predicted_direction") == "DOWN")
        write_health(
            bucket=ctx.bucket,
            module_name="predictor_inference",
            status="ok",
            run_date=ctx.date_str,
            duration_seconds=ctx.elapsed_seconds(),
            summary={
                "n_predictions": len(ctx.predictions),
                "n_up": n_up,
                "n_down": n_down,
            },
        )
    except Exception as _he:
        log.warning("Health status write failed: %s", _he)

    # ── Data manifest ────────────────────────────────────────────────────────
    try:
        from health_status import write_data_manifest
        write_data_manifest(
            bucket=ctx.bucket,
            module_name="predictor_inference",
            run_date=ctx.date_str,
            manifest={
                "n_predictions": len(ctx.predictions),
                "n_up": sum(1 for p in ctx.predictions if p.get("predicted_direction") == "UP"),
                "n_down": sum(1 for p in ctx.predictions if p.get("predicted_direction") == "DOWN"),
                "n_tickers_failed": len(getattr(cfg, 'FAILED_TICKERS', [])),
                "model_version": getattr(cfg, 'GBM_VERSION', 'unknown'),
            },
        )
    except Exception as _me:
        log.warning("Data manifest write failed: %s", _me)

    log.info("Predictor run complete for %s", ctx.date_str)


def _load_gbm_meta(ctx: PipelineContext) -> dict:
    """Load GBM training metadata from S3 (best-effort).

    For ``inference_mode == "meta"`` (the v3.0-meta architecture), reads
    ``cfg.META_MANIFEST_KEY`` (``predictor/weights/meta/manifest.json``)
    and returns it normalized into the legacy {trained_date, n_train, ...}
    shape so the metrics builder above doesn't need to branch.

    For non-meta GBM, retains the original behavior of reading
    ``cfg.GBM_WEIGHTS_META_KEY``. The legacy GBM key tracks the v2 single
    model that was retired with the meta-v3 cutover; if the meta manifest
    is the active source, the legacy key is stale by definition (caught
    2026-05-04 when ``metrics/latest.json`` reported ``trained_date:
    2026-03-28`` against an actual 4/28 active model).
    """
    if ctx.local:
        return {}
    try:
        import boto3 as _boto3
        _s3 = _boto3.client("s3")

        if ctx.inference_mode == "meta":
            _resp = _s3.get_object(Bucket=ctx.bucket, Key=cfg.META_MANIFEST_KEY)
            manifest = json.loads(_resp["Body"].read())
            # L4540 part 2: the manifest's top-level `date` is the LATEST TRAINING
            # RUN (the manifest refreshes every Saturday for the freshness SLA,
            # promoted or not). The SERVED model's "last trained" date is
            # `served_date` — equal to `date` on a promoting run, carried forward
            # to the prior champion on a non-promoting run — so a non-promoting
            # run no longer reports its own date over stale served weights. Fall
            # back to `date` for older manifests written before this field.
            _served_date = manifest.get("served_date") or manifest.get("date")
            log.info(
                "Meta manifest loaded: run_date=%s  served_date=%s  promoted=%s  meta_ic=%s",
                manifest.get("date"), _served_date, manifest.get("promoted"),
                manifest.get("models", {}).get("meta_model", {}).get("ic"))
            mm = manifest.get("models", {})
            iso = mm.get("isotonic_calibrator", {})
            return {
                "trained_date": _served_date,
                "run_date": manifest.get("date"),
                "served_version": manifest.get("served_version") or manifest.get("version"),
                "promoted": manifest.get("promoted"),
                "manifest_version": manifest.get("version"),
                "meta_val_ic": mm.get("meta_model", {}).get("ic"),
                "momentum_test_ic": mm.get("momentum", {}).get("test_ic"),
                "volatility_test_ic": mm.get("volatility", {}).get("test_ic"),
                "research_calibrator_n_samples": mm.get("research_calibrator", {}).get("n_samples"),
                "isotonic_ece_before": iso.get("ece_before"),
                "isotonic_ece_after": iso.get("ece_after"),
                "isotonic_n_samples": iso.get("n_samples"),
            }

        if ctx.model_type != "gbm":
            return {}
        _resp = _s3.get_object(Bucket=ctx.bucket, Key=cfg.GBM_WEIGHTS_META_KEY)
        meta = json.loads(_resp["Body"].read())
        log.info("GBM weights meta loaded: trained_date=%s  n_train=%s",
                 meta.get("trained_date"), meta.get("n_train"))
        return meta
    except Exception as _exc:
        log.debug("Training meta not found or unreadable: %s", _exc)
        return {}
