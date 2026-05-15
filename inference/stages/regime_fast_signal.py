"""Stage: regime_fast_signal — daily BOCPD fast regime-break circuit-breaker.

Stage F1 of regime-fast-signal-260515.md. **Observe-only**: this stage
writes the persisted online state + the daily fast-signal artifact, but
NO consumer (executor / predictor veto) reads it until Stage F2 flips
``regime_forced_bear_enabled``. It rides on the daily ``intensity_z`` the
predictor already computes in ``run_inference`` — no new Lambda, no new
Step Function state, one S3 GET + two PUTs.

Non-critical: a fast-signal failure must never take down daily
predictions. The whole stage is wrapped so it logs + emits a metric and
returns rather than raising.
"""
from __future__ import annotations

import json
import logging

from inference.pipeline import PipelineContext
from inference.s3_io import _s3_put_json

log = logging.getLogger(__name__)

# Mutable online-state key (NOT eval-artifact-shaped — it is rolling
# state, not a forensic artifact). The dated/forensic record is the
# fast_signal eval artifact below.
STATE_KEY = "regime/bocpd_state.json"
FAST_SIGNAL_PREFIX = "regime/fast_signal"


def _emit_metrics(art: dict) -> None:
    """Best-effort CloudWatch gauges. Mirrors run_inference's
    ``_emit_nan_feature_tickers_metric`` shape/namespace. Always emits
    (including 0) so alarm baselines are continuous."""
    try:
        import boto3

        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="AlphaEngine/Predictor",
            MetricData=[
                {"MetricName": "regime_forced_bear",
                 "Value": 1.0 if art["forced_bear"] else 0.0, "Unit": "Count"},
                {"MetricName": "regime_fast_change_confidence",
                 "Value": float(art["change_confidence"]), "Unit": "None"},
                {"MetricName": "regime_fast_intensity_z",
                 "Value": float(art["intensity_z"]), "Unit": "None"},
                {"MetricName": "regime_fast_consecutive_change_days",
                 "Value": float(art["consecutive_change_days"]), "Unit": "Count"},
                {"MetricName": "regime_fast_warmup",
                 "Value": 1.0 if art["warmup"] else 0.0, "Unit": "Count"},
            ],
        )
    except Exception as exc:  # noqa: BLE001 — observability, never blocks
        log.warning(
            "CloudWatch regime fast-signal metrics failed: %s. Values still "
            "surface in the fast_signal artifact + this stage's log.", exc,
        )


def run(ctx: PipelineContext) -> None:
    """Advance the daily fast signal. Observe-only in F1."""
    if ctx.dry_run or ctx.local:
        log.info("regime_fast_signal: skipped (dry_run/local)")
        return

    iz = ctx.regime_intensity_z
    if iz is None:
        # Upstream regime build failed (run_inference logs this at ERROR
        # with zero-fill fallback). Do NOT fabricate an observation —
        # skip this trading day's update and surface it loudly so a
        # recurring upstream failure is visible, not silently swallowed
        # as a settled "no bear" (feedback_no_silent_fails).
        log.warning(
            "regime_fast_signal: ctx.regime_intensity_z is None "
            "(upstream regime build degraded) — skipping fast-signal "
            "advance for this run. Detector state is unchanged.",
        )
        return

    try:
        import boto3

        from alpha_engine_lib.dates import now_dual
        from alpha_engine_lib.eval_artifacts import (
            eval_artifact_key,
            eval_latest_key,
            new_eval_run_id,
        )

        from regime.bocpd import BOCPDDetector
        from regime.fast_signal import (
            FastSignalTunables,
            dump_state,
            load_state,
            step,
        )

        dual = now_dual()
        run_id = new_eval_run_id()
        s3 = boto3.client("s3")

        # ── Load prior online state (missing/corrupt ⇒ cold start) ───────
        prev = None
        try:
            obj = s3.get_object(Bucket=ctx.bucket, Key=STATE_KEY)
            prev = load_state(json.loads(obj["Body"].read()))
        except s3.exceptions.NoSuchKey:
            log.warning(
                "regime_fast_signal: no prior state at s3://%s/%s — COLD "
                "START. Detector warms up over subsequent daily runs; "
                "forced_bear is suppressed until the warmup window clears. "
                "Rigorous historical calibration is the F1 backfill's job.",
                ctx.bucket, STATE_KEY,
            )
        except Exception as exc:  # noqa: BLE001 — corrupt/incompatible state
            log.warning(
                "regime_fast_signal: prior state at s3://%s/%s unreadable "
                "(%s) — treating as COLD START + rebuilding fresh. "
                "Investigate if this recurs.",
                ctx.bucket, STATE_KEY, exc,
            )

        detector = BOCPDDetector.for_daily()
        new_state, art = step(
            prev,
            intensity_z=float(iz),
            trading_day=dual.trading_day,
            calendar_date=dual.calendar_date,
            run_id=run_id,
            detector=detector,
            tunables=FastSignalTunables(),
        )

        # Stamp the discrete latch on ctx so write_output's F2 veto
        # clamp reads it in-process (no extra S3 read, no T-1 staleness).
        # Set regardless of warmup — the artifact already encodes
        # forced_bear=False during warmup, so this is consistent.
        ctx.regime_forced_bear = bool(art["forced_bear"])

        # ── Persist online state + forensic artifact + latest sidecar ────
        _s3_put_json(s3, ctx.bucket, STATE_KEY, json.dumps(dump_state(new_state)))
        _s3_put_json(
            s3, ctx.bucket,
            eval_artifact_key(FAST_SIGNAL_PREFIX, run_id),
            json.dumps(art, default=str),
        )
        _s3_put_json(
            s3, ctx.bucket,
            eval_latest_key(FAST_SIGNAL_PREFIX),
            json.dumps(art, default=str),
        )
        _emit_metrics(art)

        log.info(
            "regime_fast_signal: trading_day=%s forced_bear=%s "
            "(since=%s) change_conf=%.3f intensity_z=%.3f "
            "consec_change=%d consec_clear=%d warmup=%s observed=%s "
            "[OBSERVE-ONLY — no consumer in F1]",
            art["trading_day"], art["forced_bear"], art["forced_bear_since"],
            art["change_confidence"], art["intensity_z"],
            art["consecutive_change_days"], art["consecutive_clear_days"],
            art["warmup"], art["observed"],
        )
    except Exception as exc:  # noqa: BLE001 — non-critical, never block predictions
        log.warning(
            "regime_fast_signal stage failed: %s — predictions unaffected "
            "(observe-only). Detector state may be stale this run.",
            exc, exc_info=True,
        )
