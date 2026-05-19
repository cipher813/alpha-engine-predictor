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

# Drawdown regime leg (regime-drawdown-hysteresis-260518.md). Rolling
# online state + dated/forensic eval artifact + latest sidecar, mirroring
# the fast-signal keys. PR 2 = producer/observe only.
DRAWDOWN_STATE_KEY = "regime/drawdown_state.json"
DRAWDOWN_PREFIX = "regime/drawdown"


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


def _advance_drawdown(ctx: PipelineContext, s3, dual, run_id: str) -> None:
    """Advance the deterministic drawdown regime leg by one day.

    PR 2 (regime-drawdown-hysteresis-260518.md): producer + **observe
    only** — persists rolling state + a forensic artifact + latest
    sidecar, stamps ``ctx.drawdown_effective_regime`` for the future
    consumer PRs, and logs the counterfactual. NO consumer reads it yet
    (the executor / predictor-veto PRs, gated ``drawdown_regime_enabled``
    default-off, will). Fully self-contained + non-raising: a failure
    here must never affect the fast-signal path or predictions.

    HMM argmax is composed at the *weekly* substrate layer (PR 1's
    ``build_regime_substrate`` hook); the daily effective regime composes
    the legs available intraday — the drawdown leg + Stage-F
    ``forced_bear`` (most-protective-wins).
    """
    try:
        import json as _json

        from regime.drawdown import (
            compose_effective_regime,
            dump_state,
            load_state,
            read_eod_pnl_nav,
            seed_state,
            step as dd_step,
        )
        from alpha_engine_lib.eval_artifacts import (
            eval_artifact_key,
            eval_latest_key,
        )

        spy_series = ctx.macro.get("SPY") if ctx.macro else None
        if spy_series is None or len(spy_series.dropna()) == 0:
            log.warning(
                "drawdown leg: ctx.macro['SPY'] missing/empty — skipping "
                "this run (SPY leg needs the close series). "
                "Drawdown state unchanged.",
            )
            return

        nav_series = read_eod_pnl_nav(s3, bucket=ctx.bucket)  # None ⇒ degrade

        # ── Load prior state (missing/corrupt/schema ⇒ history seed) ─────
        # Single robust handler (no fragile except-on-getattr): any read
        # failure re-seeds from history; NoSuchKey is the benign cold
        # start, anything else is warned for investigation.
        prev = None
        try:
            obj = s3.get_object(Bucket=ctx.bucket, Key=DRAWDOWN_STATE_KEY)
            prev = load_state(_json.loads(obj["Body"].read()))
        except Exception as exc:  # noqa: BLE001 — missing/corrupt/schema
            prev = seed_state(spy_series, nav_history=nav_series)
            if type(exc).__name__ == "NoSuchKey":
                log.warning(
                    "drawdown leg: no prior state at s3://%s/%s — COLD "
                    "START, seeded from history (true trailing peak; "
                    "conservative initial tier).",
                    ctx.bucket, DRAWDOWN_STATE_KEY,
                )
            else:
                log.warning(
                    "drawdown leg: prior state unreadable (%s) — "
                    "re-seeding from history. Investigate if this recurs.",
                    exc,
                )

        spy_close = float(spy_series.dropna().iloc[-1])
        nav_close = (
            float(nav_series.iloc[-1]) if nav_series is not None else None
        )

        new_state, art = dd_step(
            prev,
            spy_close=spy_close,
            nav=nav_close,
            trading_day=dual.trading_day,
            calendar_date=dual.calendar_date,
            run_id=run_id,
        )

        composed = compose_effective_regime(
            spy_tier=new_state.spy_tier,
            excess_tier=(
                new_state.excess_tier
                if art["excess"]["available"] else None
            ),
            forced_bear=bool(ctx.regime_forced_bear),
        )
        art["effective_regime"] = composed
        # Observe-only stamp — no consumer reads this in PR 2.
        ctx.drawdown_effective_regime = composed["effective_regime"]

        _s3_put_json(
            s3, ctx.bucket, DRAWDOWN_STATE_KEY,
            json.dumps(dump_state(new_state)),
        )
        _s3_put_json(
            s3, ctx.bucket,
            eval_artifact_key(DRAWDOWN_PREFIX, run_id),
            json.dumps(art, default=str),
        )
        _s3_put_json(
            s3, ctx.bucket,
            eval_latest_key(DRAWDOWN_PREFIX),
            json.dumps(art, default=str),
        )

        log.info(
            "drawdown leg: trading_day=%s spy_tier=%s spy_dd=%.4f "
            "excess_tier=%s(avail=%s) forced_bear=%s → effective_regime=%s "
            "observed=%s [OBSERVE-ONLY — no consumer in PR 2]",
            art["trading_day"], art["spy"]["tier"],
            art["spy"]["drawdown"] if art["spy"]["drawdown"] is not None
            else float("nan"),
            art["excess"]["tier"], art["excess"]["available"],
            ctx.regime_forced_bear, composed["effective_regime"],
            art["observed"],
        )
    except Exception as exc:  # noqa: BLE001 — non-critical, never raise
        log.warning(
            "drawdown leg failed: %s — predictions + fast-signal "
            "unaffected (observe-only). Drawdown state may be stale this "
            "run.", exc, exc_info=True,
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

        # ── Drawdown regime leg (PR 2, observe-only) ─────────────────────
        # Rides the same daily rail (post-run_inference: ctx.macro["SPY"]
        # is populated; ctx.regime_forced_bear was just stamped above).
        # Fully self-contained + non-raising so it never affects the
        # fast-signal success path or predictions.
        _advance_drawdown(ctx, s3, dual, run_id)
    except Exception as exc:  # noqa: BLE001 — non-critical, never block predictions
        log.warning(
            "regime_fast_signal stage failed: %s — predictions unaffected "
            "(observe-only). Detector state may be stale this run.",
            exc, exc_info=True,
        )
