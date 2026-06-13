"""
Automated drift detection for feature distributions and prediction patterns.

Compares today's inference data against training baselines to detect model
degradation before it affects portfolio performance.

Every finding is emitted as a SEVERITY-CLASSIFIED, self-describing alert so an
operator can judge urgency from the alert alone — no S3 forensics required. Each
alert carries: a severity (INFO / WARN / CRITICAL), the metric value vs its
threshold and the distance between them, a TREND (chronic vs acute, derived from
the recent-days window), a plain-language likely-cause, and a recommended
action. The detector also records checks it could NOT run (e.g. feature drift
with no training baseline) under ``skipped_checks`` so a half-dark detector is
never mistaken for a clean bill of health.

Usage:
    python -m monitoring.drift_detector                    # check today
    python -m monitoring.drift_detector --date 2026-04-03  # check specific date
    python -m monitoring.drift_detector --alert            # send SNS on drift
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"

# Thresholds
FEATURE_ZSCORE_THRESHOLD = 3.0    # Flag features shifted by 3+ training stdevs
DIRECTION_CLUSTER_THRESHOLD = 0.80  # >80% same direction = degenerate
CONFIDENCE_MIN_MEAN = 0.45         # Mean confidence below this = collapsed model
ALPHA_MIN_STDEV = 0.001            # Alpha stdev below this = degenerate
CONSECUTIVE_DAYS_THRESHOLD = 3     # Direction clustering must persist N days

# ── Severity model ────────────────────────────────────────────────────────────
# A small, explicit ladder so the most urgent finding can set the SNS subject and
# the operator can triage by reading one word. INFO = visible-but-no-action
# (e.g. a check that was skipped); WARN = degraded / standing model-quality
# condition (advisory, no trading halt); CRITICAL = acute regression or a
# degenerate/empty model output that needs a look now.
INFO, WARN, CRITICAL = "INFO", "WARN", "CRITICAL"
_SEVERITY_ORDER = {INFO: 0, WARN: 1, CRITICAL: 2}


def _max_severity(severities: list[str]) -> str | None:
    """Return the highest severity in the list, or None if empty."""
    sevs = [s for s in severities if s in _SEVERITY_ORDER]
    return max(sevs, key=lambda s: _SEVERITY_ORDER[s]) if sevs else None


def _alert(
    *,
    code: str,
    severity: str,
    headline: str,
    detail: str,
    cause: str | None = None,
    action: str | None = None,
    **context,
) -> dict:
    """Build one structured, self-describing alert.

    ``line`` is a single human-readable string that folds severity + detail +
    (optionally) cause + action together — it is what flows into the
    backward-compatible ``alerts`` list[str] and the SNS body, so the alert
    explains its own severity without a second lookup. ``context`` carries the
    raw numbers (value/threshold/trend/…) for programmatic consumers.
    """
    parts = [f"[{severity}] {headline}: {detail}"]
    if cause:
        parts.append(f"Likely cause: {cause}")
    if action:
        parts.append(f"Action: {action}")
    line = " ".join(parts)
    return {
        "code": code,
        "severity": severity,
        "headline": headline,
        "detail": detail,
        "cause": cause,
        "action": action,
        "line": line,
        **context,
    }


def _load_json(s3, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def _load_parquet(s3, bucket: str, key: str):
    try:
        import pandas as pd
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None


def check_feature_drift(s3, bucket: str, date_str: str) -> list[dict]:
    """Compare today's feature distributions against the training baseline.

    Returns a list of structured alert dicts (see ``_alert``). The
    no-baseline-stats case is NOT a silent skip: the caller records it under
    ``skipped_checks`` (surfaced to the operator) via the sentinel returned in
    ``check_drift``; here it simply yields no alerts.
    """
    alerts: list[dict] = []

    # Load training feature stats
    stats = _load_json(s3, bucket, "predictor/metrics/training_feature_stats.json")
    if not stats:
        logger.warning("No training feature stats found — skipping feature drift check")
        return alerts

    train_means = np.array(stats["mean"])
    train_stds = np.array(stats["std"])
    feature_names = stats["features"]

    # Load today's feature snapshot
    tech_df = _load_parquet(s3, bucket, f"features/{date_str}/technical.parquet")
    if tech_df is None or tech_df.empty:
        alerts.append(_alert(
            code="feature_snapshot_missing",
            severity=WARN,
            headline="Feature snapshot missing",
            detail=f"no technical feature snapshot for {date_str} — feature drift could not be evaluated",
            cause="the daily feature snapshot was not written (upstream data/enrich gap)",
            action="confirm the day's feature pipeline ran; this check is blind until it does",
            date=date_str,
        ))
        return alerts

    # Compare each feature's cross-sectional mean against training distribution
    drifted = []
    for i, feat in enumerate(feature_names):
        if feat not in tech_df.columns:
            continue
        if train_stds[i] < 1e-10:
            continue  # Skip constant features
        today_mean = tech_df[feat].mean()
        if np.isnan(today_mean):
            continue
        zscore = abs(today_mean - train_means[i]) / train_stds[i]
        if zscore > FEATURE_ZSCORE_THRESHOLD:
            drifted.append((feat, round(float(zscore), 2)))

    if drifted:
        top5 = sorted(drifted, key=lambda x: -x[1])[:5]
        max_z = top5[0][1]
        drift_str = ", ".join(f"{f}(z={z})" for f, z in top5)
        # >5 stdevs on the worst feature is a hard distribution break, not noise.
        severity = CRITICAL if max_z >= 5.0 else WARN
        alerts.append(_alert(
            code="feature_drift",
            severity=severity,
            headline="Feature drift",
            detail=(f"{len(drifted)} feature(s) shifted >{FEATURE_ZSCORE_THRESHOLD:g} "
                    f"training stdevs (worst z={max_z}) — top: {drift_str}"),
            cause="today's feature distribution diverged from the training distribution "
                  "(regime shift, data-vendor change, or a feature-computation bug)",
            action="inspect the top-z features vs their training stats; a real distribution "
                   "break means the model is extrapolating and predictions are less trustworthy",
            n_drifted=len(drifted),
            max_zscore=max_z,
            top_features=top5,
            date=date_str,
        ))

    return alerts


def _daily_confidence_means(recent_preds: list[dict]) -> list[tuple[str, float]]:
    """Per-day mean ``prediction_confidence`` over the recent-days window
    (most-recent-first), skipping days with no confidence values."""
    out: list[tuple[str, float]] = []
    for day in recent_preds:
        cs = [p.get("prediction_confidence") for p in day["predictions"]
              if p.get("prediction_confidence") is not None]
        if cs:
            out.append((day["date"], float(np.mean(cs))))
    return out


def check_prediction_drift(s3, bucket: str, date_str: str) -> list[dict]:
    """Check prediction distribution patterns for degenerate model behavior.

    Returns a list of structured alert dicts (see ``_alert``)."""
    alerts: list[dict] = []

    # Load recent predictions (up to 5 trading days)
    target = date.fromisoformat(date_str)
    recent_preds = []
    for offset in range(7):  # Scan 7 calendar days to find 5 trading days
        d = (target - timedelta(days=offset)).isoformat()
        data = _load_json(s3, bucket, f"predictor/predictions/{d}.json")
        if data and "predictions" in data:
            preds = data["predictions"]
            recent_preds.append({"date": d, "predictions": preds})
        if len(recent_preds) >= 5:
            break

    if not recent_preds:
        alerts.append(_alert(
            code="no_recent_predictions",
            severity=CRITICAL,
            headline="No recent predictions found",
            detail=f"no predictions JSON in the 7 days ending {date_str} — the predictor produced nothing",
            cause="inference Lambda did not write predictions (deploy/coverage/staleness failure)",
            action="check the daily PredictorInference run; the executor is running blind without predictions",
            date=date_str,
        ))
        return alerts

    # Check today's predictions
    today = recent_preds[0]["predictions"]
    if not today:
        alerts.append(_alert(
            code="today_predictions_empty",
            severity=CRITICAL,
            headline="Today's predictions are empty",
            detail=f"the predictions file for {recent_preds[0]['date']} contains zero predictions",
            cause="inference produced an empty set (universe/coverage collapse)",
            action="check the latest PredictorInference run; downstream sizing has no signal",
            date=date_str,
        ))
        return alerts

    # Direction clustering (single day). Filter None — missing
    # predicted_direction shouldn't count as its own cluster.
    directions = [d for d in (p.get("predicted_direction") for p in today) if d]
    if directions:
        from collections import Counter
        counts = Counter(directions)
        dominant_pct = counts.most_common(1)[0][1] / len(directions)
        if dominant_pct > DIRECTION_CLUSTER_THRESHOLD:
            dominant = counts.most_common(1)[0][0]
            alerts.append(_alert(
                code="direction_clustering",
                severity=WARN,
                headline="Direction clustering",
                detail=(f"{dominant_pct:.0%} of names predict {dominant} today "
                        f"(>{DIRECTION_CLUSTER_THRESHOLD:.0%} same-direction floor)"),
                cause="the model has little cross-sectional discrimination today — it is "
                      "taking a near-uniform directional view instead of ranking names",
                action="tolerable as a one-off (a strong market day); watch for the persistent "
                       "variant below, which indicates a genuinely degenerate model",
                value=round(dominant_pct, 3),
                threshold=DIRECTION_CLUSTER_THRESHOLD,
                dominant_direction=dominant,
                date=date_str,
            ))

    # Check if clustering persists across multiple days
    if len(recent_preds) >= CONSECUTIVE_DAYS_THRESHOLD:
        consecutive_clustered = 0
        for day_data in recent_preds[:CONSECUTIVE_DAYS_THRESHOLD]:
            day_dirs = [d for d in (p.get("predicted_direction") for p in day_data["predictions"]) if d]
            if day_dirs:
                day_counts: dict[str, int] = {}
                for d in day_dirs:
                    day_counts[d] = day_counts.get(d, 0) + 1
                day_dominant_pct = max(day_counts.values()) / len(day_dirs)
                if day_dominant_pct > DIRECTION_CLUSTER_THRESHOLD:
                    consecutive_clustered += 1
        if consecutive_clustered >= CONSECUTIVE_DAYS_THRESHOLD:
            alerts.append(_alert(
                code="persistent_direction_clustering",
                severity=CRITICAL,
                headline="PERSISTENT direction clustering",
                detail=(f"{consecutive_clustered} consecutive days above the "
                        f"{DIRECTION_CLUSTER_THRESHOLD:.0%} same-direction floor"),
                cause="the model has collapsed to a one-directional view — it has lost "
                      "cross-sectional signal, so its rankings carry no usable alpha",
                action="treat the model as degenerate; promote a healthy challenger or revert "
                       "to a known-good champion",
                consecutive_days=consecutive_clustered,
                date=date_str,
            ))

    # Confidence collapse — with chronic-vs-acute trend so severity reflects
    # whether this is a standing weak-model condition or a sudden break.
    confidences = [p.get("prediction_confidence", 0.5) for p in today
                   if p.get("prediction_confidence") is not None]
    if confidences:
        mean_conf = float(np.mean(confidences))
        if mean_conf < CONFIDENCE_MIN_MEAN:
            deficit = (CONFIDENCE_MIN_MEAN - mean_conf) / CONFIDENCE_MIN_MEAN
            daily = _daily_confidence_means(recent_preds)
            days_below = sum(1 for _, m in daily if m < CONFIDENCE_MIN_MEAN)
            n_days = len(daily)
            lo = min((m for _, m in daily), default=mean_conf)
            hi = max((m for _, m in daily), default=mean_conf)

            # chronic = below the floor on most of the recent window; acute = only
            # today is below while the prior days sat above (a fresh collapse).
            if n_days >= CONSECUTIVE_DAYS_THRESHOLD and days_below >= CONSECUTIVE_DAYS_THRESHOLD:
                trend, trend_word = "chronic", "CHRONIC"
            elif n_days >= 2 and days_below == 1 and daily[0][1] < CONFIDENCE_MIN_MEAN:
                trend, trend_word = "acute", "ACUTE"
            else:
                trend, trend_word = "indeterminate", "trend-indeterminate"

            # A sudden collapse, or confidence under half the floor, is CRITICAL;
            # a standing chronic shortfall is an advisory WARN (model-quality, not
            # a same-day break).
            severity = CRITICAL if (trend == "acute" or mean_conf < 0.5 * CONFIDENCE_MIN_MEAN) else WARN

            trend_detail = (f"below floor {days_below}/{n_days} recent days "
                            f"(range {lo:.3f}–{hi:.3f})") if n_days else "no recent history"
            if trend == "chronic":
                cause = ("a low-IC champion producing low-conviction predictions — a standing "
                         "model-quality condition, NOT a same-day regression")
                action = ("resolve via champion replacement (the challenger pipeline), not an "
                          "inference hotfix; advisory — does not halt trading")
            elif trend == "acute":
                cause = ("a sudden drop from a healthy recent baseline — points at today's "
                         "served-model version or feature inputs, not a slow decay")
                action = ("check today's served model + inference feature inputs for a regression")
            else:
                cause = "low mean conviction; insufficient recent history to call chronic vs acute"
                action = "watch the next few days to classify; check served-model health"

            alerts.append(_alert(
                code="confidence_collapse",
                severity=severity,
                headline="Confidence collapse",
                detail=(f"mean prediction confidence {mean_conf:.3f} is {deficit:.0%} below the "
                        f"{CONFIDENCE_MIN_MEAN} floor — {trend_word} ({trend_detail})"),
                cause=cause,
                action=action,
                value=round(mean_conf, 3),
                threshold=CONFIDENCE_MIN_MEAN,
                pct_below_threshold=round(deficit, 3),
                trend=trend,
                recent_daily_means=[round(m, 3) for _, m in daily],
                date=date_str,
            ))

    # Alpha degeneration — predictions nearly constant ⇒ no cross-sectional signal.
    alphas = [p.get("predicted_alpha", 0.0) for p in today
              if p.get("predicted_alpha") is not None]
    if alphas:
        alpha_std = float(np.std(alphas))
        if alpha_std < ALPHA_MIN_STDEV:
            alerts.append(_alert(
                code="alpha_degeneration",
                severity=CRITICAL,
                headline="Alpha degeneration",
                detail=(f"predicted-alpha stdev {alpha_std:.6f} is below the {ALPHA_MIN_STDEV} floor "
                        f"— predictions are nearly constant across the universe"),
                cause="the model emits a near-identical alpha for every name — there is no "
                      "cross-sectional ranking signal left to size on",
                action="treat as degenerate; investigate the meta-model output and promote/revert "
                       "to a healthy model",
                value=round(alpha_std, 6),
                threshold=ALPHA_MIN_STDEV,
                date=date_str,
            ))

    return alerts


def _detect_skipped_checks(s3, bucket: str) -> list[dict]:
    """Surface checks that could NOT run, so a half-dark detector is visible.

    Currently: feature drift is skipped whenever the training-feature-stats
    baseline is absent. Recorded as INFO (visible, no action urgency) rather than
    flipping ``status`` to alert every run — but it is no longer SILENT."""
    skipped: list[dict] = []
    if _load_json(s3, bucket, "predictor/metrics/training_feature_stats.json") is None:
        skipped.append({
            "check": "feature_drift",
            "severity": INFO,
            "reason": ("no training-feature-stats baseline "
                       "(predictor/metrics/training_feature_stats.json) — the feature-drift "
                       "arm did not run; only prediction-pattern drift was evaluated"),
        })
    return skipped


def check_drift(bucket: str = DEFAULT_BUCKET, date_str: str | None = None) -> dict:
    """Run all drift checks. Returns a structured, severity-aware result.

    Backward-compatible: ``alerts`` remains a ``list[str]`` (each a rich,
    self-describing line). New additive fields: ``severity`` (overall max, or
    None when clean), ``alert_details`` (the structured dicts), and
    ``skipped_checks`` (checks that could not run)."""
    import boto3
    s3 = boto3.client("s3")

    if date_str is None:
        date_str = date.today().isoformat()

    details: list[dict] = []
    details.extend(check_feature_drift(s3, bucket, date_str))
    details.extend(check_prediction_drift(s3, bucket, date_str))
    skipped = _detect_skipped_checks(s3, bucket)

    overall_severity = _max_severity([d["severity"] for d in details])
    status = "ok" if not details else "alert"
    result = {
        "date": date_str,
        "status": status,
        "severity": overall_severity,            # NEW: overall max (None when clean)
        "alerts": [d["line"] for d in details],  # backward-compatible list[str]
        "alert_details": details,                # NEW: structured per-alert context
        "skipped_checks": skipped,               # NEW: checks that could not run
        "n_alerts": len(details),
    }

    # Write results to S3
    try:
        s3.put_object(
            Bucket=bucket,
            Key=f"predictor/metrics/drift_{date_str}.json",
            Body=json.dumps(result, indent=2).encode(),
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning("Drift results S3 write failed: %s", e)

    if details:
        for d in details:
            log_fn = logger.error if d["severity"] == CRITICAL else logger.warning
            log_fn("DRIFT ALERT %s", d["line"])
    else:
        logger.info("No drift detected for %s", date_str)
    for s in skipped:
        logger.info("DRIFT CHECK SKIPPED [%s] %s: %s", s["severity"], s["check"], s["reason"])

    return result


def format_alert_report(result: dict) -> str:
    """Render a human-readable, severity-led report for SNS / email / stdout.

    Leads with the overall severity and a per-severity count so urgency is the
    first thing read, then each alert as a labeled block, then any skipped
    checks. This is what makes the alert self-explanatory."""
    details = result.get("alert_details", [])
    lines: list[str] = []
    overall = result.get("severity") or "none"
    counts = {CRITICAL: 0, WARN: 0, INFO: 0}
    for d in details:
        counts[d.get("severity", WARN)] = counts.get(d.get("severity", WARN), 0) + 1
    lines.append(f"SEVERITY: {overall}")
    lines.append(f"Date: {result.get('date')}")
    lines.append(
        f"Predictor drift check — {len(details)} alert(s) "
        f"({counts[CRITICAL]} CRITICAL, {counts[WARN]} WARN)"
    )
    lines.append("")
    for d in details:
        lines.append(f"[{d['severity']}] {d['headline']}")
        lines.append(f"  • {d['detail']}")
        if d.get("cause"):
            lines.append(f"  • Likely cause: {d['cause']}")
        if d.get("action"):
            lines.append(f"  • Action: {d['action']}")
        lines.append("")
    for s in result.get("skipped_checks", []):
        lines.append(f"[{s['severity']}] Skipped check — {s['check']}: {s['reason']}")
    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(description="Check feature and prediction drift")
    parser.add_argument("--date", default=None, help="Date to check (YYYY-MM-DD)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--alert", action="store_true", help="Send SNS alert on drift")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = check_drift(bucket=args.bucket, date_str=args.date)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["alerts"]:
            print(format_alert_report(result))
        else:
            print(f"No drift detected for {result['date']}")
            for s in result.get("skipped_checks", []):
                print(f"  (skipped: {s['check']} — {s['reason']})")

    if result["alerts"] and args.alert:
        try:
            import boto3
            sns = boto3.client("sns", region_name="us-east-1")
            topic_arn = os.environ.get("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts")
            # Severity-led subject so the inbox conveys urgency + the headline
            # without opening the mail.
            headlines = "; ".join(dict.fromkeys(d["headline"] for d in result["alert_details"]))
            subject = f"Alpha Engine — Drift [{result.get('severity')}]: {headlines}"
            sns.publish(
                TopicArn=topic_arn,
                Subject=subject[:100],  # SNS subject hard limit
                Message=format_alert_report(result),
            )
        except Exception as e:
            logger.warning("SNS alert failed: %s", e)

    # Exit 1 on any drift, 0 when clean — unchanged contract for the SF
    # wrapper (which collapses all non-zero to one path). Severity is conveyed
    # in the alert content / SNS subject / drift_{date}.json `severity`, not the
    # exit code, so no unconsumed cross-repo coupling is introduced.
    sys.exit(1 if result["alerts"] else 0)


if __name__ == "__main__":
    main()
