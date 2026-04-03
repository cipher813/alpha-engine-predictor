"""
Automated drift detection for feature distributions and prediction patterns.

Compares today's inference data against training baselines to detect model
degradation before it affects portfolio performance.

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


def check_feature_drift(s3, bucket: str, date_str: str) -> list[str]:
    """Compare today's feature distributions against training baseline."""
    alerts = []

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
        alerts.append(f"Feature snapshot missing for {date_str}")
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
            drifted.append((feat, round(zscore, 2)))

    if drifted:
        top5 = sorted(drifted, key=lambda x: -x[1])[:5]
        drift_str = ", ".join(f"{f}(z={z})" for f, z in top5)
        alerts.append(f"Feature drift: {len(drifted)} features shifted >3 stdevs — top: {drift_str}")

    return alerts


def check_prediction_drift(s3, bucket: str, date_str: str) -> list[str]:
    """Check prediction distribution patterns for degenerate model behavior."""
    alerts = []

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
        alerts.append("No recent predictions found")
        return alerts

    # Check today's predictions
    today = recent_preds[0]["predictions"]
    if not today:
        alerts.append("Today's predictions are empty")
        return alerts

    # Direction clustering (single day)
    directions = [p.get("predicted_direction", "FLAT") for p in today]
    if directions:
        from collections import Counter
        counts = Counter(directions)
        dominant_pct = counts.most_common(1)[0][1] / len(directions)
        if dominant_pct > DIRECTION_CLUSTER_THRESHOLD:
            dominant = counts.most_common(1)[0][0]
            alerts.append(
                f"Direction clustering: {dominant_pct:.0%} predict {dominant} today"
            )

    # Check if clustering persists across multiple days
    if len(recent_preds) >= CONSECUTIVE_DAYS_THRESHOLD:
        consecutive_clustered = 0
        for day_data in recent_preds[:CONSECUTIVE_DAYS_THRESHOLD]:
            day_dirs = [p.get("predicted_direction", "FLAT") for p in day_data["predictions"]]
            if day_dirs:
                day_counts = {}
                for d in day_dirs:
                    day_counts[d] = day_counts.get(d, 0) + 1
                day_dominant_pct = max(day_counts.values()) / len(day_dirs)
                if day_dominant_pct > DIRECTION_CLUSTER_THRESHOLD:
                    consecutive_clustered += 1
        if consecutive_clustered >= CONSECUTIVE_DAYS_THRESHOLD:
            alerts.append(
                f"PERSISTENT direction clustering: {consecutive_clustered} consecutive days"
            )

    # Confidence collapse
    confidences = [p.get("prediction_confidence", 0.5) for p in today
                   if p.get("prediction_confidence") is not None]
    if confidences:
        mean_conf = np.mean(confidences)
        if mean_conf < CONFIDENCE_MIN_MEAN:
            alerts.append(f"Confidence collapse: mean={mean_conf:.3f} (threshold={CONFIDENCE_MIN_MEAN})")

    # Alpha degeneration
    alphas = [p.get("predicted_alpha", 0.0) for p in today
              if p.get("predicted_alpha") is not None]
    if alphas:
        alpha_std = np.std(alphas)
        if alpha_std < ALPHA_MIN_STDEV:
            alerts.append(f"Alpha degeneration: stdev={alpha_std:.6f} (threshold={ALPHA_MIN_STDEV})")

    return alerts


def check_drift(bucket: str = DEFAULT_BUCKET, date_str: str | None = None) -> dict:
    """Run all drift checks. Returns {status, alerts, date}."""
    import boto3
    s3 = boto3.client("s3")

    if date_str is None:
        date_str = date.today().isoformat()

    all_alerts = []
    all_alerts.extend(check_feature_drift(s3, bucket, date_str))
    all_alerts.extend(check_prediction_drift(s3, bucket, date_str))

    status = "ok" if not all_alerts else "alert"
    result = {
        "date": date_str,
        "status": status,
        "alerts": all_alerts,
        "n_alerts": len(all_alerts),
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

    if all_alerts:
        for alert in all_alerts:
            logger.warning("DRIFT ALERT: %s", alert)
    else:
        logger.info("No drift detected for %s", date_str)

    return result


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
            print(f"DRIFT DETECTED ({result['n_alerts']} alerts):")
            for alert in result["alerts"]:
                print(f"  - {alert}")
        else:
            print(f"No drift detected for {result['date']}")

    if result["alerts"] and args.alert:
        try:
            import boto3
            sns = boto3.client("sns", region_name="us-east-1")
            topic_arn = os.environ.get("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts")
            sns.publish(
                TopicArn=topic_arn,
                Subject="Alpha Engine — Drift Detection Alert",
                Message="\n".join([f"- {a}" for a in result["alerts"]]),
            )
        except Exception as e:
            logger.warning("SNS alert failed: %s", e)

    sys.exit(1 if result["alerts"] else 0)


if __name__ == "__main__":
    main()
