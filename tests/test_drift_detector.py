"""Tests for monitoring.drift_detector — feature + prediction drift checks.

check_feature_drift / check_prediction_drift now return a list of STRUCTURED
alert dicts (``{code, severity, headline, detail, cause, action, line, ...}``)
so each alert is self-describing (severity + distance-from-threshold + trend +
cause + action). ``check_drift`` keeps ``alerts`` as a backward-compatible
list[str] (the rendered ``line`` of each) and adds ``severity``,
``alert_details`` and ``skipped_checks``.
"""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detector import (
    ALPHA_MIN_STDEV,
    CONFIDENCE_MIN_MEAN,
    CONSECUTIVE_DAYS_THRESHOLD,
    CRITICAL,
    DIRECTION_CLUSTER_THRESHOLD,
    FEATURE_ZSCORE_THRESHOLD,
    INFO,
    WARN,
    _load_json,
    _load_parquet,
    _max_severity,
    check_drift,
    check_feature_drift,
    check_prediction_drift,
    format_alert_report,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _lines(alerts):
    """Rendered ``line`` of each structured alert (what flows to alerts/SNS)."""
    return [a["line"] for a in alerts]


def _codes(alerts):
    return [a["code"] for a in alerts]


# ── S3 helpers ──────────────────────────────────────────────────────────────


def _s3_with_json(payload):
    """An S3 client that returns json bytes for any key."""
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()
    s3.get_object.return_value = {"Body": body}
    return s3


def _s3_with_routes(routes):
    """routes: dict[key, "json"|"parquet"|"missing", payload]."""
    s3 = MagicMock()

    def get_object(*, Bucket, Key):
        if Key not in routes:
            raise RuntimeError(f"NoSuchKey: {Key}")
        kind, payload = routes[Key]
        body = MagicMock()
        if kind == "json":
            body.read.return_value = json.dumps(payload).encode()
        elif kind == "parquet":
            buf = io.BytesIO()
            payload.to_parquet(buf)
            buf.seek(0)
            body.read.return_value = buf.read()
        else:
            raise RuntimeError(f"Unknown kind {kind}")
        return {"Body": body}

    s3.get_object.side_effect = get_object
    return s3


def test_load_json_success():
    s3 = _s3_with_json({"hello": "world"})
    assert _load_json(s3, "bucket", "key") == {"hello": "world"}


def test_load_json_failure_returns_none():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("NoSuchKey")
    assert _load_json(s3, "bucket", "key") is None


def test_load_parquet_success():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    s3 = _s3_with_routes({"k.parquet": ("parquet", df)})
    loaded = _load_parquet(s3, "bucket", "k.parquet")
    pd.testing.assert_frame_equal(loaded, df)


def test_load_parquet_failure_returns_none():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("NoSuchKey")
    assert _load_parquet(s3, "bucket", "missing.parquet") is None


# ── severity helper ─────────────────────────────────────────────────────────


def test_max_severity_orders_correctly():
    assert _max_severity([INFO, WARN, CRITICAL]) == CRITICAL
    assert _max_severity([INFO, WARN]) == WARN
    assert _max_severity([INFO]) == INFO
    assert _max_severity([]) is None


# ── check_feature_drift ────────────────────────────────────────────────────


def test_check_feature_drift_no_stats_returns_empty():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("missing")
    assert check_feature_drift(s3, "bucket", "2026-04-01") == []


def test_check_feature_drift_missing_snapshot_alerts():
    stats = {"mean": [50.0], "std": [10.0], "features": ["rsi_14"]}
    s3 = _s3_with_routes({
        "predictor/metrics/training_feature_stats.json": ("json", stats),
        # no feature snapshot for the date → triggers "Feature snapshot missing"
    })
    alerts = check_feature_drift(s3, "bucket", "2026-04-01")
    assert "feature_snapshot_missing" in _codes(alerts)
    assert any("Feature snapshot missing" in ln for ln in _lines(alerts))
    assert alerts[0]["severity"] == WARN


def test_check_feature_drift_flags_zscore_above_threshold():
    stats = {
        "mean": [50.0, 0.0, 100.0],
        "std": [10.0, 1.0, 5.0],
        "features": ["rsi_14", "momentum", "price"],
    }
    # today's snapshot: rsi shifted by 4 stdevs, momentum and price stable
    snapshot = pd.DataFrame({
        "rsi_14": [90.0, 90.0, 90.0],     # mean=90 vs train 50, z=4 → flag
        "momentum": [0.1, 0.0, -0.1],     # mean=0 vs train 0 → stable
        "price": [101.0, 99.0, 100.0],    # mean=100 vs train 100 → stable
    })
    s3 = _s3_with_routes({
        "predictor/metrics/training_feature_stats.json": ("json", stats),
        "features/2026-04-01/technical.parquet": ("parquet", snapshot),
    })
    alerts = check_feature_drift(s3, "bucket", "2026-04-01")
    assert len(alerts) == 1
    assert alerts[0]["code"] == "feature_drift"
    assert "rsi_14" in alerts[0]["line"]
    assert "Feature drift" in alerts[0]["line"]
    # z=4 < 5 → WARN (not a hard break)
    assert alerts[0]["severity"] == WARN
    assert alerts[0]["max_zscore"] == pytest.approx(4.0, abs=0.01)


def test_check_feature_drift_extreme_zscore_is_critical():
    stats = {"mean": [50.0], "std": [10.0], "features": ["rsi_14"]}
    snapshot = pd.DataFrame({"rsi_14": [120.0, 120.0, 120.0]})  # z=7 → CRITICAL
    s3 = _s3_with_routes({
        "predictor/metrics/training_feature_stats.json": ("json", stats),
        "features/2026-04-01/technical.parquet": ("parquet", snapshot),
    })
    alerts = check_feature_drift(s3, "bucket", "2026-04-01")
    assert alerts[0]["severity"] == CRITICAL


def test_check_feature_drift_skips_constant_features():
    """Features with std < 1e-10 are skipped (would divide by zero)."""
    stats = {"mean": [50.0], "std": [1e-15], "features": ["constant"]}
    snapshot = pd.DataFrame({"constant": [999.0, 999.0]})  # huge shift but skipped
    s3 = _s3_with_routes({
        "predictor/metrics/training_feature_stats.json": ("json", stats),
        "features/2026-04-01/technical.parquet": ("parquet", snapshot),
    })
    alerts = check_feature_drift(s3, "bucket", "2026-04-01")
    assert alerts == []


def test_check_feature_drift_skips_missing_or_nan_features():
    stats = {"mean": [50.0, 0.0], "std": [10.0, 1.0], "features": ["rsi", "missing_col"]}
    snapshot = pd.DataFrame({"rsi": [55.0, 60.0]})  # missing_col absent
    s3 = _s3_with_routes({
        "predictor/metrics/training_feature_stats.json": ("json", stats),
        "features/2026-04-01/technical.parquet": ("parquet", snapshot),
    })
    alerts = check_feature_drift(s3, "bucket", "2026-04-01")
    assert alerts == []


# ── check_prediction_drift ─────────────────────────────────────────────────


def _make_preds(directions=None, confidences=None, alphas=None, n=20):
    """Build a predictions JSON payload."""
    if directions is None:
        directions = ["UP"] * n
    if confidences is None:
        confidences = [0.6] * n
    if alphas is None:
        alphas = [0.05] * n
    preds = []
    for i, (d, c, a) in enumerate(zip(directions, confidences, alphas)):
        preds.append({
            "ticker": f"T{i}",
            "predicted_direction": d,
            "prediction_confidence": c,
            "predicted_alpha": a,
        })
    return {"predictions": preds}


def test_check_prediction_drift_no_recent_alerts():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("missing")
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert _codes(alerts) == ["no_recent_predictions"]
    assert alerts[0]["severity"] == CRITICAL


def test_check_prediction_drift_empty_today_alerts():
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", {"predictions": []}),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert _codes(alerts) == ["today_predictions_empty"]
    assert alerts[0]["severity"] == CRITICAL
    assert any("empty" in ln.lower() for ln in _lines(alerts))


def test_check_prediction_drift_flags_single_day_clustering():
    """90% UP on today only → single-day cluster alert (WARN)."""
    preds = _make_preds(directions=["UP"] * 18 + ["DOWN"] * 2)
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    cluster = [a for a in alerts if a["code"] == "direction_clustering"]
    assert cluster and cluster[0]["dominant_direction"] == "UP"
    assert cluster[0]["severity"] == WARN
    # Single day → no persistent cluster
    assert "persistent_direction_clustering" not in _codes(alerts)


def test_check_prediction_drift_persistent_clustering_alert():
    """N consecutive trading days all clustered → PERSISTENT alert (CRITICAL)."""
    clustered_preds = _make_preds(directions=["UP"] * 19 + ["DOWN"] * 1)

    routes = {}
    days = ["2026-04-15", "2026-04-14", "2026-04-13", "2026-04-12", "2026-04-11"]
    for d in days:
        routes[f"predictor/predictions/{d}.json"] = ("json", clustered_preds)

    s3 = _s3_with_routes(routes)
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    persistent = [a for a in alerts if a["code"] == "persistent_direction_clustering"]
    assert persistent and persistent[0]["severity"] == CRITICAL
    assert str(CONSECUTIVE_DAYS_THRESHOLD) in persistent[0]["line"]


def test_confidence_collapse_chronic_is_warn():
    """Below the floor every recent day → CHRONIC, WARN (standing weak model)."""
    routes = {}
    for d in ["2026-04-15", "2026-04-14", "2026-04-13", "2026-04-12", "2026-04-11"]:
        routes[f"predictor/predictions/{d}.json"] = ("json", _make_preds(
            directions=["UP", "DOWN", "FLAT"] * 7,  # diversified to avoid clustering
            confidences=[0.27] * 21,                # chronically below floor
            n=21,
        ))
    s3 = _s3_with_routes(routes)
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    cc = [a for a in alerts if a["code"] == "confidence_collapse"]
    assert cc, "expected a confidence_collapse alert"
    assert cc[0]["trend"] == "chronic"
    assert cc[0]["severity"] == WARN
    assert "CHRONIC" in cc[0]["line"]
    # carries the distance from threshold for triage
    assert cc[0]["pct_below_threshold"] == pytest.approx((0.45 - 0.27) / 0.45, abs=0.01)


def test_confidence_collapse_acute_is_critical():
    """Healthy prior days, only today collapses → ACUTE, CRITICAL."""
    routes = {
        "predictor/predictions/2026-04-15.json": ("json", _make_preds(
            directions=["UP", "DOWN", "FLAT"] * 7, confidences=[0.20] * 21, n=21)),
    }
    for d in ["2026-04-14", "2026-04-13", "2026-04-12"]:
        routes[f"predictor/predictions/{d}.json"] = ("json", _make_preds(
            directions=["UP", "DOWN", "FLAT"] * 7, confidences=[0.70] * 21, n=21))
    s3 = _s3_with_routes(routes)
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    cc = [a for a in alerts if a["code"] == "confidence_collapse"]
    assert cc and cc[0]["trend"] == "acute"
    assert cc[0]["severity"] == CRITICAL
    assert "ACUTE" in cc[0]["line"]


def test_check_prediction_drift_flags_alpha_degeneration():
    preds = _make_preds(
        directions=["UP", "DOWN", "FLAT"] * 7,
        confidences=[0.6] * 21,
        alphas=[0.05] * 21,  # zero stdev
    )
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    deg = [a for a in alerts if a["code"] == "alpha_degeneration"]
    assert deg and deg[0]["severity"] == CRITICAL


def test_check_prediction_drift_clean_no_alerts():
    rng = np.random.default_rng(7)
    preds = _make_preds(
        directions=list(rng.choice(["UP", "DOWN", "FLAT"], 30)),
        confidences=list(rng.uniform(0.55, 0.85, 30)),
        alphas=list(rng.normal(0.0, 0.02, 30)),
        n=30,
    )
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert alerts == []


def test_check_prediction_drift_ignores_none_directions():
    """A prediction with predicted_direction=None should NOT contribute to clustering counts."""
    payload = {"predictions": [
        {"ticker": f"T{i}", "predicted_direction": None,
         "prediction_confidence": 0.6, "predicted_alpha": 0.05}
        for i in range(20)
    ]}
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", payload),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert "direction_clustering" not in _codes(alerts)


# ── check_drift (top-level orchestrator) ───────────────────────────────────


def test_check_drift_ok_when_no_alerts():
    rng = np.random.default_rng(11)
    preds = _make_preds(
        directions=list(rng.choice(["UP", "DOWN"], 30)),
        confidences=list(rng.uniform(0.55, 0.85, 30)),
        alphas=list(rng.normal(0.0, 0.02, 30)),
        n=30,
    )
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
        # No training_feature_stats → feature drift skipped (recorded, not silent)
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")

    assert result["status"] == "ok"
    assert result["alerts"] == []
    assert result["n_alerts"] == 0
    assert result["severity"] is None
    assert result["date"] == "2026-04-15"
    # feature-drift skip is surfaced, not silent
    assert any(s["check"] == "feature_drift" for s in result["skipped_checks"])
    # Result persisted to S3
    fake_s3.put_object.assert_called_once()
    args = fake_s3.put_object.call_args.kwargs
    assert args["Key"] == "predictor/metrics/drift_2026-04-15.json"


def test_check_drift_alert_status_and_severity():
    preds = _make_preds(directions=["UP"] * 20)  # 100% clustering → WARN single-day
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")

    assert result["status"] == "alert"
    assert result["n_alerts"] >= 1
    assert result["severity"] in (WARN, CRITICAL)
    assert any("Direction clustering" in a for a in result["alerts"])
    # alerts stays a list[str]; structured detail is additive
    assert all(isinstance(a, str) for a in result["alerts"])
    assert all("severity" in d for d in result["alert_details"])


def test_check_drift_severity_is_max_across_alerts():
    """A CRITICAL alpha-degeneration outranks a WARN clustering → overall CRITICAL."""
    preds = _make_preds(directions=["UP"] * 20, alphas=[0.05] * 20)  # cluster WARN + alpha CRITICAL
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")
    assert result["severity"] == CRITICAL


def test_check_drift_swallows_put_object_failure():
    fake_s3 = _s3_with_routes({})
    fake_s3.put_object = MagicMock(side_effect=RuntimeError("S3 down"))
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")
    assert result["status"] == "alert"  # missing preds → "No recent predictions"


def test_format_alert_report_is_severity_led():
    preds = _make_preds(directions=["UP"] * 20, alphas=[0.05] * 20)
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")
    report = format_alert_report(result)
    assert report.startswith("SEVERITY: ")
    assert "CRITICAL" in report
    # each alert renders its labeled block
    assert "Likely cause:" in report
    assert "Action:" in report
