"""Tests for monitoring.drift_detector — feature + prediction drift checks."""

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
    DIRECTION_CLUSTER_THRESHOLD,
    FEATURE_ZSCORE_THRESHOLD,
    _load_json,
    _load_parquet,
    check_drift,
    check_feature_drift,
    check_prediction_drift,
)


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
    assert any("Feature snapshot missing" in a for a in alerts)


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
    assert "rsi_14" in alerts[0]
    assert "Feature drift" in alerts[0]


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
    assert alerts == ["No recent predictions found"]


def test_check_prediction_drift_empty_today_alerts():
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", {"predictions": []}),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert any("empty" in a.lower() for a in alerts)


def test_check_prediction_drift_flags_single_day_clustering():
    """90% UP on today only → single-day cluster alert."""
    preds = _make_preds(directions=["UP"] * 18 + ["DOWN"] * 2)
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert any("Direction clustering" in a and "UP" in a for a in alerts)
    # Single day → no persistent cluster
    assert not any("PERSISTENT" in a for a in alerts)


def test_check_prediction_drift_persistent_clustering_alert():
    """N consecutive trading days all clustered → PERSISTENT alert."""
    clustered_preds = _make_preds(directions=["UP"] * 19 + ["DOWN"] * 1)

    routes = {}
    # Lay down 5 trading-day files going back from 2026-04-15
    days = ["2026-04-15", "2026-04-14", "2026-04-13", "2026-04-12", "2026-04-11"]
    for d in days:
        routes[f"predictor/predictions/{d}.json"] = ("json", clustered_preds)

    s3 = _s3_with_routes(routes)
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert any("PERSISTENT" in a for a in alerts)
    assert any(str(CONSECUTIVE_DAYS_THRESHOLD) in a for a in alerts)


def test_check_prediction_drift_flags_confidence_collapse():
    preds = _make_preds(
        directions=["UP", "DOWN", "FLAT"] * 7,  # diversified to avoid clustering
        confidences=[0.40] * 21,                # below CONFIDENCE_MIN_MEAN
    )
    s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    alerts = check_prediction_drift(s3, "bucket", "2026-04-15")
    assert any("Confidence collapse" in a for a in alerts)


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
    assert any("Alpha degeneration" in a for a in alerts)


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
    # No direction info → no direction-clustering alert, but possible alpha degeneration
    assert not any("Direction clustering" in a for a in alerts)


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
        # No training_feature_stats → check_feature_drift returns []
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")

    assert result["status"] == "ok"
    assert result["alerts"] == []
    assert result["n_alerts"] == 0
    assert result["date"] == "2026-04-15"
    # Result should be persisted to S3
    fake_s3.put_object.assert_called_once()
    args = fake_s3.put_object.call_args.kwargs
    assert args["Key"] == "predictor/metrics/drift_2026-04-15.json"


def test_check_drift_alert_status_when_drift_present():
    preds = _make_preds(directions=["UP"] * 20)  # 100% clustering
    fake_s3 = _s3_with_routes({
        "predictor/predictions/2026-04-15.json": ("json", preds),
    })
    fake_s3.put_object = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        result = check_drift(bucket="bucket", date_str="2026-04-15")

    assert result["status"] == "alert"
    assert result["n_alerts"] >= 1
    assert any("Direction clustering" in a for a in result["alerts"])


def test_check_drift_swallows_put_object_failure():
    fake_s3 = _s3_with_routes({})
    fake_s3.put_object = MagicMock(side_effect=RuntimeError("S3 down"))
    with patch("boto3.client", return_value=fake_s3):
        # Should not raise even though put_object fails
        result = check_drift(bucket="bucket", date_str="2026-04-15")
    assert result["status"] == "alert"  # missing preds → "No recent predictions"
