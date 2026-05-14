"""Tests for regime/substrate.py — orchestrator + S3 writer.

Covers:
- End-to-end substrate assembly against a fitted HMM + synthetic history
- Schema invariants (required keys, probability normalization, types)
- Canonical-shape writer (run_id artifact + latest.json sidecar)
- Round-trip read after write
- Guardrail flag computation
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from regime.hmm import HMMRegimeClassifier
from regime.substrate import (
    REGIME_FEATURES,
    SCHEMA_VERSION,
    _compute_guardrail_flags,
    build_regime_substrate,
    read_regime_substrate,
    write_regime_substrate,
)


pytest.importorskip("hmmlearn")
pytest.importorskip("scipy")


class _FakeS3:
    """In-memory S3 stub — supports put_object + get_object only."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict:
        self._objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self._objects:
            raise KeyError(f"no object at {Bucket}/{Key}")
        return {"Body": io.BytesIO(self._objects[(Bucket, Key)])}


def _build_history(n: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Two-regime synthetic history — first half calm, second half stressed.
    # Gives the HMM enough separation to discover meaningful states.
    halves = []
    for i, (mean_ret, mean_vix, mean_hy) in enumerate([
        (0.02, 14.0, 320.0),
        (-0.02, 28.0, 600.0),
    ]):
        halves.append(pd.DataFrame({
            "spy_20d_return": rng.normal(mean_ret, 0.03, n // 2),
            "vix_level": rng.normal(mean_vix, 4.0, n // 2),
            "vix_term_slope": rng.normal(0.8 if i == 0 else -0.3, 0.4, n // 2),
            "hy_oas_bps": rng.normal(mean_hy, 80.0, n // 2),
            "yield_curve_slope": rng.normal(0.4 if i == 0 else -0.2, 0.3, n // 2),
            "market_breadth": rng.normal(0.7 if i == 0 else 0.35, 0.1, n // 2).clip(0, 1),
        }))
    return pd.concat(halves, ignore_index=True)


def _build_fitted_hmm(history: pd.DataFrame) -> HMMRegimeClassifier:
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    return HMMRegimeClassifier(n_states=3, random_state=0).fit(history, feat_cols)


def test_substrate_payload_has_required_top_level_keys() -> None:
    history = _build_history()
    hmm = _build_fitted_hmm(history)
    current = {f: float(history[f].iloc[-1]) for f in REGIME_FEATURES}
    payload = build_regime_substrate(
        feature_history=history,
        current_features=current,
        hmm=hmm,
        run_id="2605141200",
        calendar_date="2026-05-14",
        trading_day="2026-05-13",
    )
    required = {
        "calendar_date", "trading_day", "run_id", "schema_version",
        "hmm", "composite", "bocpd", "guardrails", "features", "model_metadata",
    }
    assert required <= set(payload)
    assert payload["schema_version"] == SCHEMA_VERSION


def test_substrate_hmm_probs_sum_to_one() -> None:
    history = _build_history()
    hmm = _build_fitted_hmm(history)
    current = {f: float(history[f].iloc[-1]) for f in REGIME_FEATURES}
    payload = build_regime_substrate(
        feature_history=history, current_features=current, hmm=hmm,
    )
    probs = payload["hmm"]["probs"]
    assert {"bear", "neutral", "bull"} == set(probs.keys())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert payload["hmm"]["argmax"] in {"bear", "neutral", "bull"}


def test_substrate_composite_intensity_inverted_to_risk_on_orientation() -> None:
    """Composite module returns positive=risk-off; substrate orchestrator
    must invert to positive=risk-on for the downstream contract."""
    history = _build_history()
    hmm = _build_fitted_hmm(history)
    # Stressed-regime current row → composite intensity_z should be NEGATIVE
    # in substrate output (orientation: positive=risk-on, negative=risk-off).
    stressed = {
        "spy_20d_return": -0.10,
        "vix_level": 40.0,
        "vix_term_slope": -0.6,
        "hy_oas_bps": 850.0,
        "yield_curve_slope": -0.6,
        "market_breadth": 0.20,
    }
    payload = build_regime_substrate(
        feature_history=history, current_features=stressed, hmm=hmm,
    )
    assert payload["composite"]["intensity_z"] < -0.5, (
        f"stressed input should yield negative (risk-off) intensity in substrate output; "
        f"got {payload['composite']['intensity_z']}"
    )


def test_substrate_guardrail_flags_severity_floor_under_stress() -> None:
    """High VIX + negative SPY 30d return → guardrail forces at least 'bear'."""
    stressed = {
        "spy_20d_return": -0.10,
        "spy_30d_return": -0.15,  # -15% in 30d
        "vix_level": 35.0,
        "hy_oas_bps": 850.0,
    }
    flags = _compute_guardrail_flags(stressed)
    assert flags["vix_bear_breached"] is True
    assert flags["spy_30d_bear_breached"] is True
    assert flags["active_severity_floor"] == "bear"


def test_substrate_guardrail_flags_no_floor_under_calm() -> None:
    calm = {
        "spy_20d_return": 0.03,
        "spy_30d_return": 0.05,
        "vix_level": 12.0,
        "hy_oas_bps": 280.0,
    }
    flags = _compute_guardrail_flags(calm)
    assert flags["vix_bear_breached"] is False
    assert flags["spy_30d_bear_breached"] is False
    assert flags["active_severity_floor"] is None


def test_substrate_writer_publishes_dated_artifact_and_latest_sidecar() -> None:
    history = _build_history()
    hmm = _build_fitted_hmm(history)
    current = {f: float(history[f].iloc[-1]) for f in REGIME_FEATURES}
    payload = build_regime_substrate(
        feature_history=history, current_features=current, hmm=hmm,
        run_id="2605141200", calendar_date="2026-05-14", trading_day="2026-05-13",
    )
    s3 = _FakeS3()
    keys = write_regime_substrate(payload, s3_client=s3, bucket="test-bucket", prefix="regime")
    assert keys["artifact_key"] == "regime/2605141200.json"
    assert keys["latest_key"] == "regime/latest.json"
    # Round-trip: read back via consumer API
    read_back = read_regime_substrate(s3_client=s3, bucket="test-bucket", prefix="regime")
    assert read_back is not None
    assert read_back["run_id"] == "2605141200"
    assert read_back["hmm"]["argmax"] in {"bear", "neutral", "bull"}


def test_substrate_read_returns_none_when_no_artifact() -> None:
    s3 = _FakeS3()
    assert read_regime_substrate(s3_client=s3, bucket="empty", prefix="regime") is None


def test_substrate_features_block_contains_inputs() -> None:
    history = _build_history()
    hmm = _build_fitted_hmm(history)
    current = {f: float(history[f].iloc[-1]) for f in REGIME_FEATURES}
    payload = build_regime_substrate(
        feature_history=history, current_features=current, hmm=hmm,
    )
    # Every REGIME_FEATURE that was in `current` should round-trip
    # through the features block for forensic replay.
    for feat in REGIME_FEATURES:
        assert feat in payload["features"]
        assert payload["features"][feat] == pytest.approx(current[feat])


def test_substrate_writer_does_not_call_hmm_smoother() -> None:
    """Look-ahead discipline: the substrate orchestrator must NEVER
    invoke ``predict_proba_smoother``. The smoother is reserved for
    retrospective T1 ground-truth labeling.

    We pin the contract by monkey-patching the smoother to raise — if
    the orchestrator calls it, the test fails loudly.
    """
    history = _build_history()
    hmm = _build_fitted_hmm(history)

    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "substrate orchestrator called predict_proba_smoother — "
            "look-ahead discipline violated! Smoother is for retrospective use only."
        )

    hmm.predict_proba_smoother = _forbidden  # type: ignore[method-assign]
    current = {f: float(history[f].iloc[-1]) for f in REGIME_FEATURES}
    # Should succeed — orchestrator uses filter only.
    payload = build_regime_substrate(
        feature_history=history, current_features=current, hmm=hmm,
    )
    assert "hmm" in payload
