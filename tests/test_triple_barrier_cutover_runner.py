"""Tests for Stage 3 PR 4 cutover-gate runner.

Validates the end-to-end Stage 3 cutover-gate pipeline:
- ``compute_realized_alpha_for_pairs`` joins forward sector-neutral
  residual log-returns into prediction pairs
- ``run_gate`` orchestrates load → realize → validate → write
- Edge cases: future-dated predictions (forward window unclosed) get
  None realized_alpha; missing price data is silently skipped;
  benchmark fallback to SPY when sector_map is empty
- The default 21d horizon matches cfg.FORWARD_DAYS — gate measures
  predictive power on the canonical fixed-horizon ground truth axis,
  not the variant's training target (LdP-conservative)

Plan: alpha-engine-docs/private/triple-barrier-260510.md
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.variant_cutover_gate import compute_realized_alpha_for_pairs


def _make_price_series(daily_log_ret: float, n: int = 80, start: str = "2026-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=n)
    close = 100.0 * np.exp(np.cumsum(np.full(n, daily_log_ret)))
    return pd.Series(close, index=idx)


# ── compute_realized_alpha_for_pairs ─────────────────────────────────────


class TestComputeRealizedAlphaForPairs:

    def test_outperforming_stock_yields_positive_realized_alpha(self):
        # Stock 0.005/day, benchmark 0.001/day → daily residual 0.004.
        # Over 21 trading days → realized = 21 × 0.004 = 0.084.
        prices = {
            "AAPL": _make_price_series(0.005, n=60),
            "XLK": _make_price_series(0.001, n=60),
        }
        sector_map = {"AAPL": "XLK"}
        pairs = [{"date": "2026-01-15", "ticker": "AAPL"}]
        n_filled = compute_realized_alpha_for_pairs(
            pairs, horizon_days=21,
            prices_by_ticker=prices, sector_map=sector_map,
        )
        assert n_filled == 1
        assert pairs[0]["realized_alpha"] is not None
        assert math.isclose(pairs[0]["realized_alpha"], 0.084, abs_tol=1e-9)

    def test_underperforming_stock_yields_negative_realized_alpha(self):
        prices = {
            "TSLA": _make_price_series(-0.002, n=60),
            "XLK": _make_price_series(0.001, n=60),
        }
        sector_map = {"TSLA": "XLK"}
        pairs = [{"date": "2026-01-15", "ticker": "TSLA"}]
        compute_realized_alpha_for_pairs(
            pairs, horizon_days=21,
            prices_by_ticker=prices, sector_map=sector_map,
        )
        # daily residual = -0.003; 21d → -0.063
        assert math.isclose(pairs[0]["realized_alpha"], -0.063, abs_tol=1e-9)

    def test_unrealized_pairs_get_none_realized_alpha(self):
        # 60-row series; pair date is at end → no 21d forward room.
        prices = {
            "AAPL": _make_price_series(0.005, n=60),
            "XLK": _make_price_series(0.001, n=60),
        }
        sector_map = {"AAPL": "XLK"}
        # Pair date is the second-to-last row → t+21 past end
        last_date = prices["AAPL"].index[-2].strftime("%Y-%m-%d")
        pairs = [{"date": last_date, "ticker": "AAPL"}]
        n_filled = compute_realized_alpha_for_pairs(
            pairs, horizon_days=21,
            prices_by_ticker=prices, sector_map=sector_map,
        )
        # Pair should NOT have realized_alpha set
        assert n_filled == 0
        assert "realized_alpha" not in pairs[0]

    def test_missing_price_silently_skipped(self):
        prices = {"AAPL": _make_price_series(0.005, n=60), "XLK": _make_price_series(0.001, n=60)}
        sector_map = {"AAPL": "XLK", "MISSING_TKR": "XLK"}
        pairs = [
            {"date": "2026-01-15", "ticker": "AAPL"},
            {"date": "2026-01-15", "ticker": "MISSING_TKR"},
        ]
        n_filled = compute_realized_alpha_for_pairs(
            pairs, horizon_days=21,
            prices_by_ticker=prices, sector_map=sector_map,
        )
        # AAPL filled; MISSING_TKR skipped because no price series exists
        assert n_filled == 1

    def test_spy_fallback_when_no_sector_mapping(self):
        prices = {
            "JOY": _make_price_series(0.003, n=60),
            "SPY": _make_price_series(0.001, n=60),
        }
        sector_map = {}  # empty — JOY isn't mapped
        pairs = [{"date": "2026-01-15", "ticker": "JOY"}]
        compute_realized_alpha_for_pairs(
            pairs, horizon_days=21,
            prices_by_ticker=prices, sector_map=sector_map,
        )
        # daily residual = 0.002; 21d → 0.042
        assert math.isclose(pairs[0]["realized_alpha"], 0.042, abs_tol=1e-9)

    def test_pandas_timestamp_date_input(self):
        # Per-row dates from the SF training path are pd.Timestamp,
        # not ISO strings. Both inputs must work.
        prices = {
            "AAPL": _make_price_series(0.005, n=60),
            "XLK": _make_price_series(0.001, n=60),
        }
        sector_map = {"AAPL": "XLK"}
        pred_date = pd.Timestamp("2026-01-15")
        pairs = [{"date": pred_date, "ticker": "AAPL"}]
        n_filled = compute_realized_alpha_for_pairs(
            pairs, horizon_days=21,
            prices_by_ticker=prices, sector_map=sector_map,
        )
        assert n_filled == 1
        assert math.isclose(pairs[0]["realized_alpha"], 0.084, abs_tol=1e-9)

    def test_holiday_dates_snap_back_via_ffill(self):
        # Pair date on a non-trading day (e.g., a Saturday) — ffill
        # should snap to the prior trading day, NOT skip the pair.
        prices = {
            "AAPL": _make_price_series(0.005, n=60),
            "XLK": _make_price_series(0.001, n=60),
        }
        sector_map = {"AAPL": "XLK"}
        # Find a Saturday inside the price index range
        sat = pd.Timestamp("2026-01-17")  # known Saturday
        assert sat.dayofweek == 5
        pairs = [{"date": sat, "ticker": "AAPL"}]
        n_filled = compute_realized_alpha_for_pairs(
            pairs, horizon_days=21,
            prices_by_ticker=prices, sector_map=sector_map,
        )
        # Should still fill — ffill snaps to Friday
        assert n_filled == 1


# ── run_gate orchestration (mocked S3) ───────────────────────────────────


class _FakeS3:
    """Minimal boto3-S3-shaped fake for run_gate tests."""

    class _ExceptionsNamespace:
        class NoSuchKey(Exception):
            pass

    def __init__(self, objects: dict):
        # objects keyed by (Bucket, Key) → bytes payload
        self.objects = objects
        self.put_calls: list[dict] = []
        self.exceptions = self._ExceptionsNamespace()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey(f"no key {Key}")
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType})
        return {}


class TestRunGate:

    def _make_predictions_payload(self, date: str, predictions: list[dict]) -> bytes:
        return json.dumps({"date": date, "predictions": predictions}).encode("utf-8")

    def test_run_gate_writes_result_to_s3_with_full_pipeline(self):
        from analysis.triple_barrier_cutover_runner import run_gate

        bucket = "test-bucket"
        # Build 30+ pairs with a clear positive lift for the variant
        prices = {
            "AAPL": _make_price_series(0.005, n=80),
            "MSFT": _make_price_series(0.004, n=80),
            "XLK": _make_price_series(0.001, n=80),
            "SPY": _make_price_series(0.001, n=80),
        }
        sector_map = {"AAPL": "XLK", "MSFT": "XLK"}

        # Pre-build prediction pairs (skip the S3 prediction-load path
        # for this test by passing them in directly).
        rng = np.random.default_rng(42)
        date_strs = [
            (datetime.date(2026, 1, 5) + datetime.timedelta(days=i)).isoformat()
            for i in range(0, 7)  # 7 dates × ~20 tickers each = ~140 pairs
        ]
        prediction_pairs = []
        for date_str in date_strs:
            for ticker in ("AAPL", "MSFT") * 60:  # 120 of each ticker × dates → many pairs
                prediction_pairs.append({
                    "date": date_str,
                    "ticker": ticker,
                    # baseline = noise; variant = realized + small noise (ground-truth proxy)
                    "predicted_alpha": float(rng.normal(0, 0.02)),
                    "meta_alpha_tb": 0.04 + float(rng.normal(0, 0.01)),
                    "realized_alpha": None,
                })

        fake_s3 = _FakeS3({})
        payload = run_gate(
            bucket=bucket,
            prediction_pairs=prediction_pairs,
            prices_by_ticker=prices,
            sector_map=sector_map,
            s3_client=fake_s3,
            n_days=42,
            horizon_days=21,
            threshold=1.15,
            write_to_s3=True,
        )

        # Result JSON has core gate keys + runner metadata
        for key in (
            "passed", "baseline_ic", "variant_ic", "relative_lift", "n_pairs",
            "reason", "baseline_field", "variant_field",
            "run_date", "window_days", "horizon_days",
            "n_pairs_loaded", "n_realized_filled",
        ):
            assert key in payload, f"missing key: {key}"

        # S3 write occurred at the default key shape
        assert len(fake_s3.put_calls) == 1
        put = fake_s3.put_calls[0]
        assert put["Bucket"] == bucket
        assert "predictor/variant_gates/triple_barrier_" in put["Key"]
        assert put["Key"].endswith(".json")
        # Body roundtrips
        body_payload = json.loads(put["Body"].decode("utf-8"))
        assert body_payload["baseline_field"] == "predicted_alpha"
        assert body_payload["variant_field"] == "meta_alpha_tb"

    def test_run_gate_no_write_skips_s3_put(self):
        from analysis.triple_barrier_cutover_runner import run_gate

        prices = {
            "AAPL": _make_price_series(0.005, n=60),
            "XLK": _make_price_series(0.001, n=60),
            "SPY": _make_price_series(0.001, n=60),
        }
        prediction_pairs = [{
            "date": "2026-01-15", "ticker": "AAPL",
            "predicted_alpha": 0.01, "meta_alpha_tb": 0.02,
            "realized_alpha": None,
        }]
        fake_s3 = _FakeS3({})
        run_gate(
            bucket="test",
            prediction_pairs=prediction_pairs,
            prices_by_ticker=prices,
            sector_map={"AAPL": "XLK"},
            s3_client=fake_s3,
            write_to_s3=False,
        )
        assert fake_s3.put_calls == []

    def test_run_gate_insufficient_data_returns_neutral_verdict(self):
        from analysis.triple_barrier_cutover_runner import run_gate

        # No prediction history → gate returns insufficient-data verdict
        prediction_pairs: list[dict] = []
        fake_s3 = _FakeS3({})
        payload = run_gate(
            bucket="test",
            prediction_pairs=prediction_pairs,
            prices_by_ticker={},
            sector_map={},
            s3_client=fake_s3,
            write_to_s3=False,
        )
        assert payload["passed"] is False
        assert payload["n_pairs"] == 0
        assert "insufficient data" in payload["reason"]
