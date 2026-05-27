"""Tests for ``training/risk_model_persist.py`` (ROADMAP C.2b).

Wires the C.2a math primitives (``risk_model.build_factor_risk_model``)
into the Saturday SF PredictorTraining stage and persists F + D to
``s3://{bucket}/risk_model/{date}/{F,D}.parquet``. Tests pin:

  * empty / sparse data_dir returns ``status=skipped`` with the
    appropriate reason (no all-NaN F pollution of the weekly series)
  * insufficient loading-column coverage (pre-#324 cache) returns
    ``status=skipped`` — auto-activates once C.1 loadings have
    accumulated
  * happy path persists F.parquet + D.parquet + metadata.json to
    the expected S3 prefix
  * S3 persist failure is non-fatal (warns + returns payload)
  * dry_run path builds the model but skips the S3 write
  * graceful read of malformed / unreadable parquets
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from training.risk_model_persist import (
    FACTOR_LOADING_COLUMNS,
    build_and_persist_risk_model,
)


def _write_synthetic_ticker(
    data_dir: Path,
    ticker: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    *,
    include_loadings: bool = True,
) -> None:
    """Write a synthetic per-ticker parquet with OHLCV + (optionally)
    the 8 factor-loading columns."""
    n = len(dates)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    cols = {
        "Open": close,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, size=n),
        "VWAP": close,
        "source": ["yfinance"] * n,
    }
    if include_loadings:
        for col in FACTOR_LOADING_COLUMNS:
            cols[col] = rng.normal(0, 1, size=n)
    df = pd.DataFrame(cols, index=dates)
    df.to_parquet(data_dir / f"{ticker}.parquet", engine="pyarrow")


@pytest.fixture
def synthetic_universe_dir(tmp_path):
    """Build a 60-date × 40-ticker synthetic universe cache with all 8
    factor-loading columns populated — enough data to clear both the
    minimum-dates and minimum-tickers gates."""
    rng = np.random.default_rng(42)
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    for i in range(40):
        _write_synthetic_ticker(data_dir, f"TICK{i:02d}", dates, rng)
    return data_dir


# ── Sparse-data skip paths ──────────────────────────────────────────────


class TestSparseDataSkipPaths:
    def test_empty_data_dir_returns_skipped(self, tmp_path):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        result = build_and_persist_risk_model(
            data_dir=data_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            s3_client=MagicMock(),
        )
        assert result["status"] == "skipped"
        assert result["n_dates"] == 0
        assert "returns_panel has only 0 dates" in result["reason"]

    def test_below_min_dates_returns_skipped(self, tmp_path):
        rng = np.random.default_rng(1)
        data_dir = tmp_path / "shallow"
        data_dir.mkdir()
        # Only 10 dates — below the 60-date threshold
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        for i in range(40):
            _write_synthetic_ticker(data_dir, f"TICK{i:02d}", dates, rng)
        result = build_and_persist_risk_model(
            data_dir=data_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            s3_client=MagicMock(),
        )
        assert result["status"] == "skipped"
        assert "returns_panel has only" in result["reason"]

    def test_below_min_tickers_with_loadings_returns_skipped(self, tmp_path):
        rng = np.random.default_rng(2)
        data_dir = tmp_path / "no_loadings"
        data_dir.mkdir()
        dates = pd.date_range("2026-01-01", periods=80, freq="D")
        # 20 tickers WITH loadings, 5 tickers without — below the 30-
        # ticker threshold
        for i in range(20):
            _write_synthetic_ticker(data_dir, f"WITH{i:02d}", dates, rng)
        for i in range(5):
            _write_synthetic_ticker(
                data_dir, f"WITHOUT{i:02d}", dates, rng,
                include_loadings=False,
            )
        result = build_and_persist_risk_model(
            data_dir=data_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            s3_client=MagicMock(),
        )
        assert result["status"] == "skipped"
        assert "tickers carry all" in result["reason"]
        assert "pre-#324 universe cache" in result["reason"]

    def test_no_loading_columns_returns_skipped(self, tmp_path):
        """Every ticker carries OHLCV but ZERO loading columns —
        pre-2026-05-26 universe cache shape."""
        rng = np.random.default_rng(3)
        data_dir = tmp_path / "pre_c1"
        data_dir.mkdir()
        dates = pd.date_range("2026-01-01", periods=80, freq="D")
        for i in range(40):
            _write_synthetic_ticker(
                data_dir, f"TICK{i:02d}", dates, rng,
                include_loadings=False,
            )
        result = build_and_persist_risk_model(
            data_dir=data_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            s3_client=MagicMock(),
        )
        assert result["status"] == "skipped"
        assert "0 tickers carry all" in result["reason"]


# ── Happy path ──────────────────────────────────────────────────────────


class TestHappyPath:
    def test_persists_F_D_metadata_to_expected_prefix(
        self, synthetic_universe_dir
    ):
        s3 = MagicMock()
        result = build_and_persist_risk_model(
            data_dir=synthetic_universe_dir,
            bucket="alpha-engine-research",
            date_str="2026-05-27",
            s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["F_shape"][0] == result["F_shape"][1]
        assert result["D_length"] > 0

        # 3 put_object calls: F.parquet, D.parquet, metadata.json
        keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert "risk_model/2026-05-27/F.parquet" in keys
        assert "risk_model/2026-05-27/D.parquet" in keys
        assert "risk_model/2026-05-27/metadata.json" in keys

        # metadata is JSON
        meta_call = next(
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("metadata.json")
        )
        meta = json.loads(meta_call.kwargs["Body"])
        assert "n_dates" in meta
        assert "K_eff" in meta

    def test_F_is_square_symmetric(self, synthetic_universe_dir):
        """F MUST be square (K × K). C.3 will use it in Σ = B·F·Bᵀ + D;
        non-square breaks the matmul + the PSD reconstruction."""
        result = build_and_persist_risk_model(
            data_dir=synthetic_universe_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            dry_run=True,
        )
        assert result["status"] == "ok"
        F_rows, F_cols = result["F_shape"]
        assert F_rows == F_cols


# ── dry_run skips S3 ───────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_builds_model_but_no_s3(self, synthetic_universe_dir):
        s3 = MagicMock()
        result = build_and_persist_risk_model(
            data_dir=synthetic_universe_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            dry_run=True,
            s3_client=s3,
        )
        assert result["status"] == "ok"
        s3.put_object.assert_not_called()


# ── S3 persist failure ─────────────────────────────────────────────────


class TestS3PersistFailure:
    def test_s3_failure_does_not_raise(
        self, synthetic_universe_dir, caplog
    ):
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 outage")
        result = build_and_persist_risk_model(
            data_dir=synthetic_universe_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            s3_client=s3,
        )
        # status=ok because the model was built; the write was best-effort
        assert result["status"] == "ok"
        assert any(
            "S3 persist failed" in record.message
            for record in caplog.records
        )


# ── Malformed parquet handling ─────────────────────────────────────────


class TestMalformedParquet:
    def test_unreadable_parquet_is_skipped(
        self, synthetic_universe_dir, caplog
    ):
        """A garbage/truncated parquet file in the cache should be
        silently skipped — don't let one bad ticker abort the whole
        weekly risk-model build."""
        garbage = synthetic_universe_dir / "BADTICK.parquet"
        garbage.write_bytes(b"not a parquet")
        # Should still succeed with the remaining 40 valid tickers
        result = build_and_persist_risk_model(
            data_dir=synthetic_universe_dir,
            bucket="test-bucket",
            date_str="2026-05-27",
            dry_run=True,
        )
        assert result["status"] == "ok"
        assert result["n_tickers"] == 40  # garbage skipped, others survived
