"""Tests for the ``universe_lib`` parameter on store.arctic_reader.

PR7-step-7b: a total-return SHADOW run reads the scratch ``universe_crsp``
library instead of the canonical ``universe``. The default
(``universe_lib="universe"``) must be byte-identical to live (it routes
through the lib's ``open_universe_lib`` chokepoint); the shadow value opens
the named scratch library directly. The macro library is always ``macro``.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def fake_arcticdb(monkeypatch):
    """Patch ``arcticdb.Arctic`` so the lib's ``open_arctic`` chokepoint (which
    every open routes through) returns a controllable mock instead of S3."""
    import arcticdb as _real_arcticdb
    fake = MagicMock(name="arcticdb")
    monkeypatch.setattr(_real_arcticdb, "Arctic", fake.Arctic)
    return fake


def _df(n=3):
    return pd.DataFrame(
        {"Close": list(range(100, 100 + n)), "Volume": [1e6] * n},
        index=pd.date_range("2026-01-01", periods=n),
    )


def _mock_library(symbols):
    lib = MagicMock()
    lib.list_symbols.return_value = symbols

    def read(symbol):
        item = MagicMock()
        item.data = _df()
        return item

    lib.read.side_effect = read
    return lib


def test_default_reads_universe_library(fake_arcticdb, tmp_path):
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst
    universe = _mock_library(["AAPL", "MSFT"])
    macro = _mock_library(["SPY"])
    requested = []

    def get_library(name):
        requested.append(name)
        return {"universe": universe, "macro": macro}[name]

    arctic_inst.get_library.side_effect = get_library

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("no sector_map")
    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        n = download_from_arctic("bucket-x", tmp_path)  # default universe_lib

    assert n == 3  # 2 universe + 1 macro
    assert "universe" in requested
    assert "universe_crsp" not in requested
    assert "macro" in requested


def test_shadow_reads_universe_crsp_library(fake_arcticdb, tmp_path):
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst
    crsp = _mock_library(["AAPL", "MSFT", "GOOGL"])
    macro = _mock_library(["SPY"])
    requested = []

    def get_library(name):
        requested.append(name)
        return {"universe_crsp": crsp, "macro": macro}[name]

    arctic_inst.get_library.side_effect = get_library

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("no sector_map")
    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        n = download_from_arctic("bucket-x", tmp_path, universe_lib="universe_crsp")

    assert n == 4  # 3 crsp universe + 1 macro
    assert "universe_crsp" in requested
    assert "universe" not in requested  # never touches the live universe lib
    assert "macro" in requested
    assert (tmp_path / "AAPL.parquet").exists()


def test_shadow_named_library_open_failure_raises(fake_arcticdb, tmp_path):
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst
    arctic_inst.get_library.side_effect = RuntimeError("library missing")

    fake_s3 = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        with pytest.raises(RuntimeError, match="universe_crsp"):
            download_from_arctic("bucket-x", tmp_path, universe_lib="universe_crsp")
