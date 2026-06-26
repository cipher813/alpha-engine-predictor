"""Tests for store.arctic_reader.download_from_arctic — ArcticDB → local parquet."""

import json
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def fake_arcticdb(monkeypatch):
    """Replace the ``arcticdb`` module singleton's ``Arctic`` class with a
    controllable mock. Patches at the singleton level so the
    ``alpha_engine_lib.arcticdb._import_arcticdb()`` lazy-import path
    (used by ``open_universe_lib`` / ``open_macro_lib``) picks up the
    mock. Without singleton-level patching, a lib-routed call would
    bypass the mock and hit real S3 (post-L2771 chokepoint migration)."""
    import arcticdb as _real_arcticdb
    fake = MagicMock(name="arcticdb")
    # Patch on the actual arcticdb module attribute — caught by the
    # lib's `open_arctic(...)` chokepoint that all opens now route through.
    monkeypatch.setattr(_real_arcticdb, "Arctic", fake.Arctic)
    return fake


def _df(n=3):
    return pd.DataFrame(
        {"Close": list(range(100, 100 + n)), "Volume": [1e6] * n},
        index=pd.date_range("2026-01-01", periods=n),
    )


def _mock_library(symbols, get_df=lambda s: _df(), raise_on=None):
    """Build a mock ArcticDB library object."""
    lib = MagicMock()
    lib.list_symbols.return_value = symbols

    def read(symbol):
        if raise_on and symbol in raise_on:
            raise RuntimeError(f"simulated read failure on {symbol}")
        item = MagicMock()
        item.data = get_df(symbol)
        return item

    lib.read.side_effect = read
    return lib


def test_download_from_arctic_writes_universe_and_macro_parquets(fake_arcticdb, tmp_path):
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst

    universe_lib = _mock_library(["AAPL", "MSFT", "GOOGL"])
    macro_lib = _mock_library(["SPY", "VIX"])

    def get_library(name):
        return {"universe": universe_lib, "macro": macro_lib}[name]

    arctic_inst.get_library.side_effect = get_library

    fake_s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps({"AAPL": "Tech", "MSFT": "Tech"}).encode()
    fake_s3.get_object.return_value = {"Body": body}

    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        n = download_from_arctic("bucket-x", tmp_path)

    # 3 universe + 2 macro = 5 parquet files
    assert n == 5
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "AAPL.parquet" in files
    assert "SPY.parquet" in files
    assert "sector_map.json" in files
    sector_map = json.loads((tmp_path / "sector_map.json").read_text())
    assert sector_map["AAPL"] == "Tech"


def test_download_from_arctic_skips_empty_dataframes(fake_arcticdb, tmp_path):
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst

    def get_df(symbol):
        return pd.DataFrame() if symbol == "EMPTY" else _df()

    universe_lib = _mock_library(["AAPL", "EMPTY"], get_df=get_df)
    macro_lib = _mock_library([])
    arctic_inst.get_library.side_effect = lambda n: {"universe": universe_lib, "macro": macro_lib}[n]

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("no sector_map")

    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        n = download_from_arctic("bucket-x", tmp_path)

    assert n == 1  # only AAPL written
    assert (tmp_path / "AAPL.parquet").exists()
    assert not (tmp_path / "EMPTY.parquet").exists()


def test_download_from_arctic_per_ticker_read_failure_continues(fake_arcticdb, tmp_path):
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst

    universe_lib = _mock_library(["AAPL", "MSFT", "BAD"], raise_on={"BAD"})
    macro_lib = _mock_library([])
    arctic_inst.get_library.side_effect = lambda n: {"universe": universe_lib, "macro": macro_lib}[n]

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("no sector_map")
    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        n = download_from_arctic("bucket-x", tmp_path)

    # BAD failed, but AAPL + MSFT wrote → n=2
    assert n == 2
    assert (tmp_path / "AAPL.parquet").exists()
    assert not (tmp_path / "BAD.parquet").exists()


def test_download_from_arctic_macro_read_failure_continues(fake_arcticdb, tmp_path):
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst

    universe_lib = _mock_library(["AAPL"])
    macro_lib = _mock_library(["SPY", "VIX"], raise_on={"VIX"})
    arctic_inst.get_library.side_effect = lambda n: {"universe": universe_lib, "macro": macro_lib}[n]

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("no sector_map")
    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        n = download_from_arctic("bucket-x", tmp_path)

    assert n == 2  # AAPL + SPY (VIX failed)
    assert (tmp_path / "SPY.parquet").exists()
    assert not (tmp_path / "VIX.parquet").exists()


def test_download_from_arctic_progress_log_every_200(fake_arcticdb, tmp_path, caplog):
    """Log a progress line every 200 symbols."""
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst
    symbols = [f"T{i}" for i in range(420)]
    universe_lib = _mock_library(symbols)
    macro_lib = _mock_library([])
    arctic_inst.get_library.side_effect = lambda n: {"universe": universe_lib, "macro": macro_lib}[n]

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("no sector_map")
    with patch("boto3.client", return_value=fake_s3):
        with caplog.at_level("INFO"):
            from store.arctic_reader import download_from_arctic
            n = download_from_arctic("bucket-x", tmp_path)

    assert n == 420
    progress_lines = [r for r in caplog.records if "Written" in r.message and "symbols" in r.message]
    assert len(progress_lines) == 2  # 200/420 and 400/420


def test_download_from_arctic_uri_region_from_env(monkeypatch, fake_arcticdb, tmp_path):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    arctic_inst = MagicMock()
    fake_arcticdb.Arctic.return_value = arctic_inst
    arctic_inst.get_library.side_effect = lambda n: _mock_library([])

    fake_s3 = MagicMock()
    fake_s3.get_object.side_effect = RuntimeError("no sector_map")
    with patch("boto3.client", return_value=fake_s3):
        from store.arctic_reader import download_from_arctic
        download_from_arctic("bucket-x", tmp_path)

    # universe + macro libs each open their own connection via the lib
    # chokepoint, so Arctic is constructed once per library open.
    assert fake_arcticdb.Arctic.called
    uri = fake_arcticdb.Arctic.call_args.args[0]
    assert "s3.eu-west-1.amazonaws.com:bucket-x" in uri
    assert "path_prefix=arcticdb" in uri
