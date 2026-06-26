"""Wave-3 PR3-wave-2 reader-migration regression suite for
``analysis/triple_barrier_cutover_runner.py``.

Pins the per-ticker price-cache read + sector_map fallback chain so a
future "drop the legacy entry" refactor (PR4 cutover) only edits the
fallback tuples in one place — these tests then flip to expect a single
prefix.

Composes with:
  * alpha-engine-data #272 (sector_map.json write-both gap fix — now at
    3 keys: data/, predictor/, AND reference/).
  * alpha-engine-predictor #181 (the regime/features._read_parquet_close
    chokepoint — mirrors this file's inline fallback chain).
"""
from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.triple_barrier_cutover_runner import (
    _load_prices_from_s3,
    _load_sector_map_from_s3,
)


def _fake_get_object_factory(payload_by_key: dict):
    """Build an ``s3.get_object`` stub that returns Body bytes per key,
    or raises ``NoSuchKey``-shaped ``Exception`` on miss."""

    class _Body:
        def __init__(self, b: bytes):
            self._b = b

        def read(self) -> bytes:
            return self._b

    def _get_object(*, Bucket: str, Key: str):
        if Key not in payload_by_key:
            raise FileNotFoundError(f"NoSuchKey: {Key}")
        return {"Body": _Body(payload_by_key[Key])}

    return _get_object


def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    return buf.getvalue()


def _tiny_close_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.DatetimeIndex(["2026-04-20", "2026-04-21", "2026-04-22"]),
    )


# ── Wave-3 price-cache read chain ──────────────────────────────────────────


def test_load_prices_resolves_reference_and_never_consults_legacy():
    """Wave-3 PR4 cutover (DONE): the price-cache read chain is
    ``reference/price_cache/`` ONLY. The legacy ``predictor/price_cache/``
    prefix is dropped and must NOT be consulted, even when a stale copy
    still lingers under it pre-``aws s3 rm``.
    """
    payload = {
        "reference/price_cache/AAPL.parquet": _df_to_parquet_bytes(_tiny_close_frame()),
        "predictor/price_cache/AAPL.parquet": _df_to_parquet_bytes(_tiny_close_frame()),
    }
    s3 = MagicMock()
    seen_keys: list[str] = []

    def _stub(*, Bucket: str, Key: str):
        seen_keys.append(Key)
        return _fake_get_object_factory(payload)(Bucket=Bucket, Key=Key)

    s3.get_object = _stub

    out = _load_prices_from_s3("b", ["AAPL"], s3_client=s3)
    assert "AAPL" in out
    assert seen_keys == ["reference/price_cache/AAPL.parquet"], (
        "Post-cutover the read chain is reference-only."
    )
    assert "predictor/price_cache/AAPL.parquet" not in seen_keys


def test_load_prices_drops_ticker_present_only_under_legacy():
    """Wave-3 PR4 cutover: with the legacy fallback gone, a ticker whose
    parquet exists ONLY under ``predictor/price_cache/`` is invisible —
    the reference miss is the end of the chain and the ticker is dropped.
    """
    payload = {
        "predictor/price_cache/MSFT.parquet": _df_to_parquet_bytes(_tiny_close_frame()),
    }
    s3 = MagicMock()
    seen_keys: list[str] = []

    def _stub(*, Bucket: str, Key: str):
        seen_keys.append(Key)
        return _fake_get_object_factory(payload)(Bucket=Bucket, Key=Key)

    s3.get_object = _stub

    out = _load_prices_from_s3("b", ["MSFT"], s3_client=s3)
    assert "MSFT" not in out
    assert seen_keys == ["reference/price_cache/MSFT.parquet"], (
        "Only the reference prefix is consulted; legacy is never tried."
    )


def test_load_prices_skips_ticker_absent_in_both_prefixes():
    """A ticker missing from BOTH prefixes is silently dropped per the
    pre-Wave-3 contract (``caller's downstream realized-alpha join
    filters those rows``)."""
    s3 = MagicMock()
    s3.get_object = _fake_get_object_factory({})

    out = _load_prices_from_s3("b", ["GHOST"], s3_client=s3)
    assert "GHOST" not in out
    assert out == {}


# ── sector_map.json 3-key fallback ─────────────────────────────────────────


def test_sector_map_prefers_reference_key():
    """Post data #272 sector_map.json is written to 3 keys (data/,
    predictor/, reference/). The reader prefers the new home — when
    reference/ has the file it's returned without consulting the
    legacy keys.
    """
    payload = {
        "reference/price_cache/sector_map.json": json.dumps({"AAPL": "XLK"}).encode(),
        "data/sector_map.json": json.dumps({"AAPL": "XLY-STALE"}).encode(),
        "predictor/price_cache/sector_map.json": json.dumps({"AAPL": "XLF-STALE"}).encode(),
    }
    s3 = MagicMock()
    s3.get_object = _fake_get_object_factory(payload)

    out = _load_sector_map_from_s3("b", s3_client=s3)
    assert out == {"AAPL": "XLK"}, "reference/ wins when present."


def test_sector_map_falls_back_to_data_then_legacy_predictor():
    """When reference/ is missing the reader cascades through the
    historical fallbacks (data/sector_map.json → predictor/...)."""
    # reference absent → data/ wins
    payload = {
        "data/sector_map.json": json.dumps({"AAPL": "XLY"}).encode(),
        "predictor/price_cache/sector_map.json": json.dumps({"AAPL": "XLF-STALE"}).encode(),
    }
    s3 = MagicMock()
    s3.get_object = _fake_get_object_factory(payload)
    assert _load_sector_map_from_s3("b", s3_client=s3) == {"AAPL": "XLY"}

    # reference + data absent → legacy predictor/ wins
    payload2 = {
        "predictor/price_cache/sector_map.json": json.dumps({"AAPL": "XLF"}).encode(),
    }
    s3.get_object = _fake_get_object_factory(payload2)
    assert _load_sector_map_from_s3("b", s3_client=s3) == {"AAPL": "XLF"}


def test_sector_map_returns_empty_when_no_key_present():
    """All three keys absent → empty dict + WARN log; caller falls back
    to SPY benchmark per the existing contract."""
    s3 = MagicMock()
    s3.get_object = _fake_get_object_factory({})

    out = _load_sector_map_from_s3("b", s3_client=s3)
    assert out == {}


# ── Dead-constant guard ────────────────────────────────────────────────────


def test_config_no_longer_exports_price_cache_key_constant():
    """``PRICE_CACHE_KEY`` had zero callers as of 2026-05-19 — deleting
    it prevents a future ``from config import PRICE_CACHE_KEY`` that
    would silently bypass the Wave-3 read-prefix chain."""
    import config as predictor_config

    assert not hasattr(predictor_config, "PRICE_CACHE_KEY"), (
        "PRICE_CACHE_KEY was deleted in Wave-3 PR3-wave-2. If you "
        "need a price-cache S3 key template, use the read-prefix "
        "chain in ``regime/features._read_parquet_close`` or its "
        "analysis-tool mirrors — never a bare module-level constant."
    )
