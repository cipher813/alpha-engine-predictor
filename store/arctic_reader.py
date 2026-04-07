"""
store/arctic_reader.py — Load universe data from ArcticDB for predictor training.

Reads per-ticker DataFrames (OHLCV + 53 pre-computed features) from the
ArcticDB universe library and writes them as parquets to a local cache
directory. This makes ArcticDB data compatible with the existing
build_regression_arrays() pipeline in dataset.py.

Also writes macro series (SPY, VIX, etc.) and sector_map.json so the
downstream pipeline finds everything it expects.

Usage:
    from store.arctic_reader import download_from_arctic

    n_files = download_from_arctic(bucket, local_dir)
"""

from __future__ import annotations

import json
import logging
import os
import time

import arcticdb as adb
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
ARCTIC_PREFIX = "arcticdb"


def _get_arctic(bucket: str) -> adb.Arctic:
    """Create ArcticDB connection."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = f"s3s://s3.{region}.amazonaws.com:{bucket}?path_prefix={ARCTIC_PREFIX}&aws_auth=true"
    return adb.Arctic(uri)


def download_from_arctic(
    bucket: str,
    local_dir: str | os.PathLike,
) -> int:
    """
    Read all universe + macro symbols from ArcticDB and write as parquets
    to local_dir, matching the format download_price_cache() produces.

    The key difference: ArcticDB DataFrames include pre-computed feature
    columns alongside OHLCV. build_regression_arrays() in dataset.py
    detects these and skips inline compute_features().

    Returns the number of files written.
    """
    t0 = time.time()
    local_dir = str(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    arctic = _get_arctic(bucket)
    universe = arctic.get_library("universe")
    macro_lib = arctic.get_library("macro")

    n_written = 0

    # Write stock tickers from universe library
    symbols = universe.list_symbols()
    log.info("[data_source=arcticdb] Reading %d universe symbols...", len(symbols))

    for i, ticker in enumerate(symbols):
        try:
            df = universe.read(ticker).data
            if df.empty:
                continue
            out_path = os.path.join(local_dir, f"{ticker}.parquet")
            df.to_parquet(out_path, engine="pyarrow", compression="snappy")
            n_written += 1
        except Exception as exc:
            log.debug("Failed to read %s: %s", ticker, exc)

        if (i + 1) % 200 == 0:
            log.info("  Written %d/%d symbols", i + 1, len(symbols))

    # Write macro series from macro library
    for key in macro_lib.list_symbols():
        try:
            df = macro_lib.read(key).data
            if df.empty:
                continue
            out_path = os.path.join(local_dir, f"{key}.parquet")
            df.to_parquet(out_path, engine="pyarrow", compression="snappy")
            n_written += 1
        except Exception as exc:
            log.debug("Failed to read macro %s: %s", key, exc)

    # Write sector_map.json from S3 (not stored in ArcticDB)
    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="data/sector_map.json")
        sector_map = json.loads(obj["Body"].read())
        map_path = os.path.join(local_dir, "sector_map.json")
        with open(map_path, "w") as f:
            json.dump(sector_map, f)
        log.info("[data_source=arcticdb] Wrote sector_map.json (%d mappings)", len(sector_map))
    except Exception as exc:
        log.warning("Failed to load sector_map.json from S3: %s", exc)

    elapsed = time.time() - t0
    log.info(
        "[data_source=arcticdb] Cache populated in %.1fs: %d files written to %s",
        elapsed, n_written, local_dir,
    )
    return n_written
