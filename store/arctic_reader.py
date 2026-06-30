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

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
ARCTIC_PREFIX = "arcticdb"


def download_from_arctic(
    bucket: str,
    local_dir: str | os.PathLike,
    universe_lib: str = "universe",
) -> int:
    """
    Read all universe + macro symbols from ArcticDB and write as parquets
    to local_dir, matching the legacy per-ticker OHLCV parquet format
    (the now-removed S3 download_price_cache() fallback produced the same
    shape; ArcticDB is canonical since PR #6).

    The key difference: ArcticDB DataFrames include pre-computed feature
    columns alongside OHLCV. build_regression_arrays() in dataset.py
    detects these and skips inline compute_features().

    Parameters
    ----------
    universe_lib : str
        ArcticDB library to read the per-ticker stock universe from. Default
        ``"universe"`` (the canonical production library — live behaviour,
        unchanged: opened via the lib's ``open_universe_lib`` chokepoint). A
        total-return SHADOW run (PR7-7b) passes ``"universe_crsp"`` — the
        scratch library ne-data (#554) builds on a clean CRSP total-return
        basis (``Close`` = split-adjusted level + a new ``total_return_close``
        column, with the 53 features RECOMPUTED on the total-return series
        under the SAME column names). The macro library is always ``"macro"``
        regardless of basis.

    Returns the number of files written.
    """
    t0 = time.time()
    local_dir = str(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    from nousergon_lib.arcticdb import (
        open_universe_lib, open_macro_lib, open_arctic,
    )
    if universe_lib == "universe":
        # Preserve the live chokepoint exactly (its RuntimeError wrapping).
        universe = open_universe_lib(bucket)
    else:
        # Shadow basis — open the named scratch library directly.
        arctic = open_arctic(bucket)
        try:
            universe = arctic.get_library(universe_lib)
        except Exception as exc:
            raise RuntimeError(
                f"ArcticDB {universe_lib!r} library open failed on bucket "
                f"{bucket!r}: {exc}"
            ) from exc
    macro_lib = open_macro_lib(bucket)

    n_written = 0

    # Write stock tickers from universe library
    symbols = universe.list_symbols()
    log.info(
        "[data_source=arcticdb] Reading %d symbols from library '%s'...",
        len(symbols), universe_lib,
    )

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
