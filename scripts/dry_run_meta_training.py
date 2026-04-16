"""
dry_run_meta_training.py — Local dry-run of meta-trainer post-PR-34 retirement.

Exercises the full walk-forward + prod-fit + meta-ridge training loop
WITHOUT uploading to S3. Validates that the Tier 0 regime retirement
left no dangling references, the 12-feature META_FEATURES schema lands
in the saved meta_model sidecar, and the feature-importance metadata
populates.

Scoped to 100 tickers + macro (from legacy S3 parquet cache, not
ArcticDB — macOS arcticdb client has an unrelated allocator crash on
S3 init). Same data content ArcticDB serves; integration failures
(dangling refs, schema mismatches) fire on the same code paths
regardless of ticker count. Data-dependent failures specific to the
full 900-ticker universe would not surface here.

Usage:
    python scripts/dry_run_meta_training.py
"""

from __future__ import annotations

# IMPORT ORDER MATTERS. arcticdb must prime its aws-c-common / S3 client
# BEFORE pyarrow loads, or the two bundled aws-c-common copies collide and
# the Arctic S3Storage constructor segfaults on macOS (harmless on Lambda's
# Linux runtime, so production is unaffected). Empirically: if pandas is
# imported before arcticdb.Arctic() is called, the next Arctic S3 operation
# crashes with `aws_fatal_assert: allocator != ((void*)0)`.
#
# Priming a dummy Arctic instance here guarantees arcticdb's S3 client
# initializes cleanly before any downstream pandas/pyarrow imports run.
import os as _os  # noqa: E402
import arcticdb as _adb  # noqa: E402
_adb.Arctic(
    f"s3s://s3.{_os.environ.get('AWS_REGION', 'us-east-1')}.amazonaws.com:"
    f"{_os.environ.get('ALPHA_ENGINE_BUCKET', 'alpha-engine-research')}"
    f"?path_prefix=arcticdb&aws_auth=true"
).list_libraries()

import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

BUCKET = os.environ.get("ALPHA_ENGINE_BUCKET", "alpha-engine-research")
ARCTIC_PREFIX = "arcticdb"
N_TICKERS = int(os.environ.get("N_TICKERS", "100"))
MACRO_KEYS = ["SPY", "VIX", "VIX3M", "TNX", "IRX"]


def pull_from_arcticdb(local_dir: Path) -> int:
    """Download 100 random tickers + all macro series + sector_map.json
    from ArcticDB to a local directory. ArcticDB is the authoritative source
    for training features — the universe parquets carry the 15 pre-computed
    feature columns meta_trainer requires (atr_14_pct, dist_from_52w_*, etc.)
    that the legacy S3 parquet cache lacks."""
    import numpy as np
    import pandas as pd
    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = f"s3s://s3.{region}.amazonaws.com:{BUCKET}?path_prefix={ARCTIC_PREFIX}&aws_auth=true"
    arctic = _adb.Arctic(uri)
    universe = arctic.get_library("universe")
    macro_lib = arctic.get_library("macro")

    # Universe ticker sample
    all_syms = universe.list_symbols()
    rng = np.random.default_rng(42)
    n = min(N_TICKERS, len(all_syms))
    chosen = list(rng.choice(all_syms, size=n, replace=False))
    log.info("ArcticDB universe: %d symbols; sampling %d for dry-run", len(all_syms), n)

    n_written = 0
    t0 = time.time()
    for ticker in chosen:
        try:
            df = universe.read(str(ticker)).data
            if df.empty:
                continue
            df.to_parquet(local_dir / f"{ticker}.parquet", engine="pyarrow", compression="snappy")
            n_written += 1
        except Exception as e:
            log.warning("Failed universe %s: %s", ticker, e)

    # Macro series — all five
    for key in MACRO_KEYS:
        try:
            df = macro_lib.read(key).data
            if df.empty:
                log.warning("Macro %s empty", key)
                continue
            df.to_parquet(local_dir / f"{key}.parquet", engine="pyarrow", compression="snappy")
            n_written += 1
        except Exception as e:
            log.warning("Failed macro %s: %s", key, e)

    # sector_map.json from S3 (not in ArcticDB per arctic_reader.py)
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET, Key="data/sector_map.json")
        (local_dir / "sector_map.json").write_bytes(obj["Body"].read())
    except Exception as e:
        log.warning("Failed sector_map.json: %s", e)

    log.info("Wrote %d files in %.1fs to %s", n_written, time.time() - t0, local_dir)
    return n_written


def main() -> int:
    import numpy as np
    from model.meta_model import META_FEATURES
    from training.meta_trainer import run_meta_training

    tmp_dir = Path(tempfile.mkdtemp(prefix="dry_run_train_"))
    log.info("Temp data dir: %s", tmp_dir)

    n_files = pull_from_arcticdb(tmp_dir)
    if n_files < N_TICKERS:
        log.error("Not enough files downloaded (%d < %d); aborting", n_files, N_TICKERS)
        return 1

    date_str = "2026-04-16-dry-run"
    log.info("Invoking run_meta_training(dry_run=True) ...")
    t0 = time.time()
    try:
        result = run_meta_training(
            data_dir=str(tmp_dir),
            bucket=BUCKET,
            date_str=date_str,
            dry_run=True,
        )
    except Exception as e:
        log.exception("run_meta_training raised: %s", e)
        return 1
    elapsed = time.time() - t0

    log.info("=" * 60)
    log.info("DRY RUN COMPLETE in %.1fs", elapsed)
    log.info("=" * 60)

    # Validate key fields
    failures = []

    # No dangling regime keys
    unexpected_regime = [k for k in result.keys() if k.startswith("regime")]
    if unexpected_regime:
        failures.append(f"result dict still contains regime keys: {unexpected_regime}")

    # Meta model training produced something useful
    meta_ic = result.get("meta_model_ic")
    if meta_ic is None or not np.isfinite(meta_ic):
        failures.append(f"meta_model_ic missing or non-finite: {meta_ic}")

    # Feature schema
    coefs = result.get("meta_coefficients", {})
    coef_features = set(coefs.keys()) - {"intercept"}
    expected_features = set(META_FEATURES)
    unexpected_coefs = coef_features - expected_features
    missing_coefs = expected_features - coef_features
    if unexpected_coefs:
        failures.append(f"meta_coefficients has unexpected features: {unexpected_coefs}")
    if missing_coefs:
        failures.append(f"meta_coefficients missing expected features: {missing_coefs}")

    # Importance populated
    imp = result.get("meta_importance", {})
    if not imp or "standardized_coef" not in imp:
        failures.append("meta_importance not populated")
    else:
        imp_features = set(imp["standardized_coef"].keys())
        if imp_features != expected_features:
            failures.append(
                f"importance feature keys mismatch: "
                f"only-in-imp={imp_features - expected_features}, "
                f"only-in-expected={expected_features - imp_features}"
            )

    # Summary
    log.info("Result summary:")
    log.info("  elapsed:          %.1fs", elapsed)
    log.info("  meta_model_ic:    %.6f", meta_ic or 0)
    log.info("  momentum_test_ic: %.6f", result.get("momentum_test_ic", 0))
    log.info("  volatility_test_ic: %.6f", result.get("volatility_test_ic", 0))
    log.info("  promoted:         %s", result.get("promoted"))
    log.info("  n_train:          %d", result.get("n_train", 0))
    log.info("  meta features:    %d (expected %d)", len(coef_features), len(expected_features))

    if failures:
        log.error("=" * 60)
        log.error("DRY RUN FAILED — %d validation errors", len(failures))
        log.error("=" * 60)
        for f in failures:
            log.error("  - %s", f)
        return 1

    log.info("=" * 60)
    log.info("DRY RUN PASSED — all validations green")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
