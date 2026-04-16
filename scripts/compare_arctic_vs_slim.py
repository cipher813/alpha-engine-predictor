#!/usr/bin/env python3
"""
scripts/compare_arctic_vs_slim.py — Phase 7a parity check.

Reads the same N tickers from both price paths (ArcticDB direct vs slim-cache +
daily_closes delta) and diffs the resulting DataFrames. Used to verify the
cutover is safe before flipping USE_ARCTIC_INFERENCE=1 in production.

Pass criterion: for each ticker, the OHLCV columns over the shared date range
match within tolerance (default 1e-6 absolute, 1e-4 relative). Mismatches are
reported per ticker with the first few differing rows.

Usage:
    python scripts/compare_arctic_vs_slim.py                           # 20 tickers
    python scripts/compare_arctic_vs_slim.py --n 100                   # 100 tickers
    python scripts/compare_arctic_vs_slim.py --tickers AAPL MSFT NVDA  # specific
    python scripts/compare_arctic_vs_slim.py --date 2026-04-15         # historical

Exits 0 if all tickers match, 1 on any divergence. Wire into a CI check or run
ad-hoc before merge.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to sys.path so `import config`, `import inference...` work from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from inference.stages.load_prices import (
    load_price_data_from_arctic,
    load_price_data_from_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("parity")

DEFAULT_SAMPLE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                  "META", "TSLA", "BRK.B", "JPM", "V",
                  "WMT", "UNH", "XOM", "JNJ", "PG",
                  "MA", "HD", "CVX", "ABBV", "LLY"]


def _align_overlap(arctic_df: pd.DataFrame, slim_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice both frames to their shared date range and common columns."""
    shared_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                   if c in arctic_df.columns and c in slim_df.columns]
    shared_index = arctic_df.index.intersection(slim_df.index)
    return arctic_df.loc[shared_index, shared_cols], slim_df.loc[shared_index, shared_cols]


def _diff_report(ticker: str, arctic_df: pd.DataFrame, slim_df: pd.DataFrame,
                 *, atol: float, rtol: float) -> tuple[bool, str]:
    """Return (matches, description)."""
    if arctic_df.empty or slim_df.empty:
        return False, f"{ticker}: empty frame — arctic={len(arctic_df)} slim={len(slim_df)}"

    a, s = _align_overlap(arctic_df, slim_df)
    if a.empty:
        return False, f"{ticker}: no overlapping date range or columns"

    diff_mask = ~np.isclose(a.values, s.values, atol=atol, rtol=rtol, equal_nan=True)
    if not diff_mask.any():
        return True, f"{ticker}: {len(a)} rows × {len(a.columns)} cols match"

    n_bad_cells = int(diff_mask.sum())
    n_bad_rows  = int(diff_mask.any(axis=1).sum())
    # Find the first diverging row for context
    first_bad_row = a.index[diff_mask.any(axis=1)][0]
    first_bad_a = a.loc[first_bad_row].to_dict()
    first_bad_s = s.loc[first_bad_row].to_dict()
    return False, (
        f"{ticker}: {n_bad_cells} cell diffs across {n_bad_rows} rows. "
        f"First diff at {first_bad_row.date()}: arctic={first_bad_a} slim={first_bad_s}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="sample size from default list")
    p.add_argument("--tickers", nargs="+", help="override sample with explicit list")
    p.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "alpha-engine-research"))
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-4)
    args = p.parse_args()

    tickers = args.tickers or DEFAULT_SAMPLE[: args.n]
    log.info("Parity check: %d tickers, date=%s, bucket=%s", len(tickers), args.date, args.bucket)

    log.info("Reading from ArcticDB ...")
    arctic_prices, arctic_macro = load_price_data_from_arctic(tickers, args.date, args.bucket)

    log.info("Reading from slim cache ...")
    slim_prices, slim_macro = load_price_data_from_cache(tickers, args.date, args.bucket)
    if slim_prices is None:
        log.error("Slim cache returned None — cannot compare. Check S3 access.")
        return 1

    # Per-ticker comparison
    ok = 0
    bad = []
    for ticker in tickers:
        if ticker not in arctic_prices:
            bad.append(f"{ticker}: missing from ArcticDB output")
            continue
        if ticker not in slim_prices:
            bad.append(f"{ticker}: missing from slim cache output")
            continue
        matches, msg = _diff_report(
            ticker, arctic_prices[ticker], slim_prices[ticker],
            atol=args.atol, rtol=args.rtol,
        )
        if matches:
            log.info("OK   %s", msg)
            ok += 1
        else:
            log.warning("DIFF %s", msg)
            bad.append(msg)

    # Macro comparison
    shared_macro_keys = set(arctic_macro) & set(slim_macro)
    for key in sorted(shared_macro_keys):
        a = arctic_macro[key].dropna()
        s = slim_macro[key].dropna()
        shared_idx = a.index.intersection(s.index)
        if shared_idx.empty:
            bad.append(f"macro[{key}]: no overlapping dates")
            continue
        if np.allclose(a.loc[shared_idx], s.loc[shared_idx], atol=args.atol, rtol=args.rtol):
            log.info("OK   macro[%s]: %d rows match", key, len(shared_idx))
        else:
            bad.append(f"macro[{key}]: values diverge across {len(shared_idx)} shared dates")

    log.info("─" * 60)
    log.info("Summary: %d/%d tickers match, %d divergences", ok, len(tickers), len(bad))
    if bad:
        log.error("Divergences:")
        for m in bad:
            log.error("  %s", m)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
