"""
data/dataset.py — PyTorch Dataset and DataLoader construction.

Loads all parquet files from a local cache directory, computes features and
labels for each ticker, concatenates all samples, performs a time-based
70/15/15 train/val/test split, z-score normalizes features (fit on train only),
and returns DataLoaders ready for training.

SPY handling: SPY.parquet is loaded first from data_dir and its Close series
is passed to both compute_features() (for return_vs_spy_5d) and compute_labels()
(for relative return labeling). If SPY.parquet is absent, features fall back to
return_vs_spy_5d=0.0 and labels fall back to absolute returns.

VIX / sector ETF handling: VIX.parquet and sector ETF parquets (XLK, XLF, etc.)
are loaded when present. sector_map.json maps each ticker to its sector ETF.
If these files are absent the corresponding features default to neutral values.

Normalization statistics are saved to data/norm_stats.json so they can be
loaded at inference time to normalize live feature vectors consistently.

build_datasets() returns a 4-tuple: (train_loader, val_loader, test_loader,
test_forward_returns) where test_forward_returns is a float32 numpy array of
the forward return for each test sample — used to compute Pearson IC in train.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)



def _parquet_engine() -> str:
    """Return best available parquet engine (pyarrow preferred, fastparquet fallback)."""
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except ImportError:
        return "fastparquet"


def _load_ticker_parquet(path: Path) -> pd.DataFrame:
    """Load a single parquet file and return its DataFrame. Returns empty on error."""
    try:
        df = pd.read_parquet(path, engine=_parquet_engine())
        idx = pd.to_datetime(df.index)
        # Always normalize to timezone-naive UTC so compute_features' reindex() calls
        # don't raise TypeError when mixing pyarrow-written (tz-naive) and
        # fastparquet-written (potentially tz-aware) parquets.
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df.index = idx
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        # Deduplicate on date index. Mirrors inference's defensive handling
        # at inference/stages/load_prices.py:403. 2026-04-15 smoke tests
        # showed 904/909 tickers failing feature computation with
        # "cannot reindex on an axis with duplicate labels" — ArcticDB reads
        # are emitting same-date rows for essentially every ticker. Upstream
        # fix belongs in alpha-engine-data's ArcticDB write path; this is
        # the consistency fix that brings training into alignment with the
        # inference path that already expects and handles the duplicates.
        if df.index.has_duplicates:
            n_before = len(df)
            df = df[~df.index.duplicated(keep="last")]
            log.warning(
                "Deduplicated %d duplicate date rows in %s (kept last: %d → %d). "
                "Upstream ArcticDB write is emitting duplicates — file against alpha-engine-data.",
                n_before - len(df), path.name, n_before, len(df),
            )
        return df
    except Exception as exc:
        log.warning("Failed to load %s: %s", path.name, exc)
        return pd.DataFrame()


def cross_sectional_rank_normalize(
    X: np.ndarray,
    dates: list,
) -> np.ndarray:
    """
    Rank-normalize each feature within each date's cross-section.

    For each unique date, ranks all tickers' values for each feature
    and scales to (0, 1) percentile rank.  This makes features comparable
    across time — a percentile of 0.9 means the same thing in 2018 and 2024.

    No temporal leakage: ranks are computed per-date (cross-sectional only).

    Parameters
    ----------
    X : shape (N, n_features) — raw feature values, sorted by date
    dates : list of pd.Timestamp, length N, sorted ascending

    Returns
    -------
    X_ranked : shape (N, n_features) — rank-normalized features in (0, 1)
    """
    # Build date → row-indices mapping
    date_to_indices: dict[object, list[int]] = {}
    for i, d in enumerate(dates):
        date_to_indices.setdefault(d, []).append(i)

    X_ranked = X.copy()
    n_features = X.shape[1]

    for indices in date_to_indices.values():
        n = len(indices)
        if n <= 1:
            # Single ticker on this date: assign midpoint percentile
            X_ranked[indices[0], :] = 0.5
            continue
        idx_arr = np.array(indices)
        for f in range(n_features):
            vals = X[idx_arr, f]
            # argsort-of-argsort gives ranks (0-based); average ties
            order = vals.argsort()
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(n, dtype=np.float32)
            # Handle ties: average the ranks for equal values
            unique_vals, inverse = np.unique(vals, return_inverse=True)
            if len(unique_vals) < n:
                for uv_idx in range(len(unique_vals)):
                    mask = inverse == uv_idx
                    if mask.sum() > 1:
                        ranks[mask] = ranks[mask].mean()
            # Scale to (0, 1): rank / (n - 1) maps to [0, 1]
            X_ranked[idx_arr, f] = ranks / max(n - 1, 1)

    return X_ranked

def load_norm_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load saved normalization statistics from a JSON file.

    Parameters
    ----------
    path : str
        Path to norm_stats.json (written by build_datasets).

    Returns
    -------
    (mean, std) — both np.ndarray of shape (N_FEATURES,).
    """
    norm_path = Path(path)
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found at {path}. "
            "Run build_datasets() first or load from a model checkpoint."
        )
    stats = json.loads(norm_path.read_text())
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    return mean, std
