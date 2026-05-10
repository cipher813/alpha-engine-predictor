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

            # NaN-aware ranking: rank only the non-NaN values, then
            # assign every NaN entry 0.5 (cross-sectional median rank).
            # ``np.argsort`` places NaN at the END of the sorted order, so
            # the legacy "argsort + arange" path mapped NaN inputs to rank
            # = n-1 → percentile 1.0. That's actively misleading: a
            # short-history ticker with NaN momentum_20d ended up at the
            # top of the cross-section across every NaN feature
            # simultaneously, and the GBM learned a spurious
            # "rank=1.0 cluster" pattern. 2026-05-09 weekly training:
            # momentum subsample IC was -0.17 vs baseline +0.32 on the
            # 467-row short-history slice, blocking promotion despite
            # meta-IC of 0.0764. Treating NaN as "no signal → median
            # rank" puts those rows on the neutral line where they
            # belong, lets the GBM learn from the actual non-NaN
            # population, and lets the meta-Ridge weigh the component
            # honestly.
            nan_mask_f = np.isnan(vals)
            if nan_mask_f.all():
                # Every row is NaN for this feature today (e.g. an alt
                # data feature missing universe-wide on a given date) —
                # neutral 0.5 for everyone, no rank computable.
                X_ranked[idx_arr, f] = 0.5
                continue

            non_nan_idx_local = np.where(~nan_mask_f)[0]
            non_nan_vals = vals[non_nan_idx_local]
            n_non_nan = non_nan_idx_local.size

            if n_non_nan <= 1:
                # Only one non-NaN row — give it the midpoint and the
                # rest 0.5. No meaningful ranking with N=1.
                X_ranked[idx_arr, f] = 0.5
                continue

            order = non_nan_vals.argsort()
            local_ranks = np.empty_like(order, dtype=np.float32)
            local_ranks[order] = np.arange(n_non_nan, dtype=np.float32)
            # Handle ties: average the ranks for equal values within
            # the non-NaN slice.
            unique_vals, inverse = np.unique(non_nan_vals, return_inverse=True)
            if len(unique_vals) < n_non_nan:
                for uv_idx in range(len(unique_vals)):
                    mask = inverse == uv_idx
                    if mask.sum() > 1:
                        local_ranks[mask] = local_ranks[mask].mean()
            # Scale non-NaN ranks to (0, 1).
            scaled_non_nan = local_ranks / max(n_non_nan - 1, 1)

            ranks_full = np.full(n, 0.5, dtype=np.float32)
            ranks_full[non_nan_idx_local] = scaled_non_nan
            X_ranked[idx_arr, f] = ranks_full

    return X_ranked


def time_series_zscore_normalize(
    X: np.ndarray,
    dates: list,
    window: int = 252,
    min_periods: int = 60,
) -> np.ndarray:
    """Point-in-time rolling z-score for time-series-only features.

    For each unique date d, normalizes column values using rolling stats
    from the [d-window, d-1] window — strictly past data, no look-ahead.
    Returns NaN for dates that don't have ``min_periods`` of trailing
    history; downstream LightGBM consumers handle NaN natively.

    Used for MARKET-WIDE features (VIX, yield curve, breadth, etc.) that
    are constant across tickers on a given date — cross-sectional rank-norm
    would degenerate to 0.5 for every row, but time-series z-score
    preserves the cross-temporal signal (regime is high vs low VIX).
    Trees discover interactions with rank-normed per-ticker features
    automatically through sequential splits — institutional SOTA pattern.

    Args:
        X: shape (N, n_features) — raw feature values, sorted by date.
            Macro features are constant across tickers on a date; this
            function deduplicates per-date before computing rolling stats.
        dates: list of pd.Timestamp, length N, sorted ascending
        window: rolling window size (default 252 trading days = 1 year)
        min_periods: minimum trailing periods to compute z-score
            (default 60 = 3 months — gates early dates with thin history)

    Returns:
        X_zscored : shape (N, n_features) — point-in-time rolling z-scores
            in float32. NaN for dates with insufficient history.
    """
    if X.shape[0] == 0:
        return X.astype(np.float32)
    if len(dates) != X.shape[0]:
        raise ValueError(
            f"dates length {len(dates)} does not match X rows {X.shape[0]}"
        )

    # Build a date-indexed DataFrame; deduplicate to first-occurrence per
    # date so the rolling window operates on N_dates rows, not N_total
    # rows (macros are constant within a date — replicating across
    # tickers would distort the rolling mean toward heavily-tickered
    # dates).
    df_full = pd.DataFrame(X, index=pd.DatetimeIndex(dates))
    per_date = df_full.loc[~df_full.index.duplicated(keep="first")].sort_index()

    # Rolling stats use PAST data only via shift(1) — at date d, the
    # rolling window is [d-window, d-1], excluding d itself. No look-ahead.
    rolling = per_date.rolling(window, min_periods=min_periods)
    rolling_mean = rolling.mean().shift(1)
    rolling_std = rolling.std().shift(1)
    z_per_date = (per_date - rolling_mean) / rolling_std

    # Map z-scored values back to every (date, ticker) row in original
    # order. ``reindex(dates)`` broadcasts each unique-date z-score to
    # all rows on that date.
    z_for_each_row = z_per_date.reindex(pd.DatetimeIndex(dates)).to_numpy(
        dtype=np.float32,
    )
    return z_for_each_row


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
