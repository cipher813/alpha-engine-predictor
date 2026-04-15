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


def _compute_xsect_dispersion(data_path: Path) -> pd.Series | None:
    """
    Compute cross-sectional dispersion: std dev of daily returns across tickers.

    Returns a pd.Series indexed by date with the daily cross-sectional std dev.
    High dispersion = stock-picking environment; low = factor-driven.
    """
    _SKIP = {
        "SPY", "VIX", "TNX", "IRX", "GLD", "USO",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        "VIX3M",
    }
    parquets = sorted(data_path.glob("*.parquet"))
    returns_by_date: dict[pd.Timestamp, list[float]] = {}
    for path in parquets:
        ticker = path.stem
        if ticker in _SKIP:
            continue
        try:
            df = _load_ticker_parquet(path)
            if df.empty or "Close" not in df.columns or len(df) < 10:
                continue
            rets = df["Close"].pct_change().dropna()
            for dt, r in rets.items():
                if np.isfinite(r):
                    returns_by_date.setdefault(dt, []).append(r)
        except Exception:
            continue

    if not returns_by_date:
        return None

    dispersion = {}
    for dt, rets in returns_by_date.items():
        if len(rets) >= 10:
            dispersion[dt] = float(np.std(rets))

    if not dispersion:
        return None

    series = pd.Series(dispersion).sort_index()
    log.info("Cross-sectional dispersion computed: %d dates, mean=%.4f",
             len(series), series.mean())
    return series




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


def build_regression_arrays(
    data_dir: str,
    config_module,
    feature_list: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load parquets, compute features + labels, return unsplit arrays.

    Same pipeline as build_regression_datasets() but stops before splitting,
    normalization, or DataLoader construction. Used by walk-forward evaluation
    which performs its own splitting.

    Returns
    -------
    (X_all, fwd_all, all_dates)
        X_all : np.ndarray, shape (N, n_features) — raw features (not normalized)
        fwd_all : np.ndarray, shape (N,) — forward_return_5d (winsorized)
        all_dates : list — one pd.Timestamp per sample, sorted ascending
    """
    from data.feature_engineer import compute_features
    from data.label_generator import compute_labels

    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}.")

    # Load reference series (same logic as build_regression_datasets)
    spy_series = None
    spy_path = data_path / "SPY.parquet"
    if spy_path.exists():
        spy_df = _load_ticker_parquet(spy_path)
        if not spy_df.empty and "Close" in spy_df.columns:
            spy_series = spy_df["Close"].astype(float)

    vix_series = None
    vix_path = data_path / "VIX.parquet"
    if vix_path.exists():
        vix_df = _load_ticker_parquet(vix_path)
        if not vix_df.empty and "Close" in vix_df.columns:
            vix_series = vix_df["Close"].astype(float)

    def _load_close(fn):
        p = data_path / fn
        if not p.exists():
            return None
        d = _load_ticker_parquet(p)
        if d.empty or "Close" not in d.columns:
            return None
        return d["Close"].astype(float)

    tnx_series = _load_close("TNX.parquet")
    irx_series = _load_close("IRX.parquet")
    gld_series = _load_close("GLD.parquet")
    uso_series = _load_close("USO.parquet")

    # VIX3M (3-month VIX) for term structure slope
    vix3m_series = _load_close("VIX3M.parquet")

    # Cross-sectional dispersion: std dev of daily returns across all tickers.
    # Pre-computed as a single pass before per-ticker feature computation.
    xsect_dispersion = _compute_xsect_dispersion(data_path)

    sector_map: dict[str, str] = {}
    sector_map_path = data_path / "sector_map.json"
    if sector_map_path.exists():
        sector_map = json.loads(sector_map_path.read_text())

    sector_etf_cache: dict[str, pd.Series] = {}
    for etf_symbol in set(sector_map.values()):
        etf_path = data_path / f"{etf_symbol}.parquet"
        if etf_path.exists():
            etf_df = _load_ticker_parquet(etf_path)
            if not etf_df.empty and "Close" in etf_df.columns:
                sector_etf_cache[etf_symbol] = etf_df["Close"].astype(float)

    _SKIP = {
        "SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    }

    all_rows: list[tuple] = []
    _feat_cols_needed = feature_list or config_module.FEATURES
    _arcticdb_skipped_compute = 0

    for path in parquet_files:
        ticker_name = path.stem
        if ticker_name in _SKIP:
            continue
        raw_df = _load_ticker_parquet(path)
        if raw_df.empty or len(raw_df) < 265:
            continue
        sector_etf_sym = sector_map.get(ticker_name)
        sector_etf_series = sector_etf_cache.get(sector_etf_sym) if sector_etf_sym else None
        try:
            # If the parquet already has pre-computed features (from ArcticDB),
            # skip compute_features() — just compute labels on the existing data.
            has_precomputed = all(f in raw_df.columns for f in _feat_cols_needed)
            if has_precomputed:
                featured_df = raw_df
                _arcticdb_skipped_compute += 1
            else:
                featured_df = compute_features(
                    raw_df, spy_series=spy_series, vix_series=vix_series,
                    sector_etf_series=sector_etf_series, tnx_series=tnx_series,
                    irx_series=irx_series, gld_series=gld_series, uso_series=uso_series,
                    vix3m_series=vix3m_series, xsect_dispersion=xsect_dispersion,
                )
            labeled_df = compute_labels(
                featured_df,
                forward_days=config_module.FORWARD_DAYS,
                up_threshold=config_module.UP_THRESHOLD,
                down_threshold=config_module.DOWN_THRESHOLD,
                benchmark_returns=sector_etf_series if sector_etf_series is not None else spy_series,
            )
        except Exception:
            continue
        if labeled_df.empty:
            continue
        _feat_cols = feature_list or config_module.FEATURES
        features_arr = labeled_df[_feat_cols].to_numpy(dtype=np.float32)
        fwd_returns_arr = labeled_df["forward_return_5d"].to_numpy(dtype=np.float32)
        dates = labeled_df.index
        for j in range(len(dates)):
            all_rows.append((dates[j], features_arr[j], float(fwd_returns_arr[j])))

    if _arcticdb_skipped_compute:
        log.info("[data_source=arcticdb] %d tickers used pre-computed features — skipped compute_features()", _arcticdb_skipped_compute)

    if not all_rows:
        raise ValueError("No valid samples generated.")

    all_rows.sort(key=lambda r: r[0])
    all_dates = [r[0] for r in all_rows]
    X_all = np.stack([r[1] for r in all_rows], axis=0)
    fwd_all = np.array([r[2] for r in all_rows], dtype=np.float32)

    # Winsorize
    label_clip = getattr(config_module, "LABEL_CLIP", None)
    if label_clip is not None:
        fwd_all = np.clip(fwd_all, -label_clip, label_clip)

    # Cross-sectional rank normalization: per-date, per-feature.
    # Converts raw feature values to percentiles [0, 1] within each day's
    # cross-section of tickers.  No temporal leakage — ranks are per-date only.
    X_all = cross_sectional_rank_normalize(X_all, all_dates)
    log.info("Applied cross-sectional rank normalization (%d unique dates)",
             len(set(all_dates)))

    log.info("build_regression_arrays: %d samples, %d features, %d unique dates",
             len(fwd_all), X_all.shape[1], len(set(all_dates)))
    return X_all, fwd_all, all_dates




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
