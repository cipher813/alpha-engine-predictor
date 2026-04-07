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

# PyTorch is imported lazily — only needed for DataLoader-based functions
# (build_datasets, build_regression_datasets), not for build_regression_arrays.
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

log = logging.getLogger(__name__)


# ── PyTorch-dependent classes (only defined when torch is available) ──────────
# These are used by build_datasets() and build_regression_datasets() which
# return DataLoaders. build_regression_arrays() does NOT need them.

if _HAS_TORCH:

    class PredictorDataset(Dataset):
        """
        PyTorch Dataset wrapping (feature_vector, direction_label) pairs.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
            Feature matrix — technical indicators per sample.
        y : np.ndarray, shape (N,)
            Integer direction labels: 0=DOWN, 1=FLAT, 2=UP.
        """

        def __init__(self, X: np.ndarray, y: np.ndarray, date_indices: np.ndarray | None = None) -> None:
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}"
                )
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int64)
            self.date_indices = date_indices  # for DateGroupedSampler; not returned by __getitem__

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.FloatTensor(self.X[idx]),
                torch.LongTensor([self.y[idx]])[0],
            )


    class RegressionDataset(Dataset):
        """
        PyTorch Dataset wrapping (feature_vector, forward_return) pairs for
        regression-mode training (directly predicting continuous 5-day returns).

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
            Feature matrix — technical indicators per sample.
        y : np.ndarray, shape (N,)
            Continuous forward return values (float32).
        date_indices : np.ndarray or None
            Integer date group IDs for DateGroupedSampler. Not returned by __getitem__.
        """

        def __init__(self, X: np.ndarray, y: np.ndarray, date_indices: np.ndarray | None = None) -> None:
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}"
                )
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
            self.date_indices = date_indices  # for DateGroupedSampler; not returned by __getitem__

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.FloatTensor(self.X[idx]),
                torch.FloatTensor([self.y[idx]])[0],
            )


    class DateGroupedSampler(Sampler):
        """
        Batch sampler that yields all sample indices for a single trading date
        as one batch. Aligns mini-batch ICLoss with true cross-sectional IC.

        Each batch contains all stocks from one date (~770-900 samples).
        Date order is shuffled for training; kept sequential for val/test.

        Use with DataLoader(batch_sampler=...) — mutually exclusive with
        batch_size, shuffle, sampler, and drop_last.
        """

        def __init__(self, date_indices: np.ndarray, shuffle: bool = True, seed: int = 42) -> None:
            self.shuffle = shuffle
            self.seed = seed
            self._epoch = 0

            # Build mapping: date_group_id → list of sample indices
            self._date_to_indices: dict[int, list[int]] = {}
            for sample_idx, date_id in enumerate(date_indices):
                self._date_to_indices.setdefault(int(date_id), []).append(sample_idx)
            self._date_ids = sorted(self._date_to_indices.keys())

        def __iter__(self):
            date_order = list(self._date_ids)
            if self.shuffle:
                rng = np.random.RandomState(self.seed + self._epoch)
                rng.shuffle(date_order)
            for date_id in date_order:
                yield self._date_to_indices[date_id]

        def __len__(self) -> int:
            return len(self._date_ids)

        def set_epoch(self, epoch: int) -> None:
            """Set epoch for deterministic per-epoch shuffling."""
            self._epoch = epoch


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


def build_datasets(
    data_dir: str,
    config_module,
    norm_stats_path: str = "data/norm_stats.json",
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Build train/val/test DataLoaders from parquet files in data_dir.

    Steps:
    1. Load SPY.parquet for relative-return features and labels.
    2. Load all other .parquet files from data_dir.
    3. Compute features + labels (relative to SPY) for each ticker.
    4. Concatenate all samples into a single array, sorted by date.
    5. Time-based split: first 70% → train, next 15% → val, last 15% → test.
    6. Z-score normalize features using train-set statistics.
    7. Save norm stats to norm_stats_path.
    8. Return (train_loader, val_loader, test_loader, test_forward_returns).

    Parameters
    ----------
    data_dir : str
        Path to directory containing per-ticker parquet files.
    config_module :
        The config module (or any object) with BATCH_SIZE, TRAIN_FRAC,
        VAL_FRAC, FEATURES, FORWARD_DAYS, UP_THRESHOLD, DOWN_THRESHOLD attrs.
    norm_stats_path : str
        Where to save the normalization statistics JSON.

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader, test_forward_returns)
        test_forward_returns : np.ndarray, shape (n_test,)
            The forward_return_5d values for each test sample (relative to SPY
            when SPY data is available, absolute otherwise). Used to compute
            Pearson IC in train.py.
    """
    from data.feature_engineer import compute_features
    from data.label_generator import compute_labels

    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir}. "
            "Run data/bootstrap_fetcher.py first."
        )

    # ── Load SPY for relative features and labels ─────────────────────────────
    spy_series: pd.Series | None = None
    spy_path = data_path / "SPY.parquet"
    if spy_path.exists():
        spy_df = _load_ticker_parquet(spy_path)
        if not spy_df.empty and "Close" in spy_df.columns:
            spy_series = spy_df["Close"].astype(float)
            log.info("SPY loaded (%d rows) — using relative returns for features and labels", len(spy_series))
    else:
        log.warning(
            "SPY.parquet not found in %s — return_vs_spy_5d=0.0, labels=absolute return. "
            "Re-run bootstrap with SPY in the ticker list.",
            data_dir,
        )

    # ── Load VIX for market-regime feature ────────────────────────────────────
    vix_series: pd.Series | None = None
    vix_path = data_path / "VIX.parquet"
    if vix_path.exists():
        vix_df = _load_ticker_parquet(vix_path)
        if not vix_df.empty and "Close" in vix_df.columns:
            vix_series = vix_df["Close"].astype(float)
            log.info("VIX loaded (%d rows) — vix_level feature enabled", len(vix_series))
    else:
        log.warning("VIX.parquet not found — vix_level will be 1.0 (neutral). Run bootstrap to download.")

    # ── Load macro rate / commodity series (v1.3 features) ───────────────────
    def _load_close_series_cls(filename: str) -> pd.Series | None:
        p = data_path / filename
        if not p.exists():
            log.warning("%s not found — related macro feature will use neutral default.", filename)
            return None
        df = _load_ticker_parquet(p)
        if df.empty or "Close" not in df.columns:
            return None
        return df["Close"].astype(float)

    tnx_series = _load_close_series_cls("TNX.parquet")
    irx_series = _load_close_series_cls("IRX.parquet")
    gld_series = _load_close_series_cls("GLD.parquet")
    uso_series = _load_close_series_cls("USO.parquet")

    # ── Load sector map and sector ETF close series ───────────────────────────
    # sector_map: ticker → ETF symbol (e.g. "AAPL" → "XLK")
    sector_map: dict[str, str] = {}
    sector_map_path = data_path / "sector_map.json"
    if sector_map_path.exists():
        sector_map = json.loads(sector_map_path.read_text())
        log.info("Sector map loaded: %d ticker→ETF mappings", len(sector_map))
    else:
        log.warning("sector_map.json not found — sector_vs_spy_5d=0.0 for all tickers.")

    # Pre-load all sector ETF close series into a dict for fast per-ticker lookup
    sector_etf_cache: dict[str, pd.Series] = {}
    for etf_symbol in set(sector_map.values()):
        etf_path = data_path / f"{etf_symbol}.parquet"
        if etf_path.exists():
            etf_df = _load_ticker_parquet(etf_path)
            if not etf_df.empty and "Close" in etf_df.columns:
                sector_etf_cache[etf_symbol] = etf_df["Close"].astype(float)
    if sector_etf_cache:
        log.info("Loaded %d sector ETF series: %s", len(sector_etf_cache), sorted(sector_etf_cache))

    # Reference tickers that should not be treated as universe stocks
    _SKIP_TICKERS = {
        "SPY", "VIX",
        "TNX", "IRX", "GLD", "USO",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    }

    log.info("Loading %d parquet files from %s", len(parquet_files), data_dir)

    # (date, features, label, forward_return_5d)
    all_rows: list[tuple[pd.Timestamp, np.ndarray, int, float]] = []

    for i, path in enumerate(parquet_files):
        # Skip market reference files (SPY, VIX, macro series, sector ETFs)
        ticker_name = path.stem
        if ticker_name in _SKIP_TICKERS:
            continue

        raw_df = _load_ticker_parquet(path)
        if raw_df.empty or len(raw_df) < 265:
            # Need at least 252 rows for dist_from_52w_high + 5 forward days + buffer
            continue

        # Resolve sector ETF for this ticker (None → sector_vs_spy_5d = 0.0)
        sector_etf_sym = sector_map.get(ticker_name)
        sector_etf_series = sector_etf_cache.get(sector_etf_sym) if sector_etf_sym else None

        try:
            featured_df = compute_features(
                raw_df,
                spy_series=spy_series,
                vix_series=vix_series,
                sector_etf_series=sector_etf_series,
                tnx_series=tnx_series,
                irx_series=irx_series,
                gld_series=gld_series,
                uso_series=uso_series,
            )
            labeled_df = compute_labels(
                featured_df,
                forward_days=config_module.FORWARD_DAYS,
                up_threshold=config_module.UP_THRESHOLD,
                down_threshold=config_module.DOWN_THRESHOLD,
                benchmark_returns=sector_etf_series if sector_etf_series is not None else spy_series,
            )
        except Exception as exc:
            log.warning("Feature/label computation failed for %s: %s", path.name, exc)
            continue

        if labeled_df.empty:
            continue

        features_arr = labeled_df[config_module.FEATURES].to_numpy(dtype=np.float32)
        labels_arr = labeled_df["direction_int"].to_numpy(dtype=np.int64)
        fwd_returns_arr = labeled_df["forward_return_5d"].to_numpy(dtype=np.float32)
        dates = labeled_df.index

        for j in range(len(dates)):
            all_rows.append((dates[j], features_arr[j], labels_arr[j], float(fwd_returns_arr[j])))

        if (i + 1) % 50 == 0:
            log.info("  Processed %d / %d tickers (%d samples so far)", i + 1, len(parquet_files), len(all_rows))

    if not all_rows:
        raise ValueError(
            "No valid samples were generated. Check parquet files and feature computation."
        )

    # ── Sort all samples by date (time-based split requires this) ─────────────
    all_rows.sort(key=lambda r: r[0])
    all_dates = [r[0] for r in all_rows]
    X_all = np.stack([r[1] for r in all_rows], axis=0)
    y_all = np.array([r[2] for r in all_rows], dtype=np.int64)
    fwd_all = np.array([r[3] for r in all_rows], dtype=np.float32)

    N = len(y_all)
    log.info("Total samples: %d (date range: %s → %s)", N, all_dates[0], all_dates[-1])

    # ── Time-based split with purge gaps ────────────────────────────────────
    # Forward-return labels use FORWARD_DAYS of future prices. Without a
    # purge gap, the last train label and first val label share 4 of 5
    # price days (~80% overlap). We skip FORWARD_DAYS unique dates at each
    # boundary to eliminate this information leakage.
    purge_days = getattr(config_module, "FORWARD_DAYS", 5)
    n_train = int(N * config_module.TRAIN_FRAC)
    n_val_raw = int(N * config_module.VAL_FRAC)

    # --- Purge gap 1: train → val ---
    train_end_date = all_dates[n_train - 1]
    unique_post_train = sorted(set(d for d in all_dates[n_train:] if d > train_end_date))
    if len(unique_post_train) >= purge_days:
        purge_cutoff_1 = unique_post_train[purge_days - 1]
        val_start = next(i for i in range(n_train, N) if all_dates[i] > purge_cutoff_1)
    else:
        val_start = n_train

    # --- Val slice ---
    val_end = val_start + n_val_raw
    if val_end > N:
        val_end = N

    # --- Purge gap 2: val → test ---
    val_end_date = all_dates[min(val_end - 1, N - 1)]
    unique_post_val = sorted(set(d for d in all_dates[val_end:] if d > val_end_date))
    if len(unique_post_val) >= purge_days:
        purge_cutoff_2 = unique_post_val[purge_days - 1]
        test_start = next(i for i in range(val_end, N) if all_dates[i] > purge_cutoff_2)
    else:
        test_start = val_end

    n_purged_1 = val_start - n_train
    n_purged_2 = test_start - val_end
    log.info(
        "Purge gaps: train_end=%s → val_start=%s (%d samples, %d dates skipped) | "
        "val_end=%s → test_start=%s (%d samples, %d dates skipped)",
        train_end_date, all_dates[val_start] if val_start < N else "END",
        n_purged_1, purge_days,
        val_end_date, all_dates[test_start] if test_start < N else "END",
        n_purged_2, purge_days,
    )

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]

    X_val = X_all[val_start:val_end]
    y_val = y_all[val_start:val_end]

    X_test = X_all[test_start:]
    y_test = y_all[test_start:]
    fwd_test = fwd_all[test_start:]

    log.info(
        "Split: train=%d  val=%d  test=%d  (purged=%d)",
        len(y_train),
        len(y_val),
        len(y_test),
        n_purged_1 + n_purged_2,
    )

    # ── Z-score normalization (fit on train, apply to all) ────────────────────
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)  # avoid divide-by-zero

    X_train_norm = (X_train - feat_mean) / feat_std
    X_val_norm = (X_val - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std

    # ── Save normalization statistics ─────────────────────────────────────────
    norm_stats = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "features": config_module.FEATURES,
        "n_train_samples": int(n_train),
    }
    norm_path = Path(norm_stats_path)
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.write_text(json.dumps(norm_stats, indent=2))
    log.info("Norm stats saved to %s", norm_stats_path)

    # ── Build datasets and loaders ────────────────────────────────────────────
    batch_size = config_module.BATCH_SIZE
    train_dataset = PredictorDataset(X_train_norm, y_train)
    val_dataset = PredictorDataset(X_val_norm, y_val)
    test_dataset = PredictorDataset(X_test_norm, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # OK to shuffle within-split; temporal order is preserved by the split boundary
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, fwd_test


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


def build_regression_datasets(
    data_dir: str,
    config_module,
    norm_stats_path: str = "data/norm_stats.json",
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Build train/val/test DataLoaders for regression-mode training.

    Identical pipeline to build_datasets() but returns RegressionDataset
    instances where labels are continuous forward_return_5d values (float32)
    rather than integer direction classes.  Huber/MSE loss is applied directly
    to the continuous return target, which aligns the training objective with
    the Pearson IC evaluation metric.

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader, test_forward_returns)
        test_forward_returns : np.ndarray, shape (n_test,)
            Same as build_datasets() — used to compute IC in train.py.
    """
    from data.feature_engineer import compute_features
    from data.label_generator import compute_labels

    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir}. "
            "Run data/bootstrap_fetcher.py first."
        )

    # ── Load reference series (SPY, VIX, macro, sector ETFs) ──────────────────
    spy_series: pd.Series | None = None
    spy_path = data_path / "SPY.parquet"
    if spy_path.exists():
        spy_df = _load_ticker_parquet(spy_path)
        if not spy_df.empty and "Close" in spy_df.columns:
            spy_series = spy_df["Close"].astype(float)
            log.info("SPY loaded (%d rows)", len(spy_series))

    vix_series: pd.Series | None = None
    vix_path = data_path / "VIX.parquet"
    if vix_path.exists():
        vix_df = _load_ticker_parquet(vix_path)
        if not vix_df.empty and "Close" in vix_df.columns:
            vix_series = vix_df["Close"].astype(float)

    def _load_close_series(filename: str) -> pd.Series | None:
        p = data_path / filename
        if not p.exists():
            log.warning("%s not found — related macro feature will use neutral default.", filename)
            return None
        df = _load_ticker_parquet(p)
        if df.empty or "Close" not in df.columns:
            return None
        return df["Close"].astype(float)

    tnx_series = _load_close_series("TNX.parquet")
    irx_series = _load_close_series("IRX.parquet")
    gld_series = _load_close_series("GLD.parquet")
    uso_series = _load_close_series("USO.parquet")

    sector_map: dict[str, str] = {}
    sector_map_path = data_path / "sector_map.json"
    if sector_map_path.exists():
        sector_map = json.loads(sector_map_path.read_text())
        log.info("Sector map loaded: %d ticker→ETF mappings", len(sector_map))
    else:
        log.warning("sector_map.json not found — labels fall back to SPY-relative.")

    sector_etf_cache: dict[str, pd.Series] = {}
    for etf_symbol in set(sector_map.values()):
        etf_path = data_path / f"{etf_symbol}.parquet"
        if etf_path.exists():
            etf_df = _load_ticker_parquet(etf_path)
            if not etf_df.empty and "Close" in etf_df.columns:
                sector_etf_cache[etf_symbol] = etf_df["Close"].astype(float)
    if sector_etf_cache:
        log.info(
            "Loaded %d sector ETF series for sector-neutral labels: %s",
            len(sector_etf_cache), sorted(sector_etf_cache),
        )

    _SKIP_TICKERS = {
        "SPY", "VIX",
        "TNX", "IRX", "GLD", "USO",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    }

    log.info("Regression mode: loading %d parquet files from %s", len(parquet_files), data_dir)
    all_rows: list[tuple[pd.Timestamp, np.ndarray, float]] = []

    for i, path in enumerate(parquet_files):
        ticker_name = path.stem
        if ticker_name in _SKIP_TICKERS:
            continue

        raw_df = _load_ticker_parquet(path)
        if raw_df.empty or len(raw_df) < 265:
            continue

        sector_etf_sym = sector_map.get(ticker_name)
        sector_etf_series = sector_etf_cache.get(sector_etf_sym) if sector_etf_sym else None

        try:
            featured_df = compute_features(
                raw_df,
                spy_series=spy_series,
                vix_series=vix_series,
                sector_etf_series=sector_etf_series,
                tnx_series=tnx_series,
                irx_series=irx_series,
                gld_series=gld_series,
                uso_series=uso_series,
            )
            labeled_df = compute_labels(
                featured_df,
                forward_days=config_module.FORWARD_DAYS,
                up_threshold=config_module.UP_THRESHOLD,
                down_threshold=config_module.DOWN_THRESHOLD,
                benchmark_returns=sector_etf_series if sector_etf_series is not None else spy_series,
            )
        except Exception as exc:
            log.warning("Feature/label computation failed for %s: %s", path.name, exc)
            continue

        if labeled_df.empty:
            continue

        features_arr = labeled_df[config_module.FEATURES].to_numpy(dtype=np.float32)
        fwd_returns_arr = labeled_df["forward_return_5d"].to_numpy(dtype=np.float32)
        dates = labeled_df.index

        for j in range(len(dates)):
            all_rows.append((dates[j], features_arr[j], float(fwd_returns_arr[j])))

        if (i + 1) % 50 == 0:
            log.info("  Processed %d / %d tickers (%d samples so far)", i + 1, len(parquet_files), len(all_rows))

    if not all_rows:
        raise ValueError("No valid samples were generated.")

    all_rows.sort(key=lambda r: r[0])
    all_dates = [r[0] for r in all_rows]
    X_all = np.stack([r[1] for r in all_rows], axis=0)
    fwd_all = np.array([r[2] for r in all_rows], dtype=np.float32)

    # ── Label winsorization ───────────────────────────────────────────────────
    # Clip extreme forward returns to reduce gradient noise from earnings gaps,
    # M&A events, and biotech FDA moves that can be ±30–50% over 5 days.
    # These outliers distort ICLoss away from the typical ±2–4% signal range.
    label_clip = getattr(config_module, "LABEL_CLIP", None)
    if label_clip is not None:
        n_clipped = int((np.abs(fwd_all) > label_clip).sum())
        pct_clipped = 100.0 * n_clipped / max(len(fwd_all), 1)
        fwd_all = np.clip(fwd_all, -label_clip, label_clip)
        log.info(
            "Label winsorization (±%.0f%%): clipped %d / %d samples (%.2f%%)",
            label_clip * 100, n_clipped, len(fwd_all), pct_clipped,
        )
        p01, p05, p95, p99 = np.percentile(fwd_all, [1, 5, 95, 99])
        log.info(
            "Label distribution after clip — p1=%.3f  p5=%.3f  p95=%.3f  p99=%.3f",
            p01, p05, p95, p99,
        )

    # Build integer date-group IDs for DateGroupedSampler
    unique_dates = sorted(set(all_dates))
    date_to_id = {d: i for i, d in enumerate(unique_dates)}
    date_ids_all = np.array([date_to_id[d] for d in all_dates], dtype=np.int64)

    N = len(fwd_all)
    log.info("Regression dataset: %d samples, %d unique dates", N, len(unique_dates))

    # Time-based split with date-boundary alignment and purge gaps.
    # Forward-return labels use FORWARD_DAYS of future prices. Without a
    # purge gap, the last train label and first val label share 4 of 5
    # price days (~80% overlap). We skip FORWARD_DAYS unique dates at each
    # boundary to eliminate this information leakage.
    purge_days = getattr(config_module, "FORWARD_DAYS", 5)

    n_train = int(N * config_module.TRAIN_FRAC)
    n_val_raw = int(N * config_module.VAL_FRAC)

    # Advance train boundary to date boundary (don't split a date across sets)
    while n_train < N and all_dates[n_train] == all_dates[n_train - 1]:
        n_train += 1

    # --- Purge gap 1: train → val ---
    train_end_date = all_dates[n_train - 1]
    unique_post_train = sorted(set(d for d in all_dates[n_train:] if d > train_end_date))
    if len(unique_post_train) >= purge_days:
        purge_cutoff_1 = unique_post_train[purge_days - 1]
        val_start = next(i for i in range(n_train, N) if all_dates[i] > purge_cutoff_1)
    else:
        val_start = n_train

    # --- Val slice with date-boundary alignment ---
    val_end = val_start + n_val_raw
    if val_end > N:
        val_end = N
    while val_end < N and all_dates[val_end] == all_dates[val_end - 1]:
        val_end += 1

    # --- Purge gap 2: val → test ---
    val_end_date = all_dates[min(val_end - 1, N - 1)]
    unique_post_val = sorted(set(d for d in all_dates[val_end:] if d > val_end_date))
    if len(unique_post_val) >= purge_days:
        purge_cutoff_2 = unique_post_val[purge_days - 1]
        test_start = next(i for i in range(val_end, N) if all_dates[i] > purge_cutoff_2)
    else:
        test_start = val_end

    n_purged_1 = val_start - n_train
    n_purged_2 = test_start - val_end
    log.info(
        "Purge gaps: train_end=%s → val_start=%s (%d samples, %d dates skipped) | "
        "val_end=%s → test_start=%s (%d samples, %d dates skipped)",
        train_end_date, all_dates[val_start] if val_start < N else "END",
        n_purged_1, purge_days,
        val_end_date, all_dates[test_start] if test_start < N else "END",
        n_purged_2, purge_days,
    )

    X_train = X_all[:n_train]
    fwd_train = fwd_all[:n_train]
    X_val = X_all[val_start:val_end]
    fwd_val = fwd_all[val_start:val_end]
    X_test = X_all[test_start:]
    fwd_test = fwd_all[test_start:]

    # Split and re-index date IDs within each set (0-based contiguous)
    def _reindex(ids: np.ndarray) -> np.ndarray:
        uniq = np.unique(ids)
        m = {old: new for new, old in enumerate(uniq)}
        return np.array([m[x] for x in ids], dtype=np.int64)

    date_ids_train = _reindex(date_ids_all[:n_train])
    date_ids_val = _reindex(date_ids_all[val_start:val_end])
    date_ids_test = _reindex(date_ids_all[test_start:])

    log.info(
        "Regression split: train=%d  val=%d  test=%d  (purged=%d)",
        len(fwd_train), len(fwd_val), len(fwd_test),
        n_purged_1 + n_purged_2,
    )

    # Z-score normalize (fit on train, apply to all)
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)

    X_train_norm = (X_train - feat_mean) / feat_std
    X_val_norm = (X_val - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std

    # Save norm stats
    norm_stats = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "features": config_module.FEATURES,
        "n_train_samples": int(n_train),
        "mode": "regression",
    }
    norm_path = Path(norm_stats_path)
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.write_text(json.dumps(norm_stats, indent=2))
    log.info("Norm stats saved to %s (regression mode)", norm_stats_path)

    # Date-grouped DataLoaders: each batch = all stocks on one trading day
    train_sampler = DateGroupedSampler(date_ids_train, shuffle=True)
    val_sampler = DateGroupedSampler(date_ids_val, shuffle=False)
    test_sampler = DateGroupedSampler(date_ids_test, shuffle=False)

    log.info(
        "DateGroupedSampler: train=%d dates  val=%d dates  test=%d dates  avg_batch=%.0f",
        len(train_sampler), len(val_sampler), len(test_sampler),
        len(fwd_train) / max(len(train_sampler), 1),
    )

    train_loader = DataLoader(
        RegressionDataset(X_train_norm, fwd_train, date_indices=date_ids_train),
        batch_sampler=train_sampler, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        RegressionDataset(X_val_norm, fwd_val, date_indices=date_ids_val),
        batch_sampler=val_sampler, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        RegressionDataset(X_test_norm, fwd_test, date_indices=date_ids_test),
        batch_sampler=test_sampler, num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, test_loader, fwd_test


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
