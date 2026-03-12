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
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

log = logging.getLogger(__name__)


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


def _load_ticker_parquet(path: Path) -> pd.DataFrame:
    """Load a single parquet file and return its DataFrame. Returns empty on error."""
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df
    except Exception as exc:
        log.warning("Failed to load %s: %s", path.name, exc)
        return pd.DataFrame()


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
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    }

    log.info("Loading %d parquet files from %s", len(parquet_files), data_dir)

    # (date, features, label, forward_return_5d)
    all_rows: list[tuple[pd.Timestamp, np.ndarray, int, float]] = []

    for i, path in enumerate(parquet_files):
        # Skip market reference files (SPY, VIX, sector ETFs)
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
            )
            labeled_df = compute_labels(
                featured_df,
                forward_days=config_module.FORWARD_DAYS,
                up_threshold=config_module.UP_THRESHOLD,
                down_threshold=config_module.DOWN_THRESHOLD,
                spy_returns=spy_series,
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

    # ── Time-based split ──────────────────────────────────────────────────────
    n_train = int(N * config_module.TRAIN_FRAC)
    n_val = int(N * config_module.VAL_FRAC)
    # test gets the remainder (most recent data)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]

    X_val = X_all[n_train : n_train + n_val]
    y_val = y_all[n_train : n_train + n_val]

    X_test = X_all[n_train + n_val :]
    y_test = y_all[n_train + n_val :]
    fwd_test = fwd_all[n_train + n_val :]

    log.info(
        "Split: train=%d  val=%d  test=%d",
        len(y_train),
        len(y_val),
        len(y_test),
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

    # ── Load reference series (SPY, VIX, sector ETFs) ─────────────────────────
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

    _SKIP_TICKERS = {
        "SPY", "VIX",
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
            )
            labeled_df = compute_labels(
                featured_df,
                forward_days=config_module.FORWARD_DAYS,
                up_threshold=config_module.UP_THRESHOLD,
                down_threshold=config_module.DOWN_THRESHOLD,
                spy_returns=spy_series,
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

    # Build integer date-group IDs for DateGroupedSampler
    unique_dates = sorted(set(all_dates))
    date_to_id = {d: i for i, d in enumerate(unique_dates)}
    date_ids_all = np.array([date_to_id[d] for d in all_dates], dtype=np.int64)

    N = len(fwd_all)
    log.info("Regression dataset: %d samples, %d unique dates", N, len(unique_dates))

    # Time-based split with date-boundary alignment
    n_train = int(N * config_module.TRAIN_FRAC)
    n_val = int(N * config_module.VAL_FRAC)

    # Advance split points to date boundaries (don't split a date across sets)
    while n_train < N and all_dates[n_train] == all_dates[n_train - 1]:
        n_train += 1
    val_end = n_train + n_val
    while val_end < N and all_dates[val_end] == all_dates[val_end - 1]:
        val_end += 1
    n_val = val_end - n_train

    X_train = X_all[:n_train]
    fwd_train = fwd_all[:n_train]
    X_val = X_all[n_train : n_train + n_val]
    fwd_val = fwd_all[n_train : n_train + n_val]
    X_test = X_all[n_train + n_val :]
    fwd_test = fwd_all[n_train + n_val :]

    # Split and re-index date IDs within each set (0-based contiguous)
    def _reindex(ids: np.ndarray) -> np.ndarray:
        uniq = np.unique(ids)
        m = {old: new for new, old in enumerate(uniq)}
        return np.array([m[x] for x in ids], dtype=np.int64)

    date_ids_train = _reindex(date_ids_all[:n_train])
    date_ids_val = _reindex(date_ids_all[n_train : n_train + n_val])
    date_ids_test = _reindex(date_ids_all[n_train + n_val :])

    log.info("Regression split: train=%d  val=%d  test=%d", len(fwd_train), len(fwd_val), len(fwd_test))

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
