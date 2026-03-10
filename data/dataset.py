"""
data/dataset.py — PyTorch Dataset and DataLoader construction.

Loads all parquet files from a local cache directory, computes features and
labels for each ticker, concatenates all samples, performs a time-based
70/15/15 train/val/test split, z-score normalizes features (fit on train only),
and returns DataLoaders ready for training.

Normalization statistics are saved to data/norm_stats.json so they can be
loaded at inference time to normalize live feature vectors consistently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class PredictorDataset(Dataset):
    """
    PyTorch Dataset wrapping (feature_vector, direction_label) pairs.

    Parameters
    ----------
    X : np.ndarray, shape (N, 8)
        Feature matrix — 8 technical indicators per sample.
    y : np.ndarray, shape (N,)
        Integer direction labels: 0=DOWN, 1=FLAT, 2=UP.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}"
            )
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.X[idx]),
            torch.LongTensor([self.y[idx]])[0],
        )


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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from parquet files in data_dir.

    Steps:
    1. Load all .parquet files from data_dir.
    2. Compute features + labels for each ticker.
    3. Concatenate all samples into a single array, sorted by date.
    4. Time-based split: first 70% → train, next 15% → val, last 15% → test.
    5. Z-score normalize features using train-set statistics.
    6. Save norm stats to norm_stats_path.
    7. Return (train_loader, val_loader, test_loader).

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
    tuple of (train_loader, val_loader, test_loader)
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

    log.info("Loading %d parquet files from %s", len(parquet_files), data_dir)

    all_rows: list[tuple[pd.Timestamp, np.ndarray, int]] = []

    for i, path in enumerate(parquet_files):
        raw_df = _load_ticker_parquet(path)
        if raw_df.empty or len(raw_df) < 220:
            # Need at least 200 rows for MA200 + 20 rows of labels
            continue

        try:
            featured_df = compute_features(raw_df)
            labeled_df = compute_labels(
                featured_df,
                forward_days=config_module.FORWARD_DAYS,
                up_threshold=config_module.UP_THRESHOLD,
                down_threshold=config_module.DOWN_THRESHOLD,
            )
        except Exception as exc:
            log.warning("Feature/label computation failed for %s: %s", path.name, exc)
            continue

        if labeled_df.empty:
            continue

        features_arr = labeled_df[config_module.FEATURES].to_numpy(dtype=np.float32)
        labels_arr = labeled_df["direction_int"].to_numpy(dtype=np.int64)
        dates = labeled_df.index

        for j in range(len(dates)):
            all_rows.append((dates[j], features_arr[j], labels_arr[j]))

        if (i + 1) % 50 == 0:
            log.info("  Processed %d / %d tickers (%d samples so far)", i + 1, len(parquet_files), len(all_rows))

    if not all_rows:
        raise ValueError(
            "No valid samples were generated. Check parquet files and feature computation."
        )

    # ── Sort all samples by date (time-based split requires this) ─────────────
    all_rows.sort(key=lambda r: r[0])
    all_dates = [r[0] for r in all_rows]
    X_all = np.stack([r[1] for r in all_rows], axis=0)  # (N, 8)
    y_all = np.array([r[2] for r in all_rows], dtype=np.int64)  # (N,)

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

    log.info(
        "Split: train=%d  val=%d  test=%d",
        len(y_train),
        len(y_val),
        len(y_test),
    )

    # ── Z-score normalization (fit on train, apply to all) ────────────────────
    feat_mean = X_train.mean(axis=0)  # shape (8,)
    feat_std = X_train.std(axis=0)    # shape (8,)
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

    return train_loader, val_loader, test_loader


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
