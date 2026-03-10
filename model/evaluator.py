"""
model/evaluator.py — Offline evaluation metrics.

Provides:
  evaluate()            — Accuracy, per-class accuracy, confusion matrix on a DataLoader.
  compute_ic()          — Pearson IC between predicted signal and actual return.
  compute_rolling_ic()  — Rolling 20-day IC over time.
  compute_hit_rate()    — Fraction of correct directional predictions.
  compute_direction_sharpe() — Annualized Sharpe of long-UP / short-DOWN strategy.

All functions take plain Python lists/dicts rather than PyTorch objects so they
can also be called on inference outputs loaded from JSON.
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """
    Run the model on test_loader and compute accuracy metrics.

    Parameters
    ----------
    model :       Trained model in eval mode.
    test_loader : DataLoader for the test split.
    device :      Torch device string.

    Returns
    -------
    dict with keys:
        accuracy          (float)      : Overall fraction correct.
        per_class_accuracy (dict)      : {label: float} for DOWN, FLAT, UP.
        confusion_matrix  (list[list]) : 3×3 row=actual, col=predicted.
        n_samples         (int)        : Total test samples.
    """
    from config import CLASS_LABELS  # ["DOWN", "FLAT", "UP"]

    model.eval()
    model.to(device)

    n_classes = len(CLASS_LABELS)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=-1)

            for actual, predicted in zip(y_batch.cpu().numpy(), preds.cpu().numpy()):
                confusion[actual][predicted] += 1

    n_total = int(confusion.sum())
    n_correct = int(confusion.diagonal().sum())
    accuracy = n_correct / max(n_total, 1)

    per_class: dict[str, float] = {}
    for i, label in enumerate(CLASS_LABELS):
        row_sum = int(confusion[i].sum())
        per_class[label] = float(confusion[i][i]) / max(row_sum, 1)

    return {
        "accuracy": round(accuracy, 4),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
        "n_samples": n_total,
    }


def compute_ic(
    predictions: list[dict],
    actuals: list[float],
) -> float:
    """
    Pearson Information Coefficient (IC) between the predicted directional signal
    and the actual 5-day forward return.

    The predicted signal used is (p_up - p_down), which is a continuous value
    in [-1, +1]. IC = Pearson correlation with actual return.

    Parameters
    ----------
    predictions : list of dicts, each with keys 'p_up' and 'p_down'.
    actuals :     list of actual 5-day forward returns (floats).

    Returns
    -------
    float — Pearson IC. Returns 0.0 if computation fails (e.g., insufficient data).
    """
    if len(predictions) != len(actuals) or len(predictions) < 5:
        log.warning("compute_ic: insufficient data (%d predictions)", len(predictions))
        return 0.0

    try:
        signal = np.array([p["p_up"] - p["p_down"] for p in predictions], dtype=float)
        actual = np.array(actuals, dtype=float)

        # Remove NaN pairs
        mask = np.isfinite(signal) & np.isfinite(actual)
        if mask.sum() < 5:
            return 0.0

        ic = float(np.corrcoef(signal[mask], actual[mask])[0, 1])
        return round(ic, 6) if math.isfinite(ic) else 0.0

    except Exception as exc:
        log.warning("compute_ic failed: %s", exc)
        return 0.0


def compute_rolling_ic(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Compute rolling IC over a DataFrame with columns 'signal' and 'actual_return'.

    Parameters
    ----------
    df :     DataFrame with columns 'signal' (p_up - p_down) and 'actual_return'.
             Index should be a DatetimeIndex or sortable date index.
    window : Rolling window size (default 20 trading days).

    Returns
    -------
    pd.Series of rolling IC values, same index as df.
    """
    if "signal" not in df.columns or "actual_return" not in df.columns:
        raise ValueError("DataFrame must have 'signal' and 'actual_return' columns")

    def _pearson(x: pd.Series) -> float:
        if len(x) < 5:
            return float("nan")
        arr = np.array(x.tolist())
        s = arr[:, 0]
        r = arr[:, 1]
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < 5:
            return float("nan")
        corr = np.corrcoef(s[mask], r[mask])[0, 1]
        return float(corr) if math.isfinite(corr) else float("nan")

    combined = df[["signal", "actual_return"]].copy()
    rolling_ic = combined.apply(lambda row: (row["signal"], row["actual_return"]), axis=1)
    return rolling_ic.rolling(window=window).apply(
        lambda vals: _pearson(pd.Series(list(vals))),
        raw=False,
    )


def compute_hit_rate(
    predictions: list[dict],
    actuals: list[float],
    threshold: float = 0.0,
    up_threshold: float = 0.01,
    down_threshold: float = -0.01,
) -> float:
    """
    Fraction of predictions where the predicted direction matched the actual direction.

    Parameters
    ----------
    predictions :    list of dicts with 'predicted_direction' key (UP/FLAT/DOWN).
    actuals :        list of actual 5-day returns.
    threshold :      Not used directly (kept for API compatibility).
    up_threshold :   Return threshold above which actual is UP. Default +1%.
    down_threshold : Return threshold below which actual is DOWN. Default -1%.

    Returns
    -------
    float — hit rate in [0, 1].
    """
    if len(predictions) != len(actuals) or not predictions:
        return 0.0

    n_correct = 0
    n_valid = 0

    for pred, actual_return in zip(predictions, actuals):
        if not math.isfinite(actual_return):
            continue

        predicted_dir = pred.get("predicted_direction", "FLAT")

        if actual_return > up_threshold:
            actual_dir = "UP"
        elif actual_return < down_threshold:
            actual_dir = "DOWN"
        else:
            actual_dir = "FLAT"

        if predicted_dir == actual_dir:
            n_correct += 1
        n_valid += 1

    return round(n_correct / max(n_valid, 1), 4)


def compute_direction_sharpe(
    predictions: list[dict],
    actuals: list[float],
    annualization_factor: float = 252.0,
) -> float:
    """
    Annualized Sharpe ratio of a simple long-UP / short-DOWN strategy.

    Strategy:
    - If predicted_direction == UP  → +1 (long position)
    - If predicted_direction == DOWN → -1 (short position)
    - If predicted_direction == FLAT → 0 (no position)

    Daily PnL = position × actual_5d_return (treating 5-day return as per-period return).

    Parameters
    ----------
    predictions :         list of dicts with 'predicted_direction'.
    actuals :             list of actual 5-day forward returns.
    annualization_factor : Number of trading days per year. Default 252.

    Returns
    -------
    float — annualized Sharpe ratio. Returns 0.0 on insufficient data.
    """
    if len(predictions) != len(actuals) or len(predictions) < 10:
        return 0.0

    returns: list[float] = []

    for pred, actual_return in zip(predictions, actuals):
        if not math.isfinite(actual_return):
            continue

        direction = pred.get("predicted_direction", "FLAT")
        if direction == "UP":
            position = 1.0
        elif direction == "DOWN":
            position = -1.0
        else:
            position = 0.0

        returns.append(position * actual_return)

    if len(returns) < 10:
        return 0.0

    arr = np.array(returns)
    mean_ret = arr.mean()
    std_ret = arr.std(ddof=1)

    if std_ret == 0 or not math.isfinite(std_ret):
        return 0.0

    # Scale from 5-day periods to annual: there are ~252/5 = 50.4 periods/year
    periods_per_year = annualization_factor / 5.0
    sharpe = (mean_ret / std_ret) * math.sqrt(periods_per_year)
    return round(sharpe, 4) if math.isfinite(sharpe) else 0.0
