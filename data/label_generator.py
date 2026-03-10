"""
data/label_generator.py — Forward-return label computation.

Given a DataFrame of OHLCV + computed features, this module computes the
5-trading-day forward return for each row and bins it into three classes:

    UP   — forward_return > +1%   → label 2
    FLAT — forward_return in [-1%, +1%]  → label 1
    DOWN — forward_return < -1%   → label 0

The integer mapping (DOWN=0, FLAT=1, UP=2) matches CLASS_LABELS in config.py
and the model output neuron indices.

Rows where the forward return cannot be computed (i.e., the last `forward_days`
rows of the series) are dropped — they have no valid label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_labels(
    df: pd.DataFrame,
    forward_days: int = 5,
    up_threshold: float = 0.01,
    down_threshold: float = -0.01,
) -> pd.DataFrame:
    """
    Append forward-return labels to a featured DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at minimum a 'Close' column.
        Typically output of feature_engineer.compute_features().
    forward_days : int
        Number of trading days ahead to compute the forward return.
        Default is 5 (one calendar week).
    up_threshold : float
        Minimum return to classify as UP. Default +1% (0.01).
    down_threshold : float
        Maximum return to classify as DOWN. Default -1% (-0.01).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns appended:
        - forward_return_5d  (float)  : (close[t+N] / close[t]) - 1
        - direction          (str)    : "UP", "FLAT", or "DOWN"
        - direction_int      (int)    : 2, 1, or 0 respectively
        Rows where forward_return_5d is NaN (last forward_days rows) are
        dropped, since they have no label.
    """
    if df.empty:
        df = df.copy()
        df["forward_return_5d"] = pd.Series(dtype=float)
        df["direction"] = pd.Series(dtype=str)
        df["direction_int"] = pd.Series(dtype=int)
        return df

    df = df.copy()
    close = df["Close"].astype(float)

    # Forward return: (future price / current price) - 1
    # shift(-forward_days) aligns the future price to the current row
    future_close = close.shift(-forward_days)
    df["forward_return_5d"] = (future_close / close) - 1.0

    # Drop rows where the forward return is undefined (end of series)
    df = df.dropna(subset=["forward_return_5d"])

    # Bin into direction classes
    conditions = [
        df["forward_return_5d"] > up_threshold,
        df["forward_return_5d"] < down_threshold,
    ]
    choices_str = ["UP", "DOWN"]
    df["direction"] = np.select(conditions, choices_str, default="FLAT")

    # Integer labels: DOWN=0, FLAT=1, UP=2 (matches CLASS_LABELS order in config)
    label_map = {"DOWN": 0, "FLAT": 1, "UP": 2}
    df["direction_int"] = df["direction"].map(label_map).astype(int)

    return df


def label_distribution(df: pd.DataFrame) -> dict[str, float]:
    """
    Return the class distribution as proportions.
    Useful for checking class imbalance before training.

    Returns
    -------
    dict with keys "UP", "FLAT", "DOWN" and float proportion values.
    """
    if "direction" not in df.columns or df.empty:
        return {"UP": 0.0, "FLAT": 0.0, "DOWN": 0.0}

    counts = df["direction"].value_counts(normalize=True)
    return {
        "UP": float(counts.get("UP", 0.0)),
        "FLAT": float(counts.get("FLAT", 0.0)),
        "DOWN": float(counts.get("DOWN", 0.0)),
    }
