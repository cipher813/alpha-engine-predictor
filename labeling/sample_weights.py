"""LdP Ch. 4 sample-weight helpers for overlapping-label training.

Per López de Prado *Advances in Financial Machine Learning* Ch. 4.4-4.5,
overlapping forward-window labels (typical for the triple-barrier method)
violate the IID assumption that vanilla loss-weighted fitting relies on.
Two labels whose windows overlap in time share path information, so their
gradient contributions are correlated. Naive equal-weight fitting double-
counts that overlap and inflates effective sample size, which shows up as
artificially-tight in-sample IC.

The fix: weight each training row by its **average uniqueness** —
the mean over its window of ``1 / concurrency``, where concurrency at
time ``t`` is the number of labels whose window covers ``t``. Highly-
overlapping rows get small weights; isolated rows get full weight.

Weights are NOT normalized — sklearn's ``Ridge.fit(sample_weight=...)``
and LightGBM's ``train(sample_weight=...)`` both accept unnormalized
weights and rescale internally.
"""
from __future__ import annotations

import numpy as np


def _avg_uniqueness_single_group(
    window_starts: np.ndarray,
    window_lengths: np.ndarray,
) -> np.ndarray:
    """Average-uniqueness weights for a single group (e.g., one ticker).

    Caller is responsible for grouping — different groups don't share
    timeline so concurrency must be computed within each group only.

    Args:
        window_starts: (n,) integer array — start position of each label's
            window in some shared integer timeline.
        window_lengths: (n,) integer array — length of each label's window.

    Returns:
        (n,) float64 array of weights in ``(0, 1]``.
    """
    n = len(window_starts)
    if n == 0:
        return np.array([], dtype=np.float64)

    starts = np.asarray(window_starts, dtype=np.int64)
    lengths = np.asarray(window_lengths, dtype=np.int64)
    ends = starts + lengths  # exclusive end

    min_t = int(starts.min())
    max_t = int(ends.max())
    span = max_t - min_t

    # Concurrency at each timeline position
    concurrency = np.zeros(span, dtype=np.int64)
    for s, e in zip(starts, ends):
        concurrency[s - min_t : e - min_t] += 1

    # Per-label average uniqueness — mean of 1/c_t over the label's window
    weights = np.empty(n, dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        c_window = concurrency[s - min_t : e - min_t]
        # All positions in c_window are guaranteed >= 1 (this label
        # contributes itself), so 1.0 / c_window is finite.
        weights[i] = float(np.mean(1.0 / c_window))
    return weights


def average_uniqueness_weights(
    window_starts: np.ndarray,
    window_length: int | np.ndarray = 21,
    group_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Compute LdP Ch. 4.4 average-uniqueness sample weights.

    Per LdP Ch. 4.4: weight each label by the mean over its window of
    ``1 / concurrency``, where concurrency at time ``t`` is the count
    of labels whose window covers ``t``. Higher weight ⇒ less overlap
    with neighbors ⇒ more independent gradient information.

    Args:
        window_starts: (n,) integer array — start position of each label's
            window in trading-day-index space. Positions need not be
            contiguous.
        window_length: scalar or (n,) integer array — window length.
            Scalar broadcasts to all rows. Default 21 (matches the
            current alpha label horizon).
        group_ids: (n,) array or None — when provided, concurrency is
            computed within each group separately. Use this to keep
            different tickers from polluting each other's uniqueness
            (one ticker's window doesn't overlap with another ticker's
            window in any meaningful path-dependence sense). When None,
            all rows share a single timeline.

    Returns:
        (n,) float64 array of weights in ``(0, 1]``. Unnormalized — pass
        directly to sklearn / LightGBM ``sample_weight``.
    """
    n = len(window_starts)
    weights = np.empty(n, dtype=np.float64)
    if n == 0:
        return weights

    starts = np.asarray(window_starts, dtype=np.int64)
    if np.isscalar(window_length):
        lengths = np.full(n, int(window_length), dtype=np.int64)
    else:
        lengths = np.asarray(window_length, dtype=np.int64)
        if lengths.shape != (n,):
            raise ValueError(
                f"window_length array length {lengths.shape} != n_starts {n}"
            )

    if group_ids is None:
        return _avg_uniqueness_single_group(starts, lengths)

    group_ids_arr = np.asarray(group_ids)
    if len(group_ids_arr) != n:
        raise ValueError(
            f"group_ids length {len(group_ids_arr)} != n_starts {n}"
        )
    for g in np.unique(group_ids_arr):
        idx = np.where(group_ids_arr == g)[0]
        weights[idx] = _avg_uniqueness_single_group(starts[idx], lengths[idx])
    return weights
