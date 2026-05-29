"""López de Prado triple-barrier labels (regime-agnostic).

Two label flavors share the same path-walking core:

- ``triple_barrier_class_labels`` — 3-class direction labels (up / neutral
  / down). Reusable for SPY-based regime classification or per-ticker
  direction classifiers under any future continuous-conditioning rebuild.

- ``triple_barrier_alpha_labels`` — continuous alpha target: the realized
  cumulative log-return at first-touch (capped at the touched barrier)
  or at window-end (uncapped). Used by Stage 3 of the regime-conditioning
  rebuild to replace fixed-horizon L1 alpha targets — barriers can be
  regime-vol-scaled so the label adapts to vol regime continuously.

References:
- López de Prado, *Advances in Financial Machine Learning*, Ch. 3.4-3.5
- "Triple-barrier method": first-touch profit-take / stop-loss / time
  horizon barriers replace fixed-horizon labels with path-aware ones.
"""
from __future__ import annotations

import numpy as np


TRIPLE_BARRIER_SENTINEL: int = -1


def triple_barrier_class_labels(
    log_returns: np.ndarray,
    forward_window: int = 21,
    up_barrier_pct: float = 0.05,
    down_barrier_pct: float = 0.05,
) -> np.ndarray:
    """Generate 3-class triple-barrier classification labels.

    For each row, looks forward ``forward_window`` rows; classifies by
    which barrier hits first:

      - 2 (up-class)   : up barrier hit first, OR window-end cumulative
                         return > +up_barrier_pct / 2
      - 0 (down-class) : down barrier hit first, OR window-end cumulative
                         return < -down_barrier_pct / 2
      - 1 (neutral)    : neither barrier hit, window-end cumulative
                         return in [-down/2, +up/2]

    Tail rows where the forward window extends past data end get
    ``TRIPLE_BARRIER_SENTINEL`` (= -1).

    Args:
        log_returns: (n,) log-return series (any underlying — SPY for
            market-regime use, per-ticker for stock-direction use)
        forward_window: trading days forward to check
        up_barrier_pct: cumulative log-return threshold for up-class hit
        down_barrier_pct: cumulative log-return threshold for down-class hit
            (positive value; the function negates internally)

    Returns:
        (n,) int8 array — class label per row, sentinel (-1) for tail.
    """
    n = len(log_returns)
    labels = np.full(n, TRIPLE_BARRIER_SENTINEL, dtype=np.int8)
    for i in range(n - forward_window):
        cum = 0.0
        up_hit = False
        down_hit = False
        for j in range(forward_window):
            cum += log_returns[i + 1 + j]
            if cum >= up_barrier_pct:
                up_hit = True
                break
            if cum <= -down_barrier_pct:
                down_hit = True
                break
        if up_hit:
            labels[i] = 2
        elif down_hit:
            labels[i] = 0
        else:
            if cum > up_barrier_pct / 2:
                labels[i] = 2
            elif cum < -down_barrier_pct / 2:
                labels[i] = 0
            else:
                labels[i] = 1
    return labels


def triple_barrier_alpha_labels(
    log_returns: np.ndarray,
    forward_window: int = 21,
    up_barrier_pct: float | np.ndarray = 0.05,
    down_barrier_pct: float | np.ndarray = 0.05,
) -> np.ndarray:
    """Generate continuous triple-barrier alpha labels.

    For each row, looks forward ``forward_window`` rows; returns the
    realized cumulative log-return at the moment the path:

      - first touches the up barrier → label = +up_barrier_pct (capped)
      - first touches the down barrier → label = -down_barrier_pct (capped)
      - reaches the time-out window → label = window-end cumulative return

    Tail rows where the forward window extends past data end get NaN.

    The capping at touched barriers reflects the realistic execution
    model — a triple-barrier strategy exits at the touched barrier, so
    the realized return is the barrier value, not the post-touch path.
    Time-out rows return uncapped because no exit triggered.

    For Stage 3 of the regime-conditioning rebuild, callers pass per-row
    vol-scaled barrier widths (LdP Ch. 3.4): ``barrier = k × σ_t`` where
    ``σ_t`` is a trailing-window vol estimate. Barriers may be scalar
    (uniform across all rows, original Stage 0a behavior) or an ndarray
    of length ``n`` (per-row). NaN barriers at any row produce a NaN
    label at that row, propagating the underlying vol-estimate gap.

    Args:
        log_returns: (n,) log-return series
        forward_window: time-out barrier in trading days
        up_barrier_pct: profit-take barrier (cumulative log return).
            Scalar broadcasts across all rows; ndarray of length n
            applies row-wise. NaN at row i → label[i] = NaN.
        down_barrier_pct: stop-loss barrier as positive number; the
            function negates internally. Scalar or ndarray as above.

    Returns:
        (n,) float64 array — realized return label per row, NaN for tail
        and for rows where either barrier is NaN.
    """
    n = len(log_returns)
    up_arr = np.broadcast_to(np.asarray(up_barrier_pct, dtype=np.float64), (n,))
    down_arr = np.broadcast_to(np.asarray(down_barrier_pct, dtype=np.float64), (n,))
    labels = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - forward_window):
        up_i = up_arr[i]
        down_i = down_arr[i]
        if np.isnan(up_i) or np.isnan(down_i):
            continue  # NaN barrier → NaN label (already initialized)
        cum = 0.0
        hit = False
        for j in range(forward_window):
            cum += log_returns[i + 1 + j]
            if cum >= up_i:
                labels[i] = up_i
                hit = True
                break
            if cum <= -down_i:
                labels[i] = -down_i
                hit = True
                break
        if not hit:
            labels[i] = cum
    return labels


def triple_barrier_touch_order(
    log_returns: np.ndarray,
    forward_window: int = 21,
    up_barrier_pct: float | np.ndarray = 0.05,
    down_barrier_pct: float | np.ndarray = 0.05,
    timeout_policy: str = "nan",
) -> np.ndarray:
    """Generate triple-barrier *touch-order* meta-labels (LdP Ch. 3.6).

    For each row, looks forward ``forward_window`` rows and records WHICH
    horizontal barrier the path touches first:

      - 1.0 : up (profit) barrier touched first
      - 0.0 : down (stop) barrier touched first
      - time-out (neither touched within the window): governed by
        ``timeout_policy``:

          * ``"nan"`` (default) → NaN; the row is excluded from training,
            giving the pure first-touch conditional target "given a barrier
            was touched, which one" (López de Prado's meta-label).
          * ``"sign"`` → 1.0 if window-end cumulative return > 0 else 0.0;
            retains time-out rows by labelling them with realized direction,
            preserving sample size at the cost of mixing touched vs untouched
            outcomes.

    Tail rows where the forward window extends past data end get NaN. NaN
    barriers at any row propagate to a NaN label at that row (mirrors
    :func:`triple_barrier_alpha_labels`).

    This is the supervision target for the Task B meta-label classifier, whose
    calibrated ``P(up before down)`` feeds executor position sizing. The barrier
    widths are configurable so the meta-label can be aligned to EITHER the
    predictor's 21d / vol-scaled label barriers (default) or — once the Task A
    coherence diagnostic reports — the executor's realized execution barriers.

    Args:
        log_returns: (n,) log-return series.
        forward_window: time-out barrier in trading days.
        up_barrier_pct: profit-take barrier (cumulative log return). Scalar
            broadcasts across all rows; ndarray of length n applies row-wise;
            NaN at row i → NaN label at row i.
        down_barrier_pct: stop-loss barrier as a positive number; the function
            negates internally. Scalar or ndarray as above.
        timeout_policy: ``"nan"`` | ``"sign"`` — how to label time-out rows.

    Returns:
        (n,) float64 array — 1.0 / 0.0 / NaN per row.

    Raises:
        ValueError: if ``timeout_policy`` is not ``"nan"`` or ``"sign"``.
    """
    if timeout_policy not in ("nan", "sign"):
        raise ValueError(
            f"timeout_policy must be 'nan' or 'sign', got {timeout_policy!r}"
        )
    n = len(log_returns)
    up_arr = np.broadcast_to(np.asarray(up_barrier_pct, dtype=np.float64), (n,))
    down_arr = np.broadcast_to(np.asarray(down_barrier_pct, dtype=np.float64), (n,))
    labels = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - forward_window):
        up_i = up_arr[i]
        down_i = down_arr[i]
        if np.isnan(up_i) or np.isnan(down_i):
            continue  # NaN barrier → NaN label (already initialized)
        cum = 0.0
        touched = 0  # 0 = neither, 1 = up first, -1 = down first
        for j in range(forward_window):
            cum += log_returns[i + 1 + j]
            if cum >= up_i:
                touched = 1
                break
            if cum <= -down_i:
                touched = -1
                break
        if touched == 1:
            labels[i] = 1.0
        elif touched == -1:
            labels[i] = 0.0
        elif timeout_policy == "sign":
            labels[i] = 1.0 if cum > 0 else 0.0
        # else timeout_policy == "nan": leave NaN
    return labels
