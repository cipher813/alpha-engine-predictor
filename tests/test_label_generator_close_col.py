"""Tests for the ``close_col`` parameter on data.label_generator.

PR7-step-7b: a total-return SHADOW run labels off ``total_return_close``
instead of the split-adjusted-level ``Close``. The default (``"Close"``) is
the live behaviour and must be byte-identical. The market-relative label must
be total-return on BOTH legs — the stock leg via ``close_col`` and the
benchmark leg via the basis of the ``benchmark_returns`` series the caller
passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.label_generator import (
    compute_labels,
    compute_triple_barrier_alpha_labels,
    compute_triple_barrier_touch_labels,
)


def _frame(close, total_return_close, n):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Close": close, "total_return_close": total_return_close}, index=idx
    )


def test_default_close_col_uses_Close_unchanged():
    # Distinct Close vs total_return_close series so a column swap is detectable.
    n = 8
    close = [100.0 + i for i in range(n)]
    trc = [200.0 + 3 * i for i in range(n)]
    df = _frame(close, trc, n)

    out_default = compute_labels(df, forward_days=2)
    # The forward return must match the Close column, NOT total_return_close.
    expected = (df["Close"].shift(-2) / df["Close"] - 1.0).dropna()
    np.testing.assert_allclose(
        out_default["forward_return_5d"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-9,
    )


def test_close_col_total_return_uses_total_return_close():
    n = 8
    close = [100.0 + i for i in range(n)]
    trc = [200.0 + 3 * i for i in range(n)]
    df = _frame(close, trc, n)

    out_tr = compute_labels(df, forward_days=2, close_col="total_return_close")
    expected = (
        df["total_return_close"].shift(-2) / df["total_return_close"] - 1.0
    ).dropna()
    np.testing.assert_allclose(
        out_tr["forward_return_5d"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-9,
    )
    # And it differs from the split-adjusted-level basis (proves the swap).
    out_default = compute_labels(df, forward_days=2)
    assert not np.allclose(
        out_tr["forward_return_5d"].to_numpy(),
        out_default["forward_return_5d"].to_numpy(),
    )


def test_close_col_market_relative_both_legs_total_return():
    """Stock leg via close_col + benchmark leg via the TR benchmark series →
    the market-relative label is total-return on BOTH legs."""
    n = 8
    close = [100.0 + i for i in range(n)]
    trc = [100.0 + 2.0 * i for i in range(n)]
    df = _frame(close, trc, n)

    # Benchmark passed on the SAME (total-return) basis.
    bench_tr = pd.Series([50.0 + 0.5 * i for i in range(n)], index=df.index)

    out = compute_labels(
        df, forward_days=2, close_col="total_return_close",
        benchmark_returns=bench_tr,
    )
    stock_fwd = df["total_return_close"].shift(-2) / df["total_return_close"] - 1.0
    bench_fwd = bench_tr.shift(-2) / bench_tr - 1.0
    expected = (stock_fwd - bench_fwd).dropna()
    np.testing.assert_allclose(
        out["forward_return_5d"].to_numpy(), expected.to_numpy(), rtol=1e-9
    )


def test_missing_close_col_raises_loud():
    n = 6
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    df = pd.DataFrame({"Close": [100.0 + i for i in range(n)]}, index=idx)
    # total_return_close absent → fail loud rather than silently mis-label.
    import pytest

    with pytest.raises(KeyError, match="total_return_close"):
        compute_labels(df, forward_days=2, close_col="total_return_close")


def test_triple_barrier_labels_accept_close_col():
    """The parallel triple-barrier alpha + touch labels also honour close_col so
    the whole label chain is on one basis on a shadow run."""
    n = 60
    close = [100.0 + np.sin(i / 3.0) for i in range(n)]
    trc = [300.0 + 2.0 * np.sin(i / 3.0) for i in range(n)]
    df = _frame(close, trc, n)
    bench = pd.Series([50.0 + 0.1 * i for i in range(n)], index=df.index)

    a_default = compute_triple_barrier_alpha_labels(df, benchmark_returns=bench)
    a_tr = compute_triple_barrier_alpha_labels(
        df, benchmark_returns=bench, close_col="total_return_close"
    )
    assert "triple_barrier_alpha_21d" in a_default.columns
    assert "triple_barrier_alpha_21d" in a_tr.columns
    # Different price basis → different (non-identical) label series.
    assert not np.allclose(
        np.nan_to_num(a_default["triple_barrier_alpha_21d"].to_numpy()),
        np.nan_to_num(a_tr["triple_barrier_alpha_21d"].to_numpy()),
    )

    t_tr = compute_triple_barrier_touch_labels(
        df, benchmark_returns=bench, close_col="total_return_close"
    )
    assert "triple_barrier_touch_21d" in t_tr.columns
