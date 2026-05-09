"""Regression tests for the inference-vs-training feature distribution fix.

Locks the 2026-05-09 audit finding: training rank-normalizes L1 features
via ``cross_sectional_rank_normalize`` per-date before fitting the GBMs;
inference must apply the same transform to today's full ticker batch
or the GBM's split thresholds (learned on percentile ranks in (0, 1))
get raw feature values they never saw at training time.

The pre-fix symptom was leaf clustering: Friday 2026-05-08 production
predictions collapsed to 7 unique p_up values across 27 tickers, 5
saturated at 0.010 — the SaturdayHealthCheck DriftDetection alarm fired
"93% predict DOWN today". After the fix, GBM inputs are percentile
ranks (0..1 over today's batch) so the learned splits land in the
ranges they were trained against, restoring per-ticker variance.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Direct test of cross_sectional_rank_normalize on a single-date batch ────


def test_cross_sectional_rank_normalize_handles_single_date_batch():
    """Inference passes one date for every row (today's cross-section).
    The function must rank within that single group rather than treating
    each row as its own date — otherwise every row collapses to 0.5
    (the n=1 single-date branch) and the GBM gets a constant input."""
    from data.dataset import cross_sectional_rank_normalize

    # 5 tickers, 1 feature, all on the same date (today)
    X = np.array(
        [[10.0],
         [20.0],
         [30.0],
         [40.0],
         [50.0]],
        dtype=np.float32,
    )
    today_dates = ["2026-05-09"] * 5

    ranked = cross_sectional_rank_normalize(X, today_dates)
    expected = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
    assert np.allclose(ranked, expected, atol=1e-6), (
        f"Single-date batch must rank within the group; got {ranked.ravel()}"
    )


def test_single_ticker_single_date_collapses_to_05():
    """Edge case: only one ticker present in inference batch (e.g. a
    debugging/integration run with one symbol). The function correctly
    assigns 0.5 across the row — there's no meaningful cross-section
    with N=1, so neutral is the only sensible answer."""
    from data.dataset import cross_sectional_rank_normalize

    X = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    ranked = cross_sectional_rank_normalize(X, ["2026-05-09"])
    assert np.all(ranked == 0.5)


# ── Integration: simulate the inference rank-norm step end-to-end ────────────


def _fake_gbm_remembering_input():
    """Trivial GBM stub whose `predict` returns the input value verbatim
    (per-row mean). Lets a test verify what scaled inputs reach the GBM."""

    class _StubScorer:
        _val_ic = 0.5  # ≥0.02 → activates the GBM path (not fallback)

        def predict(self, X: np.ndarray) -> np.ndarray:
            # Return per-row mean so we can assert the inputs were rank-normed
            return X.mean(axis=1)

    return _StubScorer()


def test_inference_batch_rank_norm_produces_full_rank_spread():
    """Pre-fix simulation: pass raw feature values into the GBM stub
    one ticker at a time → GBM sees raw values, predictions reflect
    raw scale (no cross-sectional normalization).

    Post-fix simulation: pass rank-normed batch slice → GBM sees
    percentile ranks in (0, 1), predictions span (0, 1).

    This is a pure-function check: it verifies the math, not the
    full pipeline. The pipeline-level wiring is at
    `inference/stages/run_inference.py:336+` and exercised by the
    Saturday-SF integration runs.
    """
    from data.dataset import cross_sectional_rank_normalize

    n_tickers = 27  # Friday 2026-05-08 batch size
    n_features = 4
    rng = np.random.default_rng(seed=0)
    # Simulate raw momentum features in widely different units
    # (5d returns ~ ±0.1, MA50 ratio ~ 0.95–1.05, RSI 0–100, etc.)
    X_raw = np.column_stack([
        rng.normal(0, 0.05, n_tickers).astype(np.float32),         # m5
        rng.normal(0, 0.10, n_tickers).astype(np.float32),         # m20
        rng.normal(1.0, 0.05, n_tickers).astype(np.float32),       # ma50
        rng.uniform(20, 80, n_tickers).astype(np.float32),         # rsi
    ])

    today_dates = ["2026-05-09"] * n_tickers
    X_ranked = cross_sectional_rank_normalize(X_raw, today_dates)

    # Each feature column must span (0, 1)
    for f in range(n_features):
        col = X_ranked[:, f]
        assert col.min() == 0.0, f"feature {f} min {col.min()}"
        assert col.max() == 1.0, f"feature {f} max {col.max()}"

    # Per-row mean (what the stub GBM would predict on rank-normed input)
    # must span more than a tiny range — pre-fix raw-value inputs would
    # cluster predictions into a few leaves; post-fix every ticker's
    # rank vector is unique so per-row means span (0, 1) too.
    row_means = X_ranked.mean(axis=1)
    assert row_means.max() - row_means.min() > 0.3, (
        f"row-mean spread {row_means.max() - row_means.min():.3f} too tight; "
        "rank-norm should distribute predictions across the cross-section"
    )

    # Sanity: at least 20 unique row means (healthy variance) — the
    # 2026-05-08 production failure was 7 unique p_up across 27 tickers,
    # which is the leaf-clustering signature this fix prevents.
    assert len(set(np.round(row_means, 6))) >= 20


# ── Idempotence under the NaN-aware fix from PR #107 ────────────────────────


def test_short_history_ticker_gets_neutral_05_in_inference_batch():
    """Compose with the NaN-aware fix from PR #107: a short-history
    ticker (NaN raw feature) gets rank=0.5 in the inference batch
    instead of rank=1.0 (top decile, the pre-fix bug). The GBM then
    sees a neutral input for that ticker rather than the spurious
    top-decile signal that produced -0.17 IC during training."""
    from data.dataset import cross_sectional_rank_normalize

    # 5 tickers, the third has NaN in feature 0 (e.g. SNDK with
    # insufficient history for momentum_20d)
    X = np.array(
        [[1.0, 1.0],
         [2.0, 2.0],
         [np.nan, 3.0],   # short-history ticker
         [4.0, 4.0],
         [5.0, 5.0]],
        dtype=np.float32,
    )
    ranked = cross_sectional_rank_normalize(X, ["2026-05-09"] * 5)

    assert ranked[2, 0] == 0.5, (
        f"short-history NaN must rank to 0.5 (median), got {ranked[2, 0]}"
    )
    # Other rows in feature 0 still spread over the non-NaN cross-section
    assert ranked[0, 0] == 0.0
    assert ranked[4, 0] == 1.0
