"""Tests for regime/hmm.py — HMM regime classifier.

Covers:
- Fit + filter inference shape + normalization invariants
- State pinning across re-fits (bear < neutral < bull by spy_20d_return)
- Filter vs smoother distinction (look-ahead discipline)
- Insufficient-data guardrail
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.hmm import (
    HMMRegimeClassifier,
    PIN_FEATURE,
    REGIME_STATES,
)


pytest.importorskip("hmmlearn")


def _synthetic_regime_features(
    n_bear: int = 80,
    n_neutral: int = 100,
    n_bull: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a feature history with three statistically-separable
    regimes. Emission distributions are deliberately tight so the HMM
    has a clean shot at recovering ground truth — these tests check
    implementation correctness, not real-market separability.

    Returns a DataFrame with three feature columns + a clear regime
    label (column ``_truth``, only used for in-test verification —
    HMM never sees it).
    """
    rng = np.random.default_rng(seed)
    bear = pd.DataFrame({
        "spy_20d_return": rng.normal(-0.08, 0.015, n_bear),
        "vix_level": rng.normal(38.0, 3.0, n_bear),
        "hy_oas_bps": rng.normal(800.0, 50.0, n_bear),
        "_truth": ["bear"] * n_bear,
    })
    neutral = pd.DataFrame({
        "spy_20d_return": rng.normal(0.005, 0.010, n_neutral),
        "vix_level": rng.normal(17.0, 2.0, n_neutral),
        "hy_oas_bps": rng.normal(400.0, 30.0, n_neutral),
        "_truth": ["neutral"] * n_neutral,
    })
    bull = pd.DataFrame({
        "spy_20d_return": rng.normal(0.06, 0.010, n_bull),
        "vix_level": rng.normal(11.0, 1.5, n_bull),
        "hy_oas_bps": rng.normal(280.0, 25.0, n_bull),
        "_truth": ["bull"] * n_bull,
    })
    df = pd.concat([bear, neutral, bull], ignore_index=True)
    return df


def test_hmm_fit_and_filter_shape() -> None:
    df = _synthetic_regime_features()
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    hmm = HMMRegimeClassifier(n_states=3, random_state=0).fit(df, feat_cols)
    probs = hmm.predict_proba_filter(df)
    assert probs.shape == (len(df), 3)


def test_hmm_filter_probabilities_sum_to_one() -> None:
    df = _synthetic_regime_features()
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    hmm = HMMRegimeClassifier(n_states=3, random_state=0).fit(df, feat_cols)
    probs = hmm.predict_proba_filter(df)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_hmm_state_pinning_bear_lt_neutral_lt_bull() -> None:
    """Pinned state order means bear < neutral < bull on PIN_FEATURE."""
    df = _synthetic_regime_features()
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    hmm = HMMRegimeClassifier(n_states=3, random_state=0).fit(df, feat_cols)
    means = hmm.pinned_state_means(df)
    # bear < neutral < bull strictly
    assert means[0] < means[1] < means[2], (
        f"pinned means not monotone: {means}"
    )


def test_hmm_state_pinning_stable_across_seeds() -> None:
    """Different random_state should not swap state labels — pinning
    must produce the same (bear, neutral, bull) order even when EM
    converges to different local optima with index-permuted states."""
    df = _synthetic_regime_features()
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    means_by_seed = []
    for seed in (0, 1, 7, 42):
        hmm = HMMRegimeClassifier(n_states=3, random_state=seed).fit(df, feat_cols)
        means = hmm.pinned_state_means(df)
        means_by_seed.append(means)
        assert means[0] < means[1] < means[2], f"seed={seed} pinning broke: {means}"


def test_hmm_filter_uses_only_past_observations() -> None:
    """The filter posterior at row t must depend only on rows 0..t.

    Concretely: changing observations AFTER row t cannot affect the
    filtered posterior at row t. This is the look-ahead-discipline
    invariant — if it fails, we have leakage from future data.
    """
    df = _synthetic_regime_features()
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    hmm = HMMRegimeClassifier(n_states=3, random_state=0).fit(df, feat_cols)

    cut = len(df) // 2
    probs_full = hmm.predict_proba_filter(df)
    probs_truncated = hmm.predict_proba_filter(df.iloc[: cut + 1])

    np.testing.assert_allclose(
        probs_full[: cut + 1],
        probs_truncated,
        atol=1e-9,
        err_msg="filter posterior changed when future observations were added — leakage!",
    )


def test_hmm_smoother_differs_from_filter() -> None:
    """The smoother uses future observations — its output at any
    interior row t should differ from the filter's output at row t,
    confirming the two are not aliases.

    Uses a deliberately noisy fixture with overlapping emission
    distributions — when regimes are perfectly separable (as in the
    main fixture) the filter is already certain everywhere and the
    smoother has no uncertainty to refine. Real-world macro data is
    more like this noisy fixture than the tight main fixture.
    """
    rng = np.random.default_rng(7)
    n_per = 40
    noisy = pd.concat([
        pd.DataFrame({
            "spy_20d_return": rng.normal(-0.02, 0.04, n_per),
            "vix_level": rng.normal(25.0, 6.0, n_per),
            "hy_oas_bps": rng.normal(550.0, 100.0, n_per),
        }),
        pd.DataFrame({
            "spy_20d_return": rng.normal(0.01, 0.03, n_per),
            "vix_level": rng.normal(20.0, 5.0, n_per),
            "hy_oas_bps": rng.normal(450.0, 80.0, n_per),
        }),
        pd.DataFrame({
            "spy_20d_return": rng.normal(0.03, 0.035, n_per),
            "vix_level": rng.normal(15.0, 4.0, n_per),
            "hy_oas_bps": rng.normal(350.0, 70.0, n_per),
        }),
    ], ignore_index=True)
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    hmm = HMMRegimeClassifier(n_states=3, random_state=0).fit(noisy, feat_cols)
    filtered = hmm.predict_proba_filter(noisy)
    smoothed = hmm.predict_proba_smoother(noisy)
    max_abs_diff = np.abs(filtered - smoothed).max()
    assert max_abs_diff > 0.01, (
        f"smoother and filter outputs are too similar (max diff {max_abs_diff}) — "
        f"smoother implementation likely broken."
    )


def test_hmm_raises_on_insufficient_data() -> None:
    df = _synthetic_regime_features(n_bear=5, n_neutral=5, n_bull=5)
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    with pytest.raises(ValueError, match="Insufficient observations"):
        HMMRegimeClassifier(n_states=3).fit(df, feat_cols)


def test_hmm_raises_when_pin_feature_missing() -> None:
    """Fitting without the pin feature must fail loudly — silent
    fallback would risk label-switching across refits."""
    df = pd.DataFrame({
        "vix_level": np.random.default_rng(0).normal(18.0, 5.0, 100),
        "hy_oas_bps": np.random.default_rng(0).normal(400.0, 100.0, 100),
    })
    with pytest.raises(ValueError, match=PIN_FEATURE):
        HMMRegimeClassifier(n_states=3).fit(df, ["vix_level", "hy_oas_bps"])


def test_hmm_filter_recovers_regime_argmax_on_synthetic() -> None:
    """On synthetic data with clear regime separation, the HMM filter's
    argmax should largely agree with the ground-truth regime label.

    This is a sanity check — not a guarantee of production accuracy.
    Threshold (60%) is conservative; tightly-separated synthetic
    regimes typically yield 85%+."""
    df = _synthetic_regime_features()
    feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
    hmm = HMMRegimeClassifier(n_states=3, random_state=0).fit(df, feat_cols)
    probs = hmm.predict_proba_filter(df)
    predicted = np.array([REGIME_STATES[i] for i in probs.argmax(axis=1)])
    truth = df["_truth"].to_numpy()
    accuracy = (predicted == truth).mean()
    assert accuracy > 0.60, f"HMM argmax accuracy on synthetic regimes = {accuracy:.3f}"
