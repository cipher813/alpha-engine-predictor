"""B.1 — BayesianRidge cutover in MetaModel.

Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §B.1

The MetaModel switched from sklearn.linear_model.Ridge to BayesianRidge so
the meta-stack emits a posterior predictive variance alongside its point
estimate. The new ``predicted_alpha_std`` field is the load-bearing input
to the executor's α̂-uncertainty penalty (workstream B.3) — without it, the
optimizer treats α̂=0.05±0.002 and α̂=0.05±0.04 identically.

These tests confirm:
  • BayesianRidge predictions track Ridge predictions within ±0.005 IC
    on the same synthetic panel (acceptance gate from the plan).
  • ``predict_single_with_std`` returns a positive posterior std for a
    fitted BayesianRidge model.
  • Loading a pre-BayesianRidge pickled Ridge model gracefully returns
    (mean, None) so the 1-week post-cutover soak (until next training
    cycle promotes a BayesianRidge) doesn't break inference.
  • Metrics surface BR's Type-II ML learned α + λ for run-to-run
    diagnostics.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from model.meta_model import META_FEATURES, MetaModel


def _synth_training_data(feature_list, n=300, seed=0):
    """Synthetic panel with a weak linear signal — same shape as the schema
    tests so BR has something to fit. Larger n=300 so BR's Type-II ML
    iteration converges cleanly without the overflow warnings that occur
    on very-small-N synthetic data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(feature_list)))
    y = 0.01 * X[:, 0] + 0.005 * X[:, 1] + rng.normal(scale=0.02, size=n)
    return X, y


def test_bayesian_ridge_ic_within_tolerance_of_ridge():
    """Acceptance gate from the plan: BR predictions must track Ridge IC
    within ±0.005 on the same panel. Validates the cutover doesn't degrade
    the meta-stack's signal quality."""
    from sklearn.linear_model import Ridge

    X, y = _synth_training_data(META_FEATURES, n=500, seed=42)
    mm_br = MetaModel().fit(X, y, feature_names=META_FEATURES)

    ridge = Ridge(alpha=1.0, fit_intercept=True).fit(X, y)
    ridge_preds = ridge.predict(X)
    ridge_ic = float(np.corrcoef(ridge_preds, y)[0, 1])

    # BR's val_ic (in-sample) vs Ridge's IC on the same panel
    delta = abs(mm_br._val_ic - ridge_ic)
    assert delta < 0.005, (
        f"BayesianRidge IC ({mm_br._val_ic:.4f}) deviates from "
        f"Ridge IC ({ridge_ic:.4f}) by {delta:.4f} > 0.005 tolerance"
    )


def test_predict_single_with_std_returns_positive_std():
    """A fitted BayesianRidge model emits a strictly positive posterior std
    via predict_single_with_std — confirms the new API works end-to-end."""
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel().fit(X, y, feature_names=META_FEATURES)

    features = {name: 0.1 for name in META_FEATURES}
    mean, std = mm.predict_single_with_std(features)

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0, f"Posterior std must be > 0; got {std}"
    # std should be in the ballpark of the noise scale (0.02) used to
    # generate the synthetic data — same order of magnitude
    assert 0.001 < std < 0.5, f"std={std} outside plausible range"


def test_predict_single_and_with_std_means_agree():
    """The mean from predict_single_with_std must equal predict_single — same
    underlying model, same point estimate."""
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel().fit(X, y, feature_names=META_FEATURES)

    features = {name: 0.1 for name in META_FEATURES}
    mean_std, _ = mm.predict_single_with_std(features)
    mean_only = mm.predict_single(features)

    assert mean_std == pytest.approx(mean_only, abs=1e-12)


def test_legacy_ridge_pickle_returns_none_std():
    """Backwards-compat gate: loading a pre-BR pickled Ridge model must
    return (mean, None) from predict_single_with_std rather than raise.

    This covers the 1-week soak window between this PR landing and the
    first BR-trained model being promoted by the Saturday training cycle.
    Inference must keep serving with std=None — executor's uncertainty
    penalty (B.3) treats None as 'fall back to point-estimate sizing.'
    """
    from sklearn.linear_model import Ridge

    X, y = _synth_training_data(META_FEATURES)
    legacy_ridge = Ridge(alpha=1.0, fit_intercept=True).fit(X, y)

    mm = MetaModel()
    mm._model = legacy_ridge
    mm._fitted = True
    mm._feature_names = list(META_FEATURES)

    features = {name: 0.1 for name in META_FEATURES}
    mean, std = mm.predict_single_with_std(features)

    assert isinstance(mean, float)
    assert std is None, (
        "Legacy Ridge model must return std=None — sklearn's Ridge.predict "
        "doesn't accept return_std kwarg, so predict_single_with_std should "
        "catch TypeError and fall through"
    )


def test_metrics_include_learned_hyperparameters():
    """BayesianRidge's Type-II ML learned α + λ must appear in metrics()
    for run-to-run comparison. None values when fit was skipped."""
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel().fit(X, y, feature_names=META_FEATURES)

    m = mm.metrics()
    assert m["type"] == "meta_model_bayesian_ridge"
    assert m["learned_alpha_noise_precision"] is not None
    assert m["learned_lambda_weight_precision"] is not None
    assert m["learned_alpha_noise_precision"] > 0
    assert m["learned_lambda_weight_precision"] > 0


def test_learned_hyperparameters_roundtrip_save_load(tmp_path):
    """Save → load preserves the learned α + λ via the .meta.json sidecar."""
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel().fit(X, y, feature_names=META_FEATURES)

    pkl_path = tmp_path / "mm.pkl"
    mm.save(pkl_path)

    mm2 = MetaModel.load(pkl_path)
    assert mm2._learned_alpha == pytest.approx(mm._learned_alpha, rel=1e-6)
    assert mm2._learned_lambda == pytest.approx(mm._learned_lambda, rel=1e-6)


def test_predict_single_with_std_raises_when_unfitted():
    """Calling on an unfitted MetaModel must raise — no silent fail per
    no-silent-fails policy."""
    mm = MetaModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        mm.predict_single_with_std({"f0": 0.1})


def test_predict_single_with_std_handles_missing_features():
    """Missing features in the input dict are treated as 0.0 — matches
    predict_single's lenient policy so a single missing macro doesn't
    error a whole inference batch."""
    X, y = _synth_training_data(META_FEATURES)
    mm = MetaModel().fit(X, y, feature_names=META_FEATURES)

    partial = {META_FEATURES[0]: 0.5}  # only one feature provided
    mean, std = mm.predict_single_with_std(partial)
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0
