"""Tests for ``MetaModel.fit`` sample-weight pass-through.

LdP Ch. 4.4 average-uniqueness weights need to flow into sklearn Ridge.
This test pins the contract end-to-end:
- sklearn Ridge accepts sample_weight and produces a different fit than
  unweighted on the same data
- NaN sample-weight rows are filtered out alongside NaN feature/label rows
- Unweighted (sample_weight=None) path is unchanged from the pre-PR-2 fit
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_model import MetaModel


def _make_synthetic(n_samples: int = 200, n_features: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    true_coef = np.array([0.5, -0.3, 0.2, 0.0, 0.0])[:n_features]
    y = X @ true_coef + rng.normal(0, 0.1, size=n_samples)
    return X, y


class TestSampleWeightPassthrough:

    def test_unweighted_path_unchanged(self):
        # When sample_weight=None, fit should behave exactly as the
        # pre-PR-2 path. Reasonable smoke: coefficients are non-zero,
        # in-sample IC is high.
        X, y = _make_synthetic()
        model = MetaModel(alpha=1.0)
        model.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])
        assert model._fitted
        assert abs(model._val_ic) > 0.5  # synthetic linear → easy IC

    def test_sample_weight_changes_fit(self):
        # Heavily downweight a structured subset → fit shifts. Compare to
        # equal-weight fit by Frobenius distance of coefficient vectors.
        X, y = _make_synthetic(n_samples=200)
        weights = np.ones(200)
        weights[:100] = 0.01  # downweight first half by 100×

        model_unweighted = MetaModel(alpha=1.0)
        model_unweighted.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])
        coef_unweighted = np.array(list(
            v for k, v in model_unweighted._coefficients.items() if k != "intercept"
        ))

        model_weighted = MetaModel(alpha=1.0)
        model_weighted.fit(
            X, y,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
            sample_weight=weights,
        )
        coef_weighted = np.array(list(
            v for k, v in model_weighted._coefficients.items() if k != "intercept"
        ))
        # Coefficients must differ — heavy downweight on half the data
        # produces a meaningfully different Ridge fit.
        diff = np.linalg.norm(coef_unweighted - coef_weighted)
        assert diff > 0.01

    def test_nan_sample_weight_rows_filtered(self):
        # NaN sample weight rows must be dropped alongside NaN X/y rows,
        # not crash sklearn.
        X, y = _make_synthetic(n_samples=200)
        weights = np.ones(200)
        weights[10:20] = np.nan
        model = MetaModel(alpha=1.0)
        model.fit(
            X, y,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
            sample_weight=weights,
        )
        # 200 rows minus 10 NaN-weight rows → 190 trained
        assert model._n_samples == 190

    def test_uniform_weights_close_to_unweighted(self):
        # Ridge.fit with uniform weights ≈ unweighted to within sklearn's
        # documented sample_weight × L2-penalty interaction. Coefficients
        # should match within ~1% — the uniform-rescale shifts the
        # effective regularization strength slightly but doesn't change
        # the fit shape.
        X, y = _make_synthetic(n_samples=200)
        weights = np.full(200, 2.5)

        model_a = MetaModel(alpha=1.0)
        model_a.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])

        model_b = MetaModel(alpha=1.0)
        model_b.fit(
            X, y,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
            sample_weight=weights,
        )

        # Coefficients within 1% absolute (synthetic-data tolerance).
        for k, v_a in model_a._coefficients.items():
            v_b = model_b._coefficients[k]
            assert abs(v_a - v_b) < 0.01, f"coef {k} drifted: {v_a} vs {v_b}"
