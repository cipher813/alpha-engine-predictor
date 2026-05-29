"""Behavioral tests for model.meta_label_classifier.MetaLabelClassifier (Task B).

Calibrated López de Prado meta-label classifier: P(up barrier before down).
Uses real sklearn on synthetic separable data.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_label_classifier import MetaLabelClassifier


def _separable_dataset(n: int = 240, seed: int = 0):
    """Two Gaussian blobs in feature-0 → linearly separable binary target."""
    rng = np.random.default_rng(seed)
    n_feat = 3
    X = rng.normal(0, 1.0, size=(n, n_feat))
    # Class 1 shifted +2 on feature 0; class 0 shifted -2.
    y = (rng.random(n) > 0.5).astype(int)
    X[:, 0] += np.where(y == 1, 2.0, -2.0)
    names = ["research_calibrator_prob", "momentum_score", "expected_move"]
    return X, y.astype(float), names


class TestFit:
    def test_fits_and_separates(self):
        X, y, names = _separable_dataset()
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        assert clf.is_fitted
        m = clf.metrics()
        assert m["n_samples"] == len(y)
        assert m["auc"] > 0.8  # separable → strong AUC
        assert m["type"] == "meta_label_classifier_calibrated_logistic"
        assert m["feature_names"] == names

    def test_predict_in_unit_interval(self):
        X, y, names = _separable_dataset()
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        probs = clf.predict(X)
        assert probs.shape == (len(y),)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_predict_single_uses_schema(self):
        X, y, names = _separable_dataset()
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        # A clearly class-1 sample (feature 0 strongly positive) → prob > 0.5
        p_hi = clf.predict_single({"research_calibrator_prob": 5.0,
                                    "momentum_score": 0.0, "expected_move": 0.0})
        p_lo = clf.predict_single({"research_calibrator_prob": -5.0,
                                    "momentum_score": 0.0, "expected_move": 0.0})
        assert 0.0 <= p_lo <= 1.0 and 0.0 <= p_hi <= 1.0
        assert p_hi > p_lo

    def test_missing_feature_defaults_zero(self):
        X, y, names = _separable_dataset()
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        # empty dict → all features default 0.0, no crash
        p = clf.predict_single({})
        assert 0.0 <= p <= 1.0

    def test_sample_weight_accepted(self):
        X, y, names = _separable_dataset()
        w = np.ones(len(y))
        clf = MetaLabelClassifier().fit(X, y, feature_names=names, sample_weight=w)
        assert clf.is_fitted


class TestSkipConditions:
    def test_insufficient_samples_not_fitted(self):
        X, y, names = _separable_dataset(n=30)
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        assert not clf.is_fitted
        with pytest.raises(RuntimeError):
            clf.predict(X)

    def test_single_class_not_fitted(self):
        X, _, names = _separable_dataset(n=120)
        y = np.ones(120)  # only one class
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        assert not clf.is_fitted

    def test_nan_rows_dropped(self):
        X, y, names = _separable_dataset()
        X[:20, 0] = np.nan  # 20 invalid rows
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        assert clf.is_fitted
        assert clf.metrics()["n_samples"] == len(y) - 20


class TestPersistence:
    def test_save_load_roundtrip(self):
        X, y, names = _separable_dataset()
        clf = MetaLabelClassifier().fit(X, y, feature_names=names)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "meta_label_classifier.pkl")
            clf.save(path)
            assert os.path.exists(path)
            assert os.path.exists(path + ".meta.json")
            loaded = MetaLabelClassifier.load(path)
            assert loaded.is_fitted
            assert loaded.metrics()["feature_names"] == names
            # predictions identical after roundtrip
            np.testing.assert_allclose(clf.predict(X), loaded.predict(X), rtol=1e-9)
