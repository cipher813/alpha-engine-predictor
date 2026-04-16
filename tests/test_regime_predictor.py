"""Tests for model/regime_predictor.py: metrics helper + OOS metric plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model.regime_predictor import (
    REGIME_LABELS,
    RegimePredictor,
    compute_classification_metrics,
)


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["per_class_precision"] == [1.0, 1.0, 1.0]
        assert m["per_class_recall"] == [1.0, 1.0, 1.0]
        assert m["class_order"] == REGIME_LABELS

    def test_majority_class_predictor_flagged_by_macro_f1(self):
        """A model that always predicts 'neutral' scores high accuracy on
        neutral-heavy labels but has zero recall for bear and bull — macro-F1
        is the guard that catches this collapse. Regression-style test for
        the exact failure mode the gate protects against."""
        y_true = np.array([1] * 60 + [0] * 20 + [2] * 20)  # 60% neutral
        y_pred = np.array([1] * 100)  # always predicts neutral
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(0.60, abs=1e-4)
        # Recall for bear and bull must be 0 — nothing was routed to those classes
        assert m["per_class_recall"][0] == 0.0  # bear
        assert m["per_class_recall"][2] == 0.0  # bull
        # Macro-F1 well below accuracy — this is the promotion gate's honest signal
        assert m["macro_f1"] < 0.30

    def test_confusion_matrix_orientation(self):
        """Row = true class, column = predicted class, order bear/neutral/bull."""
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0])
        m = compute_classification_metrics(y_true, y_pred)
        cm = np.array(m["confusion_matrix"])
        assert cm[0, 0] == 1  # 1 bear→bear
        assert cm[0, 1] == 1  # 1 bear→neutral
        assert cm[1, 1] == 1  # 1 neutral→neutral
        assert cm[2, 0] == 1  # 1 bull→bear

    def test_empty_input(self):
        m = compute_classification_metrics(np.array([]), np.array([]))
        assert m["accuracy"] == 0.0
        assert m["n_samples"] == 0


class TestRegimePredictorOOSRoundtrip:
    def test_oos_metrics_roundtrip_through_save_load(self, tmp_path):
        """OOS metrics attached via set_oos_metrics() must survive save+load
        so inference-side consumers (LLM cross-check, dashboard) see the same
        values the trainer computed."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=(n, len(RegimePredictor.FEATURE_NAMES)))
        y = rng.integers(0, 3, size=n)

        rp = RegimePredictor()
        rp.fit(X, y)

        oos = {
            "accuracy": 0.42,
            "macro_f1": 0.39,
            "confusion_matrix": [[10, 5, 2], [3, 20, 4], [1, 6, 9]],
            "per_class_precision": [0.71, 0.65, 0.60],
            "per_class_recall": [0.59, 0.74, 0.56],
            "n_samples": 60,
        }
        rp.set_oos_metrics(oos)

        pkl_path = tmp_path / "rp.pkl"
        rp.save(pkl_path)

        meta = json.loads(Path(str(pkl_path) + ".meta.json").read_text())
        assert meta["oos"]["accuracy"] == 0.42
        assert meta["oos"]["macro_f1"] == 0.39
        assert meta["in_sample"]["accuracy"] == pytest.approx(rp._accuracy, abs=1e-6)

        rp2 = RegimePredictor.load(pkl_path)
        assert rp2._oos_metrics["accuracy"] == 0.42
        assert rp2._oos_metrics["macro_f1"] == 0.39
        assert rp2._train_metrics["confusion_matrix"] == meta["in_sample"]["confusion_matrix"]

    def test_empty_oos_metrics_serialize_cleanly(self, tmp_path):
        """Missing OOS metrics (e.g., training without the fold-aggregated collector)
        must not break save/load — sidecar just carries an empty dict."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(150, len(RegimePredictor.FEATURE_NAMES)))
        y = rng.integers(0, 3, size=150)
        rp = RegimePredictor().fit(X, y)

        pkl_path = tmp_path / "rp.pkl"
        rp.save(pkl_path)
        rp2 = RegimePredictor.load(pkl_path)
        assert rp2._oos_metrics == {}
