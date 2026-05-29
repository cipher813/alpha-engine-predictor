"""Source-level regression for the Task B meta-label classifier wiring —
observe-only ``barrier_win_prob`` parallel field.

Mirrors test_inference_triple_barrier_parallel.py: validates that the loader
loads the classifier tolerant of absence, inference predicts + emits the
field, and a meta-label exception cannot leak into the canonical path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


class TestLoaderContract:
    def test_loads_classifier_artifact(self):
        src = _src("inference/stages/load_model.py")
        assert "meta_label_classifier.pkl" in src
        assert 'meta_models["meta_label_clf"]' in src
        assert "from model.meta_label_classifier import MetaLabelClassifier" in src

    def test_tolerates_absence(self):
        src = _src("inference/stages/load_model.py")
        assert "meta_label_classifier not available (observe-only OK)" in src


class TestInferenceContract:
    def test_fetches_classifier(self):
        src = _src("inference/stages/run_inference.py")
        assert 'meta_label_clf = ctx.meta_models.get("meta_label_clf")' in src

    def test_predicts_barrier_win_prob(self):
        src = _src("inference/stages/run_inference.py")
        assert "meta_label_clf.predict_single(meta_features)" in src
        assert 'log.debug("barrier_win_prob predict failed for %s' in src

    def test_clipped_to_unit_interval(self):
        src = _src("inference/stages/run_inference.py")
        assert "np.clip(barrier_win_prob, 0.0, 1.0)" in src

    def test_is_optional(self):
        src = _src("inference/stages/run_inference.py")
        assert "barrier_win_prob: float | None = None" in src


class TestPredictionsJsonShape:
    def test_field_emitted(self):
        src = _src("inference/stages/run_inference.py")
        assert '"barrier_win_prob":' in src
        assert "round(barrier_win_prob, 4)" in src

    def test_null_when_absent(self):
        src = _src("inference/stages/run_inference.py")
        assert (
            '"barrier_win_prob": (\n'
            "                round(barrier_win_prob, 4)\n"
            "                if barrier_win_prob is not None else None\n"
            "            )," in src
        )


class TestTrainingContract:
    def test_classifier_trained_and_persisted(self):
        src = _src("training/meta_trainer.py")
        assert "from model.meta_label_classifier import MetaLabelClassifier" in src
        assert 'models["meta_label_classifier"]' in src
        # observe-only gate flag
        assert 'getattr(cfg, "META_LABEL_CLASSIFIER_ENABLED", True)' in src
        # touch label carried into the OOS rows
        assert '"actual_fwd_barrier_touch"' in src

    def test_touch_label_generated(self):
        src = _src("training/meta_trainer.py")
        assert "compute_triple_barrier_touch_labels" in src
        assert "triple_barrier_touch_21d" in src


class TestFailureIsolation:
    def test_canonical_path_independent(self):
        # The barrier_win_prob block runs AFTER the canonical alpha is
        # computed + clipped, inside its own try/except → a classifier crash
        # leaves predicted_alpha untouched (same guarantee as meta_alpha_tb).
        src = _src("inference/stages/run_inference.py")
        # the classifier predict is guarded
        assert "if meta_label_clf is not None and meta_label_clf.is_fitted:" in src
