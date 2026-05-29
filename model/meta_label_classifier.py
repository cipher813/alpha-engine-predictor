"""
model/meta_label_classifier.py — López de Prado meta-label classifier (Task B).

Predicts ``P(up/profit barrier touched before down/stop barrier)`` from the
SAME ``META_FEATURES`` vector the L2 Ridge stacker consumes (the Layer-1 outputs
+ research context + raw macros). This is the meta-labeling companion to the
primary alpha model (LdP *Advances in Financial Machine Learning* Ch. 3.6):
the primary model says *what* to trade; the meta-label says *how confident the
profit barrier is reached before the stop* → a sizing signal.

The output is a CALIBRATED probability (Platt/sigmoid over a logistic base)
because it feeds executor position sizing **linearly** — a miscalibrated score
would systematically distort position sizes. Trained on out-of-fold Layer-1
predictions (same OOS rows as the Ridge), supervised on
``labeling.triple_barrier.triple_barrier_touch_order``.

OBSERVE-ONLY at introduction: inference emits ``barrier_win_prob`` into
predictions.json; no live consumer reads it until the executor sizing gate
flips (Task B2). Mirrors the ``meta_model_tb`` observe-only lifecycle.

Interface deliberately mirrors ``model.meta_model.MetaModel`` (fit / predict /
predict_single / is_fitted / metrics / save / load) so the training and
inference plumbing treat it the same way.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from model.meta_model import META_FEATURES

log = logging.getLogger(__name__)

# Minimum valid rows before a fit is attempted. Below this, calibrated CV is
# unreliable and we skip (leaving the model unfitted → inference emits null).
_MIN_FIT_SAMPLES = 50


class MetaLabelClassifier:
    """Calibrated binary classifier: P(up barrier before down barrier)."""

    def __init__(self):
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._auc = 0.0
        self._brier = 0.0
        self._base_rate = 0.5  # P(y=1) in the training set; the naive baseline
        self._feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "MetaLabelClassifier":
        """Fit a calibrated logistic classifier on Layer-1 OOS outputs.

        Parameters
        ----------
        X : (N, n_features) — stacked Layer-1 outputs + macros (META_FEATURES).
        y : (N,) — binary touch-order labels (1=up first, 0=down first). NaN
            rows (timeouts under the "nan" policy, tail) are dropped here.
        feature_names : names for the persisted feature schema.
        sample_weight : (N,) optional — LdP Ch. 4.4 average-uniqueness weights,
            routed to the base logistic estimator so heavily-overlapping
            consecutive-day labels don't double-count.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
            valid &= np.isfinite(sample_weight)
            sample_weight = sample_weight[valid]
        X = X[valid]
        y = y[valid].astype(int)

        if len(y) < _MIN_FIT_SAMPLES or len(np.unique(y)) < 2:
            log.warning(
                "MetaLabelClassifier: %d valid rows / %d classes (need >=%d rows, "
                "2 classes) — skipping fit",
                len(y), len(np.unique(y)), _MIN_FIT_SAMPLES,
            )
            return self

        self._n_samples = len(y)
        self._base_rate = float(y.mean())
        names = feature_names or META_FEATURES[: X.shape[1]]
        self._feature_names = list(names)

        # Logistic base (interpretable, robust at small N) + class_weight to
        # absorb base-rate skew. Wrapped in CalibratedClassifierCV (sigmoid/
        # Platt, internal CV) so predict_proba is a calibrated probability —
        # load-bearing because the executor scales position size by it.
        base = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self._model.fit(X, y, sample_weight=sample_weight)
        self._fitted = True

        # In-sample diagnostics (honest walk-forward AUC is the backtester's
        # job; these are the run-to-run observability read, like MetaModel's
        # _val_ic). Baseline = always predict base rate → AUC 0.5, Brier =
        # base_rate*(1-base_rate).
        probs = self._model.predict_proba(X)[:, 1]
        if len(np.unique(y)) == 2:
            self._auc = float(roc_auc_score(y, probs))
        self._brier = float(brier_score_loss(y, probs))

        log.info(
            "MetaLabelClassifier fitted: n=%d  base_rate=%.3f  AUC=%.4f  Brier=%.4f "
            "(baseline AUC=0.500, Brier=%.4f)",
            self._n_samples, self._base_rate, self._auc, self._brier,
            self._base_rate * (1.0 - self._base_rate),
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P(up barrier before down barrier) per row. (N, n_features) → (N,)."""
        if not self._fitted:
            raise RuntimeError("MetaLabelClassifier not fitted")
        return self._model.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]

    def predict_single(self, features: dict) -> float:
        """P(up before down) for one ticker given a feature dict.

        Uses the persisted feature schema so a model trained on an older
        META_FEATURES list still serves after new features are appended.
        """
        feat_names = self._feature_names or META_FEATURES
        x = np.array([[features.get(f, 0.0) for f in feat_names]], dtype=np.float64)
        return float(self.predict(x)[0])

    def metrics(self) -> dict:
        return {
            "type": "meta_label_classifier_calibrated_logistic",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "auc": round(self._auc, 6),
            "brier": round(self._brier, 6),
            "base_rate": round(self._base_rate, 6),
            "feature_names": self._feature_names,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        Path(str(path) + ".meta.json").write_text(json.dumps(self.metrics(), indent=2))
        log.info("MetaLabelClassifier saved to %s (AUC=%.4f)", path, self._auc)

    @classmethod
    def load(cls, path: str | Path) -> "MetaLabelClassifier":
        path = Path(path)
        clf = cls()
        with open(path, "rb") as f:
            clf._model = pickle.load(f)
        clf._fitted = True
        meta_path = Path(str(path) + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            clf._n_samples = meta.get("n_samples", 0)
            clf._auc = meta.get("auc", 0.0)
            clf._brier = meta.get("brier", 0.0)
            clf._base_rate = meta.get("base_rate", 0.5)
            clf._feature_names = meta.get("feature_names", []) or []
        log.info(
            "MetaLabelClassifier loaded from %s (AUC=%.4f, n_features=%d)",
            path, clf._auc, len(clf._feature_names),
        )
        return clf
