"""
model/regime_predictor.py — Market regime predictor.

Classifies the current market environment into bull/neutral/bear using
macro indicators (SPY returns, VIX, yield curve, market breadth). Outputs
posterior probabilities rather than hard classifications, enabling the
meta-model to blend strategies proportionally during transitions.

Starts with multinomial logistic regression (simple, interpretable).
Upgrade path: Bayesian HMM for regime persistence modeling.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Regime labels derived from subsequent 20-day SPY return
REGIME_LABELS = ["bear", "neutral", "bull"]
REGIME_MAP = {"bear": 0, "neutral": 1, "bull": 2}

# Thresholds for labeling historical regimes from SPY 20d forward return
BEAR_THRESHOLD = -0.03   # SPY down >3% over next 20d → bear
BULL_THRESHOLD = 0.03    # SPY up >3% over next 20d → bull


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> dict:
    """
    Compute classification metrics for regime predictions.

    Returns dict with:
      - accuracy: overall classification accuracy
      - confusion_matrix: 3×3 matrix (rows=true, cols=predicted), class order bear/neutral/bull
      - per_class_precision: [bear, neutral, bull]
      - per_class_recall: [bear, neutral, bull]
      - per_class_support: [bear, neutral, bull] row counts
      - macro_f1: unweighted mean of per-class F1 (balanced metric,
                  guards against majority-class-only classifiers)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    n = len(y_true)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    accuracy = float((y_true == y_pred).mean()) if n > 0 else 0.0

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    support = cm.sum(axis=1)  # row sums = true-class counts
    pred_counts = cm.sum(axis=1)  # reused below

    for k in range(n_classes):
        col_sum = cm[:, k].sum()
        row_sum = cm[k, :].sum()
        precision[k] = float(cm[k, k] / col_sum) if col_sum > 0 else 0.0
        recall[k] = float(cm[k, k] / row_sum) if row_sum > 0 else 0.0

    f1 = np.zeros(n_classes)
    for k in range(n_classes):
        denom = precision[k] + recall[k]
        f1[k] = float(2 * precision[k] * recall[k] / denom) if denom > 0 else 0.0
    macro_f1 = float(f1.mean())

    return {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_precision": [round(p, 4) for p in precision],
        "per_class_recall": [round(r, 4) for r in recall],
        "per_class_support": [int(s) for s in support],
        "macro_f1": round(macro_f1, 4),
        "n_samples": int(n),
        "class_order": REGIME_LABELS,
    }


class RegimePredictor:
    """Predict market regime probabilities from macro indicators."""

    # Feature names expected in the input array (order matters)
    FEATURE_NAMES = [
        "spy_20d_return",
        "spy_20d_vol",
        "vix_level",
        "vix_term_slope",
        "yield_curve_slope",
        "market_breadth",
    ]

    def __init__(self):
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._accuracy = None          # in-sample training accuracy (legacy field)
        self._train_metrics: dict = {}  # full in-sample metrics (confusion matrix, F1, etc.)
        self._oos_metrics: dict = {}    # walk-forward OOS metrics (populated by caller)

    def build_features(
        self,
        spy_series: pd.Series,
        vix_series: pd.Series | None = None,
        vix3m_series: pd.Series | None = None,
        tnx_series: pd.Series | None = None,
        irx_series: pd.Series | None = None,
        all_close_prices: dict[str, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """
        Build regime feature matrix from macro time series.

        Returns DataFrame indexed by date with columns matching FEATURE_NAMES.
        """
        df = pd.DataFrame(index=spy_series.index)

        # SPY 20-day return
        df["spy_20d_return"] = (spy_series / spy_series.shift(20)) - 1.0

        # SPY 20-day realized volatility (annualized)
        log_ret = np.log(spy_series / spy_series.shift(1))
        df["spy_20d_vol"] = log_ret.rolling(20).std() * np.sqrt(252)

        # VIX level (normalized by baseline ~20)
        if vix_series is not None:
            vix_aligned = vix_series.reindex(df.index, method="ffill")
            df["vix_level"] = vix_aligned / 20.0
        else:
            df["vix_level"] = 1.0

        # VIX term structure slope
        if vix_series is not None and vix3m_series is not None:
            vix_aligned = vix_series.reindex(df.index, method="ffill")
            vix3m_aligned = vix3m_series.reindex(df.index, method="ffill")
            df["vix_term_slope"] = (vix_aligned - vix3m_aligned) / 20.0
        else:
            df["vix_term_slope"] = 0.0

        # Yield curve slope (10Y - 3M, normalized)
        if tnx_series is not None and irx_series is not None:
            tnx_aligned = tnx_series.reindex(df.index, method="ffill")
            irx_aligned = irx_series.reindex(df.index, method="ffill")
            df["yield_curve_slope"] = (tnx_aligned - irx_aligned) / 10.0
        else:
            df["yield_curve_slope"] = 0.0

        # Market breadth: % of stocks above 50-day MA
        if all_close_prices and len(all_close_prices) >= 10:
            breadth_by_date: dict[pd.Timestamp, list[int]] = {}
            for ticker, close_s in all_close_prices.items():
                if close_s is None or len(close_s) < 50:
                    continue
                ma50 = close_s.rolling(50).mean()
                above = (close_s > ma50).astype(int)
                for dt, val in above.items():
                    if pd.notna(val) and dt in df.index:
                        breadth_by_date.setdefault(dt, []).append(int(val))

            breadth_series = pd.Series(
                {dt: np.mean(vals) for dt, vals in breadth_by_date.items() if len(vals) >= 10},
                dtype=float,
            )
            df["market_breadth"] = breadth_series.reindex(df.index, method="ffill").fillna(0.5)
        else:
            df["market_breadth"] = 0.5  # neutral default

        return df.dropna()

    def build_labels(self, spy_series: pd.Series) -> pd.Series:
        """
        Label each date with a regime based on subsequent 20-day SPY return.

        bear (0):    SPY 20d forward return < -3%
        neutral (1): SPY 20d forward return in [-3%, +3%]
        bull (2):    SPY 20d forward return > +3%
        """
        fwd_20d = (spy_series.shift(-20) / spy_series) - 1.0
        labels = pd.Series(1, index=spy_series.index, dtype=int)  # default neutral
        labels[fwd_20d < BEAR_THRESHOLD] = 0
        labels[fwd_20d > BULL_THRESHOLD] = 2
        return labels.dropna()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegimePredictor":
        """
        Fit multinomial logistic regression.

        Parameters
        ----------
        X : array of shape (N, 6) — macro features
        y : array of shape (N,) — regime labels (0=bear, 1=neutral, 2=bull)
        """
        from sklearn.linear_model import LogisticRegression

        # class_weight='balanced' adjusts the loss to treat minority-class
        # errors proportionally more costly — n_samples / (n_classes * count_k)
        # for class k. Without this, the 16%/49%/35% bear/neutral/bull split
        # under L2=1.0 regularization collapses the bear coefficient to zero
        # and the classifier never predicts bear on held-out folds. Smoke
        # test 2026-04-16 confirmed that failure mode; re-run after adding
        # this lifts bear recall above the per-class floor gate below.
        self._model = LogisticRegression(
            C=1.0, solver="lbfgs",
            max_iter=1000, random_state=42,
            class_weight="balanced",
        )
        self._model.fit(X, y)
        self._fitted = True
        self._n_samples = len(y)

        # In-sample metrics (confusion matrix, per-class precision/recall, macro-F1).
        # Retain self._accuracy scalar for legacy consumers (manifest, result dict).
        preds = self._model.predict(X)
        self._train_metrics = compute_classification_metrics(y, preds)
        self._accuracy = self._train_metrics["accuracy"]

        log.info(
            "RegimePredictor fitted: n=%d  in-sample_acc=%.2f%%  macro_f1=%.3f  classes=%s",
            self._n_samples, self._accuracy * 100, self._train_metrics["macro_f1"],
            dict(zip(*np.unique(y, return_counts=True))),
        )
        cm = np.array(self._train_metrics["confusion_matrix"])
        log.info("  In-sample confusion matrix (rows=true, cols=pred, order bear/neutral/bull):")
        for i, row in enumerate(cm):
            log.info("    %s: %s", REGIME_LABELS[i], row.tolist())
        return self

    def set_oos_metrics(self, metrics: dict) -> None:
        """
        Attach walk-forward OOS evaluation metrics to this model instance.

        Called by the trainer after aggregating per-fold regime OOS predictions.
        Persisted via save() to the meta.json sidecar. Consumers (manifest,
        promotion gate, email) read these as the trustworthy accuracy signal
        — self._accuracy remains in-sample and is diagnostic only.
        """
        self._oos_metrics = metrics or {}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.

        Returns array of shape (N, 3) with columns [P(bear), P(neutral), P(bull)].
        """
        if not self._fitted:
            raise RuntimeError("RegimePredictor not fitted")
        return self._model.predict_proba(X)

    def predict_single(self, features: dict) -> dict:
        """
        Predict regime for a single observation (today's macro data).

        Parameters
        ----------
        features : dict with keys matching FEATURE_NAMES

        Returns
        -------
        {"regime_bear": float, "regime_neutral": float, "regime_bull": float}
        """
        x = np.array([[features.get(f, 0.0) for f in self.FEATURE_NAMES]])
        proba = self.predict_proba(x)[0]
        return {
            "regime_bear": round(float(proba[0]), 4),
            "regime_neutral": round(float(proba[1]), 4),
            "regime_bull": round(float(proba[2]), 4),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        # Legacy scalar `accuracy` retained for backwards-compatibility with
        # downstream readers (manifest, email, dashboard) that predate the
        # walk-forward split. It mirrors in_sample.accuracy.
        meta = {
            "type": "regime_predictor",
            "n_samples": self._n_samples,
            "accuracy": round(self._accuracy, 4) if self._accuracy else None,
            "features": self.FEATURE_NAMES,
            "in_sample": self._train_metrics,
            "oos": self._oos_metrics,
        }
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        log.info(
            "RegimePredictor saved to %s (in_sample_acc=%.3f, oos_acc=%s)",
            path, self._accuracy or 0.0,
            self._oos_metrics.get("accuracy", "n/a"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegimePredictor":
        path = Path(path)
        rp = cls()
        with open(path, "rb") as f:
            rp._model = pickle.load(f)
        rp._fitted = True
        meta_path = Path(str(path) + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            rp._n_samples = meta.get("n_samples", 0)
            rp._accuracy = meta.get("accuracy")
            rp._train_metrics = meta.get("in_sample", {}) or {}
            rp._oos_metrics = meta.get("oos", {}) or {}
        log.info("RegimePredictor loaded from %s", path)
        return rp
