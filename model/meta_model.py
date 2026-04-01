"""
model/meta_model.py — Meta-model that stacks Layer 1 specialized model outputs.

Ridge regression that learns the optimal combination of:
  - Research calibrator P(signal correct)
  - Momentum model score
  - Volatility model expected move
  - Regime predictor P(bull), P(bear)
  - Research composite score and context

Intentionally simple (ridge, not GBM) to avoid overfitting on ~10 inputs.
Trained on out-of-fold predictions from Layer 1 walk-forward validation.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Meta-model input features (order must match at training and inference)
META_FEATURES = [
    "research_calibrator_prob",   # P(research signal is correct)
    "momentum_score",             # momentum model output (continuous)
    "expected_move",              # volatility model output (continuous)
    "regime_bull",                # P(bull regime)
    "regime_bear",                # P(bear regime)
    "research_composite_score",   # raw research score (0-100, normalized to 0-1)
    "research_conviction",        # rising=1, stable=0, declining=-1
    "sector_macro_modifier",      # sector modifier from research (0.7-1.3, centered at 1)
]


class MetaModel:
    """Ridge regression stacker for Layer 1 model outputs."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._val_ic = 0.0
        self._coefficients: dict[str, float] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "MetaModel":
        """
        Fit ridge regression on Layer 1 OOS outputs.

        Parameters
        ----------
        X : array of shape (N, n_meta_features) — stacked Layer 1 outputs
        y : array of shape (N,) — actual forward returns or binary outcomes
        feature_names : names for coefficient reporting
        """
        from sklearn.linear_model import Ridge

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Remove NaN rows
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        if len(y) < 20:
            log.warning("MetaModel: only %d valid samples (need 20+) — skipping fit", len(y))
            return self

        self._n_samples = len(y)
        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model.fit(X, y)
        self._fitted = True

        # Store coefficients for interpretability
        names = feature_names or META_FEATURES[:X.shape[1]]
        self._coefficients = {
            name: round(float(coef), 6)
            for name, coef in zip(names, self._model.coef_)
        }
        self._coefficients["intercept"] = round(float(self._model.intercept_), 6)

        # Compute training IC
        preds = self._model.predict(X)
        if np.std(preds) > 1e-10 and np.std(y) > 1e-10:
            self._val_ic = float(np.corrcoef(preds, y)[0, 1])

        log.info(
            "MetaModel fitted: n=%d  IC=%.4f  alpha=%.1f",
            self._n_samples, self._val_ic, self.alpha,
        )
        for name, coef in sorted(self._coefficients.items(), key=lambda x: -abs(x[1])):
            if name != "intercept":
                log.info("  %s: %.4f", name, coef)

        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict final alpha scores. Shape (N, n_features) → (N,)."""
        if not self._fitted:
            raise RuntimeError("MetaModel not fitted")
        return self._model.predict(np.asarray(X, dtype=np.float64))

    def predict_single(self, features: dict) -> float:
        """Predict for a single ticker given a feature dict."""
        x = np.array([[features.get(f, 0.0) for f in META_FEATURES]])
        return float(self.predict(x)[0])

    def metrics(self) -> dict:
        return {
            "type": "meta_model_ridge",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "val_ic": round(self._val_ic, 6),
            "alpha": self.alpha,
            "coefficients": self._coefficients,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        meta = self.metrics()
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        log.info("MetaModel saved to %s (IC=%.4f)", path, self._val_ic)

    @classmethod
    def load(cls, path: str | Path) -> "MetaModel":
        path = Path(path)
        mm = cls()
        with open(path, "rb") as f:
            mm._model = pickle.load(f)
        mm._fitted = True
        meta_path = Path(str(path) + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            mm._n_samples = meta.get("n_samples", 0)
            mm._val_ic = meta.get("val_ic", 0.0)
            mm.alpha = meta.get("alpha", 1.0)
            mm._coefficients = meta.get("coefficients", {})
        log.info("MetaModel loaded from %s (IC=%.4f)", path, mm._val_ic)
        return mm
