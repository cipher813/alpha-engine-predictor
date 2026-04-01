"""
model/catboost_scorer.py — CatBoost regression model for alpha prediction.

Mirrors the GBMScorer interface (fit, predict, save, load, feature_importance)
to enable seamless ensemble blending with LightGBM. CatBoost uses ordered
boosting and native categorical handling, producing complementary predictions
that reduce ensemble variance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CATBOOST_VERSION = "v1.0.0"


class CatBoostScorer:
    """CatBoost regression model for cross-sectional alpha prediction."""

    def __init__(
        self,
        params: dict | None = None,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 50,
        verbose: int = 0,
    ):
        self.params = params or self._default_params()
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self._model = None
        self._feature_names: list[str] = []
        self._best_iteration: int = 0
        self._val_ic: float = 0.0

    @staticmethod
    def _default_params() -> dict:
        return {
            "loss_function": "RMSE",
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "thread_count": 4,
            "verbose": 0,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs,
    ) -> "CatBoostScorer":
        """Train CatBoost regressor with early stopping."""
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError:
            raise ImportError("catboost is not installed. Run: pip install catboost")

        self._feature_names = list(feature_names) if feature_names else [
            f"f{i}" for i in range(X_train.shape[1])
        ]

        merged_params = {**self._default_params(), **(self.params or {})}
        merged_params["iterations"] = self.n_estimators
        merged_params.pop("verbose", None)

        self._model = CatBoostRegressor(**merged_params)

        train_pool = Pool(X_train, label=y_train, feature_names=self._feature_names)
        val_pool = Pool(X_val, label=y_val, feature_names=self._feature_names)

        self._model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
        )

        self._best_iteration = self._model.best_iteration_ or self.n_estimators

        # Compute val IC
        val_preds = self._model.predict(X_val)
        if len(val_preds) > 1 and np.std(val_preds) > 1e-10 and np.std(y_val) > 1e-10:
            self._val_ic = float(np.corrcoef(val_preds, y_val)[0, 1])
        else:
            self._val_ic = 0.0

        log.info(
            "CatBoostScorer trained: best_iter=%d  val_IC=%.4f  n_features=%d",
            self._best_iteration, self._val_ic, X_train.shape[1],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous alpha scores. Shape (N, n_features) -> (N,)."""
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict(X).astype(np.float64)

    def feature_importance(self, importance_type: str = "PredictionValuesChange") -> dict:
        """Return feature importance dict. Default type is PredictionValuesChange (gain equivalent)."""
        if self._model is None:
            raise RuntimeError("Model not fitted")
        scores = self._model.get_feature_importance(type=importance_type)
        return dict(zip(self._feature_names, scores.tolist()))

    def save(self, path: str | Path) -> None:
        """Save CatBoost model to native format + metadata JSON."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path))

        meta = {
            "catboost_version": CATBOOST_VERSION,
            "feature_names": self._feature_names,
            "best_iteration": self._best_iteration,
            "val_ic": round(self._val_ic, 6),
            "params": self.params,
            "n_estimators": self.n_estimators,
        }
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        log.info("CatBoostScorer saved to %s  (val_IC=%.4f)", path, self._val_ic)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostScorer":
        """Load a previously saved CatBoostScorer."""
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("catboost is not installed. Run: pip install catboost")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CatBoost model not found: {path}")

        model = CatBoostRegressor()
        model.load_model(str(path))

        meta_path = Path(str(path) + ".meta.json")
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        scorer = cls(
            params=meta.get("params"),
            n_estimators=meta.get("n_estimators", 2000),
        )
        scorer._model = model
        scorer._feature_names = meta.get("feature_names", [])
        scorer._best_iteration = meta.get("best_iteration", 0)
        scorer._val_ic = meta.get("val_ic", 0.0)
        log.info("CatBoostScorer loaded from %s  (val_IC=%.4f)", path, scorer._val_ic)
        return scorer
