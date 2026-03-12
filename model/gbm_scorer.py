"""
model/gbm_scorer.py — LightGBM alpha scorer.

Wraps LightGBM with the same fit/predict/save/load interface as
DirectionPredictor so the two scorers compose cleanly in EnsemblePredictor.

Training objective: regression (MSE) on 5-day relative return vs SPY.
Evaluation metric: Pearson IC (same gate as MLP: >0.05).

LightGBM advantages over the MLP for tabular financial data:
  - Scale-invariant by default (no z-score normalisation needed)
  - Captures non-linear threshold interactions (e.g. RSI>70 AND dist_52w_high<-0.1)
  - Built-in feature importance (gain-based and SHAP)
  - Faster to train and tune than the MLP on CPU
  - Early stopping on val IC prevents overfitting without manual patience tuning

Upgrade path: change objective='regression' → 'lambdarank' once baseline IC is
confirmed, which directly optimises ranking (NDCG) rather than prediction error.

Usage:
    from model.gbm_scorer import GBMScorer
    scorer = GBMScorer()
    scorer.fit(X_train, y_train, X_val, y_val)
    preds = scorer.predict(X_test)   # continuous alpha scores, shape (N,)
    scorer.save('checkpoints/gbm_best.txt')

    # reload
    scorer2 = GBMScorer.load('checkpoints/gbm_best.txt')
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Bump when interface or default hyperparameters change.
GBM_VERSION = "v1.0.0"


class GBMScorer:
    """
    LightGBM wrapper for cross-sectional alpha scoring.

    Parameters
    ----------
    params : dict, optional
        LightGBM training parameters. If None, sensible financial-data
        defaults are used (see _default_params()).
    n_estimators : int
        Maximum number of boosting rounds. Early stopping will halt
        before this if val IC stops improving.
    early_stopping_rounds : int
        Stop training if val loss doesn't improve for this many rounds.
    verbose : int
        LightGBM verbosity (-1=silent, 0=warn, 1=info).
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_estimators: int = 2000,
        early_stopping_rounds: int = 50,
        verbose: int = -1,
    ) -> None:
        self.params = params or self._default_params()
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self._booster = None          # set after fit()
        self._feature_names: list[str] = []
        self._best_iteration: int = 0
        self._val_ic: float = 0.0

    # ------------------------------------------------------------------
    # Default hyperparameters
    # ------------------------------------------------------------------

    @staticmethod
    def _default_params() -> dict[str, Any]:
        """
        Conservative defaults tuned for financial cross-sectional alpha.

        Key choices:
          num_leaves=63      — moderate complexity; avoids overfit on regime-specific patterns
          min_child_samples=200 — requires 200 samples per leaf; prevents fitting micro-regimes
          feature_fraction=0.8  — subsample 80% of features per tree (Random Forest effect)
          bagging_fraction=0.8  — subsample 80% of rows per tree (reduces variance)
          bagging_freq=5        — resample every 5 rounds
          lambda_l1/l2          — L1+L2 regularisation (sparse + smooth solutions)
          learning_rate=0.05    — moderate; early stopping prevents premature halt

        These are starting defaults. train_gbm.py tunes num_leaves, min_child_samples,
        feature_fraction, and learning_rate via Optuna.
        """
        return {
            "objective": "regression",
            "metric": "mse",           # training metric; IC computed separately at eval
            "num_leaves": 63,
            "min_child_samples": 200,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "num_threads": 4,
            "verbosity": -1,
            "seed": 42,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "GBMScorer":
        """
        Train the LightGBM booster with early stopping on validation MSE.

        Parameters
        ----------
        X_train : np.ndarray, shape (N_train, n_features)
        y_train : np.ndarray, shape (N_train,) — forward_return_5d (alpha vs SPY)
        X_val   : np.ndarray, shape (N_val, n_features)
        y_val   : np.ndarray, shape (N_val,) — forward_return_5d (alpha vs SPY)
        feature_names : list of str, optional — stored for feature importance.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm\n"
                "On macOS you also need: brew install libomp"
            )

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=self._feature_names, free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, feature_name=self._feature_names,
            reference=train_data, free_raw_data=False
        )

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100 if self.verbose >= 0 else 0),
        ]

        log.info(
            "GBMScorer.fit: train=%d  val=%d  n_features=%d  n_estimators=%d  "
            "early_stopping=%d",
            len(y_train), len(y_val), X_train.shape[1],
            self.n_estimators, self.early_stopping_rounds,
        )

        self._booster = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        self._best_iteration = self._booster.best_iteration

        # Compute and log val IC at best iteration
        val_preds = self._booster.predict(X_val, num_iteration=self._best_iteration)
        self._val_ic = float(np.corrcoef(val_preds, y_val)[0, 1])

        log.info(
            "GBMScorer training complete: best_iteration=%d  val_IC=%.4f",
            self._best_iteration, self._val_ic,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous alpha scores.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        np.ndarray, shape (N,) — predicted 5-day alpha vs SPY (continuous).
        """
        if self._booster is None:
            raise RuntimeError("GBMScorer has not been fitted. Call fit() first.")
        return self._booster.predict(X, num_iteration=self._best_iteration)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """
        Return a dict mapping feature name → importance score.

        Parameters
        ----------
        importance_type : 'gain' (total gain, preferred) or 'split' (split count).
        """
        if self._booster is None:
            raise RuntimeError("Model not fitted.")
        scores = self._booster.feature_importance(importance_type=importance_type)
        return dict(zip(self._feature_names, scores.tolist()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the booster to disk.

        Saves two files:
          <path>          — LightGBM booster text format (portable, version-stable)
          <path>.meta.json — metadata: feature names, best_iteration, val_IC, version
        """
        if self._booster is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._booster.save_model(str(path))

        meta = {
            "gbm_version": GBM_VERSION,
            "feature_names": self._feature_names,
            "best_iteration": self._best_iteration,
            "val_ic": round(self._val_ic, 6),
            "params": self.params,
            "n_estimators": self.n_estimators,
        }
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        log.info("GBMScorer saved to %s  (val_IC=%.4f)", path, self._val_ic)

    @classmethod
    def load(cls, path: str | Path) -> "GBMScorer":
        """
        Load a previously saved GBMScorer from disk.

        Parameters
        ----------
        path : path to the booster file saved by save() (without .meta.json suffix).
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GBM booster not found: {path}")

        booster = lgb.Booster(model_file=str(path))

        meta_path = Path(str(path) + ".meta.json")
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        scorer = cls(
            params=meta.get("params"),
            n_estimators=meta.get("n_estimators", 2000),
        )
        scorer._booster = booster
        scorer._feature_names = meta.get("feature_names", [])
        scorer._best_iteration = meta.get("best_iteration", 0)
        scorer._val_ic = meta.get("val_ic", 0.0)

        log.info(
            "GBMScorer loaded from %s  (val_IC=%.4f  best_iter=%d)",
            path, scorer._val_ic, scorer._best_iteration,
        )
        return scorer

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = (
            f"fitted: best_iter={self._best_iteration} val_IC={self._val_ic:.4f}"
            if self._booster else "not fitted"
        )
        return f"GBMScorer({status})"
