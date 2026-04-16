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
#
# Layer-1 outputs and research context come first; raw macro features follow
# the classifier's compressed outputs (regime_bull, regime_bear). The raw
# features give Ridge a direct linear channel to the same macro signal the
# softmax classifier compresses — complementary information when the
# classifier's compression drops structure. Feature importance analysis
# (standardized coefficients + permutation) on the first post-deploy run
# tells us whether the classifier columns are pulling weight beyond what
# the raw columns already explain, informing a pruning decision.
META_FEATURES = [
    "research_calibrator_prob",   # P(research signal is correct)
    "momentum_score",             # momentum model output (continuous)
    "expected_move",              # volatility model output (continuous)
    "regime_bull",                # P(bull regime) — classifier output
    "regime_bear",                # P(bear regime) — classifier output
    "research_composite_score",   # raw research score (0-100, normalized to 0-1)
    "research_conviction",        # rising=1, stable=0, declining=-1
    "sector_macro_modifier",      # sector modifier from research (0.7-1.3, centered at 1)
    # Raw macro features (E): same 6 inputs the regime classifier consumes,
    # exposed directly to the meta ridge. Single market-wide value per date;
    # all tickers on a given date share these columns.
    "macro_spy_20d_return",       # trailing 20d SPY return
    "macro_spy_20d_vol",          # annualized 20d realized vol
    "macro_vix_level",            # VIX normalized by baseline 20
    "macro_vix_term_slope",       # (VIX - VIX3M) / 20
    "macro_yield_curve_slope",    # (TNX - IRX) / 10
    "macro_market_breadth",       # % of universe above 50d MA
]

# Raw macro feature names — used by trainer/inference to look up values from
# regime_features_df / latest_regime and to keep source-of-truth in one place.
MACRO_FEATURE_NAMES = [
    "spy_20d_return",
    "spy_20d_vol",
    "vix_level",
    "vix_term_slope",
    "yield_curve_slope",
    "market_breadth",
]
# Mapping from regime-classifier column name → meta-feature column name.
# Prefix avoids a collision if a downstream consumer ever adds its own
# `vix_level` ticker-local feature.
MACRO_FEATURE_META_MAP = {k: f"macro_{k}" for k in MACRO_FEATURE_NAMES}


class MetaModel:
    """Ridge regression stacker for Layer 1 model outputs."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._val_ic = 0.0
        self._coefficients: dict[str, float] = {}
        # Feature list the stored ridge was fit on. Persisted in .meta.json so
        # inference can adapt to the deployed schema rather than assuming the
        # module-level META_FEATURES (which changes as we extend the stack).
        # Without this, any META_FEATURES addition would break all inferences
        # against the previously-trained model.
        self._feature_names: list[str] = []

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
        self._feature_names = list(names)
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
        """Predict for a single ticker given a feature dict.

        Uses the feature list the stored ridge was fit on (self._feature_names)
        rather than the module-level META_FEATURES, so a deployed model trained
        on an older schema still serves predictions correctly after new
        features are appended to META_FEATURES in code.
        """
        feat_names = self._feature_names or META_FEATURES
        x = np.array([[features.get(f, 0.0) for f in feat_names]])
        return float(self.predict(x)[0])

    def metrics(self) -> dict:
        return {
            "type": "meta_model_ridge",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "val_ic": round(self._val_ic, 6),
            "alpha": self.alpha,
            "coefficients": self._coefficients,
            "feature_names": self._feature_names,
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
            mm._feature_names = meta.get("feature_names", []) or []
        # Backwards-compat: a model saved before feature_names was persisted
        # (pre-PR #34) has only `coefficients`. Reconstruct feature_names from
        # the coefficients dict, excluding the intercept. This keeps previously
        # deployed models serving correctly until the next training cycle
        # re-saves with the new schema.
        if not mm._feature_names and mm._coefficients:
            mm._feature_names = [k for k in mm._coefficients.keys() if k != "intercept"]
        log.info(
            "MetaModel loaded from %s (IC=%.4f, n_features=%d)",
            path, mm._val_ic, len(mm._feature_names),
        )
        return mm
