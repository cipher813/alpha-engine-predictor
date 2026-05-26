"""
model/meta_model.py — Meta-model that stacks Layer 1 specialized model outputs.

Bayesian Ridge regression that learns the optimal combination of:
  - Research calibrator P(signal correct)
  - Momentum model score
  - Volatility model expected move
  - Regime predictor P(bull), P(bear)
  - Research composite score and context

Bayesian Ridge (Tipping 2001) generalizes plain Ridge by learning the
regularization strength α via Type-II marginal likelihood, AND by exposing
a closed-form posterior predictive variance at each prediction point:

    Var[ŷ*] = σ²_n + x*ᵀ Σ_w x*

This `predicted_alpha_std` field is the load-bearing input to the
α̂-uncertainty term in the executor's MVO objective (workstream B of
optimizer-sota-upgrades-260526.md). With it, confident picks size up and
diffuse picks size down — without it, the optimizer treats α̂ = 0.05 ±
0.002 and α̂ = 0.05 ± 0.04 identically.

Intentionally simple (Bayesian ridge, not GBM) to avoid overfitting on
~10 inputs. Trained on out-of-fold predictions from Layer 1 walk-forward
validation.
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
# Layer-1 outputs and research context, plus raw macro features as direct
# ridge inputs. Previously included classifier-derived regime_bull/regime_bear
# columns from a multinomial logistic regime predictor; removed 2026-04-16
# after smoke-test walk-forward validation revealed the classifier scored
# 39.5% OOS accuracy (below always-predict-majority's 49%) with bear recall
# of 0.23 — a known-weak signal that wasn't contributing beyond what the
# raw macro features already carry. Tier 1 regime model upgrade on the
# roadmap (LightGBM + broader feature set + triple-barrier labeling); the
# hard-classified regime signal will return once it clears a named baseline.
META_FEATURES = [
    "research_calibrator_prob",   # P(research signal is correct)
    "momentum_score",             # momentum model output (continuous)
    "expected_move",              # volatility model output (continuous)
    "research_composite_score",   # raw research score (0-100, normalized to 0-1)
    "research_conviction",        # rising=1, stable=0, declining=-1
    "sector_macro_modifier",      # sector modifier from research (0.7-1.3, centered at 1)
    # Raw macro features: same 6 inputs the Tier 0 regime classifier used to
    # consume, now exposed directly to the meta ridge. Single market-wide
    # value per date; all tickers on a given date share these columns.
    "macro_spy_20d_return",       # trailing 20d SPY return
    "macro_spy_20d_vol",          # annualized 20d realized vol
    "macro_vix_level",            # VIX normalized by baseline 20
    "macro_vix_term_slope",       # (VIX - VIX3M) / 20
    "macro_yield_curve_slope",    # (TNX - IRX) / 10
    "macro_market_breadth",       # % of universe above 50d MA
    # Stage D (regime-v3 2026-05-14): AQR-style risk-on/risk-off
    # composite intensity z-score computed by ``regime.composite``.
    # Positive = risk-on, negative = risk-off (matches the substrate
    # Lambda's downstream convention). Single market-wide value per
    # date; all tickers on a given date share this column. Computed
    # in-process from the 6 macros above via z-score-weighted-sum,
    # then inverted to risk-on orientation. Equivalent to what the
    # substrate Lambda writes to substrate.json composite.intensity_z;
    # consistency by construction since both paths read the same
    # macro inputs through ``regime.composite``.
    "regime_intensity_z",
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

# Stage D — derived composite features (Stage 4 of regime-conditioning-260510
# composes with the new substrate-driven regime-v3 plan). Source column on
# ``regime_features_df`` → meta-feature name on the L2 row. Distinct map
# from MACRO_FEATURE_META_MAP because these are derived, not raw, and the
# absence of a ``regime_`` prefix on the raw macros above would collide.
REGIME_DERIVED_FEATURE_META_MAP = {
    "intensity_z": "regime_intensity_z",
}


class MetaModel:
    """Bayesian Ridge regression stacker for Layer 1 model outputs.

    Under BayesianRidge, the legacy `alpha` parameter is retained on the
    constructor for back-compat (callers like ``MetaModel(alpha=1.0)`` keep
    working) but is no longer the regularization strength — BR learns that
    via Type-II marginal likelihood. The parameter is preserved as
    metadata for run-comparison continuity with pre-BR pickled models.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._val_ic = 0.0
        self._coefficients: dict[str, float] = {}
        # Bayesian Ridge learned hyperparameters (Type-II ML estimates).
        # _learned_alpha: noise precision  (1 / σ²_n)
        # _learned_lambda: weight precision (controls effective L2 strength)
        # Both None when loaded from a pre-BayesianRidge pickled Ridge.
        self._learned_alpha: float | None = None
        self._learned_lambda: float | None = None
        # Feature list the stored ridge was fit on. Persisted in .meta.json so
        # inference can adapt to the deployed schema rather than assuming the
        # module-level META_FEATURES (which changes as we extend the stack).
        # Without this, any META_FEATURES addition would break all inferences
        # against the previously-trained model.
        self._feature_names: list[str] = []
        # Importance reports (Phase 4). Ridge coefficients alone are not
        # comparable across features that live on different scales, so we also
        # persist standardized coefficients (coef × std(feature) / std(y)) and
        # permutation importance (drop in Pearson correlation when a column
        # is shuffled). Both computed on the in-sample fit; honest walk-forward
        # importance is the backtester's job.
        self._importance: dict = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "MetaModel":
        """
        Fit ridge regression on Layer 1 OOS outputs.

        Parameters
        ----------
        X : array of shape (N, n_meta_features) — stacked Layer 1 outputs
        y : array of shape (N,) — actual forward returns or binary outcomes
        feature_names : names for coefficient reporting
        sample_weight : array of shape (N,) or None — per-row sample weights
            (optional). Used by the Stage 3 triple-barrier parallel Ridge to
            apply LdP Ch. 4.4 average-uniqueness weights, downweighting rows
            whose forward windows overlap heavily with neighbors. Sklearn
            Ridge accepts unnormalized weights and rescales internally.
        """
        from sklearn.linear_model import BayesianRidge

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Remove NaN rows
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
            valid &= np.isfinite(sample_weight)
            sample_weight = sample_weight[valid]
        X = X[valid]
        y = y[valid]

        if len(y) < 20:
            log.warning("MetaModel: only %d valid samples (need 20+) — skipping fit", len(y))
            return self

        self._n_samples = len(y)
        # BayesianRidge: uninformative Gamma hyperpriors (sklearn defaults of
        # 1e-6 on alpha_1/2, lambda_1/2) → Type-II ML drives α + λ from data.
        # Equivalent point predictions to Ridge with the learned λ, plus a
        # closed-form posterior predictive variance exposed via
        # predict(X, return_std=True). compute_score=True so the posterior
        # log-marginal-likelihood is available in metrics() for run-to-run
        # comparison.
        self._model = BayesianRidge(fit_intercept=True, compute_score=True)
        self._model.fit(X, y, sample_weight=sample_weight)
        self._fitted = True
        self._learned_alpha = float(getattr(self._model, "alpha_", None) or 0.0) or None
        self._learned_lambda = float(getattr(self._model, "lambda_", None) or 0.0) or None

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

        # Importance reporting (Phase 4). Standardized coefficients make the
        # features comparable across scales; permutation importance cross-checks
        # against the linearity assumption. Feature-importance read informs the
        # later pruning decision on classifier outputs vs. raw macro features.
        self._importance = self._compute_importance(X, y, preds)

        log.info(
            "MetaModel fitted: n=%d  IC=%.4f  alpha=%.1f",
            self._n_samples, self._val_ic, self.alpha,
        )
        # Print coefficients ordered by standardized magnitude (scale-adjusted)
        # rather than raw magnitude — raw coefficients are misleading when
        # features differ by orders of magnitude (e.g. VIX level ~ 1 vs. SPY
        # return ~ 0.01). Intercept excluded from the ranking.
        std_coef = self._importance.get("standardized_coef", {})
        ranked = sorted(std_coef.items(), key=lambda x: -abs(x[1]))
        for name, std in ranked:
            raw = self._coefficients.get(name, 0.0)
            perm = self._importance.get("permutation", {}).get(name, 0.0)
            log.info("  %-30s raw=%+.4f  std=%+.4f  perm_ic_drop=%+.4f",
                     name, raw, std, perm)

        return self

    def _compute_importance(
        self, X: np.ndarray, y: np.ndarray, preds: np.ndarray,
    ) -> dict:
        """
        Compute comparable per-feature importance metrics on the in-sample fit.

        Standardized coefficient: coef_k × std(X_k) / std(y). Scale-free,
        directly comparable across features with different units.

        Permutation importance: drop in Pearson IC between preds and y when
        column k is shuffled. Pure model-agnostic read that cross-checks the
        ridge's linearity assumption — if a classifier output explains less
        after the raw features are present, its permutation IC drop shrinks.

        Both are in-sample; walk-forward permutation on a held-out fold is the
        backtester's job. Cheap here, loud signal to read on the dashboard.
        """
        n, k = X.shape
        y_std = float(np.std(y))
        if y_std < 1e-12 or n < 20:
            return {}

        feat_std = X.std(axis=0)
        coef = self._model.coef_
        standardized = {
            name: round(float(coef[i] * feat_std[i] / y_std), 6)
            for i, name in enumerate(self._feature_names)
        }

        # Permutation importance: baseline IC minus IC with column shuffled.
        # Positive value = feature contributes positively to predictive IC.
        # Fixed seed for reproducibility within a run.
        base_ic = float(np.corrcoef(preds, y)[0, 1]) if np.std(preds) > 1e-10 else 0.0
        rng = np.random.default_rng(42)
        perm: dict[str, float] = {}
        for i, name in enumerate(self._feature_names):
            X_shuf = X.copy()
            rng.shuffle(X_shuf[:, i])
            shuf_preds = self._model.predict(X_shuf)
            if np.std(shuf_preds) > 1e-10:
                shuf_ic = float(np.corrcoef(shuf_preds, y)[0, 1])
            else:
                shuf_ic = 0.0
            perm[name] = round(base_ic - shuf_ic, 6)

        return {
            "standardized_coef": standardized,
            "permutation": perm,
            "base_ic": round(base_ic, 6),
            "y_std": round(y_std, 6),
            "feature_std": {
                name: round(float(feat_std[i]), 6)
                for i, name in enumerate(self._feature_names)
            },
        }

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

    def predict_single_with_std(self, features: dict) -> tuple[float, float | None]:
        """Predict (mean, posterior_std) for a single ticker.

        Returns ``std=None`` when the loaded sklearn estimator does not
        expose a posterior variance (legacy Ridge models pickled BEFORE the
        BayesianRidge cutover). Inference callers should treat ``None`` as
        "uncertainty signal not available" — the executor's α̂-uncertainty
        penalty term (workstream B of optimizer-sota-upgrades-260526.md)
        falls back to point-estimate sizing in that case.

        Backwards-compat is load-bearing: the first Saturday training cycle
        after this PR ships still loads the previous Ridge pickle into
        production until the new training run promotes a BayesianRidge.
        """
        if not self._fitted:
            raise RuntimeError("MetaModel not fitted")
        feat_names = self._feature_names or META_FEATURES
        x = np.array([[features.get(f, 0.0) for f in feat_names]], dtype=np.float64)
        try:
            mean, std = self._model.predict(x, return_std=True)
            return float(mean[0]), float(std[0])
        except TypeError:
            # Legacy Ridge — no return_std kwarg. Caller treats None as
            # "uncertainty unavailable" and falls back per the gradual-
            # cutover policy above.
            return float(self._model.predict(x)[0]), None

    def metrics(self) -> dict:
        return {
            "type": "meta_model_bayesian_ridge",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "val_ic": round(self._val_ic, 6),
            "alpha": self.alpha,
            "coefficients": self._coefficients,
            "feature_names": self._feature_names,
            "importance": self._importance,
            # BayesianRidge learned hyperparameters (Type-II ML). None when
            # the loaded model predates the BR cutover or fit was skipped.
            "learned_alpha_noise_precision": (
                round(self._learned_alpha, 6) if self._learned_alpha is not None else None
            ),
            "learned_lambda_weight_precision": (
                round(self._learned_lambda, 6) if self._learned_lambda is not None else None
            ),
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
            mm._importance = meta.get("importance", {}) or {}
            mm._learned_alpha = meta.get("learned_alpha_noise_precision")
            mm._learned_lambda = meta.get("learned_lambda_weight_precision")
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
