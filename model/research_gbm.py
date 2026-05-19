"""
model/research_gbm.py — research signal LightGBM scorer.

Audit Phase 3 PR 1 (2026-05-07): replaces the bucket-lookup
ResearchCalibrator (model/research_calibrator.py) with a proper
LightGBM L1 component that predicts P(research signal beats SPY at
10d) from a 9-feature input rather than a single-axis composite-score
lookup.

Per the audit:
- §3.2: "Research calibrator — *not actually a model*. It's a 6-bucket
  lookup table mapping research_composite_score ranges to empirical
  hit rates. Cannot generalize, cannot interact with other features,
  cannot adjust to regime."
- §6.2: "An actual research-side L1 model... Replacing it with a
  proper LightGBM (using research-score features + interactions with
  sector/regime context) would meaningfully diversify the L1 pool."
- §8 Phase 3: "Research LightGBM L1. Train on research_composite_score
  + research_conviction + sector_macro_modifier + 6 macro features."
- §9.4: "Pruning verification test. After the research-LGB lands, the
  Ridge's coefficient on research_composite_score should drop to
  near-zero within 1-2 training cycles if the LGB is capturing the
  signal."

Interface mirrors model/research_calibrator.ResearchCalibrator's shape
(fit / predict / save / load / metrics) so the meta-trainer integration
is a localized swap rather than an interface refactor across the
training pipeline.

Output target: P(beat SPY at 10d) — same as the bucket-lookup version.
The training label is binary (1 if signal beat SPY at 10d, 0 otherwise);
LightGBM regression objective on those binary labels gives the
fraction-correct prediction directly.

Multi-PR sequence (this PR is PR 1 of N):
- PR 1 (this): module + tests, no training/inference integration
- PR 2: train alongside bucket-lookup in meta_trainer (observe-only)
- PR 3: inference path consumes LGB + bucket-lookup in parallel
- PR 4: cutover (LGB replaces bucket-lookup as canonical)
- PR 5: META_FEATURES pruning of research_composite_score (post-Ridge
  coefficient verification per audit §9.4)
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# Canonical feature ordering — must match what meta_trainer + inference build.
# Ordered for clarity: research-derived features first, then macro context.
# Keep in lockstep with audit §8 Phase 3 deliverable spec.
RESEARCH_GBM_FEATURES: list[str] = [
    # Research-derived (3)
    "research_composite_score",   # raw research score, expected normalized 0-1
    "research_conviction",        # rising=1, stable=0, declining=-1
    "sector_macro_modifier",      # 0.7 to 1.3, centered at 1
    # Raw macro context (6) — same set the meta-Ridge consumes; gives
    # the research-LGB regime awareness so it can learn that, e.g.,
    # high-conviction signals in bear regime should be down-weighted.
    "macro_spy_20d_return",
    "macro_spy_20d_vol",
    "macro_vix_level",
    "macro_vix_term_slope",
    "macro_yield_curve_slope",
    "macro_market_breadth",
]


def _default_params() -> dict:
    """LightGBM hyperparameters for the research scorer.

    Smaller than the momentum/volatility GBMs because the research-feature
    space is much narrower (9 features, mostly slow-moving) and the
    label is noisy (binary beat-SPY-at-10d). Keeping leaves and depth
    low protects against overfitting given the typical few-thousand
    training-row count.

    Overfitting tightening (2026-05-19, ROADMAP L1816 P1 — promoted from P2):
    the 2026-05-09 promote-true retrain at the new 21d horizon showed
    ``train_ic=0.4328 / val_ic=0.0857 ⇒ ratio 5.05`` on 496 rows. With
    ``num_leaves=15 / max_depth=4`` the tree count effectively encodes
    most of the 9-feature training set verbatim — the model memorizes
    rather than generalizes. The 21d horizon shrinks the fit set vs the
    pre-cutover 5d horizon (fewer rows have finite ``actual_fwd`` labels)
    so the same capacity now sees a smaller corpus.

    Reducing to ``num_leaves=8 / max_depth=3`` caps the model class at
    a strictly smaller hypothesis space (8 ≪ 2^4 = 16 reachable terminal
    regions; depth 3 caps interaction order at 3 features). The Ridge's
    regularization currently absorbs the overfit (research_calibrator_prob
    std-coef +0.05), so this is a forward integrity fix — NOT a
    bleeding-alpha fix — and the change is bounded-risk against today's
    bounded effect. Per-fold WF GBM training (the structurally cleaner
    remediation) stays deferred until ~30 weeks corpus (~2026-06-20)
    because at 22 weeks the per-fold rows fall below the
    ``min_child_samples=30`` threshold.
    """
    return {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 30,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_threads": 4,
        "verbosity": -1,
        "seed": 42,
    }


class ResearchGBMScorer:
    """LightGBM-based research-signal scorer.

    Same interface contract as ``ResearchCalibrator`` (the bucket-lookup
    predecessor) so the meta-trainer can swap implementations with a
    localized change. The only behavioral difference is that ``fit``
    takes a feature matrix instead of a scores array, and the
    convenience ``predict_from_dict`` helper takes a dict with the
    9 expected keys for inference-time per-row scoring.
    """

    def __init__(self, params: dict | None = None, n_estimators: int = 500,
                 early_stopping_rounds: int = 30):
        self.params = params or _default_params()
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self._booster = None
        self._feature_names: list[str] = list(RESEARCH_GBM_FEATURES)
        self._best_iteration: int = 0
        self._val_ic: float = 0.0
        self._n_samples: int = 0
        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "ResearchGBMScorer":
        """Fit LightGBM on research-feature × beat-SPY-at-10d label.

        Parameters
        ----------
        X_train : (n, 9) feature matrix in RESEARCH_GBM_FEATURES order
        y_train : (n,) binary labels (1 if signal beat SPY at 10d, 0 otherwise),
                  or continuous in [0, 1] for soft-label training
        X_val   : optional validation features for early stopping
        y_val   : optional validation labels
        feature_names : override the default RESEARCH_GBM_FEATURES order
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise RuntimeError(
                "lightgbm not installed — install via requirements.txt; "
                "ResearchGBMScorer cannot be fit without it."
            ) from e

        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2-D (n_samples, n_features); got shape {X_train.shape}"
            )
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train rows ({X_train.shape[0]}) and y_train length "
                f"({y_train.shape[0]}) disagree"
            )

        if feature_names is not None:
            if len(feature_names) != X_train.shape[1]:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) does not match "
                    f"X_train cols ({X_train.shape[1]})"
                )
            self._feature_names = list(feature_names)
        else:
            self._feature_names = list(RESEARCH_GBM_FEATURES[:X_train.shape[1]])

        train_set = lgb.Dataset(
            X_train, label=y_train, feature_name=self._feature_names,
        )
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = []
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_set = lgb.Dataset(
                X_val, label=y_val, feature_name=self._feature_names,
                reference=train_set,
            )
            valid_sets.append(val_set)
            valid_names.append("val")
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds, verbose=False,
                )
            )
        callbacks.append(lgb.log_evaluation(period=0))  # silent

        self._booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._best_iteration = (
            self._booster.best_iteration or self.n_estimators
        )
        self._n_samples = int(X_train.shape[0])

        # Validation IC (Pearson correlation of predictions vs labels).
        # Used by the promotion gate to decide whether the LGB has any
        # predictive power worth keeping. NaN-safe.
        if X_val is not None and y_val is not None and len(X_val) >= 5:
            val_preds = self._booster.predict(
                X_val, num_iteration=self._best_iteration,
            )
            mask = np.isfinite(val_preds) & np.isfinite(y_val)
            if mask.sum() >= 5 and float(np.std(val_preds[mask])) > 0:
                self._val_ic = float(
                    np.corrcoef(val_preds[mask], y_val[mask])[0, 1]
                )
            else:
                self._val_ic = 0.0
        else:
            self._val_ic = 0.0

        self._fitted = True
        log.info(
            "ResearchGBMScorer fit on %d samples (best_iter=%d, val_ic=%.4f)",
            self._n_samples, self._best_iteration, self._val_ic,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict P(beat SPY at 10d) for a feature matrix.

        Mirrors GBMScorer.predict's signature: takes a (n, n_features)
        matrix, returns a (n,) array. Output is the LightGBM regression
        output clipped to [0, 1] (since the label space is binary in
        [0, 1] but the regressor can drift outside on extrapolation).
        """
        if not self._fitted or self._booster is None:
            raise RuntimeError(
                "ResearchGBMScorer not fitted — call fit() before predict()"
            )
        if X.ndim != 2:
            raise ValueError(
                f"predict expects 2-D (n_samples, n_features); got shape {X.shape}"
            )
        expected = self._booster.num_feature()
        if X.shape[1] != expected:
            raise ValueError(
                f"X has {X.shape[1]} columns but model expects {expected} "
                f"(feature_names: {self._feature_names})"
            )
        raw = self._booster.predict(X, num_iteration=self._best_iteration)
        return np.clip(raw, 0.0, 1.0)

    def predict_from_dict(self, features: dict) -> float:
        """Inference-time convenience: take a dict, return a scalar P(beat).

        Mirrors ResearchCalibrator.predict's scalar interface so the
        inference path's per-row swap is a localized change. Defensive
        on missing features: any absent key gets imputed to 0.0 (the
        booster's missing-value handling kicks in if the bagging path
        included that pattern).
        """
        if not self._fitted or self._booster is None:
            return 0.5  # neutral prior when not fitted (matches DEFAULT_PRIOR)
        row = np.array(
            [[features.get(name, 0.0) for name in self._feature_names]],
            dtype=np.float32,
        )
        return float(self.predict(row)[0])

    def metrics(self) -> dict:
        """Diagnostic metrics for the manifest + email.

        ``train_ic`` and ``train_val_ic_ratio`` are caller-supplied (the
        meta-trainer computes ``train_ic`` against the GBM's own training
        split). The scorer carries ``val_ic`` from its early-stop holdout
        and nothing else — the ratio is a *cross-split* concept the scorer
        doesn't own.
        """
        return {
            "type": "research_gbm_v1",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "n_features": len(self._feature_names),
            "feature_names": self.feature_names,
            "best_iteration": int(self._best_iteration) if self._fitted else None,
            "val_ic": round(self._val_ic, 6) if self._fitted else None,
        }

    def feature_importance(self, importance_type: str = "gain") -> dict:
        """Per-feature importance (gain or split). Empty if not fitted."""
        if not self._fitted or self._booster is None:
            return {}
        scores = self._booster.feature_importance(importance_type=importance_type)
        return dict(zip(self._feature_names, scores.tolist()))

    def save(self, path: str | Path) -> None:
        """Save booster (pickle) + .meta.json sidecar.

        Sidecar carries feature_names + best_iteration + val_ic so the
        backtester preflight + runtime smoke checks can validate the
        contract without loading the booster. Mirrors the GBMScorer
        save pattern that the audit-Phase-2a's runtime-smoke check
        depends on.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._fitted or self._booster is None:
            raise RuntimeError(
                "Cannot save unfitted ResearchGBMScorer — fit() first"
            )
        with open(path, "wb") as f:
            pickle.dump(self._booster, f)
        meta = {
            **self.metrics(),
            "params": self.params,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info(
            "ResearchGBMScorer saved to %s (val_ic=%.4f, best_iter=%d)",
            path, self._val_ic, self._best_iteration,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResearchGBMScorer":
        """Load booster + sidecar. Tolerant of missing sidecar fields
        (matches GBMScorer.load's .get-with-defaults pattern that the
        2026-05-06 force-promote audit relied on)."""
        path = Path(path)
        with open(path, "rb") as f:
            booster = pickle.load(f)

        scorer = cls()
        scorer._booster = booster

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                scorer._feature_names = list(meta.get("feature_names") or RESEARCH_GBM_FEATURES)
                scorer._best_iteration = int(meta.get("best_iteration") or 0)
                scorer._val_ic = float(meta.get("val_ic") or 0.0)
                scorer._n_samples = int(meta.get("n_samples") or 0)
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(
                    "ResearchGBMScorer sidecar at %s could not be parsed (%s) — "
                    "using booster-internal defaults",
                    meta_path, e,
                )
                scorer._feature_names = list(booster.feature_name())
                scorer._best_iteration = booster.best_iteration or booster.num_trees()
        else:
            log.warning(
                "ResearchGBMScorer sidecar at %s missing — using booster-internal defaults",
                meta_path,
            )
            scorer._feature_names = list(booster.feature_name())
            scorer._best_iteration = booster.best_iteration or booster.num_trees()

        scorer._fitted = True
        return scorer
