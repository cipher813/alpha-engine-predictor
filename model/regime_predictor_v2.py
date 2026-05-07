"""
model/regime_predictor_v2.py — Tier-1 regime classifier (audit Phase 4 PR 1).

Replaces the retired Tier-0 regime classifier at model/regime_predictor.py.
The Tier-0 model was retired 2026-04-16 after walk-forward validation
showed 39.5% OOS accuracy (below the 49% always-predict-majority baseline)
with bear-recall of 0.23. Per audit §6.2:

> "Don't add the retired regime classifier back as-is. Its OOS accuracy
> was honestly bad; adding it back without addressing the labeling and
> feature issues would just reintroduce a known-failed component."

Tier-1 design (this module) addresses the three failures:

1. **Triple-barrier labeling** instead of point-in-time labels. For each
   date, look forward ``forward_window`` trading days; classify by which
   barrier hits first:
     - bull   = SPY hits +``up_barrier_pct`` first (or window-end > +half-band)
     - bear   = SPY hits -``down_barrier_pct`` first (or window-end < -half-band)
     - neutral = otherwise
   Triple-barrier (López de Prado) gives much cleaner regime labels than
   point-in-time SPY returns because it's path-dependent.

2. **3-class LightGBM** instead of multinomial logistic. Allows non-linear
   feature interactions, robust to feature scales, native NaN handling.

3. **Honest promotion gate** per audit §8 Phase 4 spec:
     - OOS accuracy ≥ 50% (above majority-class baseline)
     - Bear-recall ≥ 0.40 (above the retired Tier-0's 0.23)
     - Per-class recall floor — ensures no class is utterly ignored

This module ships with no training/inference integration — observation-
only build. Phase 4 sequence:

- PR 1 (this): module + tests, no production wiring
- PR 2: training integration in meta_trainer (observe-only metrics +
  per-row regime predictions persisted to manifest)
- PR 3: per-regime Ridge stack module + training (3 L2 Ridges, one
  per regime, trained on regime-stratified rows)
- PR 4: inference parallel path — regime detector predicts; per-regime
  Ridges produce alpha alongside the single Ridge (observe-only)
- PR 5: cutover + gate validation (regime-conditioned ensemble IC must
  exceed single-Ridge IC by ≥ 15% relative across validation period)
- PR 6: cleanup of unused single-Ridge path post-validation

Output target: 3-class softmax probabilities (P_bull + P_neutral + P_bear = 1).
At inference, ``predict_class`` returns the argmax label as a string for
routing purposes.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# Canonical feature ordering — must match what the meta-trainer + inference
# build for the regime classifier. Includes the same 6 macro features the
# meta-Ridge consumes plus three SPY-momentum features that capture trend
# at multiple horizons (regime is fundamentally a trend-vs-volatility
# question; the existing macro features carry one half, momentum the other).
REGIME_V2_FEATURES: list[str] = [
    # 6 macro context features (same as META_FEATURES macro block)
    "macro_spy_20d_return",
    "macro_spy_20d_vol",
    "macro_vix_level",
    "macro_vix_term_slope",
    "macro_yield_curve_slope",
    "macro_market_breadth",
    # 3 SPY-momentum features at multiple horizons
    "spy_5d_return",
    "spy_60d_return",
    "spy_120d_return",
]

# Class label encoding — keep stable so saved models stay loadable. Order
# matches the column order in LightGBM's predict() output.
REGIME_CLASSES: list[str] = ["bear", "neutral", "bull"]


def make_triple_barrier_labels(
    spy_log_returns: np.ndarray,
    forward_window: int = 21,
    up_barrier_pct: float = 0.05,
    down_barrier_pct: float = 0.05,
) -> np.ndarray:
    """Generate 3-class triple-barrier labels from SPY log-return series.

    For each row, looks forward ``forward_window`` rows; classifies by
    which barrier hits first:

      - 2 (bull)   : up barrier hit first, OR window-end cumulative
                     return > up_barrier_pct / 2
      - 0 (bear)   : down barrier hit first, OR window-end cumulative
                     return < -down_barrier_pct / 2
      - 1 (neutral): neither barrier hit, window-end cumulative return
                     in [-half, +half]

    Returns int array (length matching input). Tail rows where the
    forward window extends past data end get label = -1 (sentinel for
    "no label available" — caller filters these out before fit).

    Args:
        spy_log_returns: (n,) log-return series for SPY
        forward_window: trading days forward to check
        up_barrier_pct: cumulative log-return threshold for bull hit
        down_barrier_pct: cumulative log-return threshold for bear hit
    """
    n = len(spy_log_returns)
    labels = np.full(n, -1, dtype=np.int8)
    for i in range(n - forward_window):
        cum = 0.0
        bull_hit = False
        bear_hit = False
        for j in range(forward_window):
            cum += spy_log_returns[i + 1 + j]
            if cum >= up_barrier_pct:
                bull_hit = True
                break
            if cum <= -down_barrier_pct:
                bear_hit = True
                break
        if bull_hit:
            labels[i] = 2  # bull
        elif bear_hit:
            labels[i] = 0  # bear
        else:
            # Neither barrier hit during window — classify by window-end
            # cumulative return relative to half-band.
            if cum > up_barrier_pct / 2:
                labels[i] = 2  # bull (drifted up)
            elif cum < -down_barrier_pct / 2:
                labels[i] = 0  # bear (drifted down)
            else:
                labels[i] = 1  # neutral
    return labels


def _default_params() -> dict:
    """LightGBM hyperparameters for 3-class regime classification.

    Smaller capacity than the alpha-prediction GBMs because regime
    label space is much smaller (3 classes vs continuous alpha) and
    we have far fewer rows per class than per-ticker training. Lambda
    regularization is meaningful — protects against overfit to recent
    regime patterns.
    """
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 30,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "num_threads": 4,
        "verbosity": -1,
        "seed": 42,
    }


class RegimePredictorV2:
    """Tier-1 regime classifier — 3-class LightGBM with triple-barrier labels.

    Honest gate metrics on validation set:
      - OOS accuracy (vs majority baseline ~0.49)
      - Per-class recall (especially bear, the Tier-0's weakness at 0.23)
      - Macro-F1 (guards against degenerate majority-class)
    """

    def __init__(self, params: dict | None = None, n_estimators: int = 500,
                 early_stopping_rounds: int = 30):
        self.params = params or _default_params()
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self._booster = None
        self._feature_names: list[str] = list(REGIME_V2_FEATURES)
        self._best_iteration: int = 0
        self._n_samples: int = 0
        self._fitted = False
        # Honest gate metrics — populated after fit() with validation data
        self._oos_accuracy: float | None = None
        self._per_class_recall: dict[str, float] | None = None
        self._macro_f1: float | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def class_labels(self) -> list[str]:
        return list(REGIME_CLASSES)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "RegimePredictorV2":
        """Fit 3-class LightGBM on (features, regime label) pairs.

        Args:
            X_train: (n, n_features) feature matrix
            y_train: (n,) integer labels in {0, 1, 2} (bear/neutral/bull)
            X_val:   optional validation features for early-stopping +
                     honest gate metric computation
            y_val:   optional validation labels
            feature_names: override default REGIME_V2_FEATURES order
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise RuntimeError(
                "lightgbm not installed — install via requirements.txt; "
                "RegimePredictorV2 cannot be fit without it."
            ) from e

        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2-D; got shape {X_train.shape}"
            )
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X/y length disagree: {X_train.shape[0]} vs {y_train.shape[0]}"
            )
        valid_classes = {0, 1, 2}
        unique_labels = set(int(v) for v in np.unique(y_train))
        if not unique_labels.issubset(valid_classes):
            raise ValueError(
                f"y_train labels must be in {{0, 1, 2}}; got {unique_labels}"
            )

        if feature_names is not None:
            if len(feature_names) != X_train.shape[1]:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) does not "
                    f"match X_train cols ({X_train.shape[1]})"
                )
            self._feature_names = list(feature_names)
        else:
            self._feature_names = list(REGIME_V2_FEATURES[:X_train.shape[1]])

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
        callbacks.append(lgb.log_evaluation(period=0))

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

        # Compute honest gate metrics on validation set (or train if no val).
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train
        if eval_X is not None and eval_y is not None and len(eval_X) >= 30:
            probs = self._booster.predict(
                eval_X, num_iteration=self._best_iteration,
            )
            preds = np.argmax(probs, axis=1)
            self._oos_accuracy = float(np.mean(preds == eval_y))
            self._per_class_recall = {}
            f1_per_class = []
            for cls_idx, cls_name in enumerate(REGIME_CLASSES):
                cls_mask = eval_y == cls_idx
                if cls_mask.sum() > 0:
                    recall = float(np.mean(preds[cls_mask] == cls_idx))
                else:
                    recall = float("nan")
                self._per_class_recall[cls_name] = recall
                # F1 per class for macro-F1 aggregation
                pred_mask = preds == cls_idx
                if pred_mask.sum() > 0:
                    precision = float(np.mean(eval_y[pred_mask] == cls_idx))
                else:
                    precision = 0.0
                if (
                    precision + recall > 0
                    and not np.isnan(recall)
                ):
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                f1_per_class.append(f1)
            self._macro_f1 = float(np.mean(f1_per_class))

        self._fitted = True
        log.info(
            "RegimePredictorV2 fit on %d samples (best_iter=%d, oos_acc=%s, macro_f1=%s)",
            self._n_samples, self._best_iteration,
            f"{self._oos_accuracy:.4f}" if self._oos_accuracy is not None else "n/a",
            f"{self._macro_f1:.4f}" if self._macro_f1 is not None else "n/a",
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 3-class softmax probabilities. Columns are [bear, neutral, bull]."""
        if not self._fitted or self._booster is None:
            raise RuntimeError(
                "RegimePredictorV2 not fitted — call fit() before predict_proba()"
            )
        if X.ndim != 2:
            raise ValueError(
                f"predict_proba expects 2-D; got shape {X.shape}"
            )
        return self._booster.predict(X, num_iteration=self._best_iteration)

    def predict_class(self, X: np.ndarray) -> list[str]:
        """Return string labels (bear/neutral/bull) — argmax of probabilities."""
        probs = self.predict_proba(X)
        return [REGIME_CLASSES[int(i)] for i in np.argmax(probs, axis=1)]

    def predict_class_from_dict(self, features: dict) -> str:
        """Inference-time convenience: dict in, label out. Defensive on missing
        keys (imputed to 0.0). Returns 'neutral' when not fitted."""
        if not self._fitted or self._booster is None:
            return "neutral"
        row = np.array(
            [[features.get(name, 0.0) for name in self._feature_names]],
            dtype=np.float32,
        )
        return self.predict_class(row)[0]

    def passes_promotion_gate(
        self,
        min_oos_accuracy: float = 0.50,
        min_bear_recall: float = 0.40,
        min_per_class_recall_floor: float = 0.15,
    ) -> tuple[bool, str]:
        """Audit §8 Phase 4 spec gate. Returns (pass, reason).

        Defaults:
          - oos_accuracy >= 0.50 (above majority-class baseline ~0.49)
          - bear_recall >= 0.40 (above retired Tier-0's 0.23)
          - per-class recall floor 0.15 (no class can be utterly ignored)

        Returns False with descriptive reason if not fitted, gate not
        evaluable (no per-class recall data), or any threshold missed.
        """
        if not self._fitted:
            return False, "RegimePredictorV2 not fitted"
        if self._oos_accuracy is None or self._per_class_recall is None:
            return False, "fitted but gate metrics not computed (no validation set?)"
        if self._oos_accuracy < min_oos_accuracy:
            return False, (
                f"OOS accuracy {self._oos_accuracy:.3f} < {min_oos_accuracy:.2f} "
                f"(below majority-class baseline)"
            )
        bear_recall = self._per_class_recall.get("bear", 0.0)
        if bear_recall < min_bear_recall:
            return False, (
                f"bear-recall {bear_recall:.3f} < {min_bear_recall:.2f} "
                f"(retired Tier-0 hit 0.23 — must beat that meaningfully)"
            )
        for cls, recall in self._per_class_recall.items():
            if recall < min_per_class_recall_floor:
                return False, (
                    f"{cls}-recall {recall:.3f} < {min_per_class_recall_floor:.2f} "
                    f"(class is being ignored by the model)"
                )
        return True, "all gate criteria passed"

    def metrics(self) -> dict:
        """Diagnostic metrics for the manifest + email."""
        return {
            "type": "regime_predictor_v2",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "n_features": len(self._feature_names),
            "feature_names": self.feature_names,
            "class_labels": self.class_labels,
            "best_iteration": int(self._best_iteration) if self._fitted else None,
            "oos_accuracy": (
                round(self._oos_accuracy, 6) if self._oos_accuracy is not None else None
            ),
            "per_class_recall": (
                {k: round(v, 6) if not np.isnan(v) else None
                 for k, v in self._per_class_recall.items()}
                if self._per_class_recall is not None else None
            ),
            "macro_f1": (
                round(self._macro_f1, 6) if self._macro_f1 is not None else None
            ),
        }

    def save(self, path: str | Path) -> None:
        """Save booster (pickle) + .meta.json sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._fitted or self._booster is None:
            raise RuntimeError(
                "Cannot save unfitted RegimePredictorV2 — fit() first"
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
            "RegimePredictorV2 saved to %s (oos_acc=%s, macro_f1=%s)",
            path,
            f"{self._oos_accuracy:.4f}" if self._oos_accuracy is not None else "n/a",
            f"{self._macro_f1:.4f}" if self._macro_f1 is not None else "n/a",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegimePredictorV2":
        """Load booster + sidecar. Tolerant of missing/corrupt sidecar."""
        path = Path(path)
        with open(path, "rb") as f:
            booster = pickle.load(f)

        scorer = cls()
        scorer._booster = booster

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                scorer._feature_names = list(
                    meta.get("feature_names") or REGIME_V2_FEATURES
                )
                scorer._best_iteration = int(meta.get("best_iteration") or 0)
                scorer._n_samples = int(meta.get("n_samples") or 0)
                scorer._oos_accuracy = meta.get("oos_accuracy")
                scorer._per_class_recall = meta.get("per_class_recall")
                scorer._macro_f1 = meta.get("macro_f1")
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(
                    "RegimePredictorV2 sidecar at %s could not be parsed (%s) — "
                    "using booster-internal defaults",
                    meta_path, e,
                )
                scorer._feature_names = list(booster.feature_name())
                scorer._best_iteration = booster.best_iteration or booster.num_trees()
        else:
            log.warning(
                "RegimePredictorV2 sidecar at %s missing — booster-internal defaults",
                meta_path,
            )
            scorer._feature_names = list(booster.feature_name())
            scorer._best_iteration = booster.best_iteration or booster.num_trees()

        scorer._fitted = True
        return scorer
