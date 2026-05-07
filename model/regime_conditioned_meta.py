"""
model/regime_conditioned_meta.py — per-regime Ridge stack (audit Phase 4 PR 3).

Implements the audit §8 Phase 4 spec:

> "Train *three* L2 Ridges (one per regime) on regime-tagged training rows.
> At inference, the regime detector routes to one Ridge."

Three independent MetaModel Ridges, each fit on the subset of training
rows tagged with its regime label. At inference, the regime detector
(``RegimePredictorV2``) predicts the current regime; the corresponding
Ridge produces the alpha estimate.

Why this matters: the existing single Ridge averages signal weights
across regimes. A bear-regime row's optimal weight on momentum may differ
from a bull-regime row's, but the single Ridge can only learn one
weighting. Per-regime Ridges let each model specialize.

Key design choices:

- **Module wraps three ``MetaModel`` instances** (one per regime), reusing
  the existing Ridge-fitting code rather than duplicating. The per-regime
  Ridges have the same META_FEATURES interface as the single Ridge so
  the inference swap (PR 4) is localized.

- **Fallback to unconditioned Ridge** when a regime has too few training
  rows to fit (default ``min_per_regime_rows=200``). The unconditioned
  Ridge is fit on the full pool and used as the fallback for under-
  represented regimes. This matches the audit §9.5 "fail-closed only on
  unexpected outputs" framing — under-sampled regimes route to a known-good
  fallback, not to a degenerate per-regime fit.

- **Forward-compatible with regime-detector errors**: the audit §9 open
  question 5 (system shutoff trigger on unexpected outputs) is resolved
  by routing to the unconditioned Ridge when the predicted regime label
  is unrecognized (defensive default for forward compatibility).

- **Same .meta.json sidecar shape** as MetaModel + GBMScorer so the
  backtester preflight + runtime_smoke checks already validate it.

This PR ships the module + training integration. PR 4 wires inference;
PR 5 cuts over after the gate (regime-conditioned ensemble IC > single-
Ridge IC by ≥ 15% relative across validation period) clears.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from model.meta_model import MetaModel

log = logging.getLogger(__name__)


# Regime class labels — must match RegimePredictorV2.REGIME_CLASSES.
REGIME_LABELS: list[str] = ["bear", "neutral", "bull"]


class RegimeConditionedMeta:
    """Stack of 3 Ridges, one per regime, plus an unconditioned fallback.

    Interface mirrors ``MetaModel`` (fit / predict_for_regime /
    predict_single_for_regime / save / load / metrics) so the meta_trainer
    integration in this PR + the inference swap in PR 4 are both localized.

    The ``predict_*`` methods take an explicit regime argument so callers
    don't have to manage routing logic — the module handles "regime → Ridge"
    lookup including the fallback when a regime's Ridge is missing.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_per_regime_rows: int = 200,
    ):
        self.alpha = alpha
        self.min_per_regime_rows = min_per_regime_rows
        self._ridges: dict[str, MetaModel] = {}
        self._fallback_ridge: MetaModel | None = None
        self._fitted = False
        self._n_samples_per_regime: dict[str, int] = {}
        self._val_ic_per_regime: dict[str, float] = {}
        # Feature list the ridges were fit on (all 3 + fallback share the
        # same feature schema). Persisted in .meta.json so inference can
        # adapt to deployed schema vs module-level META_FEATURES.
        self._feature_names: list[str] = []

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def regime_labels(self) -> list[str]:
        return list(REGIME_LABELS)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regimes: list[str],
        feature_names: list[str] | None = None,
    ) -> "RegimeConditionedMeta":
        """Fit per-regime Ridges + an unconditioned fallback Ridge.

        Args:
            X: (n, n_features) feature matrix (same shape as MetaModel.fit)
            y: (n,) regression target (typically actual_fwd)
            regimes: parallel list of regime labels per row (one of
                "bear" / "neutral" / "bull"); other labels route to fallback
            feature_names: column names for coefficient reporting

        Each regime gets its own Ridge fit on its subset of rows.
        Regimes with fewer than ``min_per_regime_rows`` rows are skipped;
        their inference-time predictions route to the unconditioned Ridge.
        The unconditioned Ridge is always fit on the full pool so the
        fallback is reliable.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape}")
        if X.shape[0] != len(y) or X.shape[0] != len(regimes):
            raise ValueError(
                f"X rows ({X.shape[0]}), y length ({len(y)}), regimes length "
                f"({len(regimes)}) must all agree"
            )

        self._feature_names = list(feature_names) if feature_names else []

        # Always fit the unconditioned fallback Ridge on the full pool.
        # This is the same ridge the meta-trainer's existing single-Ridge
        # path produces — predictable behavior when regime data is sparse.
        self._fallback_ridge = MetaModel(alpha=self.alpha).fit(
            X, y, feature_names=feature_names,
        )
        log.info(
            "RegimeConditionedMeta fallback Ridge fit on %d rows", len(y),
        )

        # Per-regime Ridges. Skip regimes below min_per_regime_rows.
        for regime in REGIME_LABELS:
            mask = np.array([r == regime for r in regimes])
            n_regime = int(mask.sum())
            self._n_samples_per_regime[regime] = n_regime
            if n_regime >= self.min_per_regime_rows:
                ridge = MetaModel(alpha=self.alpha).fit(
                    X[mask], y[mask], feature_names=feature_names,
                )
                self._ridges[regime] = ridge
                self._val_ic_per_regime[regime] = float(ridge._val_ic)
                log.info(
                    "RegimeConditionedMeta '%s' Ridge fit on %d rows (val_ic=%.4f)",
                    regime, n_regime, ridge._val_ic,
                )
            else:
                log.info(
                    "RegimeConditionedMeta '%s' regime: only %d rows (need >=%d) — "
                    "fallback Ridge will be used at inference",
                    regime, n_regime, self.min_per_regime_rows,
                )

        self._fitted = True
        return self

    def _select_ridge(self, regime: str) -> MetaModel:
        """Route a regime label to its Ridge (or the fallback)."""
        if regime in self._ridges:
            return self._ridges[regime]
        # Unrecognized regime label OR under-sampled regime → fallback.
        # Defensive: never raise — inference must always produce an alpha
        # even if the regime detector misbehaves.
        if self._fallback_ridge is None:
            raise RuntimeError(
                "RegimeConditionedMeta not fitted — call fit() before predict"
            )
        return self._fallback_ridge

    def predict_for_regime(
        self, X: np.ndarray, regime: str,
    ) -> np.ndarray:
        """Predict alphas for a feature matrix, routed by regime label."""
        if not self._fitted:
            raise RuntimeError(
                "RegimeConditionedMeta not fitted — call fit() before predict"
            )
        ridge = self._select_ridge(regime)
        return ridge.predict(X)

    def predict_single_for_regime(
        self, features: dict, regime: str,
    ) -> float:
        """Inference-time convenience: dict in, scalar out, routed by regime."""
        if not self._fitted:
            return 0.0
        ridge = self._select_ridge(regime)
        return float(ridge.predict_single(features))

    def predict_unconditioned(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fallback Ridge only (regime-agnostic).

        Useful for the parity comparison in the inference parallel path:
        compute both regime-conditioned and unconditioned alphas per-ticker,
        compare in observation. The PR 5 cutover decision uses this.
        """
        if not self._fitted or self._fallback_ridge is None:
            raise RuntimeError("RegimeConditionedMeta not fitted")
        return self._fallback_ridge.predict(X)

    def metrics(self) -> dict:
        """Diagnostic metrics for the manifest + email."""
        return {
            "type": "regime_conditioned_meta_v1",
            "fitted": self._fitted,
            "alpha": self.alpha,
            "min_per_regime_rows": self.min_per_regime_rows,
            "n_samples_per_regime": dict(self._n_samples_per_regime),
            "val_ic_per_regime": {
                k: round(v, 6) for k, v in self._val_ic_per_regime.items()
            },
            "regimes_with_dedicated_ridge": list(self._ridges.keys()),
            "regimes_using_fallback": [
                r for r in REGIME_LABELS if r not in self._ridges
            ],
            "fallback_val_ic": (
                round(self._fallback_ridge._val_ic, 6)
                if self._fallback_ridge is not None else None
            ),
            "feature_names": list(self._feature_names),
        }

    def save(self, path: str | Path) -> None:
        """Save all fitted Ridges + fallback as a single pickle bundle.

        Sidecar carries metrics (per-regime val_ic, sample counts, feature
        names) so the backtester preflight + runtime_smoke can validate
        without loading the full bundle.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._fitted:
            raise RuntimeError(
                "Cannot save unfitted RegimeConditionedMeta — fit() first"
            )
        bundle = {
            "alpha": self.alpha,
            "min_per_regime_rows": self.min_per_regime_rows,
            "ridges": self._ridges,
            "fallback_ridge": self._fallback_ridge,
            "n_samples_per_regime": self._n_samples_per_regime,
            "val_ic_per_regime": self._val_ic_per_regime,
            "feature_names": self._feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        meta = {
            **self.metrics(),
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info(
            "RegimeConditionedMeta saved to %s (regimes_with_dedicated_ridge=%s)",
            path, list(self._ridges.keys()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegimeConditionedMeta":
        """Load bundle. Tolerant of missing/corrupt sidecar (mirrors GBMScorer)."""
        path = Path(path)
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        scorer = cls(
            alpha=bundle.get("alpha", 1.0),
            min_per_regime_rows=bundle.get("min_per_regime_rows", 200),
        )
        scorer._ridges = bundle.get("ridges", {})
        scorer._fallback_ridge = bundle.get("fallback_ridge")
        scorer._n_samples_per_regime = bundle.get("n_samples_per_regime", {})
        scorer._val_ic_per_regime = bundle.get("val_ic_per_regime", {})
        scorer._feature_names = list(bundle.get("feature_names", []))
        scorer._fitted = scorer._fallback_ridge is not None
        return scorer
