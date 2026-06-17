"""
model/calibrator.py — Probability calibration for GBM alpha predictions.

Replaces the linear confidence mapping (p_up = 0.5 + alpha / (2 * LABEL_CLIP))
with a proper calibration model fitted on walk-forward out-of-sample predictions.

Two methods supported:
  - "platt"    : Logistic regression (Platt scaling) — parametric, stable with small N
  - "isotonic" : Isotonic regression — non-parametric, more flexible, needs more data

The calibrator maps raw continuous alpha predictions to calibrated P(UP) values
that match empirical hit rates: when the model says 70% confidence, the stock
actually goes UP ~70% of the time.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from alpha_engine_lib.quant.stats.calibration import (
    expected_calibration_error as _lib_expected_calibration_error,
)

log = logging.getLogger(__name__)


class PlattCalibrator:
    """Calibrate raw GBM alpha scores to P(direction=UP)."""

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method!r} (expected 'platt' or 'isotonic')")
        self.method = method
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._ece_before = None  # ECE of linear calibration (baseline)
        self._ece_after = None   # ECE of fitted calibrator
        self._class_weight = None  # set in fit() under method='platt'
        self._C = None             # set in fit() under method='platt'

    def fit(
        self,
        raw_alphas: np.ndarray,
        actual_up: np.ndarray,
        label_clip: float = 0.15,
        *,
        class_weight: str | dict | None = None,
        C: float = 1.0,
    ) -> "PlattCalibrator":
        """
        Fit calibrator on walk-forward OOS predictions.

        Parameters
        ----------
        raw_alphas : array of raw continuous alpha predictions (clipped to [-label_clip, label_clip])
        actual_up  : binary array, 1 if forward_return_5d > 0 else 0
        label_clip : max absolute alpha used for clipping (default 0.15)
        class_weight : passed through to ``sklearn.linear_model.LogisticRegression``
            when ``method='platt'``. Use ``'balanced'`` so the intercept doesn't
            anchor on an imbalanced marginal UP-rate (the 2026-05-23 failure
            class: OOS rows 100% bear+neutral → Platt's intercept pushed
            negative → 95% DOWN-skew across the synthetic alpha sweep).
            Ignored under ``method='isotonic'`` (isotonic is per-quantile
            non-parametric — no global intercept to rebalance).
        C : inverse L2 regularisation strength for the Platt logistic. Higher
            C = less shrinkage. Ignored under isotonic. Default 1.0 preserves
            historical behavior.
        """
        raw_alphas = np.asarray(raw_alphas, dtype=np.float64).ravel()
        actual_up = np.asarray(actual_up, dtype=np.int32).ravel()

        if len(raw_alphas) != len(actual_up):
            raise ValueError(f"Length mismatch: {len(raw_alphas)} alphas vs {len(actual_up)} labels")

        # Remove NaN/inf
        valid = np.isfinite(raw_alphas)
        raw_alphas = raw_alphas[valid]
        actual_up = actual_up[valid]

        if len(raw_alphas) < 100:
            log.warning("Calibrator: only %d valid samples (need 100+) — skipping fit", len(raw_alphas))
            return self

        self._n_samples = len(raw_alphas)
        self._class_weight = class_weight if self.method == "platt" else None
        self._C = C if self.method == "platt" else None

        # Compute baseline ECE (linear calibration)
        linear_p_up = np.clip(0.5 + raw_alphas / (2.0 * label_clip), 0.0, 1.0)
        self._ece_before = _expected_calibration_error(linear_p_up, actual_up)

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(
                C=C, class_weight=class_weight,
                solver="lbfgs", max_iter=1000,
            )
            self._model.fit(raw_alphas.reshape(-1, 1), actual_up)
        else:
            from sklearn.isotonic import IsotonicRegression
            self._model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._model.fit(raw_alphas, actual_up)

        self._fitted = True

        # Compute calibrated ECE
        calibrated_p_up = self.predict_proba(raw_alphas)
        self._ece_after = _expected_calibration_error(calibrated_p_up, actual_up)
        log.info(
            "Calibrator fitted (%s): n=%d  ECE_before=%.4f  ECE_after=%.4f  (%.1f%% reduction)",
            self.method, self._n_samples, self._ece_before, self._ece_after,
            (1 - self._ece_after / max(self._ece_before, 1e-8)) * 100,
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict_proba(self, raw_alphas: np.ndarray) -> np.ndarray:
        """
        Map raw alpha predictions to calibrated P(UP).

        Returns array of probabilities in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted — call fit() first")

        raw_alphas = np.asarray(raw_alphas, dtype=np.float64).ravel()

        if self.method == "platt":
            return self._model.predict_proba(raw_alphas.reshape(-1, 1))[:, 1]
        else:
            return self._model.predict(raw_alphas)

    def calibrate_prediction(
        self,
        raw_alpha: float,
        label_clip: float = 0.15,
    ) -> dict:
        """
        Calibrate a single prediction. Returns p_up, p_down, direction, confidence.

        Falls back to linear calibration if not fitted.

        Confidence semantics: ``|p_up - 0.5| * 2`` — distance from coin-flip on
        [0, 1]. At p_up=0.5 the model has no opinion → confidence=0.0; at
        p_up=1.0 or 0.0 the model is certain → confidence=1.0. Pre-2026-05-12
        this was ``max(p_up, p_down)`` (range [0.5, 1.0]), which treated a
        coin-flip prediction as "0.5 confident" — load-bearing on the
        DOWN-veto inversion at the 75%+ band (ROADMAP L1594).
        """
        alpha = float(np.clip(raw_alpha, -label_clip, label_clip))

        if self._fitted:
            p_up = float(self.predict_proba(np.array([alpha]))[0])
        else:
            # Linear fallback (existing behavior)
            p_up = float(np.clip(0.5 + alpha / (2.0 * label_clip), 0.0, 1.0))

        p_down = 1.0 - p_up
        direction = "UP" if p_up >= 0.5 else "DOWN"
        confidence = abs(p_up - 0.5) * 2.0

        return {
            "p_up": round(p_up, 4),
            "p_down": round(p_down, 4),
            "p_flat": 0.0,
            "predicted_direction": direction,
            "prediction_confidence": round(confidence, 4),
        }

    def metrics(self) -> dict:
        """Return calibration metrics for reporting."""
        return {
            "method": self.method,
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "ece_before": round(self._ece_before, 6) if self._ece_before is not None else None,
            "ece_after": round(self._ece_after, 6) if self._ece_after is not None else None,
            "class_weight": self._class_weight,
            "C": self._C,
        }

    def save(self, path: str | Path) -> None:
        """Save calibrator to pickle + metadata JSON.

        The sidecar includes ``deployed_at`` (ISO-8601 UTC). The backtester's
        retrain_alert uses this to apply a grace period after a fresh
        calibrator lands — during that window predictor_outcomes holds
        mixed pre/post calibrator semantics and ECE is structurally noisy.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._model, f)

        meta = self.metrics()
        meta["deployed_at"] = datetime.now(timezone.utc).isoformat()
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        log.info("Calibrator saved to %s (deployed_at=%s)", path, meta["deployed_at"])

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        """Load a previously saved calibrator."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator not found: {path}")

        meta_path = Path(str(path) + ".meta.json")
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        method = meta.get("method", "platt")
        cal = cls(method=method)

        with open(path, "rb") as f:
            cal._model = pickle.load(f)

        cal._fitted = meta.get("fitted", True)
        cal._n_samples = meta.get("n_samples", 0)
        cal._ece_before = meta.get("ece_before")
        cal._ece_after = meta.get("ece_after")
        cal._class_weight = meta.get("class_weight")
        cal._C = meta.get("C")
        log.info("Calibrator loaded from %s (method=%s, n=%d)", path, method, cal._n_samples)
        return cal


def _expected_calibration_error(
    predicted_probs: np.ndarray,
    actual_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error of ``predicted_probs`` vs ``actual_labels``.

    Thin wrapper over the fleet-canonical implementation in
    ``alpha_engine_lib.quant.stats.calibration`` so the calibrator's fit-time
    ECE (``ece_before``/``ece_after``) and the backtester's production-time ECE
    are computed by the SAME code on the SAME quantity (calibrated probability
    vs binary outcome) — the only way the two numbers are comparable as a
    calibration-drift signal. ``predicted_probs`` must be a P(UP), not a margin.

    Returns the scalar ECE (0.0 when there is no usable bin), preserving the
    prior float contract for callers.
    """
    result = _lib_expected_calibration_error(
        predicted_probs, actual_labels, n_bins=n_bins,
    )
    ece = result.get("ece")
    return float(ece) if ece is not None else 0.0
