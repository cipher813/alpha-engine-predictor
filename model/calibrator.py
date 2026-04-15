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

    def fit(
        self,
        raw_alphas: np.ndarray,
        actual_up: np.ndarray,
        label_clip: float = 0.15,
    ) -> "PlattCalibrator":
        """
        Fit calibrator on walk-forward OOS predictions.

        Parameters
        ----------
        raw_alphas : array of raw continuous alpha predictions (clipped to [-label_clip, label_clip])
        actual_up  : binary array, 1 if forward_return_5d > 0 else 0
        label_clip : max absolute alpha used for clipping (default 0.15)
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

        # Compute baseline ECE (linear calibration)
        linear_p_up = np.clip(0.5 + raw_alphas / (2.0 * label_clip), 0.0, 1.0)
        self._ece_before = _expected_calibration_error(linear_p_up, actual_up)

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
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
        """
        alpha = float(np.clip(raw_alpha, -label_clip, label_clip))

        if self._fitted:
            p_up = float(self.predict_proba(np.array([alpha]))[0])
        else:
            # Linear fallback (existing behavior)
            p_up = float(np.clip(0.5 + alpha / (2.0 * label_clip), 0.0, 1.0))

        p_down = 1.0 - p_up
        if alpha >= 0:
            direction = "UP"
            confidence = p_up
        else:
            direction = "DOWN"
            confidence = p_down

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
        log.info("Calibrator loaded from %s (method=%s, n=%d)", path, method, cal._n_samples)
        return cal


def _expected_calibration_error(
    predicted_probs: np.ndarray,
    actual_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Bins predictions into n_bins equally-spaced buckets, computes the
    absolute difference between mean predicted probability and actual
    frequency of positive outcomes in each bin, weighted by bin size.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(predicted_probs)

    for i in range(n_bins):
        mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted_probs == bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_pred = predicted_probs[mask].mean()
        avg_actual = actual_labels[mask].mean()
        ece += (n_bin / n_total) * abs(avg_pred - avg_actual)

    return float(ece)
