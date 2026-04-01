"""
model/research_calibrator.py — Research signal calibrator.

v0: Lookup table mapping research score buckets to empirical hit rates.
v1 (Phase 2): GBM on score + sub-scores + conviction + regime context.

The interface is the same for both versions — predict() returns P(signal correct).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Score buckets for v0 lookup table
DEFAULT_BUCKETS = [(0, 40), (40, 55), (55, 65), (65, 75), (75, 85), (85, 100)]
DEFAULT_PRIOR = 0.50  # neutral prior when no data for a bucket


class ResearchCalibrator:
    """Map research composite scores to P(beat SPY over 10 days)."""

    def __init__(self):
        self._lookup: dict[tuple[int, int], float] = {}
        self._bucket_counts: dict[tuple[int, int], int] = {}
        self._fitted = False
        self._n_samples = 0
        self._overall_hit_rate = None

    def fit(
        self,
        scores: np.ndarray,
        beat_spy: np.ndarray,
        buckets: list[tuple[int, int]] | None = None,
    ) -> "ResearchCalibrator":
        """
        Fit lookup table from historical signal outcomes.

        Parameters
        ----------
        scores   : array of composite research scores (0-100)
        beat_spy : binary array, 1 if signal beat SPY at 10d, 0 otherwise
        buckets  : list of (low, high) score ranges. Default: 6 buckets.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        beat_spy = np.asarray(beat_spy, dtype=np.int32).ravel()

        if len(scores) != len(beat_spy):
            raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(beat_spy)} labels")

        valid = np.isfinite(scores) & np.isfinite(beat_spy)
        scores = scores[valid]
        beat_spy = beat_spy[valid]

        self._n_samples = len(scores)
        if self._n_samples == 0:
            log.warning("ResearchCalibrator: no valid samples")
            return self

        self._overall_hit_rate = float(beat_spy.mean())
        buckets = buckets or DEFAULT_BUCKETS

        for low, high in buckets:
            mask = (scores >= low) & (scores < high)
            n = mask.sum()
            if n >= 3:  # minimum samples per bucket
                hit_rate = float(beat_spy[mask].mean())
                self._lookup[(low, high)] = hit_rate
                self._bucket_counts[(low, high)] = int(n)
            else:
                # Fall back to overall hit rate for sparse buckets
                self._lookup[(low, high)] = self._overall_hit_rate
                self._bucket_counts[(low, high)] = int(n)

        self._fitted = True
        log.info(
            "ResearchCalibrator fitted: n=%d  overall_hit_rate=%.2f%%  buckets=%d",
            self._n_samples, self._overall_hit_rate * 100, len(self._lookup),
        )
        for (lo, hi), rate in sorted(self._lookup.items()):
            n = self._bucket_counts[(lo, hi)]
            log.info("  Score %d-%d: hit_rate=%.1f%% (n=%d)", lo, hi, rate * 100, n)

        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict(self, score: float) -> float:
        """
        Return P(signal beats SPY at 10d) for a given composite score.

        Falls back to overall hit rate if score doesn't fall in any bucket,
        or DEFAULT_PRIOR if not fitted.
        """
        if not self._fitted:
            return DEFAULT_PRIOR

        for (low, high), hit_rate in self._lookup.items():
            if low <= score < high:
                return hit_rate

        # Score outside all buckets (e.g., exactly 100)
        return self._overall_hit_rate or DEFAULT_PRIOR

    def predict_batch(self, scores: np.ndarray) -> np.ndarray:
        """Vectorized predict for an array of scores."""
        return np.array([self.predict(s) for s in scores])

    def metrics(self) -> dict:
        return {
            "type": "research_calibrator_v0",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "overall_hit_rate": round(self._overall_hit_rate, 4) if self._overall_hit_rate else None,
            "buckets": {
                f"{lo}-{hi}": {"hit_rate": round(rate, 4), "n": self._bucket_counts.get((lo, hi), 0)}
                for (lo, hi), rate in sorted(self._lookup.items())
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "lookup": {f"{lo},{hi}": rate for (lo, hi), rate in self._lookup.items()},
            "bucket_counts": {f"{lo},{hi}": n for (lo, hi), n in self._bucket_counts.items()},
            "n_samples": self._n_samples,
            "overall_hit_rate": self._overall_hit_rate,
        }
        path.write_text(json.dumps(data, indent=2))
        log.info("ResearchCalibrator saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ResearchCalibrator":
        path = Path(path)
        data = json.loads(path.read_text())
        rc = cls()
        rc._lookup = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data.get("lookup", {}).items()
        }
        rc._bucket_counts = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data.get("bucket_counts", {}).items()
        }
        rc._n_samples = data.get("n_samples", 0)
        rc._overall_hit_rate = data.get("overall_hit_rate")
        rc._fitted = bool(rc._lookup)
        log.info("ResearchCalibrator loaded from %s (n=%d)", path, rc._n_samples)
        return rc
