"""Short-history subsample IC validation for Layer-1 components.

Closes ROADMAP P1 "NaN-feature handling audit + short-history subsample
validation" — subsample-validation phase. Per
``feedback_component_baseline_validation.md``:

> Before a Layer-1 model can contribute features to a downstream stacker
> (meta-model, ensemble), it MUST clear an explicit named baseline and
> have a component-level promotion gate. "The ensemble works" is not
> evidence the component works — it's evidence the component hasn't
> been isolated yet.

The full-sample test_IC at ``meta_trainer.py:942/952`` averages over
predominantly full-history rows. Short-history tickers (rows where any
L1 feature was NaN before rank-normalization) are a tiny minority and
their per-row error is invisible in the aggregate. This module isolates
that subsample, computes per-component IC vs a named simple-fallback
baseline, and supplies a hard-fail promotion gate so a deploy that
regresses on short-history tickers blocks before reaching S3.

Subsample definition: training rows whose pre-rank-norm L1 feature
vector contained any NaN. Mimics the inference-time scenario the data
layer ships (alpha-engine-data PR #78 — partial-NaN features for
short-warmup tickers like SNDK / SARO / SOLS / Q).

Named baselines (intentionally simple — the "minimum bar" a GBM must
clear to justify its complexity):

  - **Momentum**: weighted average of raw momentum features
    ``(0.4·m5 + 0.3·m20 + 0.2·ma50 + 0.1·(rsi-50)/100)`` — matches the
    direct-fallback path at ``run_inference.py:264-271`` so the gate
    asks "does the GBM beat the fallback the inference path already
    knows how to use when GBM IC < 0.02?"
  - **Volatility**: realized 20-day volatility passthrough — directly
    correlates with future absolute return.

The research calibrator gate is tracked as a separate follow-up
(different training pipeline; ``model/research_calibrator.py`` fits
on score_performance, not the X_mom/X_vol matrices).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# Min subsample size below which the gate is SKIPPED (insufficient
# statistical power to make a promotion decision). Above this floor the
# Pearson IC is meaningful enough to trust as a relative comparison.
# 30 mirrors the convention used elsewhere in the trainer
# (`pearsonr` requires N >= 3; we want a real signal floor).
MIN_SUBSAMPLE_SIZE = 30


@dataclass
class ComponentValidation:
    """Result of one component's subsample IC gate."""

    component: str
    n: int
    component_ic: float
    baseline_ic: float
    passed: bool
    skip_reason: str | None = None

    def log(self) -> None:
        if self.skip_reason:
            log.info(
                "Subsample-IC gate [%s]: SKIPPED — %s (n=%d)",
                self.component, self.skip_reason, self.n,
            )
            return
        log.info(
            "Subsample-IC gate [%s]: component_IC=%+.4f %s baseline_IC=%+.4f "
            "(n=%d) → %s",
            self.component, self.component_ic,
            ">=" if self.passed else "<", self.baseline_ic,
            self.n, "PASS" if self.passed else "BLOCK",
        )


def _safe_pearson_ic(preds: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 instead of NaN for
    constant-prediction edge cases. Matches the convention at
    ``meta_trainer.py:942`` so the gate's IC is comparable to test_IC."""
    if len(preds) < 2 or len(y) < 2:
        return 0.0
    if np.std(preds) <= 1e-10 or np.std(y) <= 1e-10:
        return 0.0
    ic = float(np.corrcoef(preds, y)[0, 1])
    return ic if np.isfinite(ic) else 0.0


def momentum_baseline_predict(
    X_mom_raw: np.ndarray, feature_names: list[str],
) -> np.ndarray:
    """Named baseline for the momentum component: weighted average of
    raw momentum features matching the direct-fallback formula at
    ``inference/stages/run_inference.py:267-271``.

    Uses ``np.nan_to_num`` with neutral defaults (0 for momentum/MA
    features, 50 for RSI) so short-history NaN rows produce a real-
    valued baseline prediction the GBM must beat.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    def _get(name: str, default: float) -> np.ndarray:
        idx = name_to_idx.get(name)
        if idx is None:
            return np.full(X_mom_raw.shape[0], default, dtype=np.float64)
        col = X_mom_raw[:, idx].astype(np.float64)
        return np.where(np.isnan(col), default, col)

    m5 = _get("momentum_5d", 0.0)
    m20 = _get("momentum_20d", 0.0)
    ma50 = _get("price_vs_ma50", 0.0)
    rsi = _get("rsi_14", 50.0)
    return 0.4 * m5 + 0.3 * m20 + 0.2 * ma50 + 0.1 * (rsi - 50) / 100


def volatility_baseline_predict(
    X_vol_raw: np.ndarray, feature_names: list[str],
) -> np.ndarray:
    """Named baseline for the volatility component: passthrough of
    raw realized 20-day volatility. Future absolute return correlates
    directly with realized volatility (volatility clustering); a GBM
    that doesn't beat this passthrough is not earning its complexity.

    Falls back to ``vol_30d`` if the 20-day feature isn't in the
    feature set; final fallback to a zero baseline (which forces a
    clean PASS unless the component itself is anti-correlated, which
    is what we want to catch).
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for cand in ("realized_vol_20d", "vol_20d", "realized_vol_30d", "vol_30d"):
        idx = name_to_idx.get(cand)
        if idx is not None:
            col = X_vol_raw[:, idx].astype(np.float64)
            return np.where(np.isnan(col), 0.0, col)
    log.warning(
        "Volatility baseline: no realized_vol_*d feature found in "
        "feature_names=%r — using zero baseline. Add a realized-vol "
        "column to VOLATILITY_FEATURES or update this baseline list.",
        feature_names,
    )
    return np.zeros(X_vol_raw.shape[0], dtype=np.float64)


def validate_component(
    component_name: str,
    component_preds: np.ndarray,
    baseline_preds: np.ndarray,
    y_true: np.ndarray,
    subsample_mask: np.ndarray,
    min_n: int = MIN_SUBSAMPLE_SIZE,
) -> ComponentValidation:
    """Run the subsample IC gate for one component.

    Parameters
    ----------
    component_name
        Human-readable label for logging (e.g. ``"momentum"``).
    component_preds, baseline_preds, y_true
        Aligned arrays of length N over the slice the caller selected
        (typically the test slice ``[val_end:]``). Must be 1-D.
    subsample_mask
        Boolean array same length as ``component_preds`` selecting the
        short-history rows.
    min_n
        Skip the gate if the subsample has fewer than this many rows
        (statistical-power floor).

    Returns
    -------
    ComponentValidation with ``passed=True`` when component_IC >=
    baseline_IC (or when the gate was skipped for size). The gate
    explicitly does NOT require positive component IC — short-history
    is a noisy slice and the right question is "does the GBM do at
    least as well as the simple fallback?", not "is the GBM clearly
    skillful?".
    """
    if not (
        component_preds.shape == baseline_preds.shape == y_true.shape
        == subsample_mask.shape
    ):
        raise ValueError(
            f"validate_component[{component_name}]: shape mismatch — "
            f"component={component_preds.shape}, baseline={baseline_preds.shape}, "
            f"y_true={y_true.shape}, mask={subsample_mask.shape}"
        )

    sub_idx = np.where(subsample_mask)[0]
    n = int(sub_idx.size)
    if n < min_n:
        return ComponentValidation(
            component=component_name, n=n,
            component_ic=0.0, baseline_ic=0.0, passed=True,
            skip_reason=f"subsample size {n} < min_n={min_n}",
        )

    component_ic = _safe_pearson_ic(component_preds[sub_idx], y_true[sub_idx])
    baseline_ic = _safe_pearson_ic(baseline_preds[sub_idx], y_true[sub_idx])
    passed = component_ic >= baseline_ic
    return ComponentValidation(
        component=component_name, n=n,
        component_ic=component_ic, baseline_ic=baseline_ic, passed=passed,
    )
