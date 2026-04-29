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

The research calibrator uses a different gate shape (time-based
holdout instead of NaN-mask subsample) — see
``validate_research_calibrator`` below. Reason: the calibrator's input
is a scalar ``score`` that LLM-derives from news/fundamentals, so it
doesn't have a "short-history" equivalent the way OHLCV-rolling-window
features do. The right question for that component is whether the
fitted calibrator beats the ``score / 100`` passthrough on a held-out
slice.
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


def research_calibrator_baseline_predict(scores: np.ndarray) -> np.ndarray:
    """Named baseline for the research calibrator: ``score / 100``
    passthrough. This is the same fallback ``research_features.py:111``
    uses when the calibrator isn't fitted, so the gate asks "does the
    fitted calibrator beat the smooth passthrough the inference path
    already knows how to use when the calibrator is absent?".
    """
    return np.asarray(scores, dtype=np.float64) / 100.0


def validate_research_calibrator(
    scores: np.ndarray,
    beat_spy: np.ndarray,
    score_dates: np.ndarray,
    holdout_frac: float = 0.2,
    min_n: int = MIN_SUBSAMPLE_SIZE,
) -> ComponentValidation:
    """Time-based holdout IC gate for the research calibrator.

    Unlike the L1 GBM gates (which use a NaN-mask subsample to mimic
    short-history tickers), the research calibrator's input is a scalar
    ``score`` that doesn't have a short-history equivalent — LLM scoring
    operates on news/fundamentals regardless of OHLCV history. The
    relevant gate is whether the fitted calibrator beats the passthrough
    baseline on a held-out slice.

    Parameters
    ----------
    scores
        1-D array of historical research scores from score_performance.
    beat_spy
        1-D array of binary outcomes (beat_spy_10d, 1=hit / 0=miss).
    score_dates
        1-D ``datetime64`` array same length as scores; used to sort
        rows for the time-based holdout.
    holdout_frac
        Fraction of rows (sorted by date) to hold out as the gate's
        test slice. Default 0.2 = last 20% by date.
    min_n
        Skip the gate if either fit-set or holdout-set is too small to
        produce meaningful IC.

    Returns
    -------
    ComponentValidation. ``passed=True`` when the held-out calibrator
    IC >= held-out baseline IC, OR when the gate was skipped for size.
    The production ``prod_calibrator`` fitted on FULL data is unaffected
    by this gate — the gate only validates that the calibrator approach
    (Platt-bucket lookup) beats the score-passthrough baseline.
    """
    from model.research_calibrator import ResearchCalibrator

    scores = np.asarray(scores, dtype=np.float64).ravel()
    beat_spy = np.asarray(beat_spy, dtype=np.int32).ravel()
    score_dates = np.asarray(score_dates).ravel()

    if not (scores.shape == beat_spy.shape == score_dates.shape):
        raise ValueError(
            f"validate_research_calibrator: shape mismatch — "
            f"scores={scores.shape}, beat_spy={beat_spy.shape}, "
            f"dates={score_dates.shape}"
        )

    n_total = scores.size
    if n_total < 2 * min_n:
        return ComponentValidation(
            component="research_calibrator", n=n_total,
            component_ic=0.0, baseline_ic=0.0, passed=True,
            skip_reason=(
                f"total {n_total} rows < 2*min_n={2*min_n} (need at "
                f"least min_n in each of fit + holdout)"
            ),
        )

    sort_idx = np.argsort(score_dates, kind="stable")
    scores_s = scores[sort_idx]
    beat_s = beat_spy[sort_idx]
    n_holdout = max(int(round(n_total * holdout_frac)), min_n)
    n_holdout = min(n_holdout, n_total - min_n)
    n_fit = n_total - n_holdout

    fit_scores = scores_s[:n_fit]
    fit_beat = beat_s[:n_fit]
    test_scores = scores_s[n_fit:]
    test_beat = beat_s[n_fit:]

    gate_calibrator = ResearchCalibrator()
    gate_calibrator.fit(fit_scores, fit_beat)
    if not gate_calibrator.is_fitted:
        return ComponentValidation(
            component="research_calibrator", n=n_total,
            component_ic=0.0, baseline_ic=0.0, passed=True,
            skip_reason=(
                f"gate calibrator failed to fit on {n_fit} rows "
                "(insufficient bucket coverage)"
            ),
        )

    component_preds = gate_calibrator.predict_batch(test_scores)
    baseline_preds = research_calibrator_baseline_predict(test_scores)

    component_ic = _safe_pearson_ic(component_preds, test_beat.astype(np.float64))
    baseline_ic = _safe_pearson_ic(baseline_preds, test_beat.astype(np.float64))
    passed = component_ic >= baseline_ic
    return ComponentValidation(
        component="research_calibrator", n=int(test_beat.size),
        component_ic=component_ic, baseline_ic=baseline_ic, passed=passed,
    )


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
