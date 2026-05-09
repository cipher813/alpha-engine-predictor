"""Deterministic momentum scorer — Layer-1 momentum component.

The momentum L1 was a LightGBM trained on 9 features. Across 16 weeks of
walk-forward validation it consistently failed to beat its named baseline
(weighted average of its 4 raw inputs): GBM val_IC ~0.01, baseline
val_IC ~0.31. The GBM was not earning its complexity — it was a
calibration tax on the meta-Ridge stack.

Per the audit-arc principle "GBMs that don't beat their named baseline
don't earn the complexity": momentum's L1 component is now the named
baseline directly. The Ridge stacker receives the deterministic score
as the ``momentum_score`` META_FEATURE. The component-level subsample
gate dissolves (component IS the baseline) and the meta-Ridge gets
a substantially stronger momentum signal.

Single source of truth for the formula. Both the training-time
vectorized path (``predict_array``) and the inference-time per-row path
(``predict_dict``) live here. Pre-2026-05-09 the same formula appeared
in two places — ``subsample_validator.momentum_baseline_predict`` and
``run_inference.py``'s GBM-fallback branch — drift between them was a
latent bug.
"""

from __future__ import annotations

import numpy as np


# Coefficients — kept here as a constant so any future tuning is one
# edit, not three. Order: m5, m20, price_vs_ma50, rsi (centered, scaled).
_W_M5 = 0.4
_W_M20 = 0.3
_W_MA50 = 0.2
_W_RSI = 0.1


def predict_array(
    X_mom_raw: np.ndarray, feature_names: list[str],
) -> np.ndarray:
    """Vectorized momentum score for a (n_rows, n_features) raw-feature
    array. Used by ``training/meta_trainer.py`` for walk-forward + final
    fit residuals — produces one momentum_score per training row.

    Uses neutral defaults (0.0 for momentum / MA features, 50 for RSI)
    so short-history NaN rows produce a real-valued score the Ridge
    can weight, rather than poisoning the meta input.
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
    return _W_M5 * m5 + _W_M20 * m20 + _W_MA50 * ma50 + _W_RSI * (rsi - 50) / 100


def predict_dict(features: dict) -> float:
    """Single-ticker momentum score for an inference-time feature dict.

    Mirrors ``predict_array`` but reads from a dict (one row's worth of
    raw features) — what ``run_inference.py`` has at hand per ticker.
    Same neutral defaults; ``None`` and ``NaN`` both degrade to default
    rather than poisoning the score.
    """
    def _safe(name: str, default: float) -> float:
        v = features.get(name)
        if v is None:
            return default
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(f) else f

    m5 = _safe("momentum_5d", 0.0)
    m20 = _safe("momentum_20d", 0.0)
    ma50 = _safe("price_vs_ma50", 0.0)
    rsi = _safe("rsi_14", 50.0)
    return _W_M5 * m5 + _W_M20 * m20 + _W_MA50 * ma50 + _W_RSI * (rsi - 50) / 100
