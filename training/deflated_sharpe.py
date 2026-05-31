"""Predictor-skill promotion battery — ROADMAP L4469 W1.3 (OBSERVE).

TWO DISTINCT LENSES on the predictor's per-date cross-sectional IC series (one
IC per date = did the model rank stocks correctly that day):

1. **Overfit-significance lens** (``deflated_sharpe_ratio``) — Bailey & López de
   Prado. Is the IC genuinely non-zero, or a false discovery from multiple
   testing? The IC information ratio ``IC-IR = mean(IC)/std(IC)`` is the Sharpe
   analog; the **Probabilistic Sharpe Ratio** PSR(SR*) is the probability the
   TRUE IC-IR exceeds SR*; the **Deflated Sharpe Ratio** sets SR* to the
   EXPECTED MAXIMUM IC-IR under the null across ``n_trials`` (~20 backtests
   manufacture a spurious 5%-significant strategy). This is a SIGNIFICANCE test,
   not a performance metric — and PSR is itself part of the house skilled-risk
   basket, so it is on-framework. IC-IR is intentionally symmetric here because
   significance testing needs the full dispersion.

2. **Downside-aware performance lens** (``downside_ic_stats``) — the house
   metric framework (``anchor-gates-on-skilled-risk-not-sharpe``; the
   evaluator-revamp stack is **Sortino + CVaR + max DD**, with Sharpe RETIRED to
   legacy) judges on DOWNSIDE risk, not symmetric volatility. Applied to the IC
   series: **Sortino-of-IC** = mean(IC) / downside-deviation(IC), **CVaR-of-IC**
   = mean of the worst k% daily ICs (the bad-ranking-day tail), and the fraction
   of negative-IC days. This is the PERFORMANCE lens that matches the basket —
   the symmetric IC-IR above is a secondary/significance stat, not the headline.

W1.4 will gate on BOTH legs (real AND downside-robust); any evaluation of the
signal as RETURNS (W3 turnover-adjusted net alpha, W5 decile spreads) uses
Sortino + CVaR + max DD, never Sharpe. OBSERVE ONLY in W1.3 — reported, not
gating. PBO via CSCV is DEFERRED to W1.3b (it measures model-*selection*
overfitting and needs a config grid; degenerate with one config per run today).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

_EULER_MASCHERONI = 0.5772156649015329


def downside_ic_stats(
    ic_series,
    *,
    target: float = 0.0,
    cvar_pct: float = 5.0,
    sortino_threshold: float = 0.0,
) -> dict:
    """Downside-aware IC performance lens (skilled-risk basket: Sortino + CVaR).

    Matches the house framework (``anchor-gates-on-skilled-risk-not-sharpe``)
    which judges on DOWNSIDE risk rather than symmetric volatility, applied to
    the predictor's per-date IC series:

    - ``sortino_of_ic`` = mean(IC) / downside_deviation(IC, target) — penalizes
      only bad-ranking days (IC < target), not upside IC dispersion.
    - ``cvar_of_ic`` = mean of the worst ``cvar_pct``% daily ICs — the expected
      IC on our worst ranking days (Expected Shortfall; CVaR is the basket's
      tail measure).
    - ``frac_negative_ic`` / ``worst_ic`` — how often / how badly we rank wrong.

    This is the PERFORMANCE lens; ``deflated_sharpe_ratio`` is the separate
    OVERFIT-significance lens. ``passes_downside_gate`` (observe) flags
    ``sortino_of_ic >= sortino_threshold``; W1.4 sets the real thresholds.
    """
    ic = np.asarray([x for x in np.asarray(ic_series, dtype=float)
                     if np.isfinite(x)], dtype=float)
    n = int(ic.size)
    if n < 5:
        return {"status": "insufficient", "n": n,
                "sortino_of_ic": None, "cvar_of_ic": float("nan"),
                "passes_downside_gate": False}
    mean_ic = float(ic.mean())
    downside = ic[ic < target] - target
    n_down = int(downside.size)
    dd = float(math.sqrt(float(np.mean(downside ** 2)))) if n_down > 0 else 0.0
    # No bad-ranking days → Sortino is "infinitely good"; report None +
    # n_downside_days=0 rather than inf (JSON-safe) so the consumer reads it as
    # a clean upside-only series, not a missing value.
    sortino = (mean_ic / dd) if dd > 1e-12 else None
    k = max(1, int(math.ceil(cvar_pct / 100.0 * n)))
    worst_k = np.sort(ic)[:k]
    cvar = float(worst_k.mean())
    frac_neg = float((ic < 0).mean())

    def _r(v, p=6):
        if v is None:
            return None
        return round(float(v), p) if np.isfinite(v) else float("nan")

    passes = bool(
        (sortino is not None and sortino >= sortino_threshold)
        or (sortino is None and mean_ic > 0)
    )
    return {
        "status": "ok",
        "n": n,
        "n_downside_days": n_down,
        "mean_ic": _r(mean_ic),
        "downside_deviation": _r(dd),
        "sortino_of_ic": _r(sortino),
        "cvar_pct": cvar_pct,
        "cvar_of_ic": _r(cvar),
        "frac_negative_ic": _r(frac_neg, 4),
        "worst_ic": _r(float(ic.min())),
        "sortino_threshold": sortino_threshold,
        "passes_downside_gate": passes,
    }


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew: float = 0.0, kurt: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """PSR(SR*) — P(true Sharpe > ``sr_benchmark``) given an observed Sharpe
    ``sr`` over ``n`` observations with ``skew`` and (non-excess) ``kurt``.
    Bailey-LdP: ``Φ( (SR - SR*)·sqrt(n-1) / sqrt(1 - skew·SR + (kurt-1)/4·SR²) )``.
    """
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    denom = math.sqrt(var) if var > 1e-12 else 1e-6
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected MAXIMUM Sharpe across ``n_trials`` independent strategies under
    the null (true Sharpe = 0), each drawn with cross-trial Sharpe std
    ``sr_std``. Bailey-LdP closed form using the Euler-Mascheroni constant:
    ``sr_std · [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]``.
    """
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return float(sr_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    ic_series,
    *,
    n_trials: int,
    sr_trials_std: float | None = None,
    threshold: float = 0.95,
) -> dict:
    """IC-IR + its Deflated Sharpe Ratio over the per-date cross-sectional IC
    series (OBSERVE).

    ``n_trials`` is the number of effective configurations searched to obtain
    this model (a conservative placeholder until cumulative trial-count tracking
    lands — W1.3b). ``sr_trials_std`` is the std of trial Sharpes; when not
    supplied it falls back to ``1/sqrt(n)`` (the single-path sampling std — a
    conservative deflation). Returns observe stats incl. ``psr_vs_zero``
    (trial-count-independent significance), ``dsr`` (deflated), and a
    ``passes_overfit_gate`` flag at ``threshold``. NOTE: this is the
    SIGNIFICANCE leg only (is the IC real?), NOT a promotion decision — W1.4
    composes it with the downside-aware performance lens (``downside_ic_stats``:
    Sortino-of-IC + CVaR) per the house skilled-risk basket.
    """
    ic = np.asarray([x for x in np.asarray(ic_series, dtype=float)
                     if np.isfinite(x)], dtype=float)
    n = int(ic.size)
    if n < 5 or ic.std(ddof=1) < 1e-12:
        return {
            "status": "insufficient", "n": n,
            "ic_ir": float("nan"), "psr_vs_zero": float("nan"),
            "dsr": float("nan"), "passes_overfit_gate": False,
        }
    sr = float(ic.mean() / ic.std(ddof=1))
    sk = float(_skew(ic))
    ku = float(_kurtosis(ic, fisher=False))  # non-excess (normal == 3)
    if sr_trials_std is None:
        sr_trials_std = 1.0 / math.sqrt(n)
    sr_star = expected_max_sharpe(n_trials, sr_trials_std)
    psr0 = probabilistic_sharpe_ratio(sr, n, sk, ku, 0.0)
    dsr = probabilistic_sharpe_ratio(sr, n, sk, ku, sr_star)

    def _r(v, p=6):
        return round(float(v), p) if np.isfinite(v) else float("nan")

    return {
        "status": "ok",
        "n": n,
        "ic_ir": _r(sr),
        "ic_mean": _r(float(ic.mean())),
        "ic_std": _r(float(ic.std(ddof=1))),
        "skew": _r(sk, 4),
        "kurtosis": _r(ku, 4),
        "n_trials": int(n_trials),
        "sr_benchmark_maxnull": _r(sr_star),
        "psr_vs_zero": _r(psr0),
        "dsr": _r(dsr),
        "threshold": threshold,
        "passes_overfit_gate": bool(np.isfinite(dsr) and dsr >= threshold),
    }
