"""
regime/bocpd.py — Bayesian online change-point detection.

Adams & MacKay 2007 algorithm. Maintains a run-length distribution
P(r_t | obs_{1:t}) — distribution over how long the current regime has
lasted. The most-likely run length collapsing to a small value signals
a regime change.

Used here as a complement to the HMM:

- HMM says   "current state is X with confidence p"
- BOCPD says "the regime changed at time t with confidence q"

The two are independent signals. HMM probabilities can stay stable
across a brief regime shift if the new state's emission distribution
overlaps the old one; BOCPD detects the shift via predictive likelihood
breakdown even when HMM is slow to update.

Implementation: closed-form Gaussian observation model with a
Normal-Inverse-Gamma conjugate prior. Constant hazard function
(default 1/100 ≈ regime change every ~100 weeks on average). Hand-rolled
rather than packaged to keep Lambda footprint small and the algorithm
explicit + testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BOCPDState:
    """Sufficient statistics for the run-length posterior + Gaussian
    observation model. Persisted between updates for online use.

    The four arrays grow by one element per ``update()`` call (one new
    possible run length is added each step). In production we cap growth
    at ``max_runlength_horizon`` (default 520 weeks ≈ 10y) — beyond that
    the run-length tail is collapsed to release memory without changing
    the change-point signal.
    """
    runlength_probs: np.ndarray   # shape (R,) — P(r_t | obs_{1:t})
    mu_t: np.ndarray              # shape (R,) — posterior mean per run length
    kappa_t: np.ndarray           # shape (R,) — pseudo-count per run length
    alpha_t: np.ndarray           # shape (R,) — IG shape per run length
    beta_t: np.ndarray            # shape (R,) — IG scale per run length


class BOCPDDetector:
    """Bayesian online change-point detector — Gaussian observation model.

    Maintains run-length posterior over a univariate observation series.
    For multivariate input, ``update()`` reduces the observation vector
    to a scalar via ``observation_projection`` (default: take the
    composite z-score scalar).

    Parameters
    ----------
    hazard
        Constant hazard rate per time step. Default 1/100. Higher values
        make the detector more sensitive (regimes change more often a
        priori).
    mu_0
        Prior mean. Default 0.0 (z-score-normalized observation).
    kappa_0
        Prior pseudo-count. Default 1.0.
    alpha_0
        Inverse-Gamma shape prior. Default 1.0.
    beta_0
        Inverse-Gamma scale prior. Default 1.0.
    max_runlength_horizon
        Cap on the run-length grid before tail collapse. Default 520.
    """

    def __init__(
        self,
        hazard: float = 1.0 / 100.0,
        mu_0: float = 0.0,
        kappa_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        max_runlength_horizon: int = 520,
    ) -> None:
        self.hazard = hazard
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.max_runlength_horizon = max_runlength_horizon

    def initial_state(self) -> BOCPDState:
        """Return the prior state — no observations yet, run length 0
        has probability 1."""
        return BOCPDState(
            runlength_probs=np.array([1.0]),
            mu_t=np.array([self.mu_0]),
            kappa_t=np.array([self.kappa_0]),
            alpha_t=np.array([self.alpha_0]),
            beta_t=np.array([self.beta_0]),
        )

    def update(self, state: BOCPDState, observation: float) -> BOCPDState:
        """Advance the run-length posterior by one observation.

        Implements equations 1-8 of Adams & MacKay 2007 for the
        Normal-Inverse-Gamma conjugate Gaussian model.
        """
        x = float(observation)

        # Predictive distribution: Student-t with parameters derived from
        # the per-run-length NIG posterior.
        df = 2.0 * state.alpha_t
        scale = np.sqrt(state.beta_t * (state.kappa_t + 1.0) / (state.alpha_t * state.kappa_t))
        pred_probs = _student_t_pdf(x, df=df, loc=state.mu_t, scale=scale)

        # Growth probabilities (no change-point): P(r_t = r_{t-1}+1)
        growth_probs = state.runlength_probs * pred_probs * (1.0 - self.hazard)
        # Change-point probability: P(r_t = 0) = sum over previous r * pred * hazard
        cp_prob = float((state.runlength_probs * pred_probs * self.hazard).sum())

        new_runlength_probs = np.concatenate([[cp_prob], growth_probs])

        # Update sufficient statistics — equations 6-7 of Adams & MacKay.
        new_mu = np.concatenate([
            [self.mu_0],
            (state.kappa_t * state.mu_t + x) / (state.kappa_t + 1.0),
        ])
        new_kappa = np.concatenate([[self.kappa_0], state.kappa_t + 1.0])
        new_alpha = np.concatenate([[self.alpha_0], state.alpha_t + 0.5])
        new_beta = np.concatenate([
            [self.beta_0],
            state.beta_t + (state.kappa_t * (x - state.mu_t) ** 2) / (2.0 * (state.kappa_t + 1.0)),
        ])

        # Normalize run-length posterior
        Z = new_runlength_probs.sum()
        if Z > 0:
            new_runlength_probs = new_runlength_probs / Z

        # Tail collapse — beyond max_runlength_horizon, fold the tail
        # mass into the last slot to keep memory bounded.
        if len(new_runlength_probs) > self.max_runlength_horizon:
            tail_mass = new_runlength_probs[self.max_runlength_horizon:].sum()
            new_runlength_probs = new_runlength_probs[:self.max_runlength_horizon].copy()
            new_runlength_probs[-1] += tail_mass
            new_mu = new_mu[:self.max_runlength_horizon]
            new_kappa = new_kappa[:self.max_runlength_horizon]
            new_alpha = new_alpha[:self.max_runlength_horizon]
            new_beta = new_beta[:self.max_runlength_horizon]

        return BOCPDState(
            runlength_probs=new_runlength_probs,
            mu_t=new_mu,
            kappa_t=new_kappa,
            alpha_t=new_alpha,
            beta_t=new_beta,
        )

    def change_signal(
        self,
        state: BOCPDState,
        *,
        min_runlength_for_change: int = 4,
        change_threshold: float = 0.5,
    ) -> dict:
        """Surface change-point signal from a run-length posterior.

        A change is declared when the run-length posterior places more
        than ``change_threshold`` mass on run lengths shorter than
        ``min_runlength_for_change``.

        Returns
        -------
        dict with:

        - ``change_signal``: bool
        - ``max_runlength_prob``: float — probability mass on most-likely
          run length (high values = settled regime)
        - ``change_confidence``: float — mass on short run lengths

        These three fields land directly in the substrate JSON.
        """
        probs = state.runlength_probs
        if len(probs) == 0:
            return {
                "change_signal": False,
                "max_runlength_prob": 0.0,
                "change_confidence": 0.0,
            }
        short_horizon = min(min_runlength_for_change, len(probs))
        change_confidence = float(probs[:short_horizon].sum())
        max_runlength_prob = float(probs.max())
        return {
            "change_signal": bool(change_confidence > change_threshold),
            "max_runlength_prob": max_runlength_prob,
            "change_confidence": change_confidence,
        }

    def run_offline(self, observations: np.ndarray) -> tuple[BOCPDState, list[dict]]:
        """Run the detector over a complete observation sequence.

        Returns the final state plus a list of per-step change-signal
        dicts. Useful for backtesting + the historical-backfill script.
        """
        state = self.initial_state()
        signals: list[dict] = []
        for obs in observations:
            state = self.update(state, float(obs))
            signals.append(self.change_signal(state))
        return state, signals


def _student_t_pdf(x: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Vectorized Student-t pdf — closed-form predictive distribution
    for the Normal-Inverse-Gamma conjugate Gaussian observation model."""
    from scipy.stats import t as student_t
    return student_t.pdf(x, df=df, loc=loc, scale=scale)
