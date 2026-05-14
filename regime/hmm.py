"""
regime/hmm.py — 3-state Gaussian HMM regime classifier.

Canonical institutional regime detection primitive (Hamilton 1989). Used
here to produce continuous posterior probabilities P(bull/neutral/bear)
at inference time — never argmax-routed (per the 2026-05-10 directive
against per-regime sub-models).

LOOK-AHEAD DISCIPLINE
---------------------
Two inference modes, structurally distinct:

- ``predict_proba_filter()`` — **Hamilton-Kim filter (forward-only)**.
  Posterior P(state_t | obs_{1:t}). Uses ONLY data up to time t. This
  is the inference-time API and the only mode that should ever be
  consumed by production substrate writes.
- ``predict_proba_smoother()`` — **forward-backward smoother**.
  Posterior P(state_t | obs_{1:T}) using full sequence. By construction
  this uses *future* observations (t+1, ..., T) to refine the estimate
  at time t. Intended **exclusively** for retrospective T1 ground-truth
  labeling per regime-v3-260514.md §5.3.3 — must NEVER be invoked at
  inference time. The substrate orchestrator does not call this method;
  only the retrospective labeling pipeline does.

A regression test pins this contract: any code path that consumes
smoothed probabilities must read from a *lagged* artifact location
(``regime/retrospective/{run_id}/``), never the contemporaneous artifact
written by the substrate orchestrator.

STATE PINNING
-------------
Raw HMM fits have label-switching: across re-fits the "bull" state may
end up at index 0 in week N and index 2 in week N+1. ``_pin_states``
sorts the fitted states by mean SPY 20-day return — highest mean = bull,
lowest = bear — and remaps so the output order is always (bear, neutral,
bull). Regression test pins this in the substrate orchestrator.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Canonical state order in returned probability vectors. Pinned post-fit
# via _pin_states() so consumers can trust positional access.
REGIME_STATES: tuple[str, ...] = ("bear", "neutral", "bull")


# The pinning feature must be present in every fit. SPY 20d return is
# the cleanest single-feature partition of regime severity historically.
PIN_FEATURE: str = "spy_20d_return"


class HMMRegimeClassifier:
    """3-state Gaussian HMM regime classifier with state pinning.

    Wraps ``hmmlearn.hmm.GaussianHMM`` with:

    - State-order pinning by mean ``spy_20d_return`` (stable across refits).
    - Filter-only inference (forward, no look-ahead) as the default
      production API.
    - Forward-backward smoother as an explicitly-named retrospective-only
      method.

    Parameters
    ----------
    n_states
        Number of latent states. Default 3 (bear/neutral/bull).
    n_iter
        EM iteration cap. Default 100.
    covariance_type
        ``hmmlearn`` covariance type. Default "full".
    random_state
        Seed for reproducibility across refits.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._model = None
        self._feature_names: list[str] | None = None
        self._state_permutation: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, feature_columns: Sequence[str]) -> "HMMRegimeClassifier":
        """Fit the HMM on a historical feature matrix.

        Parameters
        ----------
        features
            DataFrame of macro feature history. One row per observation
            (typically weekly). Must contain ``PIN_FEATURE`` and every
            entry in ``feature_columns``.
        feature_columns
            Ordered list of feature column names to feed the emission
            model. The order is preserved at inference time.

        Returns
        -------
        self
        """
        from hmmlearn.hmm import GaussianHMM

        missing = [c for c in feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if PIN_FEATURE not in features.columns:
            raise ValueError(
                f"Pinning feature '{PIN_FEATURE}' missing from features DataFrame; "
                f"required for stable state-order pinning across refits."
            )

        X = features[list(feature_columns)].dropna().to_numpy()
        if len(X) < self.n_states * 20:
            raise ValueError(
                f"Insufficient observations for HMM fit: got {len(X)}, "
                f"need at least {self.n_states * 20} for stable EM convergence."
            )

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(X)

        # Compute state permutation that orders states bear < neutral < bull
        # by mean spy_20d_return inside each state's posterior assignment.
        viterbi_states = model.predict(X)
        pin_values = features.loc[features[list(feature_columns)].dropna().index, PIN_FEATURE].to_numpy()
        state_pin_means = np.array([
            pin_values[viterbi_states == s].mean() if (viterbi_states == s).any() else np.nan
            for s in range(self.n_states)
        ])
        # Ascending sort of state pin means → bear (lowest) → neutral → bull (highest).
        # state_permutation[i] = original-state-index-now-at-position-i
        state_permutation = np.argsort(state_pin_means)

        self._model = model
        self._feature_names = list(feature_columns)
        self._state_permutation = state_permutation
        logger.info(
            "[hmm] fit complete: %d observations, pinned state order means=%s",
            len(X), state_pin_means[state_permutation].tolist(),
        )
        return self

    def predict_proba_filter(self, features: pd.DataFrame) -> np.ndarray:
        """Filter posterior probabilities — forward-only, no look-ahead.

        Computes P(state_t | obs_{1:t}) for every t in ``features``. The
        last row's probabilities are the inference-time output for the
        most recent observation. **This is the only inference API safe
        for production substrate writes.**

        Returns an array of shape ``(len(features), n_states)`` with
        rows ordered chronologically and columns in pinned order
        (bear, neutral, bull). Probabilities sum to 1.0 per row.
        """
        if self._model is None or self._state_permutation is None:
            raise RuntimeError("HMM not fitted — call fit() first.")

        X = features[self._feature_names].to_numpy()
        # hmmlearn exposes the filter via the framelogprob + forwardpass
        # API; for clarity we use the wrapper `score_samples` which
        # returns log_prob + smoothed posteriors but we need *filtered*
        # not smoothed. Re-implement filter directly using log_probs.
        log_prob_obs = self._model._compute_log_likelihood(X)  # shape (T, K)
        log_start = np.log(self._model.startprob_ + 1e-300)
        log_trans = np.log(self._model.transmat_ + 1e-300)

        T, K = log_prob_obs.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_start + log_prob_obs[0]
        for t in range(1, T):
            # log_alpha[t, j] = log_prob_obs[t, j] + logsumexp_i(log_alpha[t-1, i] + log_trans[i, j])
            log_alpha[t] = log_prob_obs[t] + _logsumexp_axis0(
                log_alpha[t - 1][:, None] + log_trans
            )
        # Normalize per row
        log_alpha_norm = log_alpha - _logsumexp_axis1(log_alpha)[:, None]
        filtered = np.exp(log_alpha_norm)
        return filtered[:, self._state_permutation]

    def predict_proba_smoother(self, features: pd.DataFrame) -> np.ndarray:
        """Forward-backward smoothed posteriors — RETROSPECTIVE USE ONLY.

        Computes P(state_t | obs_{1:T}) using the full sequence. By
        construction this uses *future* observations (t+1, ..., T) to
        refine the estimate at time t.

        **Do not call from any inference / substrate-write path.** This
        method is reserved for the retrospective T1 ground-truth labeling
        pipeline (regime-v3-260514.md §5.3.3) which writes to a
        ``regime/retrospective/`` prefix with explicit ≥8-week lag.

        Same return shape as ``predict_proba_filter`` (pinned state order).
        """
        if self._model is None or self._state_permutation is None:
            raise RuntimeError("HMM not fitted — call fit() first.")
        X = features[self._feature_names].to_numpy()
        # hmmlearn's predict_proba uses forward-backward (smoothed).
        smoothed = self._model.predict_proba(X)
        return smoothed[:, self._state_permutation]

    def log_likelihood(self, features: pd.DataFrame) -> float:
        """Sequence log-likelihood under the fitted model. Used for
        held-out validation (regime-v3-260514.md §5.5 gate #2)."""
        if self._model is None:
            raise RuntimeError("HMM not fitted — call fit() first.")
        X = features[self._feature_names].to_numpy()
        return float(self._model.score(X))

    def pinned_state_means(self, features: pd.DataFrame) -> np.ndarray:
        """Return per-state mean of ``PIN_FEATURE`` in pinned order.

        Used by tests to verify the bear < neutral < bull invariant and
        by the substrate orchestrator to surface state diagnostics.
        """
        if self._model is None or self._state_permutation is None:
            raise RuntimeError("HMM not fitted — call fit() first.")
        viterbi_states = self._model.predict(features[self._feature_names].to_numpy())
        pin_values = features[PIN_FEATURE].to_numpy()
        means = np.array([
            pin_values[viterbi_states == s].mean() if (viterbi_states == s).any() else np.nan
            for s in range(self.n_states)
        ])
        return means[self._state_permutation]


def _logsumexp_axis0(arr: np.ndarray) -> np.ndarray:
    """Numerically-stable logsumexp along axis 0."""
    m = arr.max(axis=0)
    return m + np.log(np.exp(arr - m).sum(axis=0))


def _logsumexp_axis1(arr: np.ndarray) -> np.ndarray:
    """Numerically-stable logsumexp along axis 1."""
    m = arr.max(axis=1)
    return m + np.log(np.exp(arr - m[:, None]).sum(axis=1))
