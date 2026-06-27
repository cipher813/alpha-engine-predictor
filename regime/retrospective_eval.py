"""
regime/retrospective_eval.py — Tier 1 retrospective HMM smoothing eval.

Closes Stage C.2 T1 per regime-v3-260514.md §5.3.3. Provides
8-week-lagged retrospective regime labels by running the HMM smoother
(forward-backward) over feature history that extends K weeks past
each macro agent call. Compares contemporaneous macro agent regime
calls against these retrospective ground-truth labels using an
asymmetric-loss-weighted agreement metric (bear-miss penalized more
than bull-miss, reflecting the business preference that under-
estimating risk-off conditions is worse than over-estimating them).

Why retrospective smoothing is the right ground truth
-----------------------------------------------------
Regime is a latent variable — there is no contemporaneous truth, only
proxies. The HMM forward-only filter (Hamilton-Kim) is what feeds the
macro agent at inference, using observations up through time t. The
HMM forward-backward smoother uses observations through time t+K to
refine the posterior estimate at time t. That refinement is the
closest thing to retrospective truth — structurally analogous to
how NBER dates business cycles retrospectively with lag.

CRITICAL LOOK-AHEAD DISCIPLINE
------------------------------
This module is the ONLY place in the system that calls
``HMMRegimeClassifier.predict_proba_smoother``. The inference path
(``regime/handler.py`` and the substrate Lambda) uses
``predict_proba_filter`` exclusively — pinned by regression test in
``tests/test_regime_hmm.py`` (PR #154). Mixing smoothed and filtered
posteriors at inference would inject look-ahead bias. Treat this
module as retrospective-only; its outputs land in a
``regime/retrospective/`` S3 prefix with explicit ≥K-week lag, never
in the live ``regime/`` prefix.

Asymmetric loss
---------------
``BEAR_MISS_WEIGHT`` (default 2.0) penalizes calling bull/neutral
when retrospective truth is bear more than the reverse. Symmetric
agreement is easy to game on the bull side (the bias of the
training data). The asymmetric loss reflects:
- Business cost: missed bear regime → under-defensive sizing →
  exposure to drawdown. Missed bull regime → opportunity cost (less
  painful for a relative-alpha strategy).
- Statistical cost: bear regimes are rarer in the training data, so
  a model that defaults to "bull" wins symmetric accuracy.

Output schema
-------------
Eval artifact written to ``regime/retrospective/{YYMMDDHHMM}.json`` +
``regime/retrospective/latest.json`` via the canonical
``nousergon_lib.eval_artifacts`` shape. Carries per-week pairings
(agent call vs smoothed label) and the 26-week rolling asymmetric-
loss-weighted agreement rate that's the headline T1 metric.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .hmm import HMMRegimeClassifier, PIN_FEATURE, REGIME_STATES


logger = logging.getLogger(__name__)


# Asymmetric loss weight — penalty multiplier for bear-misses (agent
# called bull/neutral when retrospective truth was bear) vs symmetric.
# Bull-misses (agent called bear when truth was bull/neutral) are
# weighted 1.0. Tunable via the ``bear_miss_weight`` parameter on
# ``score_agreement`` for sensitivity analysis.
DEFAULT_BEAR_MISS_WEIGHT: float = 2.0


# Minimum lag (weeks) between the macro agent call and the
# retrospective label window. K=8 per the doc — the smoother needs
# at least 8 future observations after the inference timestep for the
# smoothed posterior to materially refine the filter's estimate.
DEFAULT_LAG_WEEKS: int = 8


# Default rolling window (weeks) for the headline agreement metric.
# Stage A observation window is 4 weeks; 26 gives ~half a year of
# stratification once we've populated the historical baseline.
DEFAULT_ROLLING_WEEKS: int = 26


# 4-class agent taxonomy → 3-class HMM space. "caution" collapses to
# "bear" because the HMM has no caution state and the macro agent's
# caution call structurally signals bear-tilt (per regime-v3-260514.md
# §5.1's HMM-vs-macro-agent disagreement check using the same mapping).
_AGENT_TO_HMM_LABEL: dict[str, str] = {
    "bull": "bull",
    "neutral": "neutral",
    "caution": "bear",
    "bear": "bear",
}


@dataclass(frozen=True)
class RegimePairing:
    """A single (agent_call_at_T, retrospective_label_at_T) pair."""

    trading_day: pd.Timestamp
    agent_call: str               # Raw 4-class agent label
    agent_call_normalized: str    # Mapped to 3-class HMM space
    retrospective_label: str      # HMM smoother argmax at T (3-class)
    agreed: bool                  # agent_call_normalized == retrospective_label
    was_bear_miss: bool           # Agent NOT bear when retrospective IS bear
    was_bull_miss: bool           # Agent IS bear when retrospective NOT bear


def _normalize_agent_label(label: str | None) -> str | None:
    """Map agent's 4-class label into the HMM's 3-class space.

    Returns None if the input is unparseable — callers skip those
    pairings rather than guessing a label.
    """
    if not isinstance(label, str):
        return None
    return _AGENT_TO_HMM_LABEL.get(label.strip().lower())


def compute_retrospective_labels(
    feature_history: pd.DataFrame,
    *,
    hmm_feature_columns: Sequence[str],
    eval_indices: Sequence[int],
    random_state: int = 42,
) -> dict[int, str]:
    """Fit an HMM on the supplied feature history and return the
    SMOOTHED-posterior argmax label at each requested ``eval_index``.

    ``eval_indices`` are integer positions into ``feature_history``
    after dropna on ``hmm_feature_columns``. Caller is responsible
    for ensuring each ``eval_index`` is at least ``DEFAULT_LAG_WEEKS``
    rows before the end of ``feature_history`` — the smoother needs
    those future observations to refine the posterior at the eval
    point.

    Returns: ``{eval_index: "bear"|"neutral"|"bull"}``. Pinned 3-class
    HMM output (no "caution") because retrospective truth comes from
    the HMM's structural taxonomy.
    """
    if not eval_indices:
        return {}

    hmm_input = feature_history[list(hmm_feature_columns)].dropna()
    if len(hmm_input) < max(eval_indices) + 1:
        raise ValueError(
            f"feature_history has {len(hmm_input)} rows after dropna; "
            f"eval_indices max is {max(eval_indices)} — needs at least "
            f"{max(eval_indices) + DEFAULT_LAG_WEEKS} rows for valid smoothing"
        )
    if PIN_FEATURE not in feature_history.columns:
        raise ValueError(
            f"feature_history lacks PIN_FEATURE={PIN_FEATURE!r} required for "
            "stable HMM state-order pinning across refits"
        )

    hmm = HMMRegimeClassifier(n_states=3, random_state=random_state).fit(
        feature_history.loc[hmm_input.index], list(hmm_feature_columns)
    )
    smoothed = hmm.predict_proba_smoother(hmm_input)
    argmax_idx = np.argmax(smoothed, axis=1)

    return {
        int(i): REGIME_STATES[int(argmax_idx[i])]
        for i in eval_indices
        if 0 <= i < len(argmax_idx)
    }


def pair_agent_calls_with_retrospective(
    *,
    agent_calls: pd.DataFrame,
    retrospective_labels: Mapping[pd.Timestamp, str],
) -> list[RegimePairing]:
    """Join the agent's macro_regime calls with retrospective labels
    by trading_day. Skips pairings where the agent call is unparseable
    or the retrospective label is missing for that day.

    ``agent_calls`` is a DataFrame with columns ``trading_day``
    (datetime) and ``market_regime`` (4-class string). Typically built
    by reading the signals.json history from S3 or replaying from
    the research repo's archive_manager.

    ``retrospective_labels`` is a mapping from trading_day timestamp
    to the 3-class HMM smoothed argmax for that day.
    """
    pairings: list[RegimePairing] = []
    for _, row in agent_calls.iterrows():
        td = row.get("trading_day")
        agent_call = row.get("market_regime")
        if pd.isna(td):
            continue
        normalized = _normalize_agent_label(agent_call)
        if normalized is None:
            logger.info(
                "[T1] skipping pairing for %s — agent_call=%r unparseable",
                td, agent_call,
            )
            continue
        retro = retrospective_labels.get(pd.Timestamp(td))
        if retro is None:
            continue
        agreed = normalized == retro
        was_bear_miss = retro == "bear" and normalized != "bear"
        was_bull_miss = normalized == "bear" and retro != "bear"
        pairings.append(RegimePairing(
            trading_day=pd.Timestamp(td),
            agent_call=str(agent_call),
            agent_call_normalized=normalized,
            retrospective_label=retro,
            agreed=agreed,
            was_bear_miss=was_bear_miss,
            was_bull_miss=was_bull_miss,
        ))
    return pairings


def score_agreement(
    pairings: Sequence[RegimePairing],
    *,
    bear_miss_weight: float = DEFAULT_BEAR_MISS_WEIGHT,
    rolling_window_weeks: int = DEFAULT_ROLLING_WEEKS,
) -> dict[str, Any]:
    """Compute the asymmetric-loss-weighted agreement rate plus
    rolling and confusion-matrix breakdowns.

    Symmetric agreement: ``sum(agreed) / n``.
    Asymmetric agreement: weights bear-misses heavier (cost 2.0 by
    default) than other misses (cost 1.0). Higher = better.

    The headline metric returned in ``rolling_window_score`` uses
    the most-recent ``rolling_window_weeks`` pairings only — Stage A's
    4-week observation window initially, growing to a half-year
    baseline. Symmetric agreement is also reported for comparison.
    """
    if not pairings:
        return {
            "n_pairings": 0,
            "symmetric_agreement_rate": None,
            "asymmetric_weighted_agreement_rate": None,
            "bear_miss_count": 0,
            "bull_miss_count": 0,
            "rolling_window_weeks": rolling_window_weeks,
            "rolling_window_score": None,
            "confusion_matrix": {},
        }

    # Sort chronologically so rolling-window math takes the most recent N
    sorted_pairings = sorted(pairings, key=lambda p: p.trading_day)
    n = len(sorted_pairings)

    n_agreed = sum(1 for p in sorted_pairings if p.agreed)
    bear_miss_count = sum(1 for p in sorted_pairings if p.was_bear_miss)
    bull_miss_count = sum(1 for p in sorted_pairings if p.was_bull_miss)
    other_miss_count = n - n_agreed - bear_miss_count - bull_miss_count
    # other_miss = bear↔neutral or bull↔neutral disagreements that aren't
    # the asymmetric bear-vs-non-bear axis. Penalized at weight 1.0.

    symmetric_rate = n_agreed / n
    asymmetric_total_cost = (
        bear_miss_count * bear_miss_weight
        + bull_miss_count * 1.0
        + other_miss_count * 1.0
    )
    # Max possible cost = all bear-misses (worst case)
    max_cost = n * bear_miss_weight
    asymmetric_rate = 1.0 - (asymmetric_total_cost / max_cost)

    # Rolling window — most-recent N pairings
    recent = sorted_pairings[-rolling_window_weeks:]
    if recent:
        recent_agreed = sum(1 for p in recent if p.agreed)
        recent_bear_miss = sum(1 for p in recent if p.was_bear_miss)
        recent_bull_miss = sum(1 for p in recent if p.was_bull_miss)
        recent_other_miss = len(recent) - recent_agreed - recent_bear_miss - recent_bull_miss
        recent_cost = (
            recent_bear_miss * bear_miss_weight
            + recent_bull_miss * 1.0
            + recent_other_miss * 1.0
        )
        rolling_score = 1.0 - (recent_cost / (len(recent) * bear_miss_weight))
    else:
        rolling_score = None

    # Confusion matrix: rows = agent_call_normalized, cols = retrospective_label
    confusion: dict[str, dict[str, int]] = {
        a: {r: 0 for r in REGIME_STATES} for a in REGIME_STATES
    }
    for p in sorted_pairings:
        confusion[p.agent_call_normalized][p.retrospective_label] += 1

    return {
        "n_pairings": n,
        "symmetric_agreement_rate": symmetric_rate,
        "asymmetric_weighted_agreement_rate": asymmetric_rate,
        "bear_miss_count": bear_miss_count,
        "bull_miss_count": bull_miss_count,
        "other_miss_count": other_miss_count,
        "bear_miss_weight": bear_miss_weight,
        "rolling_window_weeks": rolling_window_weeks,
        "rolling_window_score": rolling_score,
        "rolling_window_size": len(recent),
        "confusion_matrix": confusion,
    }


def assemble_t1_eval_payload(
    *,
    pairings: Sequence[RegimePairing],
    score: Mapping[str, Any],
    run_id: str,
    calendar_date: str,
    trading_day: str,
    feature_window_start: str | None = None,
    feature_window_end: str | None = None,
    lag_weeks: int = DEFAULT_LAG_WEEKS,
    hmm_feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the eval-artifact JSON payload for the T1 retrospective
    eval. Includes both the headline rolling-window score and the
    per-pairing detail so the dashboard can render time series."""
    pairings_serialized = [
        {
            "trading_day": p.trading_day.isoformat() if hasattr(p.trading_day, "isoformat") else str(p.trading_day),
            "agent_call": p.agent_call,
            "agent_call_normalized": p.agent_call_normalized,
            "retrospective_label": p.retrospective_label,
            "agreed": p.agreed,
            "was_bear_miss": p.was_bear_miss,
            "was_bull_miss": p.was_bull_miss,
        }
        for p in sorted(pairings, key=lambda x: x.trading_day)
    ]
    return {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "run_id": run_id,
        "schema_version": 1,
        "eval_tier": "T1_retrospective_hmm_smoothing",
        "lag_weeks": lag_weeks,
        "score": dict(score),
        "pairings": pairings_serialized,
        "model_metadata": {
            "hmm_feature_columns": list(hmm_feature_columns) if hmm_feature_columns else None,
            "feature_window_start": feature_window_start,
            "feature_window_end": feature_window_end,
            "smoother_algorithm": "forward_backward_hamilton_kim",
            "asymmetric_loss_convention": "bear_miss=2.0_other_miss=1.0",
        },
    }
