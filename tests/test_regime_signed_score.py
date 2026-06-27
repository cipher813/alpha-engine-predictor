"""Tests for the signed continuous regime score (issue #1299).

Covers the redesign of ``effective_regime`` composition from the
most-protective categorical merge into a signed continuous [-1, 1] blend:

- per-head signed contributions on [-1, 1] (intensity_z folded in, HMM as a
  posterior expectation with the argmax demoted, downside-only drawdown legs)
- the blend renormalizes over present heads
- continuity / monotonicity of the score in intensity_z
- the STUCK-HMM defect from the issue: a stuck ``P(bear)=1.00`` argmax no
  longer forces the composition to bear once the market-grounded continuous
  heads say ~neutral
- the asymmetric protective floor (forced_bear) is preserved
- the categorical projection of the score
- backward compatibility: the legacy categorical ``effective_regime`` is
  UNCHANGED and the score is additive
- the substrate folds HMM probs + intensity_z + BOCPD into the composition
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime.drawdown import (
    compose_effective_regime,
    compose_regime_score,
    excess_tier_to_score,
    hmm_posterior_expectation,
    intensity_z_to_score,
    regime_score_to_categorical,
    spy_tier_to_score,
)


# ── per-head signed mappings ─────────────────────────────────────────────

def test_hmm_posterior_expectation_signs_and_demotes_argmax() -> None:
    # Strong bull posterior → near +1.
    assert hmm_posterior_expectation({"bull": 0.9, "neutral": 0.1, "bear": 0.0}) > 0.8
    # Strong bear posterior → near -1.
    assert hmm_posterior_expectation({"bull": 0.0, "neutral": 0.1, "bear": 0.9}) < -0.8
    # Balanced posterior → ~0 (the demotion: conviction, not argmax).
    assert abs(hmm_posterior_expectation({"bull": 0.34, "neutral": 0.33, "bear": 0.33})) < 0.05
    # Empty / missing → leg absent.
    assert hmm_posterior_expectation(None) is None
    assert hmm_posterior_expectation({}) is None


def test_drawdown_legs_are_downside_only() -> None:
    # SPY tiers can never vote positive.
    assert spy_tier_to_score("risk_on") == 0.0
    assert spy_tier_to_score("caution") < 0.0
    assert spy_tier_to_score("risk_off") == -1.0
    assert spy_tier_to_score("caution") > spy_tier_to_score("risk_off")
    # Excess leg likewise downside-only.
    assert excess_tier_to_score("risk_on") == 0.0
    assert excess_tier_to_score("alpha_bleed") == -1.0
    # Unknown / None ⇒ leg absent.
    assert spy_tier_to_score(None) is None
    assert excess_tier_to_score("nonsense") is None


def test_intensity_z_signed_zero_centered_and_bounded() -> None:
    assert intensity_z_to_score(0.0) == 0.0
    assert intensity_z_to_score(2.0) > 0.0       # risk-on positive
    assert intensity_z_to_score(-2.0) < 0.0      # risk-off negative
    # Bounded in [-1, 1] even at large magnitudes (tanh saturates to ±1).
    assert -1.0 <= intensity_z_to_score(-100.0) <= -0.99
    assert 0.99 <= intensity_z_to_score(100.0) <= 1.0
    # Non-finite / None ⇒ leg absent.
    assert intensity_z_to_score(float("nan")) is None
    assert intensity_z_to_score(None) is None


def test_intensity_z_score_is_monotone_continuous() -> None:
    zs = np.linspace(-3, 3, 25)
    scores = [intensity_z_to_score(z) for z in zs]
    # Strictly increasing (monotone) and small steps (continuous).
    diffs = np.diff(scores)
    assert np.all(diffs > 0)
    assert np.max(diffs) < 0.2


# ── blend behaviour ──────────────────────────────────────────────────────

def test_blend_renormalizes_over_present_heads() -> None:
    # Only intensity_z present → score equals that head's contribution
    # (renormalized weight = 1.0), no dilution toward 0 from absent legs.
    out = compose_regime_score(intensity_z=2.0)
    assert set(out["head_scores"]) == {"intensity_z"}
    assert abs(out["regime_score"] - intensity_z_to_score(2.0)) < 1e-9
    assert abs(sum(out["weights_used"].values()) - 1.0) < 1e-9


def test_intensity_z_now_influences_the_score() -> None:
    # THE issue: intensity_z previously never entered the composition.
    # Holding the HMM fixed at a neutral posterior, moving intensity_z from
    # risk-off to risk-on must move the score.
    hmm = {"bull": 0.33, "neutral": 0.34, "bear": 0.33}
    bear = compose_regime_score(hmm_probs=hmm, intensity_z=-2.5)["regime_score"]
    bull = compose_regime_score(hmm_probs=hmm, intensity_z=+2.5)["regime_score"]
    assert bull > bear
    assert bear < 0 < bull


def test_stuck_hmm_argmax_no_longer_dominates() -> None:
    # Reproduce the issue's documented cohort: a STUCK HMM with
    # P(bear)=1.00 at a 56-week filter run-length (a label-stability
    # artifact), but every market-grounded continuous signal says ~neutral
    # (intensity_z≈-0.22, SPY risk_on, no excess bleed, BOCPD
    # change_signal=False). Under the OLD most-protective merge the lone
    # stuck categorical head forced bear; the signed blend must NOT.
    out = compose_regime_score(
        hmm_probs={"bull": 0.0, "neutral": 0.0, "bear": 1.0},
        hmm_weeks_in_state=56,
        intensity_z=-0.22,
        spy_tier="risk_on",
        excess_tier="risk_on",
        bocpd_change_signal=False,
    )
    # The stuck HMM is demoted (posterior expectation + staleness discount),
    # so it no longer single-handedly forces a bear call when the
    # market-grounded continuous heads say ~neutral.
    assert out["regime_score_categorical"] != "bear"
    # And the staleness discount is actually engaged (< 1.0 at 56 weeks).
    assert out["hmm_staleness_discount"] < 1.0

    # Contrast: a FRESHLY transitioned high-conviction bear HMM (still in
    # grace) keeps full authority and CAN drive the call bear.
    fresh = compose_regime_score(
        hmm_probs={"bull": 0.0, "neutral": 0.0, "bear": 1.0},
        hmm_weeks_in_state=2,
        intensity_z=-0.22,
        spy_tier="risk_on",
        excess_tier="risk_on",
    )
    assert fresh["hmm_staleness_discount"] == 1.0
    assert fresh["regime_score"] < out["regime_score"]


def test_forced_bear_floor_preserved() -> None:
    # Even with maximally bullish heads, forced_bear clamps to -1 (the
    # asymmetric protective floor — confirmed multi-signal bear can never be
    # averaged away).
    out = compose_regime_score(
        hmm_probs={"bull": 1.0, "neutral": 0.0, "bear": 0.0},
        intensity_z=3.0,
        forced_bear=True,
    )
    assert out["regime_score"] == -1.0
    assert out["forced_bear_floor_applied"] is True
    assert out["regime_score_categorical"] == "bear"


def test_score_bounded_and_no_heads_is_zero() -> None:
    out = compose_regime_score()
    assert out["regime_score"] == 0.0
    assert out["regime_score_categorical"] == "neutral"
    assert out["head_scores"] == {}


def test_bocpd_is_surfaced_not_an_axis_vote() -> None:
    # BOCPD change signal is carried as an uncertainty flag, it does not
    # change the score numerically (issue #1299: not an axis vote).
    base = compose_regime_score(intensity_z=1.0, bocpd_change_signal=False)
    flagged = compose_regime_score(intensity_z=1.0, bocpd_change_signal=True)
    assert base["regime_score"] == flagged["regime_score"]
    assert flagged["bocpd_change_signal"] is True


# ── categorical projection ───────────────────────────────────────────────

def test_categorical_projection_symmetric_thresholds() -> None:
    assert regime_score_to_categorical(0.9) == "bull"
    assert regime_score_to_categorical(-0.9) == "bear"
    assert regime_score_to_categorical(0.0) == "neutral"
    assert regime_score_to_categorical(0.05) == "neutral"
    assert regime_score_to_categorical(-0.05) == "neutral"
    # Symmetric band.
    assert regime_score_to_categorical(0.5) == "bull"
    assert regime_score_to_categorical(-0.5) == "bear"


# ── backward compatibility: legacy categorical unchanged + additive ──────

def test_compose_effective_regime_legacy_categorical_unchanged() -> None:
    # The legacy most-protective categorical field must be byte-identical to
    # the pre-#1299 behaviour, even when the new signed heads disagree.
    out = compose_effective_regime(
        hmm_argmax="bear",
        spy_tier="risk_on",
        excess_tier="risk_on",
        hmm_probs={"bull": 0.0, "neutral": 0.0, "bear": 1.0},
        intensity_z=-0.1,
    )
    # Legacy: HMM argmax=bear, drawdown legs None → effective bear.
    assert out["effective_regime"] == "bear"
    # Additive signed score is present and is its own (different) opinion.
    assert "regime_score" in out
    assert "regime_score_categorical" in out
    assert -1.0 <= out["regime_score"] <= 1.0


def test_compose_effective_regime_additive_keys_present() -> None:
    out = compose_effective_regime(hmm_argmax="neutral")
    for k in (
        "effective_regime", "drivers", "drawdown_tier",
        "drawdown_protective_severity", "forced_bear",
        "regime_score", "regime_score_categorical", "regime_score_detail",
    ):
        assert k in out


# ── substrate folds the new heads in ─────────────────────────────────────

def _hist(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "spy_20d_return": rng.normal(0.01, 0.02, n),
        "vix_level": rng.normal(16.0, 3.0, n),
        "vix_term_slope": rng.normal(0.5, 0.3, n),
        "hy_oas_bps": rng.normal(350.0, 50.0, n),
        "yield_curve_slope": rng.normal(0.3, 0.2, n),
        "market_breadth": rng.normal(0.6, 0.1, n).clip(0, 1),
    })


def _fitted_hmm(history: pd.DataFrame):
    from regime.hmm import HMMRegimeClassifier

    return HMMRegimeClassifier(n_states=3, random_state=0).fit(
        history, list(history.columns)
    )


def test_substrate_folds_hmm_probs_and_intensity_into_score() -> None:
    from regime.substrate import build_regime_substrate

    hist = _hist()
    payload = build_regime_substrate(
        feature_history=hist, current_features=hist.iloc[-1].to_dict(),
        hmm=_fitted_hmm(hist), run_id="2606271200",
        calendar_date="2026-06-27", trading_day="2026-06-26",
        drawdown_block={"spy": {"tier": "risk_on"},
                        "excess": {"available": False, "tier": "risk_on"}},
    )
    eff = payload["effective_regime"]
    assert "regime_score" in eff
    assert -1.0 <= eff["regime_score"] <= 1.0
    heads = eff["regime_score_detail"]["head_scores"]
    # Both the HMM posterior and intensity_z must now be present heads — the
    # exact defect the issue calls out (they previously never entered).
    assert "hmm" in heads
    assert "intensity_z" in heads


def test_substrate_sidecar_carries_regime_score() -> None:
    from regime.substrate import build_regime_substrate, write_regime_substrate

    class _Mem:
        def __init__(self) -> None:
            self.objs: dict[str, bytes] = {}

        def put_object(self, *, Bucket, Key, Body, ContentType=None):
            self.objs[Key] = Body if isinstance(Body, bytes) else Body.encode()
            return {}

    import json as _json

    hist = _hist()
    payload = build_regime_substrate(
        feature_history=hist, current_features=hist.iloc[-1].to_dict(),
        hmm=_fitted_hmm(hist), run_id="2606271200",
        calendar_date="2026-06-27", trading_day="2026-06-26",
        drawdown_block={"spy": {"tier": "risk_on"},
                        "excess": {"available": False, "tier": "risk_on"}},
    )
    s3 = _Mem()
    res = write_regime_substrate(payload, s3_client=s3, bucket="b", prefix="regime")
    sidecar = _json.loads(s3.objs[res["latest_key"]])
    assert sidecar["regime_score"] == payload["effective_regime"]["regime_score"]
    assert sidecar["regime_score_categorical"] in {"bull", "neutral", "bear"}
    # Legacy categorical still present + unchanged.
    assert sidecar["effective_regime"] in {"bull", "neutral", "caution", "bear"}
