"""Tests for the heuristic stance classifier.

Emits continuous stance loadings (4-element softmax probability
distribution) AND the dominant discrete label (argmax) per ticker.
The continuous emission matches the institutional factor-model
pattern (Barra/AQR/Fama-French/BlackRock Aladdin derive continuous
loadings from data) while the dominant label keeps simple v1
consumers cheap.

Pure function; tests exercise the smooth-function behavior + boundary
cases without spinning up the full inference pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from model.stance_classifier import (
    STANCE_NAMES,
    StanceLiteral,
    classify_stance,
    MOMENTUM_SCALE,
    VALUE_SCALE,
    CATALYST_PEAK_DAYS,
)


class TestClassifyStanceShape:
    """The output shape contract — every call returns
    ``(dominant, loadings, catalyst_date)`` with valid types and a
    proper probability distribution."""

    def test_returns_three_tuple(self):
        result = classify_stance({"momentum_20d": 0.05})
        assert isinstance(result, tuple) and len(result) == 3

    def test_loadings_sum_to_one(self):
        """Softmax → loadings must form a proper probability
        distribution. ±1e-3 tolerance for float roundoff (matches the
        ``StanceLoadings`` validator's tolerance)."""
        for features in [
            {"momentum_20d": 0.10},
            {"momentum_20d": -0.10},
            {},  # empty
            {"momentum_20d": float("nan"), "days_to_earnings": 10},
        ]:
            _, loadings, _ = classify_stance(features)
            total = sum(loadings.values())
            assert abs(total - 1.0) < 1e-3, (
                f"loadings sum to {total} for features={features}"
            )

    def test_loadings_have_all_four_keys(self):
        """Every call returns all 4 stance keys. Pinned so a future
        feature-set refactor that drops a stance branch doesn't make
        downstream consumers crash on KeyError."""
        _, loadings, _ = classify_stance({"momentum_20d": 0.10})
        assert set(loadings.keys()) == set(STANCE_NAMES)

    def test_loadings_each_in_unit_interval(self):
        """Each loading must be in [0, 1] — proper probability."""
        _, loadings, _ = classify_stance({"momentum_20d": 0.10})
        for stance, loading in loadings.items():
            assert 0.0 <= loading <= 1.0, f"{stance}={loading} out of bounds"

    def test_dominant_is_argmax_of_loadings(self):
        """``dominant`` must equal ``argmax(loadings)`` for every call.
        Pinned so the discrete + continuous emissions stay consistent."""
        for features in [
            {"momentum_20d": 0.20, "price_vs_ma50": 1.10},
            {"momentum_20d": -0.15, "debt_to_equity": 0.3, "eps_ttm": 2.0},
            {"days_to_earnings": 10},
            {"realized_vol_20d": 0.10, "debt_to_equity": 0.3},
        ]:
            dominant, loadings, _ = classify_stance(features)
            argmax_stance = max(loadings, key=lambda k: loadings[k])
            assert dominant == argmax_stance, (
                f"dominant={dominant} != argmax({loadings})={argmax_stance}"
            )

    def test_dominant_is_in_canonical_vocabulary(self):
        """``dominant`` is always one of the 4 StanceLiteral values."""
        dominant, _, _ = classify_stance({})
        assert dominant in STANCE_NAMES


class TestStanceLoadingsBehavior:
    """Verify the smooth-function semantics — stronger features
    produce higher loadings on the matching stance."""

    def test_strong_momentum_dominates_loading(self):
        """+20% in 20d AND price > MA50 → momentum is the dominant
        loading by a wide margin."""
        _, loadings, _ = classify_stance({
            "momentum_20d": 0.20,
            "price_vs_ma50": 1.10,
            "realized_vol_20d": 0.20,
            "debt_to_equity": 0.5,
            "eps_ttm": 2.0,
        })
        assert loadings["momentum"] > 0.4, (
            f"strong momentum should dominate; got {loadings}"
        )
        assert loadings["momentum"] > loadings["value"]
        assert loadings["momentum"] > loadings["quality"]

    def test_oversold_with_quality_fundamentals_dominates_value(self):
        """-15% in 20d AND low debt AND positive EPS → value is the
        dominant loading."""
        _, loadings, _ = classify_stance({
            "momentum_20d": -0.15,
            "price_vs_ma50": 0.85,
            "debt_to_equity": 0.3,  # low leverage
            "eps_ttm": 3.0,         # profitable
            "realized_vol_20d": 0.25,
        })
        assert loadings["value"] > loadings["momentum"]
        assert loadings["value"] > loadings["catalyst"]

    def test_oversold_with_distress_does_not_dominate_value(self):
        """-15% in 20d AND high debt AND negative EPS → value loading
        is dampened by the fundamental_quality factor. Either quality
        wins (default tiebreaker) or value still wins but by less."""
        _, loadings_distressed, _ = classify_stance({
            "momentum_20d": -0.15,
            "debt_to_equity": 3.5,
            "eps_ttm": -2.0,
        })
        _, loadings_quality, _ = classify_stance({
            "momentum_20d": -0.15,
            "debt_to_equity": 0.3,
            "eps_ttm": 3.0,
        })
        # Distressed value loading must be LESS than quality-defended value loading
        assert loadings_distressed["value"] < loadings_quality["value"], (
            f"distressed value={loadings_distressed['value']} should be < "
            f"quality-defended value={loadings_quality['value']}"
        )

    def test_low_vol_low_debt_dominates_quality(self):
        """Low realized vol + low debt + flat momentum → quality is
        the dominant loading."""
        _, loadings, _ = classify_stance({
            "momentum_20d": 0.01,
            "realized_vol_20d": 0.08,
            "debt_to_equity": 0.2,
            "eps_ttm": 5.0,
        })
        assert loadings["quality"] > loadings["momentum"]
        assert loadings["quality"] > loadings["value"]

    def test_upcoming_earnings_dominates_catalyst(self):
        """Earnings 10 days out (peak of catalyst bell curve) →
        catalyst is the dominant loading."""
        _, loadings, _ = classify_stance({
            "momentum_20d": 0.02,
            "days_to_earnings": 10,
        })
        assert loadings["catalyst"] > loadings["momentum"]
        assert loadings["catalyst"] > loadings["value"]
        assert loadings["catalyst"] > loadings["quality"]

    def test_far_future_earnings_does_not_fire_catalyst(self):
        """Earnings 90 days out → catalyst loading near zero (outside
        the bell curve's effective range). Falls through to other
        stances."""
        _, loadings, _ = classify_stance({
            "momentum_20d": 0.02,
            "days_to_earnings": 90,
        })
        # catalyst raw score = 0 (gated by the 60-day max in classifier);
        # post-softmax catalyst loading ≈ 1/4 (uniform distribution when
        # all raws are zero) — but other features should win out
        assert loadings["catalyst"] < loadings["momentum"] or loadings["catalyst"] < loadings["quality"], (
            f"far-future earnings should not dominate; got {loadings}"
        )

    def test_past_earnings_does_not_fire_catalyst(self):
        """Earnings in the past (negative days) → catalyst loading
        ~zero. Defensive against a feature pipeline that emits
        negative values."""
        _, loadings, _ = classify_stance({
            "momentum_20d": 0.02,
            "days_to_earnings": -3,
        })
        # Same behavior as far-future
        assert loadings["catalyst"] <= 0.30, (
            f"past earnings should not load catalyst; got {loadings}"
        )

    def test_mixed_exposure_produces_mixed_loadings(self):
        """The institutional motivation: a ticker with both strong
        momentum AND upcoming earnings should produce loadings on
        BOTH stances (not 1.0 on one and 0.0 on others). Pinned so a
        future refactor that returns to hard-coded discrete rules
        breaks this test."""
        _, loadings, _ = classify_stance({
            "momentum_20d": 0.12,         # strong momentum
            "price_vs_ma50": 1.08,
            "days_to_earnings": 12,       # near catalyst peak
        })
        # Both should have meaningful loading (not just one near 1.0)
        assert loadings["momentum"] > 0.15
        assert loadings["catalyst"] > 0.15


class TestCatalystDateField:
    """``catalyst_date`` is the executor's exit-boundary signal."""

    def test_catalyst_date_passed_through_when_provided(self):
        _, _, catalyst_date = classify_stance({
            "days_to_earnings": 5,
            "next_earnings_date": "2026-05-16",
        })
        assert catalyst_date == "2026-05-16"

    def test_catalyst_date_none_when_field_absent(self):
        _, _, catalyst_date = classify_stance({
            "days_to_earnings": 5,
        })
        assert catalyst_date is None

    def test_catalyst_date_none_for_non_string_values(self):
        """Defensive: feature pipeline drift could emit a non-string
        value. Coerce to None rather than passing through bad data."""
        _, _, catalyst_date = classify_stance({
            "days_to_earnings": 5,
            "next_earnings_date": 20260516,  # int instead of ISO string
        })
        assert catalyst_date is None


class TestDefensivePaths:
    """Boundary cases — empty features, NaN values, missing fields."""

    def test_empty_features_returns_valid_distribution(self):
        """Empty feature row — classifier returns a valid softmax
        (probably uniform) not a crash."""
        dominant, loadings, _ = classify_stance({})
        assert dominant in STANCE_NAMES
        assert abs(sum(loadings.values()) - 1.0) < 1e-3

    def test_nan_features_treated_as_neutral(self):
        """NaN feature values must not crash the classifier. Treated as
        zero / neutral defaults so the softmax still produces a valid
        distribution."""
        dominant, loadings, _ = classify_stance({
            "momentum_20d": float("nan"),
            "price_vs_ma50": float("nan"),
            "days_to_earnings": float("nan"),
            "realized_vol_20d": float("nan"),
        })
        assert dominant in STANCE_NAMES
        assert abs(sum(loadings.values()) - 1.0) < 1e-3

    def test_pandas_series_input_works(self):
        """Real-world callers pass a pandas Series (per-ticker feature
        row from the precomputed batch). Both Series and dict must work."""
        s = pd.Series({
            "momentum_20d": 0.10,
            "price_vs_ma50": 1.05,
            "debt_to_equity": 0.5,
            "eps_ttm": 2.0,
            "realized_vol_20d": 0.20,
        })
        dominant, loadings, _ = classify_stance(s)
        assert dominant in STANCE_NAMES
        assert abs(sum(loadings.values()) - 1.0) < 1e-3


class TestStanceVocabularyContract:
    """The StanceLiteral imported from alpha-engine-lib must keep the
    4-element closed set the predictor's heuristic was calibrated on."""

    def test_lib_vocabulary_matches_arc_taxonomy(self):
        import typing
        args = typing.get_args(StanceLiteral)
        assert args == ("momentum", "value", "quality", "catalyst")

    def test_stance_names_constant_matches_literal(self):
        import typing
        assert STANCE_NAMES == typing.get_args(StanceLiteral)

    def test_scale_constants_have_expected_signs(self):
        """MOMENTUM_SCALE / VALUE_SCALE / CATALYST_PEAK_DAYS — sanity
        check the cold-start parameters. Pinned so a sign-flip
        regression breaks the test before it ships wrong loadings."""
        assert MOMENTUM_SCALE > 0
        assert VALUE_SCALE > 0
        assert 0 < CATALYST_PEAK_DAYS < 30
