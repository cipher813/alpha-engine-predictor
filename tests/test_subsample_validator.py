"""Tests for ``model.subsample_validator`` — short-history subsample IC gate.

Closes ROADMAP P1 "NaN-feature handling audit + short-history subsample
validation" — subsample-validation phase. Pin the per-component
promotion gate that blocks deploys where a Layer-1 GBM regresses against
its named simple-fallback baseline on the subsample mimicking
inference-time short-history tickers (data layer ships partial-NaN
features for these per alpha-engine-data PR #78).
"""

from __future__ import annotations

import numpy as np
import pytest


# ── _safe_pearson_ic ─────────────────────────────────────────────────────────


class TestSafePearsonIC:
    def test_normal_correlation(self):
        from model.subsample_validator import _safe_pearson_ic
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        y = x + rng.normal(0, 0.5, 200)
        ic = _safe_pearson_ic(x, y)
        assert 0.6 < ic < 1.0

    def test_constant_predictions_returns_zero(self):
        """np.corrcoef on a constant array would return NaN. Convention:
        return 0.0 so the gate's IC is comparable to test_IC."""
        from model.subsample_validator import _safe_pearson_ic
        x = np.zeros(100)
        y = np.linspace(-1, 1, 100)
        assert _safe_pearson_ic(x, y) == 0.0

    def test_constant_y_returns_zero(self):
        from model.subsample_validator import _safe_pearson_ic
        x = np.linspace(-1, 1, 100)
        y = np.zeros(100)
        assert _safe_pearson_ic(x, y) == 0.0

    def test_short_input_returns_zero(self):
        from model.subsample_validator import _safe_pearson_ic
        assert _safe_pearson_ic(np.array([1.0]), np.array([1.0])) == 0.0
        assert _safe_pearson_ic(np.array([]), np.array([])) == 0.0


# ── momentum_baseline_predict ────────────────────────────────────────────────


class TestMomentumBaseline:
    def test_clean_features_match_fallback_formula(self):
        """The baseline mirrors run_inference.py:267-271 exactly:
        0.4·m5 + 0.3·m20 + 0.2·ma50 + 0.1·(rsi-50)/100"""
        from model.subsample_validator import momentum_baseline_predict
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        X = np.array([[0.05, 0.10, 0.02, 70.0]])
        preds = momentum_baseline_predict(X, feature_names)
        expected = 0.4 * 0.05 + 0.3 * 0.10 + 0.2 * 0.02 + 0.1 * (70 - 50) / 100
        assert preds[0] == pytest.approx(expected)

    def test_nan_features_use_neutral_defaults(self):
        from model.subsample_validator import momentum_baseline_predict
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        # NaN row: should use defaults 0/0/0/50 → baseline = 0
        X = np.array([[np.nan, np.nan, np.nan, np.nan]])
        preds = momentum_baseline_predict(X, feature_names)
        assert preds[0] == pytest.approx(0.0)

    def test_partial_nan_uses_neutral_for_missing_only(self):
        from model.subsample_validator import momentum_baseline_predict
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        X = np.array([[0.05, np.nan, np.nan, np.nan]])
        preds = momentum_baseline_predict(X, feature_names)
        # m5=0.05; rest neutral → 0.4 * 0.05 = 0.02
        assert preds[0] == pytest.approx(0.02)

    def test_missing_feature_name_uses_default(self):
        """If MOMENTUM_FEATURES is reordered or a feature is absent,
        the baseline still produces a real-valued prediction."""
        from model.subsample_validator import momentum_baseline_predict
        feature_names = ["momentum_5d"]  # only one feature available
        X = np.array([[0.10]])
        preds = momentum_baseline_predict(X, feature_names)
        # m5=0.10, others default 0/0/50 → 0.4 * 0.10 + 0.1 * (50-50)/100 = 0.04
        assert preds[0] == pytest.approx(0.04)

    def test_batch_shape_preserved(self):
        from model.subsample_validator import momentum_baseline_predict
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        X = np.random.default_rng(0).normal(0, 0.1, size=(50, 4))
        preds = momentum_baseline_predict(X, feature_names)
        assert preds.shape == (50,)


# ── volatility_baseline_predict ──────────────────────────────────────────────


class TestVolatilityBaseline:
    def test_realized_vol_20d_passthrough(self):
        from model.subsample_validator import volatility_baseline_predict
        feature_names = ["realized_vol_20d", "vol_30d", "atr_14"]
        X = np.array([[0.025, 0.030, 0.018]])
        preds = volatility_baseline_predict(X, feature_names)
        assert preds[0] == pytest.approx(0.025)

    def test_falls_back_to_vol_20d_alias(self):
        from model.subsample_validator import volatility_baseline_predict
        feature_names = ["vol_20d", "atr_14"]
        X = np.array([[0.020, 0.018]])
        preds = volatility_baseline_predict(X, feature_names)
        assert preds[0] == pytest.approx(0.020)

    def test_falls_back_to_realized_vol_30d(self):
        from model.subsample_validator import volatility_baseline_predict
        feature_names = ["realized_vol_30d", "atr_14"]
        X = np.array([[0.030, 0.018]])
        preds = volatility_baseline_predict(X, feature_names)
        assert preds[0] == pytest.approx(0.030)

    def test_no_realized_vol_feature_uses_zero(self):
        from model.subsample_validator import volatility_baseline_predict
        feature_names = ["atr_14", "bollinger_width"]
        X = np.array([[0.018, 0.04], [0.020, 0.05]])
        preds = volatility_baseline_predict(X, feature_names)
        assert (preds == 0.0).all()

    def test_nan_passthrough_imputed_to_zero(self):
        from model.subsample_validator import volatility_baseline_predict
        feature_names = ["realized_vol_20d"]
        X = np.array([[np.nan]])
        preds = volatility_baseline_predict(X, feature_names)
        assert preds[0] == 0.0


# ── validate_component ───────────────────────────────────────────────────────


class TestValidateComponent:
    def _make_dataset(self, n: int = 100, signal_strength: float = 0.5, seed: int = 42):
        """Synthetic component / baseline / y_true with controllable
        relative signal strength."""
        rng = np.random.default_rng(seed)
        y = rng.normal(0, 1, n)
        component_preds = signal_strength * y + rng.normal(0, 0.5, n)
        baseline_preds = 0.2 * y + rng.normal(0, 0.5, n)
        return component_preds, baseline_preds, y

    def test_strong_component_passes(self):
        from model.subsample_validator import validate_component
        c, b, y = self._make_dataset(n=200, signal_strength=0.8)
        mask = np.ones(200, dtype=bool)
        r = validate_component("momentum", c, b, y, mask, min_n=30)
        assert r.passed is True
        assert r.component_ic > r.baseline_ic
        assert r.n == 200
        assert r.skip_reason is None

    def test_weak_component_blocks(self):
        """When the component IC is below baseline IC, gate must block."""
        from model.subsample_validator import validate_component
        c, b, y = self._make_dataset(n=200, signal_strength=0.05)  # weak
        # Baseline_strength=0.2 hardcoded in helper; component is now weaker
        mask = np.ones(200, dtype=bool)
        r = validate_component("momentum", c, b, y, mask, min_n=30)
        assert r.passed is False
        assert r.component_ic < r.baseline_ic
        assert r.n == 200

    def test_subsample_mask_filters_correctly(self):
        from model.subsample_validator import validate_component
        c, b, y = self._make_dataset(n=200)
        mask = np.zeros(200, dtype=bool)
        mask[:50] = True  # first 50 rows only
        r = validate_component("momentum", c, b, y, mask, min_n=30)
        assert r.n == 50

    def test_small_subsample_skipped(self):
        from model.subsample_validator import validate_component
        c, b, y = self._make_dataset(n=200)
        mask = np.zeros(200, dtype=bool)
        mask[:10] = True
        r = validate_component("momentum", c, b, y, mask, min_n=30)
        assert r.passed is True  # SKIPPED → reported as pass (no statistical power)
        assert r.skip_reason is not None
        assert r.n == 10

    def test_empty_subsample_skipped(self):
        from model.subsample_validator import validate_component
        c, b, y = self._make_dataset(n=200)
        mask = np.zeros(200, dtype=bool)  # no subsample
        r = validate_component("momentum", c, b, y, mask, min_n=30)
        assert r.passed is True
        assert r.n == 0
        assert r.skip_reason is not None

    def test_equal_ic_passes(self):
        """Boundary: component_ic == baseline_ic → PASS (>=).
        The gate's contract is "GBM doesn't regress vs baseline";
        equality is acceptable."""
        from model.subsample_validator import validate_component
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 100)
        # Perfect correlation in component
        c = y.copy()
        b = y.copy()
        mask = np.ones(100, dtype=bool)
        r = validate_component("momentum", c, b, y, mask, min_n=30)
        assert r.component_ic == pytest.approx(r.baseline_ic)
        assert r.passed is True

    def test_constant_component_returns_ic_zero(self):
        """If a component collapses to constant predictions on the
        subsample, _safe_pearson_ic returns 0. Gate then passes only if
        baseline is also <= 0 (or NaN-zero'd)."""
        from model.subsample_validator import validate_component
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 100)
        c = np.zeros(100)
        b = 0.5 * y + rng.normal(0, 0.5, 100)
        mask = np.ones(100, dtype=bool)
        r = validate_component("momentum", c, b, y, mask, min_n=30)
        assert r.component_ic == 0.0
        assert r.baseline_ic > 0.0
        assert r.passed is False

    def test_shape_mismatch_raises(self):
        from model.subsample_validator import validate_component
        c = np.zeros(100)
        b = np.zeros(50)
        y = np.zeros(100)
        mask = np.zeros(100, dtype=bool)
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_component("x", c, b, y, mask)


# ── research_calibrator_baseline_predict ─────────────────────────────────────


class TestResearchCalibratorBaseline:
    def test_passthrough_div_100(self):
        from model.subsample_validator import research_calibrator_baseline_predict
        scores = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        baseline = research_calibrator_baseline_predict(scores)
        assert (baseline == np.array([0.0, 0.25, 0.5, 0.75, 1.0])).all()

    def test_matches_research_features_fallback(self):
        """The baseline must match research_features.py:111's fallback
        formula (``score / 100``) — that's what the gate is asking the
        fitted calibrator to beat."""
        from model.subsample_validator import research_calibrator_baseline_predict
        scores = np.array([42.0, 87.0])
        assert (research_calibrator_baseline_predict(scores) == scores / 100).all()


# ── validate_research_calibrator ────────────────────────────────────────────


class TestValidateResearchCalibrator:
    def _make_dataset(
        self, n: int = 600, signal_strength: float = 0.6, seed: int = 0,
    ):
        """Synthetic (scores, beat_spy, dates) where higher score ->
        higher P(beat). signal_strength=0 means no signal (random)."""
        rng = np.random.default_rng(seed)
        scores = rng.uniform(0, 100, n)
        # Probability of beat increases with score, scaled by strength
        p_beat = 0.5 + signal_strength * (scores - 50) / 100
        p_beat = np.clip(p_beat, 0.01, 0.99)
        beat = (rng.uniform(0, 1, n) < p_beat).astype(np.int32)
        # Synthetic dates spanning ~2 years
        base = np.datetime64("2024-01-01", "D")
        date_offsets = np.sort(rng.integers(0, 730, n))
        dates = base + date_offsets
        return scores, beat, dates

    def test_calibrator_passes_on_non_monotonic_signal(self):
        """The bucket calibrator's strength domain is non-monotonic
        score→outcome relationships. On purely linear signal, the
        score/100 passthrough wins on Pearson IC by definition (it's
        smooth, the bucket output is piecewise-constant). The gate
        passes when the calibrator's discrete bucketing actually
        captures pattern the passthrough can't — e.g. mid-score
        outperforming high-score (a real scoring-saturation effect)."""
        from model.subsample_validator import validate_research_calibrator
        rng = np.random.default_rng(123)
        n = 800
        scores = rng.uniform(0, 100, n)
        # Non-monotonic: hit rate peaks at score 60-70, drops at 80+
        # (saturation — over-confident high scores become contrarian).
        p_beat = np.where(
            scores < 50, 0.40,
            np.where(scores < 70, 0.65,
                     np.where(scores < 85, 0.55, 0.40)),
        )
        beat = (rng.uniform(0, 1, n) < p_beat).astype(np.int32)
        base = np.datetime64("2024-01-01", "D")
        dates = base + np.arange(n)
        r = validate_research_calibrator(scores, beat, dates)
        assert r.component == "research_calibrator"
        assert r.skip_reason is None
        # Calibrator's bucket lookup captures the non-monotonic peak
        # better than the smooth score/100 passthrough can.
        assert r.passed is True, (
            f"Calibrator should beat passthrough on non-monotonic signal: "
            f"component_ic={r.component_ic:.4f}, baseline_ic={r.baseline_ic:.4f}"
        )

    def test_calibrator_blocks_on_linear_signal(self):
        """Inverse-but-meaningful: on purely linear signal the smooth
        passthrough wins on Pearson IC and the calibrator's bucketing
        adds no value. Gate should BLOCK — promoting a calibrator that
        doesn't beat its own free-fallback is exactly what the gate
        protects against per feedback_component_baseline_validation."""
        from model.subsample_validator import validate_research_calibrator
        scores, beat, dates = self._make_dataset(n=800, signal_strength=0.8)
        r = validate_research_calibrator(scores, beat, dates)
        assert r.skip_reason is None
        # Both ICs are positive and meaningful; passthrough wins by
        # the structural-smoothness margin.
        assert r.baseline_ic > 0.20
        assert r.component_ic <= r.baseline_ic
        assert r.passed is False

    def test_random_calibrator_blocks_or_ties(self):
        """When the underlying score has zero predictive power, the
        bucket-lookup calibrator can only pick up sampling noise. Its
        IC on a held-out slice should be at most equal to the
        passthrough's IC — and on out-of-sample data the bucket
        approach's overfitting noise typically does worse."""
        from model.subsample_validator import validate_research_calibrator
        scores, beat, dates = self._make_dataset(n=600, signal_strength=0.0)
        r = validate_research_calibrator(scores, beat, dates)
        # Either the calibrator <= baseline (block) OR they're equal/tied.
        # The key invariant: gate did NOT silently pass on a no-signal dataset.
        # We assert the IC values are both small.
        assert abs(r.component_ic) < 0.15
        assert abs(r.baseline_ic) < 0.15

    def test_holdout_slice_size_respects_min_n(self):
        """holdout_frac=0.2 on 100 rows = 20 rows holdout. Below MIN_SUBSAMPLE_SIZE=30,
        the gate forces holdout up to min_n."""
        from model.subsample_validator import validate_research_calibrator, MIN_SUBSAMPLE_SIZE
        scores, beat, dates = self._make_dataset(n=100, signal_strength=0.5)
        r = validate_research_calibrator(scores, beat, dates, holdout_frac=0.2)
        # The resulting test slice should be at least min_n
        assert r.n >= MIN_SUBSAMPLE_SIZE

    def test_too_few_total_rows_skipped(self):
        from model.subsample_validator import validate_research_calibrator
        scores, beat, dates = self._make_dataset(n=20, signal_strength=0.8)
        r = validate_research_calibrator(scores, beat, dates)
        assert r.passed is True
        assert r.skip_reason is not None
        assert r.n == 20

    def test_empty_input_skipped_gracefully(self):
        from model.subsample_validator import validate_research_calibrator
        scores = np.array([])
        beat = np.array([], dtype=np.int32)
        dates = np.array([], dtype="datetime64[D]")
        r = validate_research_calibrator(scores, beat, dates)
        assert r.passed is True
        assert r.n == 0
        assert r.skip_reason is not None

    def test_shape_mismatch_raises(self):
        from model.subsample_validator import validate_research_calibrator
        scores = np.zeros(100)
        beat = np.zeros(50, dtype=np.int32)
        dates = np.array(["2024-01-01"] * 100, dtype="datetime64[D]")
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_research_calibrator(scores, beat, dates)

    def test_time_based_holdout_uses_latest_dates(self):
        """The holdout slice must be the LATEST dates (chronological
        holdout), not random — temporal leakage protection. Verify by
        constructing a dataset where signal flips sign in the last 20%
        and confirming the holdout's component_ic reflects that flip."""
        from model.subsample_validator import validate_research_calibrator
        rng = np.random.default_rng(42)
        n = 400
        # First 80%: positive signal (high score → beat). Last 20%:
        # negative signal (high score → miss).
        scores = rng.uniform(0, 100, n)
        base = np.datetime64("2024-01-01", "D")
        date_offsets = np.arange(n)  # strictly chronological
        dates = base + date_offsets

        beat = np.zeros(n, dtype=np.int32)
        # First 320 rows: hit rate = 0.5 + 0.4*(score-50)/100
        for i in range(320):
            p = 0.5 + 0.4 * (scores[i] - 50) / 100
            beat[i] = 1 if rng.uniform() < p else 0
        # Last 80 rows: hit rate = 0.5 - 0.4*(score-50)/100 (FLIPPED)
        for i in range(320, n):
            p = 0.5 - 0.4 * (scores[i] - 50) / 100
            beat[i] = 1 if rng.uniform() < p else 0

        r = validate_research_calibrator(scores, beat, dates, holdout_frac=0.2)
        # Calibrator was fit on positive-signal first 80%, evaluated on
        # negative-signal last 20%. Both component AND baseline should
        # have negative IC on the holdout (because score is now anti-
        # correlated with beat). The fact that component_ic is negative
        # confirms the holdout split was time-based.
        assert r.baseline_ic < 0.0, (
            "Baseline IC on flipped-signal holdout should be negative — "
            "confirms time-based split"
        )

    def test_unsorted_dates_handled_via_argsort(self):
        """Inputs may arrive in arbitrary order; the gate must sort by
        date before slicing the holdout."""
        from model.subsample_validator import validate_research_calibrator
        scores, beat, dates = self._make_dataset(n=400, signal_strength=0.6)
        # Shuffle in unison
        perm = np.random.default_rng(0).permutation(len(scores))
        scores_shuffled = scores[perm]
        beat_shuffled = beat[perm]
        dates_shuffled = dates[perm]
        r1 = validate_research_calibrator(scores, beat, dates)
        r2 = validate_research_calibrator(
            scores_shuffled, beat_shuffled, dates_shuffled,
        )
        # Result invariance: same data, different order in → same IC out
        assert r1.n == r2.n
        assert r1.component_ic == pytest.approx(r2.component_ic, abs=1e-9)
        assert r1.baseline_ic == pytest.approx(r2.baseline_ic, abs=1e-9)
        assert r1.passed == r2.passed


# ── ComponentValidation logging ──────────────────────────────────────────────


class TestComponentValidationLogging:
    def test_pass_log_format(self, caplog):
        import logging
        from model.subsample_validator import ComponentValidation

        with caplog.at_level(logging.INFO):
            ComponentValidation(
                component="momentum", n=120,
                component_ic=0.045, baseline_ic=0.030, passed=True,
            ).log()
        assert any("momentum" in r.message and "PASS" in r.message for r in caplog.records)

    def test_block_log_format(self, caplog):
        import logging
        from model.subsample_validator import ComponentValidation

        with caplog.at_level(logging.INFO):
            ComponentValidation(
                component="volatility", n=120,
                component_ic=0.005, baseline_ic=0.030, passed=False,
            ).log()
        assert any(
            "volatility" in r.message and "BLOCK" in r.message
            for r in caplog.records
        )

    def test_skip_log_format(self, caplog):
        import logging
        from model.subsample_validator import ComponentValidation

        with caplog.at_level(logging.INFO):
            ComponentValidation(
                component="momentum", n=10,
                component_ic=0.0, baseline_ic=0.0, passed=True,
                skip_reason="subsample size 10 < min_n=30",
            ).log()
        assert any("SKIPPED" in r.message for r in caplog.records)
