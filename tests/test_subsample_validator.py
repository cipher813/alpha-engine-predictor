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
