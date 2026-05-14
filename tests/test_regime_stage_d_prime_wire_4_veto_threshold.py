"""Stage D' Wire 4 — regime-conditional predictor veto threshold.

Pins:

1. ``regime_conditional_veto_adjustment`` math: ``z*scale`` clamped to
   ``[-cap, +cap]``; ``None`` → 0.0.
2. ``get_veto_threshold`` honors ``regime_veto_enabled`` flag in S3
   predictor_params; default OFF preserves the discrete-regime path.
3. When Wire 4 is ON, the continuous adjustment REPLACES the discrete
   one (both encode the same signal — don't double-count).
4. Direction matches the discrete table: risk-off (z<0) → LOWER
   threshold (more aggressive veto); risk-on (z>0) → HIGHER threshold
   (more permissive).
5. ``PipelineContext`` carries ``regime_intensity_z`` so ``run_inference``
   can stamp it for ``write_output``'s veto path.

See ``~/Development/alpha-engine-docs/private/regime-v3-260514.md``
§6 Stage D' Wire 4 for the architectural framing.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from inference.stages.write_output import (
    get_veto_threshold,
    regime_conditional_veto_adjustment,
)


# ─────────────────────────────────────────────────────────────────────
# regime_conditional_veto_adjustment — pure math
# ─────────────────────────────────────────────────────────────────────


class TestRegimeConditionalVetoAdjustment:
    def test_none_returns_zero(self):
        """Substrate-unavailable path: None → 0.0 (no adjustment)."""
        assert regime_conditional_veto_adjustment(None) == 0.0

    def test_zero_intensity_returns_zero(self):
        """Neutral regime (z=0) → 0.0 (no adjustment)."""
        assert regime_conditional_veto_adjustment(0.0) == 0.0

    def test_positive_z_returns_positive_adjustment(self):
        """Risk-on (z>0) → POSITIVE adjustment (raises threshold)."""
        assert regime_conditional_veto_adjustment(1.0) == pytest.approx(0.10)

    def test_negative_z_returns_negative_adjustment(self):
        """Risk-off (z<0) → NEGATIVE adjustment (lowers threshold)."""
        assert regime_conditional_veto_adjustment(-1.0) == pytest.approx(-0.10)

    def test_clamped_to_positive_cap(self):
        """Extreme risk-on → clamped to +cap (default 0.20)."""
        assert regime_conditional_veto_adjustment(10.0) == 0.20

    def test_clamped_to_negative_cap(self):
        """Extreme risk-off → clamped to -cap (default -0.20)."""
        assert regime_conditional_veto_adjustment(-10.0) == -0.20

    def test_z_minus_2_matches_discrete_bear_magnitude(self):
        """At z=-2σ, continuous adjustment (-0.20) matches the discrete
        ``bear`` adjustment exactly — by design so cutover doesn't
        produce a structural break in veto rates."""
        assert regime_conditional_veto_adjustment(-2.0) == pytest.approx(-0.20)

    def test_custom_scale(self):
        """``scale`` parameter controls sensitivity per σ."""
        assert regime_conditional_veto_adjustment(1.0, scale=0.20) == pytest.approx(0.20)
        assert regime_conditional_veto_adjustment(-1.0, scale=0.20) == pytest.approx(-0.20)

    def test_custom_cap(self):
        """``cap`` parameter tightens the clamp."""
        assert regime_conditional_veto_adjustment(5.0, cap=0.05) == 0.05
        assert regime_conditional_veto_adjustment(-5.0, cap=0.05) == -0.05


# ─────────────────────────────────────────────────────────────────────
# get_veto_threshold — Wire 4 integration
# ─────────────────────────────────────────────────────────────────────


class TestGetVetoThresholdWire4Disabled:
    """``regime_veto_enabled=False`` (or absent) → legacy discrete path.
    Even if ``regime_intensity_z`` is passed, it's ignored."""

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_no_wire4_flag_in_params_uses_discrete(self, mock_load):
        """Params without ``regime_veto_enabled`` → discrete path."""
        mock_load.return_value = {"veto_confidence": 0.30}
        # Pass intensity_z but expect discrete bear path
        result = get_veto_threshold("test-bucket", "bear", regime_intensity_z=-2.0)
        assert result == pytest.approx(0.10)  # 0.30 - 0.20 (discrete bear)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_wire4_explicitly_false_uses_discrete(self, mock_load):
        mock_load.return_value = {
            "veto_confidence": 0.30,
            "regime_veto_enabled": False,
        }
        result = get_veto_threshold("test-bucket", "bull", regime_intensity_z=2.0)
        assert result == pytest.approx(0.40)  # 0.30 + 0.10 (discrete bull)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_legacy_callers_without_intensity_z_unchanged(self, mock_load):
        """Existing callers passing only (bucket, regime) still work."""
        mock_load.return_value = {"veto_confidence": 0.30}
        result = get_veto_threshold("test-bucket", "neutral")
        assert result == pytest.approx(0.30)


class TestGetVetoThresholdWire4Enabled:
    """``regime_veto_enabled=True`` → continuous path replaces discrete."""

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_continuous_replaces_discrete_when_enabled(self, mock_load):
        """Even when ``market_regime=bear``, the continuous adjustment
        from intensity_z takes precedence — they encode the same signal."""
        mock_load.return_value = {
            "veto_confidence": 0.30,
            "regime_veto_enabled": True,
        }
        # z=0 should produce NO adjustment (continuous neutral)
        # even though market_regime="bear" (discrete bear = -0.20)
        result = get_veto_threshold("test-bucket", "bear", regime_intensity_z=0.0)
        assert result == pytest.approx(0.30)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_continuous_risk_off_lowers_threshold(self, mock_load):
        mock_load.return_value = {
            "veto_confidence": 0.30,
            "regime_veto_enabled": True,
        }
        # z=-1: 0.30 + (-0.10) = 0.20
        result = get_veto_threshold("test-bucket", "", regime_intensity_z=-1.0)
        assert result == pytest.approx(0.20)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_continuous_risk_on_raises_threshold(self, mock_load):
        mock_load.return_value = {
            "veto_confidence": 0.30,
            "regime_veto_enabled": True,
        }
        # z=+1: 0.30 + 0.10 = 0.40
        result = get_veto_threshold("test-bucket", "", regime_intensity_z=1.0)
        assert result == pytest.approx(0.40)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_continuous_clamped_to_floor(self, mock_load):
        """Result threshold clamped to ≥0.0 even after large negative adjustment."""
        mock_load.return_value = {
            "veto_confidence": 0.10,
            "regime_veto_enabled": True,
        }
        # z=-3 → adjustment -0.20 (cap); 0.10 - 0.20 = -0.10 → clamped to 0.0
        result = get_veto_threshold("test-bucket", "", regime_intensity_z=-3.0)
        assert result == 0.0

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_continuous_clamped_to_ceiling(self, mock_load):
        mock_load.return_value = {
            "veto_confidence": 0.75,
            "regime_veto_enabled": True,
        }
        # z=+3 → adjustment +0.20 (cap); 0.75 + 0.20 = 0.95 → clamped to 0.80
        result = get_veto_threshold("test-bucket", "", regime_intensity_z=3.0)
        assert result == 0.80

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_intensity_z_none_falls_back_to_discrete(self, mock_load):
        """Wire 4 enabled but intensity_z=None → falls back to discrete.
        Preserves behavior when the regime substrate panel couldn't be
        built (cold start, missing data, etc.)."""
        mock_load.return_value = {
            "veto_confidence": 0.30,
            "regime_veto_enabled": True,
        }
        result = get_veto_threshold("test-bucket", "bear", regime_intensity_z=None)
        # Falls back to discrete bear adjustment
        assert result == pytest.approx(0.10)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_custom_scale_from_params(self, mock_load):
        mock_load.return_value = {
            "veto_confidence": 0.30,
            "regime_veto_enabled": True,
            "regime_veto_scale": 0.20,  # 2× default sensitivity
        }
        # z=+1 with scale 0.20: 0.30 + 0.20 = 0.50
        result = get_veto_threshold("test-bucket", "", regime_intensity_z=1.0)
        assert result == pytest.approx(0.50)

    @patch("inference.stages.write_output._load_predictor_params_from_s3")
    def test_custom_cap_from_params(self, mock_load):
        mock_load.return_value = {
            "veto_confidence": 0.30,
            "regime_veto_enabled": True,
            "regime_veto_cap": 0.05,  # tighter cap
        }
        # z=+5: raw=0.50, cap=0.05 → adj=0.05; threshold=0.35
        result = get_veto_threshold("test-bucket", "", regime_intensity_z=5.0)
        assert result == pytest.approx(0.35)


# ─────────────────────────────────────────────────────────────────────
# PipelineContext schema pin
# ─────────────────────────────────────────────────────────────────────


class TestPipelineContextCarriesIntensityZ:
    """run_inference stamps intensity_z on ctx so write_output can use it."""

    def test_pipeline_context_has_regime_intensity_z_field(self):
        from inference.pipeline import PipelineContext

        ctx = PipelineContext()
        # Default None — preserves legacy callers that don't set it
        assert ctx.regime_intensity_z is None

    def test_pipeline_context_accepts_float(self):
        from inference.pipeline import PipelineContext

        ctx = PipelineContext()
        ctx.regime_intensity_z = 1.23
        assert ctx.regime_intensity_z == 1.23
