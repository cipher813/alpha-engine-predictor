"""
test_regime_drawdown_severity_axis.py

Pin the v0.42.0 Phase 2A type-system separation
(caution-regime-retirement-260528.md): drawdown leg's protective state
is an axis ORTHOGONAL to macro market_regime, exposed via the
``drawdown_protective_severity`` ordinal (0=risk_on, 1=caution,
2=risk_off/alpha_bleed) instead of aliasing into the 4-class macro
regime vocabulary.

Covers:
  - _PROTECTION_ORDER macro axis is 3-class (no caution)
  - DRAWDOWN_PROTECTIVE_SEVERITY axis is 0/1/2 ordinal
  - spy_tier_to_regime: caution → None (no macro escalation);
    risk_off → "bear" (still escalates macro because it's the
    drawdown leg's most-protective state)
  - spy_tier_to_protective_severity: risk_on=0, caution=1,
    risk_off=2, unknown→0, None→0
  - compose_effective_regime emits drawdown_tier +
    drawdown_protective_severity + the legacy effective_regime field
    for backward-compat
  - write_output.get_veto_threshold reads severity ordinal; legacy
    drawdown_effective_regime string still works via grandfather
    derivation
  - substrate.active_severity_floor only emits "bear" (caution
    floor retired — its driving signals flow continuously through
    regime_intensity_z)
"""

from __future__ import annotations

import pytest


# ── Macro axis 3-class ───────────────────────────────────────────────────


def test_macro_protection_order_is_3class():
    from regime.drawdown import _PROTECTION_ORDER

    assert set(_PROTECTION_ORDER.keys()) == {"bull", "neutral", "bear"}
    assert _PROTECTION_ORDER["bull"] < _PROTECTION_ORDER["neutral"]
    assert _PROTECTION_ORDER["neutral"] < _PROTECTION_ORDER["bear"]


# ── Drawdown axis ordinal ────────────────────────────────────────────────


def test_drawdown_protective_severity_axis_ordering():
    from regime.drawdown import DRAWDOWN_PROTECTIVE_SEVERITY

    assert DRAWDOWN_PROTECTIVE_SEVERITY["risk_on"] == 0
    assert DRAWDOWN_PROTECTIVE_SEVERITY["caution"] == 1
    assert DRAWDOWN_PROTECTIVE_SEVERITY["risk_off"] == 2
    assert DRAWDOWN_PROTECTIVE_SEVERITY["alpha_bleed"] == 2


def test_spy_tier_to_protective_severity():
    from regime.drawdown import spy_tier_to_protective_severity

    assert spy_tier_to_protective_severity("risk_on") == 0
    assert spy_tier_to_protective_severity("caution") == 1
    assert spy_tier_to_protective_severity("risk_off") == 2
    assert spy_tier_to_protective_severity(None) == 0
    assert spy_tier_to_protective_severity("unknown_tier") == 0


def test_excess_tier_to_protective_severity():
    from regime.drawdown import excess_tier_to_protective_severity

    assert excess_tier_to_protective_severity("risk_on") == 0
    assert excess_tier_to_protective_severity("alpha_bleed") == 2
    assert excess_tier_to_protective_severity(None) == 0


# ── Macro projection (spy_tier_to_regime) post-decoupling ────────────────


def test_spy_tier_to_regime_caution_no_longer_escalates_macro():
    """Drawdown caution sits on the orthogonal drawdown axis post-v0.42.0;
    it no longer aliases into the macro regime vocabulary (caution would
    coerce to "neutral" or get rejected by the 3-class enum). The
    protective semantic is preserved via the severity ordinal."""
    from regime.drawdown import spy_tier_to_regime

    assert spy_tier_to_regime("caution") is None
    assert spy_tier_to_regime("risk_on") is None
    # risk_off STILL escalates the macro axis — its protective tier IS
    # macro bear, and the bear vocabulary still exists post-retirement.
    assert spy_tier_to_regime("risk_off") == "bear"


def test_excess_tier_to_regime_unchanged():
    """Excess alpha_bleed still escalates macro to bear directly."""
    from regime.drawdown import excess_tier_to_regime

    assert excess_tier_to_regime("alpha_bleed") == "bear"
    assert excess_tier_to_regime("risk_on") is None


# ── compose_effective_regime emits both axes ─────────────────────────────


class TestComposeEmitsBothAxes:
    """The composed payload carries the macro projection
    (effective_regime + drivers, for backward compat) AND the drawdown
    axis (drawdown_tier + drawdown_protective_severity, the canonical
    SOTA input for veto / sizing consumers)."""

    def test_caution_tier_macro_neutral_severity_one(self):
        from regime.drawdown import compose_effective_regime

        out = compose_effective_regime(spy_tier="caution")
        # Macro axis: caution no longer escalates (drivers all None
        # → all-None defaults to "neutral").
        assert out["effective_regime"] == "neutral"
        # Drawdown axis: severity 1 (caution).
        assert out["drawdown_tier"] == "caution"
        assert out["drawdown_protective_severity"] == 1

    def test_risk_off_tier_macro_bear_severity_two(self):
        from regime.drawdown import compose_effective_regime

        out = compose_effective_regime(spy_tier="risk_off")
        assert out["effective_regime"] == "bear"
        assert out["drawdown_tier"] == "risk_off"
        assert out["drawdown_protective_severity"] == 2

    def test_excess_alpha_bleed_overrides_severity(self):
        from regime.drawdown import compose_effective_regime

        # spy_tier=caution (sev 1) + excess_tier=alpha_bleed (sev 2).
        # Composed severity = max = 2.
        out = compose_effective_regime(spy_tier="caution", excess_tier="alpha_bleed")
        assert out["drawdown_protective_severity"] == 2
        # Macro axis: drawdown_excess escalates to bear.
        assert out["effective_regime"] == "bear"

    def test_no_drawdown_input_severity_zero(self):
        from regime.drawdown import compose_effective_regime

        out = compose_effective_regime(hmm_argmax="neutral")
        assert out["drawdown_protective_severity"] == 0
        assert out["drawdown_tier"] is None
        assert out["effective_regime"] == "neutral"

    def test_forced_bear_field_preserved(self):
        from regime.drawdown import compose_effective_regime

        out = compose_effective_regime(forced_bear=True)
        assert out["forced_bear"] is True
        assert out["effective_regime"] == "bear"
        # forced_bear is macro-axis only — drawdown axis stays at 0.
        assert out["drawdown_protective_severity"] == 0


# ── write_output veto threshold reads severity ordinal ───────────────────


class TestVetoThresholdSeverityOrdinal:
    """Post-v0.42.0 the veto threshold reads drawdown_protective_severity
    directly. Legacy drawdown_effective_regime string still works as a
    grandfather derivation path.

    Tests pin base=0.30 + cap=0.20 explicitly in the fake params so the
    expected floor values are deterministic across environments —
    cfg.MIN_CONFIDENCE drifts between local + CI."""

    # Pinned for arithmetic determinism. Floors derived as:
    #   severity 0 → base = 0.30 (no clamp)
    #   severity 1 → base - cap/2 = 0.30 - 0.10 = 0.20 (caution floor)
    #   severity 2 → base - cap   = 0.30 - 0.20 = 0.10 (bear floor)
    _PINNED_BASE = 0.30
    _PINNED_CAP = 0.20

    @pytest.fixture
    def s3_mock(self, monkeypatch):
        # Patch the predictor_params loader to return drawdown_regime_enabled=True
        # so the severity clamp actually fires (not OBSERVE-mode). Pin
        # veto_confidence + regime_veto_cap so the floor values are
        # deterministic regardless of cfg.MIN_CONFIDENCE drift.
        from inference.stages import write_output

        def _fake_load(_bucket):
            return {
                "drawdown_regime_enabled": True,
                "veto_confidence": self._PINNED_BASE,
                "regime_veto_cap": self._PINNED_CAP,
            }

        monkeypatch.setattr(write_output, "_load_predictor_params_from_s3", _fake_load)
        return "test-bucket"

    def test_severity_2_clamps_to_bear_floor(self, s3_mock):
        from inference.stages.write_output import get_veto_threshold

        t = get_veto_threshold(s3_mock, drawdown_protective_severity=2)
        assert t == pytest.approx(0.10, abs=0.001)

    def test_severity_1_clamps_to_caution_floor(self, s3_mock):
        from inference.stages.write_output import get_veto_threshold

        t = get_veto_threshold(s3_mock, drawdown_protective_severity=1)
        assert t == pytest.approx(0.20, abs=0.001)

    def test_severity_0_no_clamp(self, s3_mock):
        from inference.stages.write_output import get_veto_threshold

        t = get_veto_threshold(s3_mock, drawdown_protective_severity=0)
        assert t == pytest.approx(0.30, abs=0.001)

    def test_legacy_string_field_grandfather_derives_severity(self, s3_mock):
        """drawdown_effective_regime="bear" or "caution" still works
        — grandfather derives severity from the string."""
        from inference.stages.write_output import get_veto_threshold

        t_bear = get_veto_threshold(s3_mock, drawdown_effective_regime="bear")
        assert t_bear == pytest.approx(0.10, abs=0.001)
        t_caution = get_veto_threshold(s3_mock, drawdown_effective_regime="caution")
        assert t_caution == pytest.approx(0.20, abs=0.001)

    def test_severity_ordinal_wins_over_legacy_string(self, s3_mock):
        """When both are supplied, the ordinal is canonical."""
        from inference.stages.write_output import get_veto_threshold

        t = get_veto_threshold(
            s3_mock,
            drawdown_effective_regime="bear",       # would imply severity 2
            drawdown_protective_severity=1,         # ordinal wins → caution floor
        )
        assert t == pytest.approx(0.20, abs=0.001)


# ── Substrate active_severity_floor retires caution hint ─────────────────


class TestSubstrateActiveSeverityFloor:
    """The substrate's ``active_severity_floor`` field is a hint mirroring
    the macro-agent regime guardrail thresholds. The macro-agent's
    soft-caution branches retired in Phase 1B; the substrate's caution
    hint follows. Only "bear" + None remain. The threshold-crossing
    boolean feature flags stay for observability."""

    def _build_guardrail_signals(self, **overrides):
        from regime.substrate import _compute_guardrail_flags, GUARDRAIL_DEFAULTS

        current = {
            "vix_level": 18.0,
            "hy_oas_bps": 350.0,
            "spy_30d_return": 0.01,
            "spy_20d_return": 0.01,
        }
        current.update(overrides)
        return _compute_guardrail_flags(current, GUARDRAIL_DEFAULTS)

    def test_bear_hint_still_fires(self):
        # VIX > vix_bear_threshold AND SPY 30d < spy_30d_bear_threshold
        flags = self._build_guardrail_signals(
            vix_level=35.0, spy_30d_return=-0.12,
        )
        assert flags["active_severity_floor"] == "bear"

    def test_caution_hint_no_longer_fires_on_vix_spy(self):
        # Pre-v0.42.0 this fired caution; now stays None.
        flags = self._build_guardrail_signals(
            vix_level=27.0, spy_30d_return=-0.07,
        )
        assert flags["active_severity_floor"] is None

    def test_caution_hint_no_longer_fires_on_hy_oas(self):
        # Pre-v0.42.0 this fired caution; now stays None.
        flags = self._build_guardrail_signals(
            hy_oas_bps=800.0,
        )
        assert flags["active_severity_floor"] is None

    def test_threshold_crossing_bools_preserved(self):
        # Observability feature flags stay — they're feature flags,
        # not regime classifications.
        flags = self._build_guardrail_signals(
            vix_level=27.0, spy_30d_return=-0.07, hy_oas_bps=600.0,
        )
        assert flags["vix_caution_breached"] is True
        assert flags["spy_30d_caution_breached"] is True
        assert flags["hy_oas_caution_breached"] is True
