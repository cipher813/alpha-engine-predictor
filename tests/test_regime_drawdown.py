"""Tests for regime/drawdown.py — the deterministic drawdown regime leg.

Covers:
- ``bear_stretches`` parity vs the pre-extraction inline algorithm
  (the single-source-of-truth invariant: production leg == eval ref)
- tiered SPY hysteresis truth table (enter/exit asymmetry, ladder
  de-escalation one rung at a time, pure-level immediacy)
- excess (book-vs-market) leg + NAV graceful-degrade
- step() idempotency + cold start
- state (de)serialization round-trip + schema-mismatch → ValueError
- most_protective + compose_effective_regime truth table
- read_eod_pnl_nav graceful-degrade matrix
- additive substrate hook (None ⇒ zero change; present ⇒ block + sidecar)
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from regime.drawdown import (
    DRAWDOWN_SCHEMA_VERSION,
    DrawdownState,
    DrawdownTunables,
    bear_stretches,
    compose_effective_regime,
    dump_state,
    load_state,
    most_protective,
    read_eod_pnl_nav,
    seed_state,
    step,
)


# ── bear_stretches parity (single source of truth) ───────────────────────

def _legacy_bear_stretches(spy: pd.Series, enter: float, exit: float):
    """The exact pre-extraction inline algorithm from
    scripts/backfill_regime_fast_signal.py::_bear_stretches. The shared
    implementation must reproduce this byte-for-byte or the F1
    calibration silently decouples from production (plan §3)."""
    peak = spy.cummax()
    dd = spy / peak - 1.0
    stretches = []
    in_bear = False
    start = None
    for ts, d in dd.items():
        if not in_bear and d <= -enter:
            in_bear, start = True, ts
        elif in_bear and d >= -exit:
            stretches.append((start, ts))
            in_bear = False
    if in_bear and start is not None:
        stretches.append((start, dd.index[-1]))
    return stretches


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_bear_stretches_matches_legacy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=900, freq="W")
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, 900))), index=idx)
    assert bear_stretches(spy, enter=0.10, exit=0.05) == _legacy_bear_stretches(
        spy, 0.10, 0.05
    )


def test_bear_stretches_unterminated_closes_at_last() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    spy = pd.Series([100, 100, 88, 87, 86], index=idx)  # never recovers
    out = bear_stretches(spy, enter=0.10, exit=0.05)
    assert len(out) == 1 and out[0][1] == idx[-1]


def test_bear_stretches_empty_series() -> None:
    assert bear_stretches(pd.Series(dtype=float)) == []


# ── tiered SPY hysteresis ────────────────────────────────────────────────

def _run_spy(prices, tun=None):
    st = None
    for i, p in enumerate(prices):
        st, art = step(
            st, spy_close=float(p), trading_day=f"d{i}",
            calendar_date=f"c{i}", run_id="r", tunables=tun,
        )
    return st, art


def test_spy_tier_enter_thresholds_pure_level() -> None:
    # Peak 100. -5% → caution, -10% → risk_off, default pure-level
    # (min_persist_days=1) so transitions are immediate.
    st, art = _run_spy([100, 97])           # -3% → still risk_on
    assert st.spy_tier == "risk_on"
    st, art = _run_spy([100, 94])           # -6% → caution
    assert st.spy_tier == "caution"
    st, art = _run_spy([100, 89])           # -11% → risk_off
    assert st.spy_tier == "risk_off"
    assert art["spy"]["regime_contribution"] == "bear"


def test_spy_asymmetric_exit_and_one_rung_deescalation() -> None:
    # Enter risk_off at -12%, then partial recovery must NOT jump
    # straight to risk_on — exit bands gate it one rung at a time.
    st, _ = _run_spy([100, 88])             # risk_off
    assert st.spy_tier == "risk_off"
    # recover to -8% (above risk_off_exit -7%? no, 8% deep > 7%) → stay
    st, _ = _run_spy([100, 88, 92])         # -8% depth → still risk_off
    assert st.spy_tier == "risk_off"
    # recover to -6% (≤7% risk_off_exit, but >3% caution_exit) → caution
    st, _ = _run_spy([100, 88, 94])
    assert st.spy_tier == "caution"
    # full recovery to -2% (≤3% caution_exit) from caution → risk_on
    st, _ = _run_spy([100, 88, 94, 98])
    assert st.spy_tier == "risk_on"


def test_spy_hysteresis_band_holds_between_thresholds() -> None:
    # caution entered at -6%; a drift to -4% stays caution (between
    # caution_exit -3% and caution_enter -5%) — the band is the debounce.
    st, _ = _run_spy([100, 94, 96])
    assert st.spy_tier == "caution"


def test_min_persist_days_gt_1_delays_then_latches() -> None:
    tun = DrawdownTunables(min_persist_days=3)
    # One -11% day is not enough to latch risk_off when persistence=3.
    st, _ = _run_spy([100, 89], tun)
    assert st.spy_tier == "risk_on"
    assert st.pending_spy_tier == "risk_off" and st.pending_spy_days == 1
    # Three consecutive risk_off-proposing days → latch.
    st, _ = _run_spy([100, 89, 89, 89], tun)
    assert st.spy_tier == "risk_off"


# ── excess (book-vs-market) leg + NAV graceful-degrade ───────────────────

def test_excess_unavailable_when_nav_none() -> None:
    st, art = step(
        None, spy_close=100.0, trading_day="d0", calendar_date="c0",
        run_id="r", nav=None,
    )
    assert art["excess"]["available"] is False
    assert art["excess"]["regime_contribution"] is None
    assert st.excess_tier == "risk_on"


def test_excess_alpha_bleed_when_book_deeper_than_market() -> None:
    # SPY flat at peak (dd 0); NAV draws down 6% → excess depth 6% ≥ 5%.
    st = None
    st, _ = step(None, spy_close=100.0, trading_day="d0",
                 calendar_date="c0", run_id="r", nav=100.0)
    st, art = step(st, spy_close=100.0, trading_day="d1",
                   calendar_date="c1", run_id="r", nav=94.0)
    assert st.excess_tier == "alpha_bleed"
    assert art["excess"]["available"] is True
    assert art["excess"]["regime_contribution"] == "bear"


def test_excess_not_triggered_when_book_tracks_market() -> None:
    st = None
    st, _ = step(None, spy_close=100.0, trading_day="d0",
                 calendar_date="c0", run_id="r", nav=100.0)
    # Both down ~6% together → excess gap ~0, no alpha_bleed.
    st, _ = step(st, spy_close=94.0, trading_day="d1",
                 calendar_date="c1", run_id="r", nav=94.0)
    assert st.excess_tier == "risk_on"


# ── step idempotency + cold start ────────────────────────────────────────

def test_step_idempotent_on_same_trading_day() -> None:
    st, _ = step(None, spy_close=100.0, trading_day="d0",
                 calendar_date="c0", run_id="r")
    obs_before = st.observations_seen
    st2, art = step(st, spy_close=50.0, trading_day="d0",
                    calendar_date="c0", run_id="r")
    assert st2 is st                       # unchanged
    assert art["observed"] is False
    assert st2.observations_seen == obs_before


def test_cold_start_seeds_peak_and_flags() -> None:
    st, art = step(None, spy_close=100.0, trading_day="d0",
                   calendar_date="c0", run_id="r")
    assert art["cold_start"] is True
    assert st.spy_peak == 100.0
    assert st.observations_seen == 1


# ── state (de)serialization ──────────────────────────────────────────────

def test_dump_load_roundtrip() -> None:
    st = DrawdownState(spy_tier="caution", spy_peak=123.4,
                       observations_seen=9, last_update_trading_day="d9")
    rt = load_state(json.loads(json.dumps(dump_state(st))))
    assert rt == st


def test_load_state_schema_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        load_state({"schema_version": DRAWDOWN_SCHEMA_VERSION + 99})


# ── composition truth table ──────────────────────────────────────────────

def test_most_protective_ordering_and_none_handling() -> None:
    assert most_protective(None, None) == "neutral"
    assert most_protective("bull", None, "caution") == "caution"
    assert most_protective("neutral", "bear", "caution") == "bear"
    assert most_protective("bull", "neutral") == "neutral"


def test_compose_effective_regime_drivers_preserved() -> None:
    out = compose_effective_regime(
        hmm_argmax="neutral", spy_tier="risk_off",
        excess_tier="risk_on", forced_bear=False,
    )
    assert out["effective_regime"] == "bear"           # drawdown_spy escalates
    assert out["drivers"]["hmm"] == "neutral"
    assert out["drivers"]["drawdown_spy"] == "bear"
    assert out["drivers"]["drawdown_excess"] is None
    # forced_bear alone also escalates
    out2 = compose_effective_regime(hmm_argmax="bull", forced_bear=True)
    assert out2["effective_regime"] == "bear"


# ── read_eod_pnl_nav graceful-degrade matrix ─────────────────────────────

class _S3:
    def __init__(self, body: bytes | None = None, raise_exc: bool = False):
        self._body, self._raise = body, raise_exc

    def get_object(self, *, Bucket, Key):
        if self._raise or self._body is None:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self._body)}


def test_eod_pnl_happy_path() -> None:
    csv = b"date,nav,alpha\n2026-05-01,100\n2026-05-02,101\n2026-05-03,99\n"
    s = read_eod_pnl_nav(_S3(csv), bucket="b")
    assert s is not None and len(s) == 3 and float(s.iloc[1]) == 101.0


def test_eod_pnl_missing_key_returns_none() -> None:
    assert read_eod_pnl_nav(_S3(raise_exc=True), bucket="b") is None


def test_eod_pnl_no_nav_column_returns_none() -> None:
    assert read_eod_pnl_nav(_S3(b"date,alpha\n2026-05-01,0.1\n"), bucket="b") is None


def test_eod_pnl_too_short_returns_none() -> None:
    assert read_eod_pnl_nav(_S3(b"date,nav\n2026-05-01,100\n"), bucket="b") is None


def test_eod_pnl_empty_returns_none() -> None:
    assert read_eod_pnl_nav(_S3(b""), bucket="b") is None


# ── cold-start seed_state ────────────────────────────────────────────────

def test_seed_state_uses_true_trailing_peak_not_today() -> None:
    # Rallied to 120 then fell to 102 (-15% from the trailing peak). A
    # naive cold start (peak=today) would report ~0%; seed_state must
    # use the 120 peak and start in risk_off.
    idx = pd.date_range("2020-01-01", periods=6, freq="W")
    spy = pd.Series([100, 110, 120, 115, 108, 102], index=idx)
    st = seed_state(spy)
    assert st.spy_peak == 120.0
    assert st.spy_tier == "risk_off"          # -15% ≥ 10% enter
    # First real step from the seed reports the true drawdown, not 0.
    st2, art = step(st, spy_close=102.0, trading_day="d0",
                     calendar_date="c0", run_id="r")
    assert art["spy"]["drawdown"] == pytest.approx(102 / 120 - 1.0)
    assert st2.spy_tier == "risk_off"


def test_seed_state_conservative_tier_thresholds() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="W")
    assert seed_state(pd.Series([100, 100, 97], index=idx)).spy_tier == "risk_on"
    assert seed_state(pd.Series([100, 100, 94], index=idx)).spy_tier == "caution"
    assert seed_state(pd.Series([100, 100, 89], index=idx)).spy_tier == "risk_off"


def test_seed_state_nav_excess_seeding() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="W")
    spy = pd.Series([100, 100, 100], index=idx)        # SPY flat at peak
    nav = pd.Series([100, 100, 93], index=idx)         # book -7% ⇒ excess 7%
    st = seed_state(spy, nav_history=nav)
    assert st.nav_peak == 100.0 and st.excess_tier == "alpha_bleed"


def test_seed_state_empty_history() -> None:
    assert seed_state(pd.Series(dtype=float)) == DrawdownState()


# ── daily stage: _advance_drawdown (observe-only) ────────────────────────

def test_stage_advances_drawdown_observe_only(monkeypatch) -> None:
    """The fast-signal stage also advances the drawdown leg: stamps
    ctx.drawdown_effective_regime + persists state/artifact/sidecar,
    while leaving predictions + the fast-signal path untouched."""
    import inference.stages.regime_fast_signal as stg
    from inference.pipeline import PipelineContext

    class _NoSuchKey(Exception):
        pass

    puts: dict[str, str] = {}

    class _FakeS3:
        class exceptions:
            NoSuchKey = _NoSuchKey

        def get_object(self, Bucket, Key):  # noqa: N803 — cold start both
            raise _NoSuchKey()

        def put_object(self, **kw):
            pass

    monkeypatch.setitem(
        __import__("sys").modules, "boto3",
        type("B", (), {"client": staticmethod(lambda svc: _FakeS3())}),
    )
    monkeypatch.setattr(stg, "_s3_put_json",
                        lambda s3, b, k, body: puts.__setitem__(k, body))
    monkeypatch.setattr(stg, "_emit_metrics", lambda art: None)

    ctx = PipelineContext(date_str="2026-05-15", dry_run=False, local=False,
                          bucket="alpha-engine-research")
    ctx.regime_intensity_z = -2.5
    # -16% from the 120 trailing peak ⇒ drawdown leg → risk_off → bear.
    ctx.macro = {"SPY": pd.Series(
        [100.0, 110.0, 120.0, 112.0, 105.0, 101.0],
        index=pd.date_range("2026-04-01", periods=6, freq="W"),
    )}
    assert ctx.drawdown_effective_regime is None  # default

    stg.run(ctx)

    assert ctx.drawdown_effective_regime == "bear"        # risk_off escalates
    assert stg.DRAWDOWN_STATE_KEY in puts
    assert any("regime/drawdown/" in k for k in puts)      # artifact + latest
    # Fast-signal path unaffected (cold-start warmup ⇒ forced_bear False).
    assert ctx.regime_forced_bear is False


def test_stage_drawdown_graceful_when_spy_missing(monkeypatch) -> None:
    """No SPY in ctx.macro ⇒ drawdown leg skips cleanly; fast-signal +
    predictions unaffected, ctx.drawdown_effective_regime stays None."""
    import inference.stages.regime_fast_signal as stg
    from inference.pipeline import PipelineContext

    class _NoSuchKey(Exception):
        pass

    class _FakeS3:
        class exceptions:
            NoSuchKey = _NoSuchKey

        def get_object(self, Bucket, Key):  # noqa: N803
            raise _NoSuchKey()

        def put_object(self, **kw):
            pass

    monkeypatch.setitem(
        __import__("sys").modules, "boto3",
        type("B", (), {"client": staticmethod(lambda svc: _FakeS3())}),
    )
    monkeypatch.setattr(stg, "_s3_put_json", lambda *a, **k: None)
    monkeypatch.setattr(stg, "_emit_metrics", lambda art: None)

    ctx = PipelineContext(date_str="2026-05-15", bucket="alpha-engine-research")
    ctx.regime_intensity_z = 0.1
    ctx.macro = {}  # no SPY

    stg.run(ctx)  # must not raise

    assert ctx.drawdown_effective_regime is None


# ── additive substrate hook ──────────────────────────────────────────────

pytest.importorskip("hmmlearn")
pytest.importorskip("scipy")


def _hist(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i, (r, v, h) in enumerate([(0.02, 14.0, 320.0), (-0.02, 28.0, 600.0)]):
        parts.append(pd.DataFrame({
            "spy_20d_return": rng.normal(r, 0.03, n // 2),
            "vix_level": rng.normal(v, 4.0, n // 2),
            "vix_term_slope": rng.normal(0.8 if i == 0 else -0.3, 0.4, n // 2),
            "hy_oas_bps": rng.normal(h, 80.0, n // 2),
            "yield_curve_slope": rng.normal(0.4 if i == 0 else -0.2, 0.3, n // 2),
            "market_breadth": rng.normal(0.7 if i == 0 else 0.35, 0.1,
                                         n // 2).clip(0, 1),
        }))
    return pd.concat(parts, ignore_index=True)


def _fitted_hmm(history: pd.DataFrame):
    from regime.hmm import HMMRegimeClassifier

    cols = list(history.columns)
    return HMMRegimeClassifier(n_states=3, random_state=0).fit(history, cols)


def test_substrate_no_drawdown_block_is_zero_change() -> None:
    from regime.substrate import build_regime_substrate

    hist = _hist()
    payload = build_regime_substrate(
        feature_history=hist, current_features=hist.iloc[-1].to_dict(),
        hmm=_fitted_hmm(hist), run_id="2605181200",
        calendar_date="2026-05-18", trading_day="2026-05-15",
    )
    assert "drawdown" not in payload
    assert "effective_regime" not in payload


def test_substrate_with_drawdown_block_adds_keys_and_composes() -> None:
    from regime.substrate import build_regime_substrate

    hist = _hist()
    block = {
        "spy": {"tier": "risk_off", "regime_contribution": "bear"},
        "excess": {"available": True, "tier": "risk_on"},
    }
    payload = build_regime_substrate(
        feature_history=hist, current_features=hist.iloc[-1].to_dict(),
        hmm=_fitted_hmm(hist), run_id="2605181200",
        calendar_date="2026-05-18", trading_day="2026-05-15",
        drawdown_block=block,
    )
    assert payload["drawdown"] == block
    # SPY risk_off → bear, regardless of the HMM argmax (most-protective).
    assert payload["effective_regime"]["effective_regime"] == "bear"


def test_substrate_sidecar_carries_effective_regime() -> None:
    from regime.substrate import build_regime_substrate, write_regime_substrate

    class _Mem:
        def __init__(self) -> None:
            self.objs: dict[str, bytes] = {}

        def put_object(self, *, Bucket, Key, Body, ContentType=None):
            self.objs[Key] = Body if isinstance(Body, bytes) else Body.encode()
            return {}

    hist = _hist()
    payload = build_regime_substrate(
        feature_history=hist, current_features=hist.iloc[-1].to_dict(),
        hmm=_fitted_hmm(hist), run_id="2605181200",
        calendar_date="2026-05-18", trading_day="2026-05-15",
        drawdown_block={"spy": {"tier": "caution"},
                        "excess": {"available": False, "tier": "risk_on"}},
    )
    s3 = _Mem()
    res = write_regime_substrate(payload, s3_client=s3, bucket="b", prefix="regime")
    sidecar = json.loads(s3.objs[res["latest_key"]])
    # The sidecar must mirror the composed payload value exactly (the
    # composition itself — most-protective over the HMM leg too — is
    # covered by the composition truth-table tests).
    assert (
        sidecar["effective_regime"]
        == payload["effective_regime"]["effective_regime"]
    )
    assert sidecar["effective_regime"] in {"bull", "neutral", "caution", "bear"}
