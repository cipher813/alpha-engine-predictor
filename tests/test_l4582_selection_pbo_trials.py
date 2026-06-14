"""L4582 — PBO + trial-count logging + two-threshold discipline (OBSERVE).

Pins the three selection-observability legs added to the promotion-gate stack:

(a) ``cscv_pbo`` — CSCV probability of backtest overfitting over an aligned
    (combo × spec) IC matrix, leave-one-combination-out; plus the zoo's
    ``_selection_pbo`` wiring (alignment by CPCV shape, champion excluded by
    construction, honest-insufficient posture).
(b) the cumulative trial ledger (``_record_trials``: dedupe by version_id,
    cumulative count, recorded-not-swallowed failure) and the per-candidate
    ``dsr_selection`` re-deflation it feeds.
(c) the registry-entry bar + DSR gate are now OBSERVABILITY ONLY (surfaced via
    ``registry_bar_pass`` / ``dsr_gate_pass`` / ``dsr_training``), NOT promotion
    blockers — config#671/#673/#1052 relative-best: a beats-champion challenger
    PROMOTES regardless of DSR (the absolute DSR-0.95 hurdle was dropped).

None of (a)/(b) changes eligibility — pinned explicitly.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz
from training.deflated_sharpe import cscv_pbo

from tests.test_model_zoo import _FakeS3


# ── (a) cscv_pbo math ────────────────────────────────────────────────────────


def test_cscv_pbo_dominant_spec_is_zero():
    # Spec 1 beats spec 0 at EVERY combo → the IS pick is always spec 1 and it
    # always ranks top OOS → PBO 0.
    matrix = [[0.01, 0.05]] * 8
    out = cscv_pbo(matrix, spec_ids=["a", "b"])
    assert out["status"] == "ok"
    assert out["pbo"] == 0.0
    assert out["selected_counts"] == {"b": 8}


def test_cscv_pbo_alternating_spec_is_half():
    # Spec A alternates 0/1 around spec B's constant 0.4: whenever the held-out
    # combo is one of A's bad rows, the IS mean (over the good rows) still picks
    # A and it underperforms OOS → half the splits are mistakes.
    a = [0.0, 1.0] * 4
    b = [0.4] * 8
    out = cscv_pbo(list(zip(a, b)), spec_ids=["a", "b"])
    assert out["status"] == "ok"
    assert out["pbo"] == pytest.approx(0.5)


def test_cscv_pbo_insufficient_specs_and_splits():
    one_spec = cscv_pbo([[0.1]] * 8)
    assert one_spec["status"] == "insufficient"
    few_rows = cscv_pbo([[0.1, 0.2]] * 3)
    assert few_rows["status"] == "insufficient"
    assert "min_splits" in few_rows["reason"]


def test_cscv_pbo_drops_nonfinite_rows():
    rows = [[0.01, 0.05]] * 6 + [[float("nan"), 0.05], [0.01, float("inf")]]
    out = cscv_pbo(rows)
    assert out["status"] == "ok"
    assert out["n_splits"] == 6  # the 2 dirty rows dropped, not fabricated


# ── manifest builder with the L4582 fields ───────────────────────────────────


def _manifest(fwd=21, mean_ic=0.15, *, downside=True, dsr=0.97,
              gate=None, ics=None, n_groups=10, k_test=2):
    gate = (dsr is not None and dsr >= 0.95) if gate is None else gate
    cpcv = {"mean_ic": mean_ic, "n_groups": n_groups, "k_test": k_test}
    if ics is not None:
        cpcv["ics"] = ics
    return {
        "forward_days": fwd,
        "meta_model_oos_ic_cpcv": cpcv,
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": downside},
            "overfit": {"passes_overfit_gate": gate, "dsr": dsr},
        },
    }


# ── (c) DSR / registry bar are OBSERVABILITY, not blockers (config#1052) ──────


def test_dsr_and_registry_bar_are_observability_not_blockers(monkeypatch):
    # config#671/#673/#1052: the DSR gate + registry bar no longer block promotion.
    # All three challengers beat the champion (0.20 > 0.10 + margin) and FAIL the
    # DSR gate (gate=False) — under relative-best they are ALL eligible. The DSR /
    # registry-bar results are surfaced for observability, never as a reason.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    monkeypatch.setattr(cfg, "WF_DSR_REGISTRY_THRESHOLD", 0.80, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _manifest(mean_ic=0.10),
        # between bars: downside passes, DSR 0.85 ∈ [0.80, 0.95)
        "predictor/registry/near-v/manifest.json": _manifest(
            mean_ic=0.20, dsr=0.85, gate=False),
        # below both bars: DSR 0.40
        "predictor/registry/far-v/manifest.json": _manifest(
            mean_ic=0.20, dsr=0.40, gate=False),
        # downside fails → registry bar fails regardless of DSR
        "predictor/registry/down-v/manifest.json": _manifest(
            mean_ic=0.20, downside=False, dsr=0.97, gate=False),
    })
    board = mz.select_winner(s3, "bkt", trained=[
        {"spec_id": "near", "version_id": "near-v", "model_version": "m"},
        {"spec_id": "far", "version_id": "far-v", "model_version": "m"},
        {"spec_id": "down", "version_id": "down-v", "model_version": "m"},
    ], margin=0.01)
    rows = {c["spec_id"]: c for c in board["candidates"]}
    # all eligible on relative-best, DSR notwithstanding
    for sid in ("near", "far", "down"):
        assert rows[sid]["reason"] == "eligible", sid
        assert rows[sid]["eligible"] is True, sid
        assert rows[sid]["dsr_gate_pass"] is False, sid      # DSR gate surfaced
    # registry_bar_pass is still computed for observability
    assert rows["near"]["registry_bar_pass"] is True         # DSR 0.85 in-band
    assert rows["far"]["registry_bar_pass"] is False         # DSR 0.40 below
    assert rows["down"]["registry_bar_pass"] is False        # downside fails
    # a winner IS selected (highest-IC eligible — all tie at 0.20, one is chosen)
    assert board["winner_version_id"] in {"near-v", "far-v", "down-v"}


def test_legacy_manifest_without_dsr_field_still_promotes_on_relative_best(monkeypatch):
    # Pre-L4582 manifests carry no overfit.dsr — dsr_training is None (not crash),
    # dsr_gate_pass reflects the gate flag, and the challenger still PROMOTES on
    # relative-best (beats champion by margin) since DSR is not a blocker.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    legacy = _manifest(mean_ic=0.20, gate=False)
    del legacy["meta_model_promotion_stats"]["overfit"]["dsr"]
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _manifest(mean_ic=0.10),
        "predictor/registry/old-v/manifest.json": legacy,
    })
    board = mz.select_winner(s3, "bkt", trained=[
        {"spec_id": "old", "version_id": "old-v", "model_version": "m"}])
    row = board["candidates"][0]
    assert row["reason"] == "eligible"
    assert row["dsr_gate_pass"] is False
    assert row["dsr_training"] is None
    assert board["winner_version_id"] == "old-v"


# ── (b) trial ledger + dsr_selection ─────────────────────────────────────────


def _trained(*pairs):
    return [{"spec_id": s, "version_id": v, "model_version": f"spec-{s}"}
            for s, v in pairs]


def test_record_trials_appends_and_dedupes():
    s3 = _FakeS3()
    n, status = mz._record_trials(s3, "bkt", "2026-06-13",
                                  _trained(("a", "a-v1"), ("b", "b-v1")))
    assert (n, status) == (2, "ok")
    # same versions again → no growth; one new version → +1
    s3.objects[mz._TRIAL_LOG_KEY] = json.loads(s3.puts[mz._TRIAL_LOG_KEY])
    n, status = mz._record_trials(s3, "bkt", "2026-06-20",
                                  _trained(("a", "a-v1"), ("a", "a-v2")))
    assert (n, status) == (3, "ok")
    ledger = json.loads(s3.puts[mz._TRIAL_LOG_KEY])
    assert [t["version_id"] for t in ledger["trials"]] == ["a-v1", "b-v1", "a-v2"]


def test_record_trials_failure_is_recorded_not_swallowed():
    class _BrokenS3(_FakeS3):
        def put_object(self, **kwargs):  # noqa: N803
            raise RuntimeError("s3 down")

    n, status = mz._record_trials(_BrokenS3(), "bkt", "2026-06-13", _trained(("a", "a-v1")))
    assert n is None
    assert status.startswith("error:")


def test_select_winner_dsr_selection_uses_cumulative_trials(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    ics = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04]
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _manifest(mean_ic=0.10),
        "predictor/registry/a-v/manifest.json": _manifest(mean_ic=0.20, ics=ics),
    })
    few = mz.select_winner(s3, "bkt", trained=_trained(("a", "a-v")),
                           n_trials_cumulative=2)
    many = mz.select_winner(s3, "bkt", trained=_trained(("a", "a-v")),
                            n_trials_cumulative=200)
    d_few = few["candidates"][0]["dsr_selection"]
    d_many = many["candidates"][0]["dsr_selection"]
    assert d_few is not None and d_many is not None
    assert d_many < d_few                       # more trials → harsher deflation
    assert few["n_trials_cumulative"] == 2
    # without a count the stat is honestly absent, not fabricated
    none = mz.select_winner(s3, "bkt", trained=_trained(("a", "a-v")))
    assert none["candidates"][0]["dsr_selection"] is None


# ── (a) zoo wiring: selection_pbo block ──────────────────────────────────────


def test_selection_pbo_across_aligned_candidates(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PBO_TARGET", 0.2, raising=False)
    dominant = [0.06, 0.07, 0.05, 0.08, 0.06, 0.07]
    weak = [0.01, 0.02, 0.00, 0.01, 0.02, 0.01]
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _manifest(mean_ic=0.10),  # champion: NO ics → excluded
        "predictor/registry/a-v/manifest.json": _manifest(mean_ic=0.20, ics=dominant),
        "predictor/registry/b-v/manifest.json": _manifest(mean_ic=0.05, ics=weak),
        # misaligned CPCV shape → dropped by name, not silently
        "predictor/registry/c-v/manifest.json": _manifest(
            mean_ic=0.10, ics=[0.1, 0.2, 0.3], n_groups=6),
    })
    board = mz.select_winner(
        s3, "bkt", trained=_trained(("a", "a-v"), ("b", "b-v"), ("c", "c-v")))
    pbo = board["selection_pbo"]
    assert pbo["status"] == "ok"
    assert pbo["pbo"] == 0.0                      # dominant spec wins every split
    assert pbo["pbo_pass"] is True
    assert pbo["pbo_target"] == 0.2
    assert pbo["spec_ids"] == ["a", "b"]
    assert pbo["dropped_misaligned_specs"] == ["c"]


def test_selection_pbo_insufficient_with_one_candidate(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _manifest(mean_ic=0.10),
        "predictor/registry/a-v/manifest.json": _manifest(
            mean_ic=0.20, ics=[0.05] * 6),
    })
    board = mz.select_winner(s3, "bkt", trained=_trained(("a", "a-v")))
    assert board["selection_pbo"]["status"] == "insufficient"
    assert board["selection_pbo"]["pbo"] is None


# ── integration: rotation writes ledger status into the leaderboard ─────────


def test_rotation_leaderboard_carries_trial_log_and_pbo(monkeypatch):
    import model.registry as reg
    from tests.test_model_zoo import _SPECS
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    ics = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06]
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _manifest(mean_ic=0.10),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/resid-v/manifest.json": _manifest(mean_ic=0.20, ics=ics),
        "predictor/registry/h60-v/manifest.json": _manifest(
            fwd=60, mean_ic=0.30, ics=[x + 0.01 for x in ics]),
    })
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [
        {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        {"version_id": "h60-v", "model_version": "spec-60d", "date": "2026-06-13"},
    ])
    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS,
        train_fn=lambda bucket, *, date_str=None, dry_run=False: {"status": "ok"},
        registered_versions=[], s3=s3, auto_promote_winner=False,
        date_str="2026-06-13",
    )
    assert board["trial_log_status"] == "ok"
    assert board["n_trials_cumulative"] == 2      # both candidates ledgered
    assert mz._TRIAL_LOG_KEY in s3.puts
    # the horizon variant still competes in the PBO matrix (selection
    # diagnostics ≠ promotion eligibility — it stays non_canonical_horizon)
    assert board["selection_pbo"]["status"] == "ok"
    written = json.loads(s3.puts[f"{mz._LEADERBOARD_PREFIX}/2026-06-13.json"])
    assert written["selection_pbo"]["pbo"] is not None
