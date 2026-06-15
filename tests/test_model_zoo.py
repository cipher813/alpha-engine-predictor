"""Tests for the model-zoo driver (L4488c).

Pins: spec resolution (active / retired / missing); the override allowlist
(disallowed keys fail loud); the save/restore context (sets cfg, restores on
exit AND on exception, removes a previously-absent attr); train_spec applies the
overrides while the injected train_fn runs and defaults the version label; and
train_all_active iterates active specs, skips retired, and continues past a
single failure.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz

_SPECS = [
    {"id": "resid", "status": "active", "overrides": {"RESIDUAL_MOMENTUM_ENABLED": True}},
    {"id": "h60", "status": "active", "model_version_label": "spec-60d",
     "overrides": {"FORWARD_DAYS": 60}},
    {"id": "old", "status": "retired", "overrides": {"FORWARD_DAYS": 90}},
]


def test_resolve_spec_active_retired_missing():
    assert mz.resolve_spec("resid", _SPECS)["id"] == "resid"
    with pytest.raises(mz.ModelSpecError, match="not active"):
        mz.resolve_spec("old", _SPECS)
    with pytest.raises(mz.ModelSpecError, match="not found"):
        mz.resolve_spec("nope", _SPECS)


def test_validate_overrides_rejects_disallowed_key():
    with pytest.raises(mz.ModelSpecError, match="allowlist"):
        mz._validate_overrides({"SOME_RANDOM_ATTR": 1})
    mz._validate_overrides({"FORWARD_DAYS": 60})  # allowed → no raise


def test_spec_overrides_sets_and_restores(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    assert cfg.FORWARD_DAYS == 21
    with mz.spec_overrides({"FORWARD_DAYS": 60}):
        assert cfg.FORWARD_DAYS == 60
    assert cfg.FORWARD_DAYS == 21  # restored


def test_spec_overrides_restores_on_exception(monkeypatch):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    with pytest.raises(RuntimeError):
        with mz.spec_overrides({"FORWARD_DAYS": 60}):
            assert cfg.FORWARD_DAYS == 60
            raise RuntimeError("boom")
    assert cfg.FORWARD_DAYS == 21  # restored despite the exception


def test_spec_overrides_removes_previously_absent_attr():
    # MODEL_VERSION_LABEL may not pre-exist on a bare cfg; the context must
    # delattr it on exit rather than leave a stale value.
    had = hasattr(cfg, "MODEL_VERSION_LABEL")
    prev = getattr(cfg, "MODEL_VERSION_LABEL", None)
    if had:
        delattr(cfg, "MODEL_VERSION_LABEL")
    try:
        with mz.spec_overrides({"MODEL_VERSION_LABEL": "spec-x"}):
            assert cfg.MODEL_VERSION_LABEL == "spec-x"
        assert not hasattr(cfg, "MODEL_VERSION_LABEL")  # removed (was absent)
    finally:
        if had:
            cfg.MODEL_VERSION_LABEL = prev


def test_train_spec_applies_overrides_and_defaults_label():
    seen = {}

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        seen["bucket"] = bucket
        seen["forward_days"] = cfg.FORWARD_DAYS              # override in effect
        seen["resid"] = cfg.RESIDUAL_MOMENTUM_ENABLED
        seen["label"] = cfg.MODEL_VERSION_LABEL
        return {"status": "ok", "model_version": cfg.MODEL_VERSION_LABEL}

    import contextlib
    base_fd = getattr(cfg, "FORWARD_DAYS", 21)
    with contextlib.ExitStack():
        out = mz.train_spec("h60", "bkt", specs=_SPECS, train_fn=_fake_train)
    assert seen["forward_days"] == 60
    assert seen["label"] == "spec-60d"        # spec's declared label
    assert out["status"] == "ok"
    assert cfg.FORWARD_DAYS == base_fd        # restored after the call


def test_train_spec_label_defaults_to_spec_id_when_unset():
    captured = {}

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        captured["label"] = cfg.MODEL_VERSION_LABEL
        return {"status": "ok"}

    mz.train_spec("resid", "bkt", specs=_SPECS, train_fn=_fake_train)
    assert captured["label"] == "spec-resid"  # no declared label → spec-<id>


def test_train_all_active_skips_retired_and_continues_on_failure():
    calls = []

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        calls.append(cfg.MODEL_VERSION_LABEL)
        if cfg.FORWARD_DAYS == 60:
            raise RuntimeError("h60 boom")
        return {"status": "ok"}

    results = mz.train_all_active("bkt", specs=_SPECS, train_fn=_fake_train)
    # Only the 2 active specs run; the retired one is skipped.
    assert set(results) == {"resid", "h60"}
    assert results["resid"]["status"] == "ok"
    assert results["h60"]["status"] == "error"  # failure captured, didn't abort
    assert "old" not in results


# ── L4488g: weekly rotation ──────────────────────────────────────────────────


def test_resolve_label_overrides_then_declared_then_default():
    # overrides.MODEL_VERSION_LABEL wins (train_spec applies it verbatim)…
    assert mz._resolve_label(
        {"id": "x", "overrides": {"MODEL_VERSION_LABEL": "spec-custom"}}
    ) == "spec-custom"
    # …else the declared model_version_label…
    assert mz._resolve_label({"id": "h60", "model_version_label": "spec-60d"}) == "spec-60d"
    # …else spec-<id>.
    assert mz._resolve_label({"id": "resid"}) == "spec-resid"


def test_select_rotation_never_trained_first_then_id_tiebreak():
    # No registry history → both active specs are maximally stale; id tiebreak.
    sel = mz.select_rotation_specs(_SPECS, [], budget=1)
    assert sel == ["h60"]                       # "h60" < "resid"
    assert mz.select_rotation_specs(_SPECS, [], budget=5) == ["h60", "resid"]
    # Retired spec never selected.
    assert "old" not in mz.select_rotation_specs(_SPECS, [], budget=5)


def test_select_rotation_prefers_never_trained_over_trained():
    # resid trained recently, h60 never → h60 (never-trained) is stalest.
    reg = [{"model_version": "spec-resid", "date": "2026-06-01"}]
    assert mz.select_rotation_specs(_SPECS, reg, budget=1) == ["h60"]


def test_select_rotation_oldest_version_first_using_newest_per_spec():
    # Both trained; resid's NEWEST version is older than h60's → resid first.
    reg = [
        {"model_version": "spec-resid", "date": "2026-05-01"},
        {"model_version": "spec-resid", "date": "2026-05-20"},  # newest for resid
        {"model_version": "spec-60d", "date": "2026-06-01"},
    ]
    assert mz.select_rotation_specs(_SPECS, reg, budget=1) == ["resid"]
    assert mz.select_rotation_specs(_SPECS, reg, budget=2) == ["resid", "h60"]


def test_select_rotation_priority_steers_cold_start():
    # All never-registered (equal staleness) → priority breaks the tie ahead of
    # the id alphabetical fallback. resid would lose the id tiebreak ("h60" <
    # "resid") but priority 10 pulls it first — the cold-start steer toward the
    # one promote-eligible variant.
    specs = [
        {"id": "resid", "status": "active", "priority": 10,
         "overrides": {"RESIDUAL_MOMENTUM_ENABLED": True}},
        {"id": "h60", "status": "active", "overrides": {"FORWARD_DAYS": 60}},
        {"id": "h90", "status": "active", "overrides": {"FORWARD_DAYS": 90}},
    ]
    assert mz.select_rotation_specs(specs, [], budget=1) == ["resid"]
    # Full cold-start order: resid (priority), then the two zero-priority
    # horizons by id.
    assert mz.select_rotation_specs(specs, [], budget=3) == ["resid", "h60", "h90"]


def test_select_rotation_priority_never_overrides_staleness():
    # resid has priority 10 AND a fresh version; the horizons are never-trained.
    # Staleness dominates → a never-trained horizon is picked before the fresh
    # high-priority resid, so priority cannot starve the round-robin.
    specs = [
        {"id": "resid", "status": "active", "priority": 10,
         "overrides": {"RESIDUAL_MOMENTUM_ENABLED": True}},
        {"id": "h60", "status": "active", "overrides": {"FORWARD_DAYS": 60}},
    ]
    reg = [{"model_version": "spec-resid", "date": "2026-06-13"}]
    assert mz.select_rotation_specs(specs, reg, budget=1) == ["h60"]


def test_select_rotation_default_priority_preserves_id_tiebreak():
    # No priority field anywhere → behavior is byte-identical to the pre-priority
    # id tiebreak (back-compat).
    assert mz.select_rotation_specs(_SPECS, [], budget=2) == ["h60", "resid"]


def test_train_weekly_rotation_trains_only_budget_stalest():
    trained = []

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        trained.append(cfg.MODEL_VERSION_LABEL)
        return {"status": "ok"}

    reg = [{"model_version": "spec-60d", "date": "2026-06-01"}]  # h60 fresh, resid never
    results = mz.train_weekly_rotation(
        "bkt", budget=1, specs=_SPECS, train_fn=_fake_train, registered_versions=reg,
    )
    # Budget=1 → only the stalest (resid, never trained) runs.
    assert set(results) == {"resid"}
    assert trained == ["spec-resid"]


def test_train_weekly_rotation_continues_past_failure():
    def _fake_train(bucket, *, date_str=None, dry_run=False):
        if cfg.FORWARD_DAYS == 60:
            raise RuntimeError("h60 boom")
        return {"status": "ok"}

    results = mz.train_weekly_rotation(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train, registered_versions=[],
    )
    assert results["h60"]["status"] == "error"
    assert results["resid"]["status"] == "ok"


# ── L4544: rotation isolation (G1 challenger-first + G2 contract restore) ─────


class _FakeS3:
    """Minimal S3 stub: get_object reads from ``objects``, put_object records to
    ``puts``. Missing keys raise (mimicking a NoSuchKey the code catches)."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg names
        if Key not in self.objects:
            raise KeyError(Key)
        body = json.dumps(self.objects[Key]).encode()
        return {"Body": io.BytesIO(body), "ContentType": "application/json"}

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        self.puts[Key] = Body


def _mk_manifest(forward_days, cpcv_ic, gate_pass, *, dsr=None):
    overfit = {"passes_overfit_gate": gate_pass}
    if dsr is not None:
        overfit["dsr"] = dsr
    return {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": (
            {"mean_ic": cpcv_ic} if cpcv_ic is not None else {"status": "error"}
        ),
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": gate_pass},
            "overfit": overfit,
        },
    }


def test_rotation_trains_every_selected_spec():
    # config#1052/#679: the former G1 guard (forcing TRAINING_AUTO_PROMOTE_ENABLED
    # off per train) is GONE — training is unconditionally challenger-first, so a
    # spec can never self-promote and there is nothing to force/restore. The
    # surviving rotation invariant is that every selected spec is trained exactly
    # once; G2 contract restore is covered by the roundtrip test below.
    trained = []

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        trained.append(bucket)
        return {"status": "ok"}

    results = mz.train_weekly_rotation(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train, registered_versions=[],
    )
    n_active = sum(1 for s in _SPECS if s.get("status", "active") == "active" and s.get("id"))
    assert len(trained) == min(5, n_active)                # one train per selected spec
    assert all(r.get("status") == "ok" for r in results.values())


def test_snapshot_and_restore_live_contract_roundtrip():
    # G2: the live champion contract is captured then restored byte-for-byte.
    champ_manifest = {"version": "champ", "forward_days": 21}
    feat = {"features": ["a", "b"]}
    s3 = _FakeS3({cfg.META_MANIFEST_KEY: champ_manifest, cfg.META_FEATURE_LIST_KEY: feat})
    saved = mz._snapshot_live_contract(s3, "bkt")
    assert set(saved) == {cfg.META_MANIFEST_KEY, cfg.META_FEATURE_LIST_KEY}
    mz._restore_live_contract(s3, "bkt", saved)
    assert json.loads(s3.puts[cfg.META_MANIFEST_KEY]) == champ_manifest
    assert json.loads(s3.puts[cfg.META_FEATURE_LIST_KEY]) == feat


# ── L4544: immediate CPCV selection ──────────────────────────────────────────


def test_select_winner_horizon_floor_and_margin(monkeypatch):
    # config#671/#673/#1052 RELATIVE-BEST: the DSR/Sortino gate is NO LONGER a
    # promotion blocker. Eligibility = right horizon AND IC > positive floor AND
    # IC >= champion + margin. The DSR-gate result is surfaced (dsr_gate_pass),
    # never blocks.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),          # champion baseline
        # DSR-gate FAILS (passes_overfit_gate=False, dsr 0.247) but beats champion
        # by margin → PROMOTES under relative-best (the exact 6/13 scenario).
        "predictor/registry/dsrfail-v/manifest.json": _mk_manifest(
            21, 0.20, False, dsr=0.247),
        "predictor/registry/h60-v/manifest.json": _mk_manifest(60, 0.30, True),    # wrong horizon
        "predictor/registry/low-v/manifest.json": _mk_manifest(21, 0.105, True),   # below champ+margin
    })
    trained = [
        {"spec_id": "dsrfail", "version_id": "dsrfail-v", "model_version": "spec-dsrfail"},
        {"spec_id": "h60", "version_id": "h60-v", "model_version": "spec-60d"},
        {"spec_id": "low", "version_id": "low-v", "model_version": "spec-low"},
    ]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    # the DSR-gate-failing challenger PROMOTES on relative-best (beats champ+margin)
    assert board["winner_version_id"] == "dsrfail-v"
    reasons = {c["spec_id"]: c["reason"] for c in board["candidates"]}
    assert reasons["h60"] == "non_canonical_horizon"        # wrong horizon, never eligible
    assert reasons["low"] == "below_champion_plus_margin"
    assert reasons["dsrfail"] == "eligible"
    # DSR is surfaced for OBSERVABILITY on the winner, but did NOT block promotion.
    win = next(c for c in board["candidates"] if c["spec_id"] == "dsrfail")
    assert win["dsr_gate_pass"] is False                    # absolute gate failed
    assert win["dsr_training"] == 0.247                     # value surfaced
    assert "promote_min_ic" in board
    assert "observe_eligible_version_ids" not in board      # observe tier collapsed


def test_select_winner_positive_ic_floor_blocks_negative_best_of_n(monkeypatch):
    # The ONLY absolute floor under relative-best: a best-of-N challenger whose
    # CPCV mean IC is <= the positive floor is noise, not edge — never promote it,
    # even though it is the best (and only) candidate and "beats" a None champion.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    s3 = _FakeS3({
        # champion manifest unreadable → champ_ic None; floor must still block.
        "predictor/registry/neg-v/manifest.json": _mk_manifest(21, -0.05, True),
    })
    trained = [{"spec_id": "neg", "version_id": "neg-v", "model_version": "spec-neg"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.005)
    c = board["candidates"][0]
    assert c["reason"] == "below_floor"
    assert c["eligible"] is False
    assert board["winner_version_id"] is None


def test_select_winner_nan_cpcv_is_no_cpcv_not_below_margin(monkeypatch):
    # L4565c: a date-starved CPCV reports mean_ic=NaN. float('nan') is not None
    # and slips past `ic is None`, then `ic < champ+margin` is False (NaN cmp) —
    # mislabeling the candidate `below_champion_plus_margin`. _cpcv_mean must
    # collapse NaN→None so it's correctly `no_cpcv` and never wins.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),                     # champion
        "predictor/registry/nan-v/manifest.json": _mk_manifest(21, float("nan"), True),
    })
    trained = [{"spec_id": "nan", "version_id": "nan-v", "model_version": "spec-nan"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    reasons = {c["spec_id"]: c["reason"] for c in board["candidates"]}
    assert reasons["nan"] == "no_cpcv"
    assert board["winner_version_id"] is None


def _run_select_fixture(monkeypatch, *, auto_promote):
    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/resid-v/manifest.json": _mk_manifest(21, 0.20, True),
        "predictor/registry/h60-v/manifest.json": _mk_manifest(60, 0.30, True),
    })
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [
        {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        {"version_id": "h60-v", "model_version": "spec-60d", "date": "2026-06-13"},
    ])
    promotes = []
    monkeypatch.setattr(reg, "promote_to_champion",
                        lambda s3c, b, vid, **k: promotes.append(vid))

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train,
        registered_versions=[], s3=s3, auto_promote_winner=auto_promote,
    )
    return board, promotes, s3


def test_run_rotation_observe_recommends_but_never_promotes(monkeypatch):
    board, promotes, s3 = _run_select_fixture(monkeypatch, auto_promote=False)
    assert board["mode"] == "observe"
    assert board["winner_version_id"] == "resid-v"          # 21d challenger beats champion
    assert board["promoted"] is None                        # observe → not executed
    assert promotes == []                                   # promote_to_champion never called
    assert any(k.startswith(mz._LEADERBOARD_PREFIX) for k in s3.puts)  # leaderboard written


def test_run_rotation_cutover_promotes_winner(monkeypatch):
    board, promotes, _ = _run_select_fixture(monkeypatch, auto_promote=True)
    assert board["mode"] == "cutover"
    assert board["winner_version_id"] == "resid-v"
    assert board["promoted"] == "resid-v"                   # cutover → executed
    assert promotes == ["resid-v"]                          # promoted exactly the winner


# ── config#1051: fail loud on an INERT rotation (0 challengers trained) ───────


def test_count_challengers_trained_excludes_errors():
    results = {
        "a": {"status": "ok"},
        "b": {"status": "error", "error": "boom"},
        "c": {"status": "ok"},
    }
    assert mz._count_challengers_trained(results) == 2
    assert mz._count_challengers_trained({}) == 0
    assert mz._count_challengers_trained({"x": {"status": "error"}}) == 0


def test_inert_rotation_alerts_when_active_specs_but_zero_trained(monkeypatch):
    """The core config#1051 closes-when #2: ≥1 active promote-eligible spec exists
    but the rotation trains 0 challengers (empty selection) → WARN + named CW
    metric + SNS, distinct from 'trained-but-lost'."""
    from alpha_engine_lib import alerts  # conftest autouse-stubs .publish

    cw_calls = []
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: cw_calls.append(n))
    inert_calls = {}

    real_inert = mz._alert_inert_rotation

    def _spy_inert(bucket, date_str, **kw):
        inert_calls.update(kw)
        inert_calls["called"] = True
        return real_inert(bucket, date_str, **kw)

    monkeypatch.setattr(mz, "_alert_inert_rotation", _spy_inert)

    # budget=0 → selection is empty even though _SPECS has 2 active specs, so the
    # rotation trains nothing (the empty-selection branch of the inert bug).
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
    })

    def _fake_train(bucket, *, date_str=None, dry_run=False):  # pragma: no cover
        raise AssertionError("train_fn must not be called when budget=0")

    board = mz.run_rotation_and_select(
        "bkt", budget=0, specs=_SPECS, train_fn=_fake_train,
        registered_versions=[], s3=s3, date_str="2026-06-13",
    )

    assert inert_calls.get("called") is True
    assert inert_calls["n_active"] == 2          # resid + h60 are active in _SPECS
    assert inert_calls["n_selected"] == 0        # budget=0 → trained nothing
    assert cw_calls == [0]                        # CW metric emitted with value 0
    # SNS alert fired (severity warning, dedup keyed on the date).
    assert alerts.publish.called
    _, kwargs = alerts.publish.call_args
    assert kwargs["severity"] == "warning"
    assert kwargs["dedup_key"] == "model_zoo_inert_2026-06-13"
    assert "0 CHALLENGERS" in kwargs["message"]


def test_all_specs_errored_is_inert_and_alerts(monkeypatch):
    """Even with budget covering the active specs, if EVERY selected spec errors
    the rotation is inert (0 usable challengers) → alert fires."""
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: None)
    alerted = []
    monkeypatch.setattr(mz, "_alert_inert_rotation",
                        lambda *a, **k: alerted.append(k))

    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
    })

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        raise RuntimeError("every train crashes")

    mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train,
        registered_versions=[], s3=s3, date_str="2026-06-13",
    )
    assert len(alerted) == 1
    assert alerted[0]["n_active"] == 2


def test_trained_but_lost_does_not_alert_inert(monkeypatch):
    """A challenger trained but lost to the champion (winner-less leaderboard,
    candidates present) is NOT inert — no inert alert, CW metric reports the
    trained count (>0)."""
    cw_calls = []
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: cw_calls.append(n))
    inert = []
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: inert.append(k))
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)

    import model.registry as reg
    # Champion CPCV beats both trained challengers → no winner, but they DID train.
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.50, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/resid-v/manifest.json": _mk_manifest(21, 0.10, True),
        "predictor/registry/h60-v/manifest.json": _mk_manifest(60, 0.12, True),
    })
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [
        {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        {"version_id": "h60-v", "model_version": "spec-60d", "date": "2026-06-13"},
    ])

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train,
        registered_versions=[], s3=s3, date_str="2026-06-13",
    )
    assert inert == []                       # NOT inert — challengers trained
    assert cw_calls == [2]                   # both active specs trained
    assert board["winner_version_id"] is None  # but neither beat the champion


def test_date_str_defaults_to_trading_day_when_none(monkeypatch):
    """config#1051 item 4: a None date_str defaults to now_dual().trading_day so
    the leaderboard / trial_log never key on null."""
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: None)
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: None)

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    # dry_run path returns a leaderboard with the resolved date — no S3 needed.
    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train, dry_run=True,
    )
    assert board["date"] is not None
    # ISO date shape YYYY-MM-DD.
    assert len(board["date"]) == 10 and board["date"][4] == "-"


# ── config#1051: config.py fail-loud guard on an empty spec roster ────────────


def test_assert_model_specs_loaded_raises_when_empty(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_SPECS", [], raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_WEEKLY_BUDGET", 1, raising=False)
    with pytest.raises(cfg.ModelSpecsEmptyError):
        cfg.assert_model_specs_loaded()


def test_assert_model_specs_loaded_ok_when_populated(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_SPECS", [{"id": "x", "status": "active"}], raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_WEEKLY_BUDGET", 1, raising=False)
    cfg.assert_model_specs_loaded()  # no raise


def test_assert_model_specs_loaded_noop_when_budget_zero(monkeypatch):
    """An empty roster with budget 0 is a legitimately disabled zoo — no raise."""
    monkeypatch.setattr(cfg, "MODEL_SPECS", [], raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_WEEKLY_BUDGET", 0, raising=False)
    cfg.assert_model_specs_loaded()  # no raise
