"""Tests for the model-zoo PARALLEL fan-out entrypoints (config#1083).

Pins the train-one-spec / select / list-rotation-specs split that lets the
Saturday Step Function fan one memory-isolated spot out per challenger:

  • train_one_spec        — trains+registers ONE spec, challenger-first (G2
                            contract snapshot/restore), raises on real failure.
  • list_rotation_spec_ids — emits the budget-N stalest active spec ids.
  • run_select_only       — selection over the registry pool, TOLERATING specs
                            whose Map iteration failed (absent → not a crash),
                            writing the leaderboard to BOTH the dated key AND
                            latest.json, sending the one digest, firing the inert
                            alert only when the pool is truly empty.

Reuses the _FakeS3 stub + _mk_manifest helper shape from test_model_zoo.py.
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


class _FakeS3:
    """Minimal S3 stub mirroring test_model_zoo.py: get_object reads ``objects``,
    put_object records to ``puts``; missing keys raise."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803
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


# ── train-one-spec ───────────────────────────────────────────────────────────


def test_train_one_spec_trains_and_applies_overrides():
    seen = {}

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        seen["bucket"] = bucket
        seen["forward_days"] = cfg.FORWARD_DAYS
        seen["label"] = cfg.MODEL_VERSION_LABEL
        return {"status": "ok", "model_version": cfg.MODEL_VERSION_LABEL}

    base_fd = getattr(cfg, "FORWARD_DAYS", 21)
    out = mz.train_one_spec("h60", "bkt", specs=_SPECS, train_fn=_fake_train)
    assert seen["forward_days"] == 60
    assert seen["label"] == "spec-60d"
    assert out["status"] == "ok"
    assert cfg.FORWARD_DAYS == base_fd            # restored after the call


def test_train_one_spec_raises_on_real_failure():
    # A real training failure must PROPAGATE so the SF Map iteration records
    # THIS spec's failure (non-zero exit) without aborting siblings.
    def _boom(bucket, *, date_str=None, dry_run=False):
        raise RuntimeError("oom in fold 3")

    with pytest.raises(RuntimeError, match="oom in fold 3"):
        mz.train_one_spec("resid", "bkt", specs=_SPECS, train_fn=_boom)


def test_train_one_spec_raises_on_unknown_spec():
    with pytest.raises(mz.ModelSpecError, match="not found"):
        mz.train_one_spec("nope", "bkt", specs=_SPECS, train_fn=lambda *a, **k: {})


# ── list-rotation-specs ──────────────────────────────────────────────────────


def test_list_rotation_spec_ids_budget_stalest():
    # No registry → both active maximally stale; id tiebreak; budget caps the list.
    assert mz.list_rotation_spec_ids("bkt", budget=1, specs=_SPECS,
                                     registered_versions=[]) == ["h60"]
    assert mz.list_rotation_spec_ids("bkt", budget=5, specs=_SPECS,
                                     registered_versions=[]) == ["h60", "resid"]
    # Retired spec never appears.
    assert "old" not in mz.list_rotation_spec_ids("bkt", budget=5, specs=_SPECS,
                                                  registered_versions=[])


def test_list_rotation_spec_ids_empty_when_no_active(monkeypatch):
    assert mz.list_rotation_spec_ids("bkt", budget=3, specs=[],
                                     registered_versions=[]) == []


# ── registry pool resolution (parallel select) ───────────────────────────────


def test_resolve_registered_specs_for_date_tolerates_missing(monkeypatch):
    # resid registered for the date; h60 NOT (its Map iteration failed). The pool
    # must contain only resid — h60 absent, not a crash.
    import model.registry as reg
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [
        {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        # a STALE prior-week resid that must be ignored for today's vintage
        {"version_id": "resid-old", "model_version": "spec-resid", "date": "2026-06-06"},
    ])
    pool = mz._resolve_registered_specs_for_date(
        _FakeS3(), "bkt", "2026-06-13", specs=_SPECS,
    )
    assert [c["spec_id"] for c in pool] == ["resid"]
    assert pool[0]["version_id"] == "resid-v"        # today's, not the stale one


def test_resolve_registered_specs_for_date_empty_on_total_failure(monkeypatch):
    import model.registry as reg
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [])
    pool = mz._resolve_registered_specs_for_date(
        _FakeS3(), "bkt", "2026-06-13", specs=_SPECS,
    )
    assert pool == []


# ── run_select_only (the parallel select entrypoint) ─────────────────────────


def _select_fixture(monkeypatch, *, registered, auto_promote=True):
    """Set up an S3 + registry where ``registered`` is the list of (spec_id,
    model_version, version_id, fwd, cpcv, gate) tuples that registered for the
    date. Returns (board, promotes, s3)."""
    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    objects = {
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
    }
    versions = []
    for sid, mv, vid, fwd, cpcv, gate in registered:
        objects[f"predictor/registry/{vid}/manifest.json"] = _mk_manifest(fwd, cpcv, gate)
        versions.append({"version_id": vid, "model_version": mv, "date": "2026-06-13"})
    s3 = _FakeS3(objects)
    # list_versions(stage="champion") is used by _current_champion_version_id;
    # return [] (no prior champion) regardless of the stage arg.
    monkeypatch.setattr(reg, "list_versions",
                        lambda s3c, b, stage=None: versions if stage != "champion" else [])
    promotes = []
    monkeypatch.setattr(reg, "promote_to_champion",
                        lambda s3c, b, vid, **k: promotes.append(vid))
    board = mz.run_select_only(
        "bkt", date_str="2026-06-13", s3=s3, specs=_SPECS,
        auto_promote_winner=auto_promote,
    )
    return board, promotes, s3


def test_run_select_only_tolerates_missing_specs_and_promotes(monkeypatch):
    # Only resid registered (h60's Map iteration failed → absent). Select must
    # rank resid against the champion, promote it, and NOT crash on absent h60.
    board, promotes, s3 = _select_fixture(
        monkeypatch,
        registered=[("resid", "spec-resid", "resid-v", 21, 0.20, True)],
        auto_promote=True,
    )
    assert board["winner_version_id"] == "resid-v"
    assert board["promoted"] == "resid-v"
    assert promotes == ["resid-v"]
    # Only the present spec (+ no base) is a candidate; h60 simply isn't there.
    cand_ids = {c["spec_id"] for c in board["candidates"]}
    assert "resid" in cand_ids
    assert "h60" not in cand_ids


def test_run_select_only_writes_both_leaderboard_keys(monkeypatch):
    board, _, s3 = _select_fixture(
        monkeypatch,
        registered=[("resid", "spec-resid", "resid-v", 21, 0.20, True)],
        auto_promote=False,
    )
    dated = f"{mz._LEADERBOARD_PREFIX}/2026-06-13.json"
    latest = f"{mz._LEADERBOARD_PREFIX}/latest.json"
    assert dated in s3.puts, "dated leaderboard key not written"
    assert latest in s3.puts, "latest.json mirror not written (config#1083 fix)"
    # Both carry the SAME board content.
    assert json.loads(s3.puts[dated])["date"] == "2026-06-13"
    assert json.loads(s3.puts[latest])["date"] == "2026-06-13"


def test_run_select_only_sends_one_digest(monkeypatch):
    sent = []
    monkeypatch.setattr(mz, "send_zoo_digest_email",
                        lambda lb, b, d, **k: sent.append(d) or True)
    _select_fixture(
        monkeypatch,
        registered=[("resid", "spec-resid", "resid-v", 21, 0.20, True)],
        auto_promote=False,
    )
    assert sent == ["2026-06-13"]            # exactly ONE consolidated digest


def test_run_select_only_inert_alert_on_empty_pool(monkeypatch):
    # No spec registered AND no base champion-arch → truly inert select → alert.
    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [])
    monkeypatch.setattr(mz, "send_zoo_digest_email", lambda *a, **k: True)
    alerted = []
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: alerted.append(k))
    cw = []
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: cw.append(n))

    s3 = _FakeS3({cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True)})
    board = mz.run_select_only("bkt", date_str="2026-06-13", s3=s3,
                               specs=_SPECS, auto_promote_winner=False)
    assert board["candidates"] == []
    assert len(alerted) == 1
    assert alerted[0]["n_active"] == 2        # 2 active specs but 0 candidates
    assert cw == [0]


def test_run_select_only_no_inert_alert_when_base_present(monkeypatch):
    # The base champion-arch alone in the pool is a candidate — NOT inert.
    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "v3.0-meta", raising=False)
    monkeypatch.setattr(mz, "send_zoo_digest_email", lambda *a, **k: True)
    alerted = []
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: alerted.append(k))
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: None)

    # Only the base champion-arch is registered (label == MODEL_VERSION_LABEL).
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        "predictor/registry/base-v/manifest.json": _mk_manifest(21, 0.12, True),
    })
    monkeypatch.setattr(reg, "list_versions",
                        lambda s3c, b, stage=None: (
                            [{"version_id": "base-v", "model_version": "v3.0-meta",
                              "date": "2026-06-13"}] if stage != "champion" else []))
    board = mz.run_select_only("bkt", date_str="2026-06-13", s3=s3,
                               specs=_SPECS, auto_promote_winner=False)
    cand_ids = {c["spec_id"] for c in board["candidates"]}
    assert "champion-arch" in cand_ids
    assert alerted == []                       # base present → not inert
