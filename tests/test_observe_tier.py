"""Tests for the observe tier (config #671 / #673 / #702 — Option B + A1 + C).

Pins:
  - registry ``observe`` stage + ``register_to_observe`` (challenger→observe is a
    stage-only patch, NEVER copies into the live weights prefix, never demotes a
    champion);
  - shadow runner enumerates BOTH challenger + observe stages;
  - ``select_winner`` ``observe_eligible`` logic (clears registry bar + beats
    champion + right horizon, but NOT full-gate);
  - ``run_rotation_and_select`` registers observe-eligibles to observe WITHOUT
    promoting the champion (independent of auto-promote);
  - A1 effective-N TIGHTENS DSR (never loosens) + surfaces the 'needs ~Y' bar.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
import model.registry as reg
from training import model_zoo as mz
from training.deflated_sharpe import deflated_sharpe_ratio


# ── shared fakes ─────────────────────────────────────────────────────────────


class _FakeS3:
    """get_object reads ``objects``; put_object records to ``puts``; copy_object
    records to ``copies``. Missing keys raise (the code catches)."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}
        self.copies = []

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        body = json.dumps(self.objects[Key]).encode()
        return {"Body": io.BytesIO(body), "ContentType": "application/json"}

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        self.puts[Key] = Body
        # keep objects consistent so a subsequent get sees the patch
        try:
            self.objects[Key] = json.loads(Body)
        except Exception:
            pass

    def copy_object(self, Bucket, Key, CopySource):  # noqa: N803
        self.copies.append((CopySource.get("Key"), Key))


def _mk_manifest(forward_days, cpcv_ic, gate_pass, *, dsr=None, registry_dsr=None,
                 ics=None, n_paths=None):
    """A manifest with the promotion-stat structure select_winner reads.

    ``registry_dsr`` (when set) drives _registry_bar_pass via the overfit DSR;
    ``gate_pass`` drives the FULL gate (both downside + overfit pass flags)."""
    overfit = {"passes_overfit_gate": gate_pass}
    if dsr is not None:
        overfit["dsr"] = dsr
    cpcv = {"mean_ic": cpcv_ic} if cpcv_ic is not None else {"status": "error"}
    if ics is not None:
        cpcv["ics"] = ics
    if n_paths is not None:
        cpcv["n_backtest_paths"] = n_paths
    return {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": cpcv,
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": gate_pass or (registry_dsr is not None)},
            "overfit": overfit,
        },
    }


# ── registry: observe stage + register_to_observe ────────────────────────────


def test_valid_stages_includes_observe():
    assert "observe" in reg.VALID_STAGES
    assert reg.SHADOW_STAGES == ("challenger", "observe")


def test_register_to_observe_patches_stage_no_live_copy():
    s3 = _FakeS3({
        "predictor/registry/c-v/_lineage.json": {
            "version_id": "c-v", "stage": "challenger",
        },
    })
    res = reg.register_to_observe(s3, "bkt", "c-v")
    assert res["changed"] is True
    # stage patched to observe
    patched = json.loads(s3.puts["predictor/registry/c-v/_lineage.json"])
    assert patched["stage"] == "observe"
    # NEVER copies into the live weights prefix — zero live allocation.
    assert s3.copies == []
    assert not any(k.startswith(reg.DEFAULT_SOURCE_PREFIX) for k in s3.puts)


def test_register_to_observe_idempotent():
    s3 = _FakeS3({
        "predictor/registry/o-v/_lineage.json": {"version_id": "o-v", "stage": "observe"},
    })
    res = reg.register_to_observe(s3, "bkt", "o-v")
    assert res["changed"] is False
    assert "predictor/registry/o-v/_lineage.json" not in s3.puts  # no re-write


def test_register_to_observe_never_demotes_champion():
    s3 = _FakeS3({
        "predictor/registry/champ-v/_lineage.json": {"version_id": "champ-v", "stage": "champion"},
    })
    res = reg.register_to_observe(s3, "bkt", "champ-v")
    assert res["changed"] is False
    assert res["stage"] == "champion"
    assert s3.puts == {}  # champion lineage untouched


def test_register_to_observe_missing_bundle_raises():
    s3 = _FakeS3({})
    with pytest.raises(reg.RegistryError, match="no registry bundle"):
        reg.register_to_observe(s3, "bkt", "ghost-v")


# ── shadow runner enumerates both stages ─────────────────────────────────────


def test_shadow_runner_lists_both_challenger_and_observe(monkeypatch):
    import inference.stages.shadow_versions as sv

    calls = []

    def _fake_list(s3c, bucket, *, stage=None, **k):
        calls.append(stage)
        return {
            "challenger": [{"version_id": "c-1"}],
            "observe": [{"version_id": "o-1"}, {"version_id": "c-1"}],  # dup c-1
        }.get(stage, [])

    monkeypatch.setattr(reg, "list_versions", _fake_list)

    class _Ctx:
        dry_run = False
        bucket = "bkt"
        explicit_tickers = None
        date_str = "2026-06-13"

        def near_timeout(self):
            return True  # stop immediately so no real inference runs

    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 10, raising=False)
    monkeypatch.setattr("boto3.client", lambda *a, **k: object())
    sv.run(_Ctx())
    # both stages enumerated; the duplicate c-1 across stages is de-duped.
    assert set(calls) == {"challenger", "observe"}


# ── select_winner: observe_eligible logic ────────────────────────────────────


def test_select_winner_marks_observe_eligible_near_miss(monkeypatch):
    # A challenger that BEATS the champion + clears the registry bar (dsr 0.85 >=
    # 0.80) but FAILS the full gate (passes_overfit_gate False) → observe-eligible,
    # NOT promotion-eligible.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "WF_DSR_REGISTRY_THRESHOLD", 0.80, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        # near-miss: beats champ (0.20 > 0.10+margin), registry bar passes via
        # downside+dsr=0.85, but full gate (overfit) fails.
        "predictor/registry/nm-v/manifest.json": _mk_manifest(
            21, 0.20, False, registry_dsr=0.85, dsr=0.85),
    })
    trained = [{"spec_id": "nm", "version_id": "nm-v", "model_version": "spec-nm"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    c = board["candidates"][0]
    assert c["reason"] == "near_miss_below_promotion_bar"
    assert c["beats_champion_by_margin"] is True
    assert c["registry_bar_pass"] is True
    assert c["observe_eligible"] is True
    assert c["eligible"] is False
    assert board["winner_version_id"] is None              # no full-gate winner
    assert board["observe_eligible_version_ids"] == ["nm-v"]


def test_select_winner_full_gate_winner_is_not_observe(monkeypatch):
    # A full-gate-pass winner is promotion-eligible — NOT observe (observe is the
    # near-miss tier). observe_eligible must be False for it.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "WF_DSR_REGISTRY_THRESHOLD", 0.80, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        "predictor/registry/w-v/manifest.json": _mk_manifest(
            21, 0.20, True, registry_dsr=0.97, dsr=0.97),
    })
    trained = [{"spec_id": "w", "version_id": "w-v", "model_version": "spec-w"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    c = board["candidates"][0]
    assert c["eligible"] is True
    assert c["observe_eligible"] is False
    assert board["winner_version_id"] == "w-v"
    assert board["observe_eligible_version_ids"] == []


def test_select_winner_loser_not_observe_eligible(monkeypatch):
    # Clears the registry bar but does NOT beat the champion → not observe-eligible
    # (observe requires demonstrated CPCV edge over the champion).
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "WF_DSR_REGISTRY_THRESHOLD", 0.80, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.20, True),
        "predictor/registry/lo-v/manifest.json": _mk_manifest(
            21, 0.10, False, registry_dsr=0.85, dsr=0.85),  # below champ
    })
    trained = [{"spec_id": "lo", "version_id": "lo-v", "model_version": "spec-lo"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    c = board["candidates"][0]
    assert c["beats_champion_by_margin"] is False
    assert c["observe_eligible"] is False
    assert board["observe_eligible_version_ids"] == []


# ── run_rotation_and_select: registers observe WITHOUT promoting champion ─────


def _rotation_fixture(monkeypatch, *, auto_promote):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "WF_DSR_REGISTRY_THRESHOLD", 0.80, raising=False)
    specs = [{"id": "nm", "status": "active", "model_version_label": "spec-nm",
              "overrides": {}}]
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        # near-miss challenger: beats champ, registry bar passes, full gate fails.
        "predictor/registry/nm-v/manifest.json": _mk_manifest(
            21, 0.20, False, registry_dsr=0.85, dsr=0.85),
        "predictor/registry/nm-v/_lineage.json": {"version_id": "nm-v", "stage": "challenger"},
    })
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: (
        [{"version_id": "nm-v", "model_version": "spec-nm", "date": "2026-06-13"}]
        if stage == "challenger" else []
    ))
    promotes = []
    monkeypatch.setattr(reg, "promote_to_champion",
                        lambda s3c, b, vid, **k: promotes.append(vid))
    observes = []
    _orig = reg.register_to_observe
    monkeypatch.setattr(reg, "register_to_observe",
                        lambda s3c, b, vid, **k: (observes.append(vid), _orig(s3c, b, vid, **k))[1])

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=specs, train_fn=_fake_train,
        registered_versions=[], s3=s3, auto_promote_winner=auto_promote,
        date_str="2026-06-13",
    )
    return board, promotes, observes, s3


def test_rotation_registers_observe_does_not_promote_observe_mode(monkeypatch):
    board, promotes, observes, s3 = _rotation_fixture(monkeypatch, auto_promote=False)
    assert board["winner_version_id"] is None              # no full-gate winner
    assert board["promoted"] is None                        # champion NOT promoted
    assert promotes == []                                   # promote_to_champion never called
    assert observes == ["nm-v"]                             # moved to observe tier
    assert board["moved_to_observe"] == ["nm-v"]
    # stage actually patched to observe in the registry
    assert json.loads(s3.puts["predictor/registry/nm-v/_lineage.json"])["stage"] == "observe"


def test_rotation_registers_observe_even_under_cutover(monkeypatch):
    # Observe is shadow-only / zero allocation, so it runs independent of the
    # auto-promote flag. A near-miss still moves to observe; the champion is NOT
    # promoted (the near-miss is not a full-gate winner).
    board, promotes, observes, _ = _rotation_fixture(monkeypatch, auto_promote=True)
    assert board["mode"] == "cutover"
    assert board["winner_version_id"] is None
    assert promotes == []                                   # near-miss is not promoted
    assert observes == ["nm-v"]


# ── A1: effective-N tightens DSR + surfaces the 'needs ~Y' bar ────────────────


def test_effective_n_tightens_dsr_never_loosens():
    rng = np.random.default_rng(7)
    ic = rng.normal(0.08, 0.25, 60)
    full = deflated_sharpe_ratio(ic, n_trials=10)
    eff = deflated_sharpe_ratio(ic, n_trials=10, n_eff=3)
    assert full["n_used"] == 60
    assert eff["n_used"] == 3                               # effective count fed the stat
    assert eff["dsr"] <= full["dsr"]                        # tighter (more honest)
    assert eff["ic_ir"] == full["ic_ir"]                    # point estimate unchanged


def test_effective_n_surfaces_needed_ic_ir():
    ic = np.full(50, 0.1) + np.random.default_rng(0).normal(0, 0.2, 50)
    out = deflated_sharpe_ratio(ic, n_trials=5, n_eff=2)
    # the IC-IR a series of this EFFECTIVE length would need to clear the bar is
    # surfaced (a positive, finite number) so 'too young' is explicit.
    assert out["ic_ir_needed_for_threshold"] > 0
    assert np.isfinite(out["ic_ir_needed_for_threshold"])


def test_effective_n_clamped_to_n_when_larger():
    # A caller passing n_eff > n must NOT loosen the gate — clamp to n.
    ic = np.random.default_rng(1).normal(0.1, 0.2, 30)
    out = deflated_sharpe_ratio(ic, n_trials=5, n_eff=999)
    assert out["n_used"] == 30


def test_effective_n_default_is_raw_n():
    ic = np.random.default_rng(2).normal(0.1, 0.2, 40)
    out = deflated_sharpe_ratio(ic, n_trials=5)
    assert out["n_used"] == 40 and out["n_eff"] is None
