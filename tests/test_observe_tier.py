"""Tests for the DORMANT observe-tier infrastructure + A1 effective-N DSR.

config#671/#673/#702/#1052. The Option-B promote-to-observe tier is RETIRED for
the core loop (config#1052 relative-best: a beats-champion challenger PROMOTES, it
does not enter an observe soak). ``select_winner``/``run_rotation_and_select`` no
longer flag or register observe-eligibles (see test_model_zoo.py for the relative-
best eligibility tests). The registry ``observe`` stage + ``register_to_observe``
are RETAINED as dormant, additive infrastructure for a possible future explicit
observe soak; these tests pin that they still behave correctly (stage-only patch,
never copies into live weights, never demotes a champion) and that the shadow
runner still enumerates both stages. A1 effective-N DSR (tightens, never loosens)
is retained observability and is pinned here too.
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


# ── observe tier COLLAPSED: select_winner emits no observe fields ─────────────


def test_select_winner_emits_no_observe_fields(monkeypatch):
    # config#1052: the observe tier is retired for the core loop. A DSR-gate-failing
    # challenger that beats the champion now PROMOTES (relative-best) rather than
    # being flagged observe_eligible — so neither the per-candidate observe_eligible
    # flag nor the leaderboard observe_eligible_version_ids plumbing exists anymore.
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        # beats champ (0.20 > 0.10+margin), DSR gate FAILS → still PROMOTES now.
        "predictor/registry/nm-v/manifest.json": _mk_manifest(
            21, 0.20, False, registry_dsr=0.85, dsr=0.85),
    })
    trained = [{"spec_id": "nm", "version_id": "nm-v", "model_version": "spec-nm"}]
    board = mz.select_winner(s3, "bkt", trained=trained, margin=0.01)
    c = board["candidates"][0]
    assert c["reason"] == "eligible"                       # promotes on relative-best
    assert c["eligible"] is True
    assert "observe_eligible" not in c                     # flag removed
    assert c["dsr_gate_pass"] is False                     # DSR gate surfaced, not blocking
    assert board["winner_version_id"] == "nm-v"
    assert "observe_eligible_version_ids" not in board     # plumbing removed


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
