"""Tests for the model-zoo driver (L4488c).

Pins: spec resolution (active / retired / missing); the override allowlist
(disallowed keys fail loud); the save/restore context (sets cfg, restores on
exit AND on exception, removes a previously-absent attr); train_spec applies the
overrides while the injected train_fn runs and defaults the version label; and
train_all_active iterates active specs, skips retired, and continues past a
single failure.
"""
from __future__ import annotations

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
