"""Tests for the champion/challenger Phase 1 shadow runner (L4469).

Covers the safety-critical contract: no-op when disabled / dry-run / supplemental
/ no-challengers; clone reuses shared data + swaps the weights prefix; each
challenger is written to its own shadow key; the time-guard stops mid-run; and a
single-challenger failure never aborts the others (the live path is untouched).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from inference.pipeline import PipelineContext
from inference.stages import shadow_versions as sv


class _FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body})


def _base_ctx(**over):
    ctx = PipelineContext()
    ctx.date_str = "2026-06-02"
    ctx.bucket = "bkt"
    ctx.dry_run = False
    ctx.start_ts = time.monotonic()  # else near_timeout() trips on start_ts=0
    ctx.soft_timeout_s = 780
    ctx.tickers = ["AAA", "BBB"]
    ctx.price_data = {"AAA": object(), "BBB": object()}
    ctx.macro = {"vix": 1.0}
    for k, v in over.items():
        setattr(ctx, k, v)
    return ctx


def test_clone_reuses_shared_and_swaps_prefix():
    ctx = _base_ctx()
    ctx.predictions = [{"ticker": "AAA"}]  # live results — must NOT carry over
    shadow = sv._clone_for_shadow(ctx, weights_prefix="predictor/registry/V/")
    assert shadow.weights_prefix_override == "predictor/registry/V/"
    assert shadow.tickers == ["AAA", "BBB"]
    assert shadow.price_data["AAA"] is ctx.price_data["AAA"]  # frames shared
    assert shadow.price_data is not ctx.price_data  # container copied
    assert shadow.predictions == []  # fresh result state


def test_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", False, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    sv.run(_base_ctx())
    assert fake.puts == []


def test_noop_in_dry_run(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    sv.run(_base_ctx(dry_run=True))
    assert fake.puts == []


def test_noop_on_supplemental_run(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    sv.run(_base_ctx(explicit_tickers=["AAA"]))
    assert fake.puts == []


def test_noop_when_no_challengers(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr("model.registry.list_versions", lambda *a, **k: [])
    sv.run(_base_ctx())
    assert fake.puts == []


def _patch_stages(monkeypatch, *, fail_on=None):
    """Patch load_model.run + run_inference.run; run_inference stamps preds."""
    import importlib

    lm = importlib.import_module("inference.stages.load_model")
    ri = importlib.import_module("inference.stages.run_inference")

    def _lm_run(ctx):
        if fail_on and ctx.weights_prefix_override and fail_on in ctx.weights_prefix_override:
            raise RuntimeError("boom")

    def _ri_run(ctx):
        ctx.predictions = [{"ticker": "AAA", "predicted_alpha": 0.01}]

    monkeypatch.setattr(lm, "run", _lm_run)
    monkeypatch.setattr(ri, "run", _ri_run)


def test_shadows_each_challenger(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 3, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch)
    sv.run(_base_ctx())

    keys = sorted(p["Key"] for p in fake.puts)
    assert keys == [
        "predictor/predictions_shadow/V1/2026-06-02.json",
        "predictor/predictions_shadow/V2/2026-06-02.json",
    ]
    body = json.loads(fake.puts[0]["Body"])
    assert body["shadow"] is True and body["n_predictions"] == 1


def test_max_n_caps_challengers(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 1, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch)
    sv.run(_base_ctx())
    assert len(fake.puts) == 1  # capped to max_n=1


def test_failure_continues_to_next_challenger(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 3, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch, fail_on="V1")  # V1 errors in load_model
    sv.run(_base_ctx())
    keys = [p["Key"] for p in fake.puts]
    assert keys == ["predictor/predictions_shadow/V2/2026-06-02.json"]


def test_time_guard_stops_run(monkeypatch):
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SHADOW_VERSIONS_MAX_N", 3, raising=False)
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: [{"version_id": "V1"}, {"version_id": "V2"}],
    )
    _patch_stages(monkeypatch)
    ctx = _base_ctx()
    monkeypatch.setattr(ctx, "near_timeout", lambda: True)  # already over budget
    sv.run(ctx)
    assert fake.puts == []  # nothing shadowed — guard fired before the first
