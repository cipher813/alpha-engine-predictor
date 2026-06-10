"""Tests for the alpha_engine_lib.phase_registry adoption in train_handler
(L4528 — predictor is the 3rd consumer of the lib phase framework, after the
backtester + alpha-engine-data).

Focus on the pieces that don't need a full 90-min training run:
  * `_build_registry` — None in dry-run; env-var recovery controls; hard caps
    read from config (best-effort);
  * `_load_training_result` — reloads the persisted summary; 404 → None;
    non-404 → raise (fail loud);
  * the `meta_training` auto-skip-reload contract driven through a real
    PhaseRegistry + in-memory fake S3: a 2nd run with weights+summary present
    skips the retrain and reloads `result`; a missing summary (half-state)
    forces a re-run.
"""

from __future__ import annotations

import io
import json

import pytest
from botocore.exceptions import ClientError

from training import train_handler
from alpha_engine_lib.phase_registry import PhaseRegistry


# ── fake S3 (markers in `objects`, artifact existence in `artifacts`) ─────────


class _FakeS3:
    def __init__(self, objects=None, artifacts=None):
        self.objects: dict[str, bytes] = dict(objects or {})
        self.artifacts: set[str] = set(artifacts or [])

    @staticmethod
    def _err(code):
        return ClientError({"Error": {"Code": code, "Message": "x"}}, "Op")

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._err("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()
        return {}

    def head_object(self, Bucket, Key):
        if Key in self.artifacts:
            return {"ContentLength": 1}
        raise self._err("404")


# ── _build_registry ──────────────────────────────────────────────────────────


def test_build_registry_none_in_dry_run():
    assert train_handler._build_registry("b", "2026-06-13", dry_run=True) is None


def test_build_registry_real_run_uses_predictor_prefix():
    reg = train_handler._build_registry("b", "2026-06-13", dry_run=False)
    assert isinstance(reg, PhaseRegistry)
    assert reg.marker_prefix == "predictor"
    assert reg.date == "2026-06-13"


def test_build_registry_env_recovery_controls(monkeypatch):
    monkeypatch.setenv("PREDICTOR_FORCE_PHASES", "meta_training")
    monkeypatch.setenv("PREDICTOR_SKIP_PHASES", "risk_model_persist")
    reg = train_handler._build_registry("b", "2026-06-13", dry_run=False)
    assert "meta_training" in reg._force_phases
    assert "risk_model_persist" in reg._explicit_skip


def test_build_registry_args_beat_env(monkeypatch):
    monkeypatch.setenv("PREDICTOR_FORCE", "1")
    reg = train_handler._build_registry(
        "b", "2026-06-13", dry_run=False, force=False, force_phases="meta_training",
    )
    # explicit force_phases arg wins; env PREDICTOR_FORCE still ORs in force_all.
    assert "meta_training" in reg._force_phases


# ── _load_training_result ────────────────────────────────────────────────────


def test_load_training_result_reads_summary(monkeypatch):
    payload = {"promoted": True, "test_ic": 0.061}
    fake = _FakeS3(objects={
        "predictor/metrics/training_summary_2026-06-13.json": json.dumps(payload).encode(),
    })
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    out = train_handler._load_training_result("b", "2026-06-13")
    assert out == payload


def test_load_training_result_missing_returns_none(monkeypatch):
    monkeypatch.setattr("boto3.client", lambda *a, **k: _FakeS3())
    assert train_handler._load_training_result("b", "2026-06-13") is None


def test_load_training_result_non_404_raises(monkeypatch):
    class _Boom(_FakeS3):
        def get_object(self, Bucket, Key):
            raise self._err("AccessDenied")

    monkeypatch.setattr("boto3.client", lambda *a, **k: _Boom())
    with pytest.raises(ClientError):
        train_handler._load_training_result("b", "2026-06-13")


# ── meta_training auto-skip-reload contract ──────────────────────────────────
#
# Drive the exact pattern _main_impl uses for the meta_training phase against a
# real PhaseRegistry, isolating it from the 90-min trainer.

_SUMMARY_KEY = "predictor/metrics/training_summary_2026-06-13.json"


def _reg(fake):
    return PhaseRegistry(
        date="2026-06-13", bucket="alpha-engine-research",
        marker_prefix="predictor", s3_client=fake,
    )


def _run_meta_phase(reg, fake, *, monkeypatch, train_fn):
    """Mirror _main_impl's meta_training block: skip→reload, else train+record."""
    summary = {"promoted": True, "test_ic": 0.06}
    # _load_training_result + _write_training_summary use boto3.client → fake.
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(train_handler, "_annotate_subsample_noise_floor", lambda *a, **k: None)

    with reg.phase("meta_training", supports_auto_skip=True) as ctx:
        if ctx.skipped:
            result = train_handler._load_training_result("alpha-engine-research", "2026-06-13")
            if result is None:
                raise RuntimeError("half-state")
            return result, "skipped"
        result = train_fn()
        train_handler._write_training_summary(result, "alpha-engine-research", "2026-06-13")
        ctx.record_artifact(_SUMMARY_KEY)
        return result, "trained"


def test_meta_training_trains_then_records_summary(monkeypatch):
    fake = _FakeS3()
    calls = []
    res, mode = _run_meta_phase(
        _reg(fake), fake, monkeypatch=monkeypatch,
        train_fn=lambda: calls.append(1) or {"promoted": True, "test_ic": 0.06},
    )
    assert mode == "trained" and calls == [1]
    # Summary persisted + marker records it as the artifact.
    assert _SUMMARY_KEY in fake.objects
    marker = json.loads(fake.objects["predictor/2026-06-13/.phases/meta_training.json"])
    assert marker["artifact_keys"] == [_SUMMARY_KEY]


def test_meta_training_auto_skips_and_reloads_when_summary_present(monkeypatch):
    # First run trains + writes the summary; flag the summary present, then a
    # fresh registry must skip the retrain and reload result from the summary.
    fake = _FakeS3()
    _run_meta_phase(
        _reg(fake), fake, monkeypatch=monkeypatch,
        train_fn=lambda: {"promoted": True, "test_ic": 0.06},
    )
    fake.artifacts.add(_SUMMARY_KEY)

    calls = []
    res, mode = _run_meta_phase(
        _reg(fake), fake, monkeypatch=monkeypatch,
        train_fn=lambda: calls.append(1) or {"x": 1},
    )
    assert mode == "skipped", "retrain must be auto-skipped"
    assert calls == [], "the 90-min trainer must NOT run on a resume"
    assert res["promoted"] is True and res["test_ic"] == 0.06  # reloaded


def test_meta_training_half_state_forces_rerun(monkeypatch):
    # Marker ok but summary artifact missing (crash between train + summary):
    # L4524 invalidates the marker → the phase re-runs rather than reloading.
    fake = _FakeS3()
    _run_meta_phase(
        _reg(fake), fake, monkeypatch=monkeypatch,
        train_fn=lambda: {"promoted": True},
    )
    # Do NOT mark the summary present → head_object 404s.
    calls = []
    res, mode = _run_meta_phase(
        _reg(fake), fake, monkeypatch=monkeypatch,
        train_fn=lambda: calls.append(1) or {"promoted": True},
    )
    assert mode == "trained" and calls == [1], "missing summary must force a re-train"
