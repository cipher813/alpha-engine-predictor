"""Tests for the nousergon_lib.phase_registry adoption in train_handler
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
from nousergon_lib.phase_registry import PhaseRegistry


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


# ── _write_training_summary fail-loud contract (config#1234) ─────────────────


def test_write_training_summary_reraises_on_s3_put_failure(monkeypatch):
    """The training-summary write is the training SSOT (executor/dashboard read
    it; the meta_training phase records it as its artifact). A swallowed write
    failure is a ghost success — so a genuine S3 PUT error must RE-RAISE
    (config#1234, mirroring crucible-research#312)."""
    class _PutBoom(_FakeS3):
        def put_object(self, Bucket, Key, Body, **kw):
            raise self._err("AccessDenied")

    fake = _PutBoom()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(train_handler, "_annotate_subsample_noise_floor", lambda *a, **k: None)

    with pytest.raises(ClientError):
        train_handler._write_training_summary(
            {"promoted": True}, "alpha-engine-research", "2026-06-13",
        )


def test_write_training_summary_annotation_failure_is_non_fatal(monkeypatch):
    """The prior-cycle subsample-noise-floor annotation is a best-effort READ of
    last cycle's latest.json (a percent-change reference only). A failure there
    must NOT block persisting the current (load-bearing) summary — the SSOT write
    still happens."""
    fake = _FakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)

    def _boom(*a, **k):
        raise RuntimeError("prior-cycle latest.json unreadable")
    monkeypatch.setattr(train_handler, "_annotate_subsample_noise_floor", _boom)

    train_handler._write_training_summary(
        {"promoted": True, "test_ic": 0.06}, "alpha-engine-research", "2026-06-13",
    )

    # Dated + latest SSOT both written despite the annotation failure.
    assert "predictor/metrics/training_summary_2026-06-13.json" in fake.objects
    assert "predictor/metrics/training_summary_latest.json" in fake.objects


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


# ── config#1170 — spec-level (not date-level) phase markers ───────────────────
#
# The bug: phase markers were keyed at DATE granularity
# (`predictor/{date}/.phases/meta_training.json`, ONE file shared by every
# zoo spec). So a same-date rerun skipped ALL specs (any spec's marker
# auto-skipped every spec) and a FAILED spec couldn't retry same-day (it saw a
# sibling's success marker). The fix namespaces the marker prefix + the
# meta_training reload artifact per challenger spec; the base champion keeps the
# spec-less path. These tests assert the per-spec idempotency the fix buys.


def test_spec_namespace_none_for_base_champion(monkeypatch):
    """The base champion (label == config default) keeps the spec-less paths."""
    import config as cfg
    monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", train_handler._cfg_default_label(cfg))
    assert train_handler._spec_namespace() is None
    reg = train_handler._build_registry("b", "2026-06-20", dry_run=False, spec=None)
    assert reg.marker_prefix == "predictor"
    assert reg._marker_key("meta_training") == "predictor/2026-06-20/.phases/meta_training.json"
    assert (
        train_handler._training_summary_key("2026-06-20", None)
        == "predictor/metrics/training_summary_2026-06-20.json"
    )


def test_spec_namespace_per_challenger_label(monkeypatch):
    """A zoo challenger derives a filesystem-safe per-spec namespace from its
    MODEL_VERSION_LABEL, and the marker key lands UNDER that namespace."""
    import config as cfg
    monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "resid-mom-v1")
    ns = train_handler._spec_namespace()
    assert ns == "resid-mom-v1"
    reg = train_handler._build_registry("b", "2026-06-20", dry_run=False, spec=ns)
    assert reg._marker_key("meta_training") == (
        "predictor/model_zoo/resid-mom-v1/2026-06-20/.phases/meta_training.json"
    )
    assert train_handler._training_summary_key("2026-06-20", ns) == (
        "predictor/model_zoo/resid-mom-v1/metrics/training_summary_2026-06-20.json"
    )


def test_spec_namespace_slugifies_unsafe_label(monkeypatch):
    """An arbitrary spec label is reduced to a single safe S3 path segment."""
    import config as cfg
    monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "spec/with bad:chars")
    assert train_handler._spec_namespace() == "spec_with_bad_chars"


# ── the idempotency contract, end-to-end across two specs on ONE date ─────────


def _spec_reg(fake, spec):
    """A PhaseRegistry keyed exactly as _build_registry(spec=...) would key it."""
    prefix = f"predictor/model_zoo/{spec}" if spec else "predictor"
    return PhaseRegistry(
        date="2026-06-20", bucket="alpha-engine-research",
        marker_prefix=prefix, s3_client=fake,
    )


def _run_spec_meta_phase(fake, spec, *, monkeypatch, train_fn):
    """Mirror _main_impl's meta_training block for a NAMED spec: the marker +
    the reload artifact are both spec-namespaced (config#1170)."""
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    monkeypatch.setattr(train_handler, "_annotate_subsample_noise_floor", lambda *a, **k: None)
    reg = _spec_reg(fake, spec)
    summary_key = train_handler._training_summary_key("2026-06-20", spec)
    with reg.phase("meta_training", supports_auto_skip=True) as ctx:
        if ctx.skipped:
            result = train_handler._load_training_result(
                "alpha-engine-research", "2026-06-20", spec,
            )
            if result is None:
                raise RuntimeError("half-state")
            return result, "skipped"
        result = train_fn()  # may raise to simulate a spec FAILURE
        train_handler._write_training_summary(
            result, "alpha-engine-research", "2026-06-20", spec,
        )
        ctx.record_artifact(summary_key)
        return result, "trained"


def test_same_date_rerun_reruns_only_incomplete_specs(monkeypatch):
    """Same-date rerun: a completed spec auto-skips (and is NOT re-trained),
    while a spec that never ran on the first pass DOES train on the rerun.

    Pre-fix this was the headline bug: spec A's date-level marker made spec B
    auto-skip even though B never ran.
    """
    fake = _FakeS3()

    # First pass: only spec A trains + completes (artifact present).
    a_calls = []
    _, mode_a = _run_spec_meta_phase(
        fake, "spec-a", monkeypatch=monkeypatch,
        train_fn=lambda: a_calls.append(1) or {"promoted": True, "test_ic": 0.06},
    )
    assert mode_a == "trained" and a_calls == [1]
    fake.artifacts.add(train_handler._training_summary_key("2026-06-20", "spec-a"))

    # Rerun (same date): A must auto-skip (already done), B must still train
    # because B has its OWN marker namespace — A's marker no longer touches it.
    a_calls2, b_calls = [], []
    _, mode_a2 = _run_spec_meta_phase(
        fake, "spec-a", monkeypatch=monkeypatch,
        train_fn=lambda: a_calls2.append(1) or {"x": 1},
    )
    _, mode_b = _run_spec_meta_phase(
        fake, "spec-b", monkeypatch=monkeypatch,
        train_fn=lambda: b_calls.append(1) or {"promoted": False, "test_ic": 0.02},
    )
    assert mode_a2 == "skipped" and a_calls2 == [], "completed spec must not double-run"
    assert mode_b == "trained" and b_calls == [1], "a never-run spec must NOT be skipped"


def test_failed_spec_can_retry_same_day(monkeypatch):
    """A spec that FAILED on the first pass can retry same-day — a sibling's
    success marker must not short-circuit it.

    Pre-fix: spec A's success wrote the shared date-level marker, so failed
    spec B saw it and skipped on retry instead of re-training.
    """
    fake = _FakeS3()

    # Spec A succeeds (writes its marker + artifact).
    _run_spec_meta_phase(
        fake, "spec-a", monkeypatch=monkeypatch,
        train_fn=lambda: {"promoted": True, "test_ic": 0.06},
    )
    fake.artifacts.add(train_handler._training_summary_key("2026-06-20", "spec-a"))

    # Spec B FAILS on its first run (train raises) — its marker is status=error,
    # no artifact written.
    with pytest.raises(RuntimeError, match="boom"):
        _run_spec_meta_phase(
            fake, "spec-b", monkeypatch=monkeypatch,
            train_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    # Same-day retry of B: it must RE-TRAIN (its error marker is not status=ok,
    # and A's success marker is in a different namespace).
    b_calls = []
    _, mode_b = _run_spec_meta_phase(
        fake, "spec-b", monkeypatch=monkeypatch,
        train_fn=lambda: b_calls.append(1) or {"promoted": False, "test_ic": 0.03},
    )
    assert mode_b == "trained" and b_calls == [1], "a failed spec must retry same-day"


def test_completed_spec_never_double_runs_on_rerun(monkeypatch):
    """A completed spec auto-skips on every subsequent same-date rerun and
    reloads ITS OWN summary (never a sibling's)."""
    fake = _FakeS3()
    _run_spec_meta_phase(
        fake, "spec-a", monkeypatch=monkeypatch,
        train_fn=lambda: {"promoted": True, "test_ic": 0.061},
    )
    fake.artifacts.add(train_handler._training_summary_key("2026-06-20", "spec-a"))
    # A different spec also completes with a DIFFERENT result.
    _run_spec_meta_phase(
        fake, "spec-b", monkeypatch=monkeypatch,
        train_fn=lambda: {"promoted": False, "test_ic": 0.019},
    )
    fake.artifacts.add(train_handler._training_summary_key("2026-06-20", "spec-b"))

    calls = []
    res_a, mode_a = _run_spec_meta_phase(
        fake, "spec-a", monkeypatch=monkeypatch,
        train_fn=lambda: calls.append("a") or {"x": 1},
    )
    res_b, mode_b = _run_spec_meta_phase(
        fake, "spec-b", monkeypatch=monkeypatch,
        train_fn=lambda: calls.append("b") or {"x": 1},
    )
    assert mode_a == "skipped" and mode_b == "skipped"
    assert calls == [], "no completed spec may re-train on a same-date rerun"
    # Each reloads its OWN summary — not a sibling's (consequence #3).
    assert res_a["test_ic"] == 0.061
    assert res_b["test_ic"] == 0.019
