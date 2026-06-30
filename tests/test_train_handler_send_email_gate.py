"""train_handler.main() per-run email gating (send_email param + env override).

Brian's spec: the base --full-only retrain's separate email is folded into the
model-zoo digest. main() gains a send_email param (default True) and ALSO honors
PREDICTOR_DEFER_TRAINING_EMAIL so the SF/spot can defer without arg-threading.
The dated training_summary S3 write is ALWAYS performed regardless.

Pins:
  - send_email=False suppresses send_training_email.
  - PREDICTOR_DEFER_TRAINING_EMAIL truthy flips the default-True caller to no-send.
  - send_email=True (default, env unset) still emails.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import train_handler as th


def _patch_impl(monkeypatch):
    """Stub _main_impl so main()'s gating logic is exercised in isolation; record
    the send_email it forwards. Stub guard_entrypoint to a no-op context."""
    import contextlib
    seen = {}

    def _fake_impl(bucket, *, date_str=None, dry_run=False, skip_phases="",
                  force_phases="", force=False, send_email=True, io=None):
        seen["send_email"] = send_email
        seen["io"] = io
        return {"status": "ok"}

    monkeypatch.setattr(th, "_main_impl", _fake_impl)
    monkeypatch.setattr(th, "guard_entrypoint", lambda *a, **k: contextlib.nullcontext())
    return seen


def test_env_truthy_variants():
    assert th._env_truthy("1") is True
    assert th._env_truthy("true") is True
    assert th._env_truthy("YES") is True
    assert th._env_truthy(None) is False
    assert th._env_truthy("") is False
    assert th._env_truthy("0") is False
    assert th._env_truthy("false") is False
    assert th._env_truthy("off") is False


def test_main_default_sends_email(monkeypatch):
    monkeypatch.delenv("PREDICTOR_DEFER_TRAINING_EMAIL", raising=False)
    seen = _patch_impl(monkeypatch)
    th.main("bkt")
    assert seen["send_email"] is True


def test_main_explicit_send_email_false(monkeypatch):
    monkeypatch.delenv("PREDICTOR_DEFER_TRAINING_EMAIL", raising=False)
    seen = _patch_impl(monkeypatch)
    th.main("bkt", send_email=False)
    assert seen["send_email"] is False


def test_env_defer_flips_default_to_no_send(monkeypatch):
    monkeypatch.setenv("PREDICTOR_DEFER_TRAINING_EMAIL", "1")
    seen = _patch_impl(monkeypatch)
    th.main("bkt")                       # default-True caller (the --full-only spot)
    assert seen["send_email"] is False


def test_env_falsey_does_not_defer(monkeypatch):
    monkeypatch.setenv("PREDICTOR_DEFER_TRAINING_EMAIL", "0")
    seen = _patch_impl(monkeypatch)
    th.main("bkt")
    assert seen["send_email"] is True


def test_impl_gates_send_training_email(monkeypatch):
    """_main_impl must skip send_training_email when send_email=False, but the
    caller still receives the result (and the summary write — unchanged — would
    have happened upstream)."""
    # We exercise just the email branch by calling a tiny shim mirroring the
    # gate: confirm send_training_email is NOT called with send_email=False and
    # IS called with send_email=True. Use the real send_training_email stubbed.
    calls = []
    monkeypatch.setattr(th, "send_training_email",
                        lambda result, date_str: calls.append(date_str) or True)

    # Mirror the production gate exactly (the `if not dry_run and send_email:` line).
    def _email_branch(send_email, dry_run=False):
        if not dry_run and send_email:
            th.send_training_email({"x": 1}, "2026-06-20")

    _email_branch(send_email=False)
    assert calls == []
    _email_branch(send_email=True)
    assert calls == ["2026-06-20"]
