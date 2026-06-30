"""Tests that train_handler.main() resolves and threads the TrainingIOSpec.

PR7-step-7b: ``main(shadow_basis="crsp")`` (or ``CRSP_SHADOW_ENABLED=true``)
must build a shadow spec and pass it to ``_main_impl``; the default call must
pass a live spec. We capture the spec by monkeypatching ``_main_impl`` so no
real training / ArcticDB / S3 runs.
"""

from __future__ import annotations

import pytest

from training import train_handler


@pytest.fixture
def capture_io(monkeypatch):
    """Replace ``_main_impl`` with a capture stub; return the captured spec."""
    captured = {}

    def _fake_main_impl(bucket, **kwargs):
        captured["io"] = kwargs.get("io")
        captured["kwargs"] = kwargs
        return {"captured": True}

    monkeypatch.setattr(train_handler, "_main_impl", _fake_main_impl)
    # Ensure no ambient env flips the default-live path.
    monkeypatch.delenv("CRSP_SHADOW_ENABLED", raising=False)
    monkeypatch.delenv("SHADOW_BASIS", raising=False)
    monkeypatch.delenv("PREDICTOR_DEFER_TRAINING_EMAIL", raising=False)
    return captured


def test_default_call_threads_live_spec(capture_io):
    train_handler.main(bucket="b", dry_run=True)
    io = capture_io["io"]
    assert io is not None
    assert io.is_shadow is False
    assert io.universe_lib == "universe"
    assert io.close_col == "Close"
    assert io.allow_live_promote is True


def test_explicit_shadow_basis_threads_shadow_spec(capture_io):
    train_handler.main(bucket="b", dry_run=True, shadow_basis="crsp")
    io = capture_io["io"]
    assert io.is_shadow is True
    assert io.universe_lib == "universe_crsp"
    assert io.close_col == "total_return_close"
    # The three independent promotion guards are all off.
    assert io.allow_live_promote is False
    assert io.register_in_zoo is False
    assert io.write_side_artifacts is False
    # Outputs isolated under the shadow prefix.
    assert io.weights_prefix == "predictor/weights_shadow/crsp/"


def test_env_flag_threads_shadow_spec(capture_io, monkeypatch):
    monkeypatch.setenv("CRSP_SHADOW_ENABLED", "true")
    train_handler.main(bucket="b", dry_run=True)
    io = capture_io["io"]
    assert io.is_shadow is True
    assert io.shadow_basis == "crsp"
