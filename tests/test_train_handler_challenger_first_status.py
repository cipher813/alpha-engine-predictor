"""L4540: the training email must distinguish a gate-PASSING challenger-first
run (registered, NOT auto-promoted by design) from a true gate FAILURE.

Before the fix, `passes_ic_gate` held `promoted` (= gate_passed AND auto_promote),
so the 2026-06-06 run — which passed every gate but did not auto-promote — was
mislabeled "Meta OOS IC +0.2079 FAIL | Not promoted" (a recurrence of L1668).
The subject's PASS/FAIL is now the QUALITY verdict, and "Registered challenger"
is shown instead of "Not promoted" for the challenger-first case.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import train_handler


def _base_result(**over):
    r = {
        "model_version": "v3.0-meta",
        "meta_model_oos_ic": 0.2079,
        "test_ic": 0.4871,
        "mse_ic": 0.0,
        "val_ic": 0.4871,
        "rank_ic": None,
        "ensemble_ic": None,
        "ensemble_enabled": False,
        "ic_ir": 1.2,
        "elapsed_s": 91.0,
        "n_train": 1043,
        "ic_positive_20": 12,
        "feature_importance_top10": [],
        "promoted_mode": None,
        "calibration": {},
        "promotion_gate_detail": {},
    }
    r.update(over)
    return r


def _capture_subject(monkeypatch, result):
    captured: dict = {}

    def _fake_send(subject, body, **kwargs):
        captured["subject"] = subject
        captured["body"] = body
        return True

    monkeypatch.setattr("krepis.email_sender.send_email", _fake_send)
    monkeypatch.setattr(cfg, "EMAIL_SENDER", "sender@example.com")
    monkeypatch.setattr(cfg, "EMAIL_RECIPIENTS", ["ops@example.com"])
    ok = train_handler.send_training_email(result, "2026-06-13")
    assert ok is True
    return captured


def test_challenger_first_gate_pass_is_not_labeled_fail(monkeypatch):
    # config#1052/#679: training is unconditionally challenger-first — every gate
    # passed, model registered as a challenger (promotion owned by the model-zoo
    # select_winner step), NOT a failure. `auto_promote_enabled` is the frozen-False
    # legacy contract field a current manifest carries.
    result = _base_result(
        promoted=False, gate_passed=True, passes_ic_gate=True,
        auto_promote_enabled=False,
        promotion_gate_detail={"promoted_blocker_reason": "challenger_first"},
    )
    cap = _capture_subject(monkeypatch, result)
    assert "PASS" in cap["subject"]
    assert "FAIL" not in cap["subject"]
    assert "Registered challenger" in cap["subject"]
    assert "Not promoted" not in cap["subject"]
    # Body promotion label reflects the registered-challenger state, not failure.
    assert "Registered challenger" in cap["body"]
    assert "IC gate failed" not in cap["body"]


def test_promoted_run_reads_pass_promoted(monkeypatch):
    # Backward-compat: a LEGACY pre-challenger-first archive manifest may carry
    # promoted=True / auto_promote_enabled=True. The email path still honors a
    # `promoted` result (e.g. a manual `model.registry --promote` restamp), so the
    # "Promoted" subject must still render when an old/restamped manifest is read.
    result = _base_result(
        promoted=True, gate_passed=True, passes_ic_gate=True,
        auto_promote_enabled=True, promoted_mode="meta",
    )
    cap = _capture_subject(monkeypatch, result)
    assert "PASS" in cap["subject"]
    assert "Promoted (meta)" in cap["subject"]
    assert "Registered challenger" not in cap["subject"]


def test_true_gate_failure_still_reads_fail(monkeypatch):
    # A genuine gate failure (e.g. output-distribution) must still read FAIL.
    result = _base_result(
        promoted=False, gate_passed=False, passes_ic_gate=False,
        auto_promote_enabled=True,
        promotion_gate_detail={"promoted_blocker_reason": "output_dist"},
    )
    cap = _capture_subject(monkeypatch, result)
    assert "FAIL" in cap["subject"]
    assert "Not promoted" in cap["subject"]
    assert "Registered challenger" not in cap["subject"]
    assert "output-distribution gate" in cap["body"]
