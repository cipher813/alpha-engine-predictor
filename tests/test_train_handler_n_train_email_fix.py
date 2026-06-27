"""config#1083 — the training email must never crash on a fold missing n_train.

Before the fix, the v2 walk-forward fold table used ``f["n_train"]:,`` /
``f["ic"]`` direct-subscript, so a fold dict lacking ``n_train`` (or ``ic`` /
``fold`` / ``test_start`` / ``test_end``) raised KeyError and crashed the WHOLE
training email — the 2nd of the two botched model-zoo rotations. send_training_email
must now build successfully on a malformed fold (default-read every field).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import train_handler as th


def _result_with_fold_missing_n_train():
    """A v2 (non-meta) result whose walk-forward fold lacks n_train (and ic)."""
    return {
        "model_version": "v2.0-gbm",            # non-meta → the v2 fold-table branch
        "test_ic": 0.05,
        "rank_ic": 0.04,
        "ensemble_ic": 0.05,
        "ic_ir": 1.0,
        "promoted": False,
        "passes_ic_gate": False,
        "walk_forward": {
            "median_ic": 0.03,
            "pct_positive": 0.6,
            "passes_wf": True,
            "folds": [
                # a complete fold + a fold MISSING n_train and ic (the landmine)
                {"fold": 1, "test_start": "2026-01-01", "test_end": "2026-01-21",
                 "n_train": 1000, "ic": 0.04},
                {"fold": 2, "test_start": "2026-02-01", "test_end": "2026-02-21"},
            ],
        },
    }


def test_send_training_email_survives_fold_without_n_train(monkeypatch):
    captured = {}

    def _fake_send(subject, plain, *, recipients=None, html=None, sender=None, region=None):
        captured["subject"] = subject
        captured["html"] = html
        captured["plain"] = plain
        return True

    # Stub the SMTP/SES chokepoint so no real email fires; the point is that
    # building the body (which dereferences each fold) does not raise.
    import krepis.email_sender as es
    import config as cfg
    monkeypatch.setattr(es, "send_email", _fake_send)
    monkeypatch.setattr(cfg, "EMAIL_SENDER", "x@y.z", raising=False)
    monkeypatch.setattr(cfg, "EMAIL_RECIPIENTS", ["a@b.c"], raising=False)

    ok = th.send_training_email(_result_with_fold_missing_n_train(), "2026-06-13")
    assert ok is True
    # The malformed fold rendered with defaults rather than raising.
    assert "train=0" in captured["plain"]        # n_train defaulted to 0
    assert captured["html"] is not None
