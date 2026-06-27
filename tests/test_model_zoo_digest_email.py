"""Model-zoo consolidated digest email + per-challenger email suppression.

Brian's spec: the model-zoo rotation sends ONE consolidated digest email
(champion scores, each challenger's scores, who was promoted) instead of a
per-challenger email storm, and the base --full-only retrain's email is folded
into the same digest (deferred via PREDICTOR_DEFER_TRAINING_EMAIL). Fallback: a
skipped/failed zoo must NOT go silent.

Pins:
  - train_spec's default trainer binds send_email=False so NO challenger emails.
  - send_zoo_digest_email builds champion + per-challenger + promotion sections
    and sends EXACTLY ONCE via the train_handler SMTP/SES chokepoint.
  - the digest is robust when a candidate is missing/errored fields.
  - run_rotation_and_select sends the digest exactly once on the real path; the
    digest never aborts the rotation on a send failure (no-silent-fails carve-out).
"""
from __future__ import annotations

import io
import json
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
]


class _FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode()),
                "ContentType": "application/json"}

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        self.puts[Key] = Body


def _mk_manifest(forward_days, cpcv_ic, gate_pass):
    return {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": {"mean_ic": cpcv_ic} if cpcv_ic is not None else {},
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": gate_pass},
            "overfit": {"passes_overfit_gate": gate_pass},
        },
    }


# ── 2a: per-challenger email suppression ─────────────────────────────────────


def test_train_spec_default_trainer_binds_send_email_false(monkeypatch):
    """The zoo's default train_fn (real train_handler.main) must be called with
    send_email=False so each challenger does NOT send its own training email."""
    seen = {}
    import training.train_handler as th

    def _fake_main(bucket, *, date_str=None, dry_run=False, send_email=True, **kw):
        seen["send_email"] = send_email
        return {"status": "ok", "model_version": cfg.MODEL_VERSION_LABEL
                if hasattr(cfg, "MODEL_VERSION_LABEL") else "x"}

    monkeypatch.setattr(th, "main", _fake_main)
    # train_fn=None → train_spec builds the partial over the real main.
    mz.train_spec("resid", "bkt", specs=_SPECS)
    assert seen["send_email"] is False


# ── 2b/2c: the consolidated digest ───────────────────────────────────────────


def _capture_digest_send(monkeypatch):
    captured = {}

    def _fake_send(subject, body, *, recipients=None, html=None, sender=None, region=None):
        captured["subject"] = subject
        captured["plain"] = body
        captured["html"] = html
        captured["n_calls"] = captured.get("n_calls", 0) + 1
        return True

    monkeypatch.setattr("krepis.email_sender.send_email", _fake_send)
    monkeypatch.setattr(cfg, "EMAIL_SENDER", "sender@example.com", raising=False)
    monkeypatch.setattr(cfg, "EMAIL_RECIPIENTS", ["ops@example.com"], raising=False)
    return captured


def test_send_zoo_digest_email_has_champion_challengers_promotion(monkeypatch):
    captured = _capture_digest_send(monkeypatch)
    s3 = _FakeS3({
        "predictor/metrics/training_summary_2026-06-20.json": {
            "model_version": "v3.0-meta", "meta_model_oos_ic": 0.21,
            "passes_ic_gate": True, "promoted": False,
        },
    })
    leaderboard = {
        "date": "2026-06-20", "mode": "cutover",
        "champion": {"forward_days": 21, "cpcv_mean_ic": 0.10},
        "margin": 0.01, "promote_min_ic": 0.0,
        "candidates": [
            {"spec_id": "resid", "cpcv_mean_ic": 0.20, "forward_days": 21,
             "passes_gate": True, "eligible": True, "reason": "eligible"},
            {"spec_id": "h60", "cpcv_mean_ic": 0.30, "forward_days": 60,
             "passes_gate": True, "eligible": False, "reason": "non_canonical_horizon"},
        ],
        "winner_version_id": "resid-v", "promoted": "resid-v",
        "reverted_from": "old-champ-v",
    }
    ok = mz.send_zoo_digest_email(leaderboard, "bkt", "2026-06-20", s3=s3)
    assert ok is True
    assert captured["n_calls"] == 1
    subj = captured["subject"]
    assert "Model-Zoo Rotation 2026-06-20" in subj
    assert "2 challengers" in subj
    assert "promoted: resid-v" in subj
    plain = captured["plain"]
    # CHAMPION section (CPCV + base enrichment)
    assert "CHAMPION" in plain and "+0.1000" in plain
    assert "v3.0-meta" in plain          # base champion-arch enrichment
    # CHALLENGERS section (each candidate + reason)
    assert "resid" in plain and "h60" in plain
    assert "non_canonical_horizon" in plain
    # PROMOTION section (old -> new)
    assert "old-champ-v" in plain and "resid-v" in plain
    # html mirrors the three sections
    html = captured["html"]
    assert "Champion" in html and "Challengers" in html and "Promotion" in html


def test_digest_renders_three_labeled_groups_no_double_count(monkeypatch):
    # #679(ii): the digest shows three CLEARLY-LABELED groups — Serving champion
    # (stale), Champion-arch (fresh baseline), Challengers — and champion-arch is
    # NOT double-counted in the challenger table.
    captured = _capture_digest_send(monkeypatch)
    s3 = _FakeS3({})
    leaderboard = {
        "date": "2026-06-20", "mode": "observe",
        "serving_champion": {"forward_days": 21, "cpcv_mean_ic": 0.10,
                             "served_version": "v3.0-meta", "served_date": "2026-05-30",
                             "cpcv_is_stale_snapshot": True},
        "champion_arch": {"version_id": "arch-v", "cpcv_mean_ic": 0.15,
                          "is_promotion_baseline": True},
        "promotion_baseline_ic": 0.15, "promotion_baseline_source": "champion_arch_fresh",
        "champion": {"forward_days": 21, "cpcv_mean_ic": 0.10},  # back-compat alias
        "margin": 0.0, "promote_min_ic": 0.0,
        "candidates": [
            {"spec_id": "champion-arch", "version_id": "arch-v", "group": "champion_arch",
             "cpcv_mean_ic": 0.15, "forward_days": 21, "passes_gate": True,
             "eligible": False, "reason": "champion_arch_baseline"},
            {"spec_id": "resid", "version_id": "resid-v", "group": "challenger",
             "cpcv_mean_ic": 0.20, "forward_days": 21, "passes_gate": True,
             "eligible": True, "reason": "eligible"},
        ],
        "winner_version_id": "resid-v", "promoted": "resid-v",
        "promoted_kind": "challenger", "reverted_from": "old-champ-v",
    }
    ok = mz.send_zoo_digest_email(leaderboard, "bkt", "2026-06-20", s3=s3)
    assert ok is True
    plain = captured["plain"]
    # Three distinct, labeled groups present.
    assert "SERVING CHAMPION" in plain
    assert "CHAMPION-ARCH" in plain
    assert "CHALLENGERS (1)" in plain                 # ONE challenger — arch excluded
    # Serving CPCV (stale) + served date; champion-arch CPCV (baseline) both shown.
    assert "2026-05-30" in plain                      # serving served-date
    assert "PROMOTION BASELINE" in plain
    # Subject counts ONE challenger (champion-arch not double-counted).
    assert "1 challengers" in captured["subject"]
    html = captured["html"]
    assert "Serving champion" in html and "Champion-arch" in html
    assert "Challengers (1)" in html


def test_digest_includes_console_deeplink(monkeypatch):
    """The digest deep-links to the console Model Zoo page, keyed by THIS
    rotation's date_str (the trading-day key) so the link opens the exact cycle.
    The slug must equal crucible-dashboard's pinned url_path ("model-zoo")."""
    captured = _capture_digest_send(monkeypatch)
    s3 = _FakeS3({})
    leaderboard = {
        "date": "2026-06-26", "mode": "cutover",
        "champion": {"forward_days": 21, "cpcv_mean_ic": 0.13},
        "candidates": [
            {"spec_id": "resid", "cpcv_mean_ic": 0.18, "forward_days": 21,
             "passes_gate": True, "eligible": True, "reason": "eligible"},
        ],
        "winner_version_id": "resid-v", "promoted": "resid-v",
    }
    ok = mz.send_zoo_digest_email(leaderboard, "bkt", "2026-06-26", s3=s3)
    assert ok is True
    expected = "https://console.nousergon.ai/model-zoo?date=2026-06-26"
    assert expected in captured["plain"]
    assert f'href="{expected}"' in captured["html"]
    # The slug constant is the cross-repo contract with the dashboard page.
    assert mz.MODEL_ZOO_SLUG == "model-zoo"


def test_digest_no_promotion_says_so(monkeypatch):
    captured = _capture_digest_send(monkeypatch)
    s3 = _FakeS3({})  # no base summary → champion section uses CPCV only
    leaderboard = {
        "date": "2026-06-20", "mode": "observe",
        "champion": {"forward_days": 21, "cpcv_mean_ic": 0.10},
        "candidates": [
            {"spec_id": "resid", "cpcv_mean_ic": 0.05, "forward_days": 21,
             "passes_gate": True, "eligible": False, "reason": "below_champion_plus_margin"},
        ],
        "winner_version_id": None, "promoted": None,
    }
    ok = mz.send_zoo_digest_email(leaderboard, "bkt", "2026-06-20", s3=s3)
    assert ok is True
    assert "promoted: none" in captured["subject"]
    assert "No promotion" in captured["plain"]


def test_digest_robust_to_missing_candidate_fields(monkeypatch):
    """A candidate missing cpcv_mean_ic / fields (e.g. a manifest read failed)
    must not crash the digest — it reports whatever exists."""
    captured = _capture_digest_send(monkeypatch)
    s3 = _FakeS3({})
    leaderboard = {
        "date": "2026-06-20", "mode": "observe",
        "champion": {},                       # champion section degrades to n/a
        "candidates": [
            {"spec_id": "broken"},            # all other fields absent
            {},                               # entirely empty candidate
        ],
        "winner_version_id": None, "promoted": None,
    }
    ok = mz.send_zoo_digest_email(leaderboard, "bkt", "2026-06-20", s3=s3)
    assert ok is True
    assert "n/a" in captured["plain"]         # missing IC rendered as n/a, no crash


def test_digest_skips_when_email_not_configured(monkeypatch):
    monkeypatch.setattr(cfg, "EMAIL_SENDER", None, raising=False)
    monkeypatch.setattr(cfg, "EMAIL_RECIPIENTS", None, raising=False)
    s3 = _FakeS3({})
    ok = mz.send_zoo_digest_email(
        {"champion": {}, "candidates": [], "winner_version_id": None, "promoted": None},
        "bkt", "2026-06-20", s3=s3,
    )
    assert ok is False


# ── run_rotation_and_select wiring: digest fires exactly once ─────────────────


def _run_fixture(monkeypatch, *, send_raises=False):
    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: None)
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: None)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/resid-v/manifest.json": _mk_manifest(21, 0.20, True),
    })
    monkeypatch.setattr(reg, "list_versions", lambda s3c, b, stage=None: [
        {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-20"},
    ])
    monkeypatch.setattr(reg, "promote_to_champion", lambda *a, **k: None)

    sends = []

    def _fake_digest(leaderboard, bucket, date_str, *, s3=None):
        sends.append(date_str)
        if send_raises:
            raise RuntimeError("smtp down")
        return True

    monkeypatch.setattr(mz, "send_zoo_digest_email", _fake_digest)

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train,
        registered_versions=[], s3=s3, date_str="2026-06-20",
    )
    return board, sends


def test_rotation_sends_digest_exactly_once(monkeypatch):
    board, sends = _run_fixture(monkeypatch)
    assert sends == ["2026-06-20"]               # exactly one digest
    assert board["digest_email"] == "sent"


def test_rotation_digest_failure_does_not_abort(monkeypatch):
    """no-silent-fails carve-out: a digest-send failure is recorded (WARN +
    leaderboard marker) but NEVER aborts the rotation."""
    board, sends = _run_fixture(monkeypatch, send_raises=True)
    assert sends == ["2026-06-20"]
    assert board["winner_version_id"] == "resid-v"   # rotation completed
    assert board["digest_email"].startswith("model_zoo_digest_email_failed")


def test_dry_run_does_not_send_digest(monkeypatch):
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: None)
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: None)
    sends = []
    monkeypatch.setattr(mz, "send_zoo_digest_email",
                        lambda *a, **k: sends.append(1) or True)

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train, dry_run=True,
    )
    assert sends == []                            # no real training → no digest
