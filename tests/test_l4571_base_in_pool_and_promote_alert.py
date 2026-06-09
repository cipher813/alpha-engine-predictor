"""L4571 — the base champion-architecture competes in the model-zoo pool, and
an auto-promotion announces itself (Telegram/SNS) with an exact revert command.

Why: auto-promote is meaningless unless the base champion-arch retrain (which
runs first on Saturday and registers as a `stage="challenger"` `v3.0-meta`
version) competes against the rotated variants in the SAME pool — else
`select_winner` only ever ranks one variant against a stale serving manifest and
a fresh-data retrain can never win. And because the realized-edge auto-demote
(L4539) is not live, revert is MANUAL — so a promotion MUST fire a loud alert
carrying the revert command, or "revert if it misbehaves" is not actionable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz
from tests.test_model_zoo import _FakeS3, _mk_manifest, _SPECS


def _stage_versions(challengers, champions):
    """A `list_versions` stub that honours the `stage` filter."""
    return lambda s3c, b, stage=None: {
        "challenger": challengers, "champion": champions
    }.get(stage, challengers + champions)


# ── _resolve_base_champion_version ─────────────────────────────────────────


class TestResolveBaseChampionVersion:
    def test_picks_todays_v3_meta_challenger(self, monkeypatch):
        import model.registry as reg
        monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "v3.0-meta", raising=False)
        monkeypatch.setattr(reg, "list_versions", _stage_versions([
            {"version_id": "base-old", "model_version": "v3.0-meta", "date": "2026-06-06"},
            {"version_id": "base-today", "model_version": "v3.0-meta", "date": "2026-06-13"},
            {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        ], []))
        out = mz._resolve_base_champion_version(None, "bkt", "2026-06-13")
        assert out == {"spec_id": "champion-arch", "model_version": "v3.0-meta",
                       "version_id": "base-today"}

    def test_none_when_no_base_registered(self, monkeypatch):
        import model.registry as reg
        monkeypatch.setattr(reg, "list_versions", _stage_versions([
            {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        ], []))
        assert mz._resolve_base_champion_version(None, "bkt", "2026-06-13") is None

    def test_read_failure_is_none(self, monkeypatch):
        import model.registry as reg
        def _boom(*a, **k):
            raise RuntimeError("registry down")
        monkeypatch.setattr(reg, "list_versions", _boom)
        assert mz._resolve_base_champion_version(None, "bkt", "2026-06-13") is None


# ── base competes in the pool + alerts on promote/observe ──────────────────


def _pool_fixture(monkeypatch, *, auto_promote, base_ic, variant_ic):
    """Champion (serving) CPCV 0.10; a base-arch retrain + one variant compete."""
    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "v3.0-meta", raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),          # serving champion
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/base-today/manifest.json": _mk_manifest(21, base_ic, True),
        "predictor/registry/resid-v/manifest.json": _mk_manifest(21, variant_ic, True),
    })
    monkeypatch.setattr(reg, "list_versions", _stage_versions(
        challengers=[
            {"version_id": "base-today", "model_version": "v3.0-meta", "date": "2026-06-13"},
            {"version_id": "resid-v", "model_version": "spec-resid", "date": "2026-06-13"},
        ],
        champions=[{"version_id": "old-champ-v", "model_version": "v3.0-meta", "date": "2026-06-06"}],
    ))
    promotes = []
    monkeypatch.setattr(reg, "promote_to_champion",
                        lambda s3c, b, vid, **k: promotes.append(vid))
    alerts_sent = []
    import alpha_engine_lib.alerts as _alerts
    monkeypatch.setattr(_alerts, "publish",
                        lambda **kw: alerts_sent.append(kw) or None)

    def _fake_train(bucket, *, date_str=None, dry_run=False):
        return {"status": "ok"}

    board = mz.run_rotation_and_select(
        "bkt", budget=5, specs=_SPECS, train_fn=_fake_train,
        registered_versions=[], s3=s3, date_str="2026-06-13",
        auto_promote_winner=auto_promote,
    )
    return board, promotes, alerts_sent


class TestBaseInPool:
    def test_base_arch_is_ranked_and_can_win(self, monkeypatch):
        # base-arch (0.30) > variant (0.15) > champion (0.10) → base-arch wins.
        board, promotes, _ = _pool_fixture(monkeypatch, auto_promote=True,
                                           base_ic=0.30, variant_ic=0.15)
        ids = {c["spec_id"] for c in board["candidates"]}
        assert "champion-arch" in ids            # base IS in the pool
        assert board["winner_version_id"] == "base-today"
        assert promotes == ["base-today"]

    def test_variant_can_beat_the_base(self, monkeypatch):
        board, _, _ = _pool_fixture(monkeypatch, auto_promote=True,
                                    base_ic=0.12, variant_ic=0.25)
        assert board["winner_version_id"] == "resid-v"


class TestPromotionAlert:
    def test_cutover_fires_alert_with_revert_command(self, monkeypatch):
        board, promotes, alerts_sent = _pool_fixture(
            monkeypatch, auto_promote=True, base_ic=0.30, variant_ic=0.15)
        assert promotes == ["base-today"]
        assert board["reverted_from"] == "old-champ-v"
        assert len(alerts_sent) == 1
        a = alerts_sent[0]
        assert a["severity"] == "warning"
        # The revert command targets the PRIOR champion's version_id, exactly.
        assert "--promote old-champ-v" in a["message"]
        assert "base-today" in a["message"]      # the new champion
        assert a["dedup_key"] == "model_zoo_promote_2026-06-13"

    def test_observe_fires_info_alert_no_promote(self, monkeypatch):
        board, promotes, alerts_sent = _pool_fixture(
            monkeypatch, auto_promote=False, base_ic=0.30, variant_ic=0.15)
        assert promotes == []                    # observe → never promotes
        assert board["promoted"] is None
        assert len(alerts_sent) == 1
        assert alerts_sent[0]["severity"] == "info"
        assert "--promote base-today" in alerts_sent[0]["message"]  # manual-promote hint

    def test_no_winner_no_alert(self, monkeypatch):
        # All candidates below champion+margin → no winner, no alert.
        board, promotes, alerts_sent = _pool_fixture(
            monkeypatch, auto_promote=True, base_ic=0.10, variant_ic=0.10)
        assert board["winner_version_id"] is None
        assert promotes == []
        assert alerts_sent == []
