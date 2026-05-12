"""Tests for the ``feature_list.json`` inference-universe disclosure
artifact written by ``training/meta_trainer.py`` alongside ``manifest.json``.

ROADMAP P2 (added 2026-05-05, closed 2026-05-12). Single source of
truth for the dashboard's Feature Store inventory page to compute the
production-vs-research delta — features in ArcticDB but not yet wired
into inference — without parsing LightGBM ``.txt`` files (which would
violate Decision 11 of the presentation revamp plan).

S3 contract: ``predictor/weights/meta/feature_list.json``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from model.meta_model import META_FEATURES
from model.research_gbm import RESEARCH_GBM_FEATURES


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


# ── Config contract ────────────────────────────────────────────────────


class TestFeatureListConfigKey:

    def test_feature_list_key_defined(self):
        assert hasattr(cfg, "META_FEATURE_LIST_KEY")
        assert cfg.META_FEATURE_LIST_KEY == (
            "predictor/weights/meta/feature_list.json"
        )

    def test_feature_list_key_colocated_with_manifest(self):
        # Lives next to manifest.json in the meta-weights prefix so the
        # dashboard can read both with a single S3 prefix listing.
        assert cfg.META_FEATURE_LIST_KEY.startswith(cfg.META_WEIGHTS_PREFIX)


# ── Source-level regression: producer wired into manifest write site ───


class TestProducerSourceContract:

    def test_feature_list_emit_present(self):
        src = _src("training/meta_trainer.py")
        # The write block has to land at the manifest-write site so a
        # successful retrain produces both artifacts atomically.
        assert "META_FEATURE_LIST_KEY" in src
        assert '"l2_features": list(META_FEATURES)' in src
        assert '"l1_features":' in src

    def test_research_gbm_features_imported(self):
        src = _src("training/meta_trainer.py")
        assert (
            "from model.research_gbm import RESEARCH_GBM_FEATURES" in src
        )

    def test_emit_carries_canonical_schema(self):
        src = _src("training/meta_trainer.py")
        # Schema per ROADMAP spec: trained_at, version, l2_features,
        # l1_features { momentum, volatility, research_calibrator }.
        assert '"trained_at": date_str' in src
        assert '"version": "v3.0-meta"' in src
        assert '"momentum":' in src
        assert '"volatility": list(cfg.VOLATILITY_FEATURES)' in src
        assert (
            '"research_calibrator": list(RESEARCH_GBM_FEATURES)' in src
        )

    def test_momentum_features_are_deterministic_baseline_inputs(self):
        # The momentum L1 is a deterministic 4-input baseline (see
        # ``model.momentum_scorer``), not the full
        # ``cfg.MOMENTUM_FEATURES`` subscription list. Emitting the
        # 9-feature subscription would misreport the consumed universe.
        src = _src("training/meta_trainer.py")
        assert (
            '"momentum": ["momentum_5d", "momentum_20d",\n'
            in src
            or '"momentum": ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]'
            in src
        )

    def test_emit_writes_to_s3_with_json_content_type(self):
        src = _src("training/meta_trainer.py")
        # Same put_object shape as the manifest write — JSON body, JSON
        # content-type, named S3 key constant.
        assert "Key=cfg.META_FEATURE_LIST_KEY" in src
        assert "ContentType=\"application/json\"" in src


# ── Schema construction smoke test ─────────────────────────────────────


class TestFeatureListSchemaShape:
    """Validates the in-memory shape that the producer constructs from
    the canonical feature-list constants. Mirrors the producer's dict
    literal so a future change to a constant ripples through.
    """

    def _build(self, date_str: str = "2026-05-12") -> dict:
        return {
            "trained_at": date_str,
            "version": "v3.0-meta",
            "l2_features": list(META_FEATURES),
            "l1_features": {
                "momentum": [
                    "momentum_5d", "momentum_20d",
                    "price_vs_ma50", "rsi_14",
                ],
                "volatility": list(cfg.VOLATILITY_FEATURES),
                "research_calibrator": list(RESEARCH_GBM_FEATURES),
            },
        }

    def test_l2_features_match_meta_features(self):
        out = self._build()
        assert out["l2_features"] == list(META_FEATURES)
        # Canonical META_FEATURES has 12 entries today (6 stack-output
        # features + 6 raw macros). Pin to catch silent extensions.
        assert len(out["l2_features"]) == 12

    def test_momentum_features_match_deterministic_formula(self):
        # Mirrors ``model.momentum_scorer.predict_array``'s consumed
        # columns: m5, m20, price_vs_ma50, rsi_14 (centered+scaled).
        out = self._build()
        assert out["l1_features"]["momentum"] == [
            "momentum_5d", "momentum_20d",
            "price_vs_ma50", "rsi_14",
        ]

    def test_volatility_features_match_cfg(self):
        out = self._build()
        assert out["l1_features"]["volatility"] == list(
            cfg.VOLATILITY_FEATURES
        )

    def test_research_calibrator_features_match_research_gbm(self):
        out = self._build()
        assert out["l1_features"]["research_calibrator"] == list(
            RESEARCH_GBM_FEATURES
        )
        # Canonical RESEARCH_GBM_FEATURES has 9 entries (3 research +
        # 6 macro). Pin to catch silent extensions.
        assert len(out["l1_features"]["research_calibrator"]) == 9

    def test_payload_is_json_serializable(self):
        out = self._build()
        # All constants must be plain lists/strings — no numpy scalars
        # leaking in from the upstream definitions.
        s = json.dumps(out)
        round_trip = json.loads(s)
        assert round_trip == out
