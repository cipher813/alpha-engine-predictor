"""Tests for the ``TRAIN_START_DATE`` diagnostic knob.

ROADMAP L1581 (added 2026-04-15, closed 2026-05-12 as diagnostic
substrate). The 2026-04-15 ArcticDB-unification PR extended the
training corpus back to 2019-04 (vs the pre-unification 2020-03);
smoke-test meta IC dropped 0.052 → 0.033. Hypothesis: pre-2020 folds
pull the Ridge coefficients toward less-predictive averages.

This knob defaults to None (no clamp). Setting it in alpha-engine-
config flips the meta-trainer to drop pre-cutoff rows from the
per-ticker labeled DataFrame so a single Sat retrain can compare
meta IC against the no-clamp baseline. The effective value is
persisted in the manifest so the comparison is auditable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


# ── Config contract ────────────────────────────────────────────────────


class TestTrainStartDateConfigKnob:

    def test_train_start_date_defined(self):
        assert hasattr(cfg, "TRAIN_START_DATE")

    def test_train_start_date_defaults_to_none(self):
        # The shipped sample yaml has ``train_start_date: null`` so the
        # default behavior — keep all history — is preserved. Diagnostic
        # only; flipped in alpha-engine-config for a single retrain.
        # Note: production predictor.yaml may override; this test pins
        # the *sample* default.
        sample = (
            _REPO / "config" / "predictor.sample.yaml"
        ).read_text()
        assert "train_start_date: null" in sample

    def test_loaded_via_get_for_back_compat(self):
        # Older deployed predictor.yaml files don't have the key. The
        # config loader must use ``.get`` so they continue to load
        # cleanly with the default (None).
        src = _src("config.py")
        assert (
            'TRAIN_START_DATE: str | None = _train_cfg.get("train_start_date")'
            in src
        )


# ── Source-level wiring contract ───────────────────────────────────────


class TestFilterAppliedToLabeled:

    def test_filter_lives_after_empty_check(self):
        # Filter is applied AFTER the existing ``labeled.empty`` guard,
        # so tickers with no labeled rows at all (rare) bail out before
        # the clamp logic touches them.
        src = _src("training/meta_trainer.py")
        # The filter block exists.
        assert "if cfg.TRAIN_START_DATE is not None:" in src
        # Uses the canonical pandas Timestamp + index-mask pattern.
        assert "_cutoff_ts = pd.Timestamp(cfg.TRAIN_START_DATE)" in src
        assert "labeled = labeled.loc[labeled.index >= _cutoff_ts]" in src

    def test_post_filter_empty_check_present(self):
        # An aggressive cutoff date can wipe out a ticker's full history;
        # the post-filter empty check routes the ticker to the same
        # reject bucket as the pre-filter case so reject summaries
        # remain accurate.
        src = _src("training/meta_trainer.py")
        # Look for the post-filter empty-check block — pin both the
        # condition and the reject-bucket append.
        idx = src.find("_cutoff_ts = pd.Timestamp(cfg.TRAIN_START_DATE)")
        assert idx > 0
        tail = src[idx:idx + 600]
        assert "labeled = labeled.loc[labeled.index >= _cutoff_ts]" in tail
        assert "if labeled.empty:" in tail
        assert "reject_empty_labeled.append(ticker)" in tail

    def test_filter_runs_before_multi_horizon_sidecar(self):
        # The multi-horizon sidecars reindex to ``labeled.index``, so
        # they automatically pick up the truncated index. But the
        # sidecars compute ``close.shift(-N)`` against ``raw_df`` —
        # which is intentionally NOT filtered — so forward labels at
        # dates just after the cutoff aren't distorted by an artificial
        # front-of-window.
        src = _src("training/meta_trainer.py")
        filter_pos = src.find("labeled = labeled.loc[labeled.index >= _cutoff_ts]")
        sidecar_pos = src.find("close_for_horizon = raw_df[close_col]")
        assert filter_pos > 0 and sidecar_pos > 0
        assert filter_pos < sidecar_pos


# ── Manifest contract ──────────────────────────────────────────────────


class TestManifestRecordsTrainStartDate:

    def test_train_start_date_in_manifest(self):
        src = _src("training/meta_trainer.py")
        # Persist the effective clamp so a clamp-vs-no-clamp comparison
        # across Sat retrains is auditable from the manifest alone.
        assert '"train_start_date":' in src
        # null when no clamp was set — preserves audit value across runs.
        assert (
            "str(cfg.TRAIN_START_DATE)\n                    if cfg.TRAIN_START_DATE is not None else None"
            in src
        )


# ── Behavior smoke ─────────────────────────────────────────────────────


class TestDateFilterBehavior:
    """Validates the pandas index-mask semantics the filter relies on
    so the diagnostic produces honest before/after row counts.
    """

    def test_inclusive_lower_bound(self):
        # ``>=`` cutoff so the date string itself is *included* in the
        # retained set — matches how operator-facing knobs typically read.
        idx = pd.DatetimeIndex(
            ["2020-03-30", "2020-03-31", "2020-04-01"]
        )
        labeled = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
        cutoff = pd.Timestamp("2020-03-31")
        out = labeled.loc[labeled.index >= cutoff]
        assert list(out.index.strftime("%Y-%m-%d")) == [
            "2020-03-31", "2020-04-01",
        ]

    def test_full_drop_produces_empty_frame(self):
        # Cutoff post-dating every available row → empty result; the
        # production code then routes the ticker to ``reject_empty_labeled``.
        idx = pd.DatetimeIndex(["2019-01-01", "2019-06-01"])
        labeled = pd.DataFrame({"x": [1, 2]}, index=idx)
        cutoff = pd.Timestamp("2025-01-01")
        out = labeled.loc[labeled.index >= cutoff]
        assert out.empty

    def test_no_cutoff_preserves_all_rows(self):
        # Sanity: a None cutoff (the default) MUST be a no-op for the
        # caller. The production code gates the mask behind
        # ``if cfg.TRAIN_START_DATE is not None`` so this is structural.
        idx = pd.DatetimeIndex(["2019-01-01", "2020-01-01", "2025-01-01"])
        labeled = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
        # Mimic the production guard.
        cutoff = None
        if cutoff is not None:
            labeled = labeled.loc[labeled.index >= pd.Timestamp(cutoff)]
        assert len(labeled) == 3
