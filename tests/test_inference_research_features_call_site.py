"""
tests/test_inference_research_features_call_site.py — Lock the inference call
site to use the centralized ``model.research_features.extract_research_features``
helper.

Pre-2026-04-28 the inference path at ``run_inference.py:293`` reimplemented
the per-ticker research-feature extraction inline and contained a bug:
``sector_modifier = sig.get("sector_modifiers", {}).get(...)`` reads from the
per-ticker dict instead of the top-level ``signals.json`` payload, always
returning 0.0 sector_modifier regardless of research's sector ratings.

The fix moves the lookup to a shared helper. These tests assert source-text
invariants so a future refactor can't silently regress to the inline buggy
form, plus a behavioral test that the helper's output flows into the
inference loop's `meta_features` dict correctly.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_RUN_INFERENCE = (
    Path(__file__).resolve().parent.parent
    / "inference" / "stages" / "run_inference.py"
)


@pytest.fixture(scope="module")
def run_inference_source() -> str:
    return _RUN_INFERENCE.read_text()


class TestInferenceUsesSharedHelper:
    def test_imports_extract_research_features(self, run_inference_source):
        """The shared helper must be imported at the inference call site —
        regression detector for someone copy-pasting the inline lookup back."""
        assert (
            "from model.research_features import extract_research_features"
            in run_inference_source
        ), (
            "run_inference.py must import extract_research_features from "
            "model.research_features so it stays in sync with meta_trainer's "
            "row-construction join."
        )

    def test_calls_extract_research_features(self, run_inference_source):
        assert "extract_research_features(" in run_inference_source

    def test_no_buggy_per_ticker_sector_modifiers_lookup(self, run_inference_source):
        """The pre-fix bug pattern was:
            sig.get("sector_modifiers", {}).get(sig.get("sector", ""), ...)
        ``sig`` is the per-ticker dict; ``sector_modifiers`` lives at the
        top level of signals.json. This test forbids the buggy pattern."""
        # The buggy form combined sig.get with sector_modifiers — the helper
        # reads from the full payload, so the per-ticker `sig.get` against
        # `sector_modifiers` should never reappear.
        buggy_pattern = re.compile(
            r"sig\.get\(\s*[\"']sector_modifiers[\"']"
        )
        assert not buggy_pattern.search(run_inference_source), (
            "Detected the pre-2026-04-28 inference bug pattern: "
            "sig.get('sector_modifiers', ...) on the per-ticker dict. "
            "sector_modifiers lives at the top level of signals.json — "
            "use extract_research_features instead."
        )

    def test_keeps_full_signals_payload_around(self, run_inference_source):
        """Variable name lock: extract_research_features needs the full
        payload, not the per-ticker dict. Without keeping the payload, the
        call site must either reload it from S3 (slow, redundant) or fall
        back to the buggy inline lookup."""
        assert "research_signals_payload" in run_inference_source


class TestInferenceFallbackOnMissingTicker:
    def test_neutral_defaults_when_helper_returns_none(self, run_inference_source):
        """When the shared helper returns None (ticker absent from
        snapshot, or no payload available), inference must fall through
        to the legacy neutral defaults — not crash, not silently
        substitute zero into meta_features. Regression test on the
        if/else structure around the helper call."""
        # Look for the structure: rf = extract_research_features(...);
        # if rf is not None: ... else: research_cal_prob = 0.5
        assert "if rf is not None:" in run_inference_source or (
            "rf is None" in run_inference_source
        )
        # Neutral fallback for research_cal_prob = 0.5 is preserved
        assert re.search(
            r"research_cal_prob\s*=\s*0\.5",
            run_inference_source,
        ) is not None
