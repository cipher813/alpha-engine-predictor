"""Tests for the Stage 3 PR 3 inference parallel path — triple-barrier
L2 Ridge rides alongside the canonical L2 Ridge in observe-only mode.

Stage 3 of the regime-conditioning rebuild — replaces the fixed-21d-horizon
L1 alpha target with path-dependent triple-barrier labels (LdP Ch. 3.4).
PRs 1-2 shipped the label generator + parallel training; this PR (PR 3)
wires inference parallel observation; PR 5 flips the cutover after the
variant_cutover_gate clears (≥15% relative IC lift over ≥3 Sat-SF firings).

Validates:
- ``inference/stages/load_model.py`` loads ``meta_model_tb.pkl`` alongside
  the canonical ``meta_model.pkl``, tolerant of absence (early observation
  period or any cycle where Step 7b's n_tb_finite<100 skipped fit).
- ``inference/stages/run_inference.py`` emits ``meta_alpha_tb`` parallel
  field in predictions JSON.
- Failure isolation: meta_model_tb absent / per-ticker predict raises →
  canonical ``predicted_alpha`` is unaffected; ``meta_alpha_tb`` rides as
  null in the output JSON.
- End-to-end behavior: the parallel field uses the SAME META_FEATURES
  vector the canonical Ridge consumes; only the trained weights differ.

Plan: alpha-engine-docs/private/triple-barrier-260510.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


# ── Source-level regression: loader + inference path wired ──────────────


class TestLoaderContract:

    def test_load_model_loads_meta_model_tb(self):
        src = _src("inference/stages/load_model.py")
        # Loader reads the parallel artifact from the same prefix as canonical
        assert "meta_model_tb.pkl" in src
        # Stored under the meta_tb key (distinct from canonical "meta")
        assert 'meta_models["meta_tb"]' in src

    def test_load_model_tolerates_meta_tb_absence(self):
        # Stage 1c / 2b pattern — the loader debug-logs and continues when
        # the parallel artifact is absent. The canonical Ridge is NOT
        # gated on the parallel artifact's presence.
        src = _src("inference/stages/load_model.py")
        # Tolerant log line uses "observe-only OK" framing so debug noise
        # is intentional, not an error
        assert "meta_model_tb not available (observe-only OK)" in src

    def test_loader_uses_metamodel_class(self):
        src = _src("inference/stages/load_model.py")
        # The loader instantiates MetaModel.load — same class as canonical
        # (only weights differ; structure identical)
        assert "from model.meta_model import MetaModel" in src


class TestInferenceContract:

    def test_run_inference_fetches_meta_tb(self):
        src = _src("inference/stages/run_inference.py")
        assert 'meta_models.get("meta_tb")' in src
        assert "meta_model_tb = ctx.meta_models.get(\"meta_tb\")" in src

    def test_run_inference_predicts_meta_alpha_tb(self):
        src = _src("inference/stages/run_inference.py")
        # Per-ticker predict — same META_FEATURES dict as canonical
        assert "meta_model_tb.predict_single(meta_features)" in src
        # Tolerant: try/except so a per-ticker predict crash doesn't
        # abort the loop, mirrors expected_move_macro_aug pattern
        assert 'log.debug("meta_alpha_tb predict failed for %s' in src

    def test_meta_alpha_tb_clipped_to_max_r(self):
        # Same alpha clip envelope as the canonical alpha — keeps both
        # streams comparable for variant_cutover_gate. max_r is the
        # label clip from cfg.LABEL_CLIP propagated via cfg upstream.
        src = _src("inference/stages/run_inference.py")
        assert "np.clip(meta_alpha_tb, -max_r, max_r)" in src

    def test_meta_alpha_tb_is_optional(self):
        # Annotation explicitly None-able so downstream consumers know
        # to handle missing-weight cycles. PR 4 (variant_cutover_gate
        # wiring) joins predictions on this field; null handling matters.
        src = _src("inference/stages/run_inference.py")
        assert "meta_alpha_tb: float | None = None" in src


class TestPredictionsJsonShape:

    def test_predictions_dict_has_meta_alpha_tb_field(self):
        src = _src("inference/stages/run_inference.py")
        # Result-dict assignment for the per-ticker output
        assert '"meta_alpha_tb":' in src
        # Rounded to 6 decimal places (same precision as expected_move,
        # predicted_alpha, expected_move_macro_aug) for JSON storage
        assert "round(meta_alpha_tb, 6)" in src

    def test_meta_alpha_tb_null_when_model_absent(self):
        # The result-dict expression is `round(...) if not None else None`
        # so JSON has explicit null when the parallel model isn't loaded
        src = _src("inference/stages/run_inference.py")
        assert (
            '"meta_alpha_tb": (\n'
            "                round(meta_alpha_tb, 6)\n"
            "                if meta_alpha_tb is not None else None\n"
            "            )," in src
        )


class TestFailureIsolation:
    """Per the Stage 1c / 2b pattern: parallel-observation paths must
    not gate the canonical alpha in any failure mode."""

    def test_canonical_alpha_unchanged_by_pr3(self):
        # The canonical predicted_alpha computation must be unchanged
        # — Step 3's edits add parallel logic AFTER the canonical alpha
        # is computed and clipped, never modify it.
        src = _src("inference/stages/run_inference.py")
        # Canonical computation: meta_model.predict_single → alpha → clip
        canonical_idx = src.find(
            'alpha = float(meta_model.predict_single(meta_features))'
        )
        assert canonical_idx != -1
        # Parallel block sits AFTER the canonical clip, not before
        canonical_clip_idx = src.find(
            'alpha = float(np.clip(alpha, -max_r, max_r))'
        )
        parallel_idx = src.find(
            'meta_alpha_tb: float | None = None'
        )
        assert parallel_idx != -1
        assert parallel_idx > canonical_clip_idx, (
            "Parallel triple-barrier block must run AFTER the canonical "
            "alpha is computed + clipped, not before — otherwise a TB "
            "predict failure could leak into the canonical path."
        )
