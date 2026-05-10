"""Source-level regression tests for Stage 3 PR 2 wiring in meta_trainer.

Pins the parallel triple-barrier L2 Ridge plumbing in
``training/meta_trainer.py``:

- Triple-barrier label is computed alongside the canonical label per ticker
- Per-row triple-barrier label flows into oos_meta_rows
- Parallel meta-Ridge fits on triple-barrier-finite rows with LdP Ch. 4.4
  average-uniqueness sample weights (per-ticker)
- Parallel artifact gets persisted to S3 alongside the canonical model

Stage 3 of the regime-conditioning rebuild — replaces the fixed-21d-horizon
L1 alpha target with path-dependent triple-barrier labels (LdP Ch. 3.4).
PR 1 shipped the label generator; this PR (PR 2) wires parallel L2
training; PR 3 wires inference parallel field; PR 5 flips the cutover
flag after the variant_cutover_gate clears.

Plan: ~/Development/alpha-engine-docs/private/triple-barrier-260510.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


class TestTripleBarrierLabelPlumbing:

    def test_compute_triple_barrier_imported(self):
        src = _src("training/meta_trainer.py")
        # Imported alongside compute_labels — single-line for visibility.
        assert (
            "from data.label_generator import compute_labels, "
            "compute_triple_barrier_alpha_labels"
        ) in src

    def test_tb_chunks_declared_alongside_fwd_chunks(self):
        src = _src("training/meta_trainer.py")
        assert "fwd_tb_chunks: list[np.ndarray] = []" in src

    def test_tb_label_appended_per_ticker(self):
        src = _src("training/meta_trainer.py")
        # Pulled via .get(...) with NaN fallback so PR 1's empty-DF path
        # doesn't crash the loop.
        assert "fwd_tb_chunks.append(" in src
        assert "labeled.get(" in src
        assert '"triple_barrier_alpha_21d"' in src

    def test_tb_call_uses_config_knobs(self):
        src = _src("training/meta_trainer.py")
        # All five config knobs (forward_window, vol_window, vol_multiplier,
        # min_periods, plus the ENFORCE_CUTOVER consumed downstream).
        assert "compute_triple_barrier_alpha_labels(" in src
        assert "forward_window=cfg.TRIPLE_BARRIER_FORWARD_WINDOW" in src
        assert "vol_window=cfg.TRIPLE_BARRIER_VOL_WINDOW" in src
        assert "vol_multiplier=cfg.TRIPLE_BARRIER_VOL_MULTIPLIER" in src
        assert "min_periods=cfg.TRIPLE_BARRIER_MIN_PERIODS" in src

    def test_tb_array_built_and_sorted(self):
        src = _src("training/meta_trainer.py")
        assert "y_fwd_tb_unsorted = np.concatenate(fwd_tb_chunks)" in src
        assert "y_fwd_tb = y_fwd_tb_unsorted[sort_idx]" in src

    def test_tb_label_clipped_alongside_canonical(self):
        src = _src("training/meta_trainer.py")
        # Same LABEL_CLIP envelope as y_fwd
        assert "y_fwd_tb = np.clip(y_fwd_tb, -cfg.LABEL_CLIP, cfg.LABEL_CLIP)" in src

    def test_tb_label_flows_into_oos_meta_row(self):
        src = _src("training/meta_trainer.py")
        # Per-row dict gets the triple-barrier label + ticker for grouping
        assert '"actual_fwd_triple_barrier": float(y_fwd_tb[idx])' in src
        assert '"ticker": ticker' in src


class TestParallelTripleBarrierRidgeWiring:

    def test_parallel_ridge_imports_average_uniqueness(self):
        src = _src("training/meta_trainer.py")
        # Lazy import inside the parallel block — keeps module-level import
        # surface unchanged for back-compat with downstream consumers
        assert "from labeling.sample_weights import average_uniqueness_weights" in src

    def test_parallel_ridge_uses_full_oos_meta_rows_pre_filter(self):
        src = _src("training/meta_trainer.py")
        # Must run BEFORE the canonical-finite filter so the TB-finite mask
        # (different from canonical-finite) operates on the full row set
        canonical_filter_idx = src.find("canonical-finite rows so downstream consumers")
        parallel_block_idx = src.find("Parallel triple-barrier L2 Ridge")
        assert parallel_block_idx != -1
        assert parallel_block_idx < canonical_filter_idx

    def test_parallel_ridge_uses_per_ticker_sample_weights(self):
        src = _src("training/meta_trainer.py")
        # group_ids must be the ticker so different tickers don't pollute
        # each other's concurrency
        assert "group_ids=tb_tickers" in src
        assert "window_length=cfg.TRIPLE_BARRIER_FORWARD_WINDOW" in src

    def test_parallel_ridge_passes_sample_weight_to_fit(self):
        src = _src("training/meta_trainer.py")
        assert "meta_model_tb = MetaModel(alpha=1.0)" in src
        assert "meta_model_tb.fit(" in src
        assert "sample_weight=tb_sample_weights" in src

    def test_parallel_ridge_min_finite_rows_gate(self):
        src = _src("training/meta_trainer.py")
        # n>=100 gate mirrors the canonical Ridge's gate
        assert "n_tb_finite >= 100" in src
        # Skip-cycle log when below gate
        assert "Triple-barrier Meta-Ridge: only %d finite-label rows" in src

    def test_parallel_ridge_emits_in_sample_ic_log(self):
        src = _src("training/meta_trainer.py")
        assert "Triple-barrier Meta-Ridge fit" in src
        assert "val_ic_tb=%.4f" in src

    def test_parallel_ridge_cross_target_diagnostic(self):
        src = _src("training/meta_trainer.py")
        # IC of TB Ridge predictions against canonical labels (parallel
        # to the canonical-vs-legacy diagnostic above)
        assert "ic_tb_ridge_vs_canonical" in src
        assert "Triple-barrier-vs-canonical diagnostic" in src


class TestParallelArtifactPersistence:

    def test_meta_model_tb_in_upload_set(self):
        src = _src("training/meta_trainer.py")
        # Conditional add — None skip mirrors prod_research_gbm /
        # prod_vol_macro_aug / prod_vol_risk_aug pattern
        assert "if meta_model_tb is not None:" in src
        assert 'models["meta_model_tb"] = (meta_model_tb, "meta_model_tb.pkl")' in src
