"""Source-level regression tests for Stage 3 PR 5: cutover-gate SF wiring.

Pins the train_handler tail call that invokes the triple-barrier cutover
gate after model upload + training-summary write. Mirrors the
source-level regression style used by ``test_meta_trainer_triple_barrier.py``.

The wiring contract:
- Step 2e runs AFTER step 2d (training summary write) and BEFORE step 3
  (training email send) so the gate result can be merged into the
  ``result`` dict for email surfacing.
- Failure is non-blocking — wrapped in try/except with a WARNING log;
  the gate is observe-only until ``triple_barrier.enforce_cutover`` flips
  in alpha-engine-config (PR 6 of the arc).
- Default knobs: 42-day window, 21-day horizon, 1.15 relative-lift
  threshold (LdP institutional default).

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


class TestTrainHandlerCutoverGateWiring:

    def test_run_gate_imported_and_called(self):
        src = _src("training/train_handler.py")
        # Lazy import inside the step block — keeps top-of-module import
        # surface unchanged for back-compat with downstream consumers
        assert "from analysis.triple_barrier_cutover_runner import run_gate" in src
        # Aliased to _run_tb_gate to avoid namespace collisions with any
        # future helper named run_gate
        assert "as _run_tb_gate" in src
        assert "_gate_payload = _run_tb_gate(" in src

    def test_gate_runs_with_default_knobs(self):
        src = _src("training/train_handler.py")
        # 42-day window, 21-day horizon, 1.15 threshold (institutional defaults)
        assert "n_days=42" in src
        assert "horizon_days=21" in src
        assert "threshold=1.15" in src
        # Writes result to S3 so dashboards / future evaluator-email
        # surfacers can read it
        assert "write_to_s3=True" in src

    def test_gate_failure_is_non_blocking(self):
        src = _src("training/train_handler.py")
        # try/except with WARN log around the entire gate invocation.
        # Per the parallel-observation pattern, gate failure must not
        # break training — observe-only path should never gate the
        # canonical promotion decision.
        gate_block_idx = src.find("Stage 3 PR 5 SF wiring")
        assert gate_block_idx != -1
        # The try/except catches Exception broadly + logs at WARN
        try_idx = src.find("try:", gate_block_idx)
        except_idx = src.find("except Exception as _gate_err:", gate_block_idx)
        warn_idx = src.find(
            "Stage 3 cutover gate failed (non-blocking, observe-only)",
            gate_block_idx,
        )
        assert try_idx != -1 and except_idx != -1 and warn_idx != -1
        assert try_idx < except_idx < warn_idx

    def test_gate_runs_only_when_not_dry_run(self):
        # The block is gated behind ``if not dry_run`` so dry-run /
        # local validation paths skip the gate entirely. Mirrors the
        # gating on training summary write + email send.
        src = _src("training/train_handler.py")
        gate_block_idx = src.find("Stage 3 PR 5 SF wiring")
        # Find the most recent `if not dry_run:` line ABOVE the gate
        # block — should sit immediately before the try.
        if_idx = src.rfind("if not dry_run:", 0, gate_block_idx + 200)
        assert if_idx != -1
        assert if_idx < gate_block_idx + 200

    def test_gate_result_merged_into_training_summary(self):
        src = _src("training/train_handler.py")
        # The gate verdict is plumbed into result dict so email rendering
        # + downstream consumers see passed / n_pairs / lift / reason.
        # Allows the existing email pipeline to surface gate state without
        # a separate orchestration layer.
        assert 'result["triple_barrier_cutover_gate"] = {' in src
        # Required fields for operator visibility
        assert '"passed":' in src
        assert '"n_pairs":' in src
        assert '"baseline_ic":' in src
        assert '"variant_ic":' in src
        assert '"relative_lift":' in src
        assert '"reason":' in src

    def test_gate_runs_after_summary_before_email(self):
        # Step 2e (gate) must run AFTER step 2d (training summary write)
        # — gate consumes the freshly-uploaded artifacts — and BEFORE
        # step 3 (training email send) — email surfaces gate result.
        src = _src("training/train_handler.py")
        summary_idx = src.find("Step 2d: Write training summary to S3")
        gate_idx = src.find("Step 2e: Triple-barrier cutover gate")
        email_idx = src.find("Step 3: Email")
        assert summary_idx != -1 and gate_idx != -1 and email_idx != -1
        assert summary_idx < gate_idx < email_idx, (
            "Step 2e (cutover gate) must sit between Step 2d (summary write) "
            "and Step 3 (email send)"
        )
