"""Source-invariant tests pinning the meta_trainer shadow-isolation wiring.

PR7-step-7b: a shadow run must (1) write weights/manifest/feature-list/oos to
the IO-spec prefixes (never the hardcoded live champion keys), (2) skip the
model-zoo registry snapshot so the shadow model can never become a challenger
the selector promotes, and (3) force ``promoted = False`` regardless of gate
outcome. ``run_meta_training`` is a ~4k-line long pole that needs a full
training fixture to execute, so we guard the wiring with source-text invariants
(mirrors ``tests/test_meta_trainer_streaming.py``) — a regression that
re-hardcodes a live key or drops a guard fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_META_TRAINER = Path(__file__).resolve().parent.parent / "training" / "meta_trainer.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _META_TRAINER.read_text()


def test_run_meta_training_accepts_io_spec(src):
    assert "io: \"TrainingIOSpec | None\" = None" in src
    assert "io = TrainingIOSpec.live()" in src
    assert "close_col = io.close_col" in src


def test_output_prefix_keys_come_from_io(src):
    # The S3-write block resolves its prefix/keys from the io spec, NOT from a
    # hardcoded live constant.
    assert "prefix = io.weights_prefix" in src
    assert "manifest_key = io.manifest_key" in src
    assert "feature_list_key = io.feature_list_key" in src


def test_manifest_and_feature_list_written_to_io_keys(src):
    assert "Key=manifest_key," in src
    assert "Key=feature_list_key," in src
    # The pre-shadow hardcoded write keys must be gone from the put_object sites.
    assert "Key=cfg.META_MANIFEST_KEY," not in src
    assert "Key=cfg.META_FEATURE_LIST_KEY," not in src


def test_oos_rows_use_io_paths(src):
    assert "io.oos_rows_key(date_str)" in src
    assert "io.oos_rows_latest_key" in src
    assert 'f"predictor/diagnostics/oos_rows/{date_str}.parquet"' not in src
    assert '"predictor/diagnostics/oos_rows/latest.parquet"' not in src


def test_registry_snapshot_skipped_when_not_register_in_zoo(src):
    # The champion/challenger snapshot — what makes a model promotable — is
    # gated on io.register_in_zoo so a shadow run never enters the pool.
    assert "if not io.register_in_zoo:" in src
    assert "skipping model-zoo registry" in src


def test_shadow_forces_promoted_false(src):
    assert "if not io.allow_live_promote:" in src
    # The forced-False guard sits in the promotion block.
    block = src[src.index("gate_passed = promoted"):]
    assert "if not io.allow_live_promote:" in block[:1500]


def test_manifest_records_shadow_basis(src):
    assert '"shadow_basis": io.shadow_basis,' in src


def test_load_close_threads_close_col_for_benchmarks(src):
    # SPY + sector ETF benchmark legs read the active basis (close_col); macro
    # legs keep "Close".
    assert '_load_close("SPY.parquet", col=close_col)' in src
    assert 'col=close_col)' in src  # sector ETF loop
    assert 'def _load_close(fn, col: str = "Close")' in src


def test_compute_labels_passes_close_col(src):
    assert re.search(r"compute_labels\([^)]*close_col=close_col", src, re.DOTALL)
