"""
tests/test_walk_forward_calibrator.py — Source-text and behavioral checks
for per-fold research_calibrator fitting (PR #56, 2026-04-28).

Pre-PR-#56 the calibrator was fit ONCE on the full ``score_performance``
history before the walk-forward loop and reused across every fold. That
allowed forward outcomes (post-train_end_date beat_spy_10d entries) to
shape ``research_calibrator_prob`` for earlier folds — minor leakage
that PR #55 explicitly noted as a follow-up. PR #56 fits a fold-local
calibrator inside the walk-forward loop using only score_performance
entries dated ``<= train_end_date``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_META_TRAINER = (
    Path(__file__).resolve().parent.parent
    / "training" / "meta_trainer.py"
)


@pytest.fixture(scope="module")
def meta_trainer_source() -> str:
    return _META_TRAINER.read_text()


class TestWalkForwardCalibratorSource:
    def test_loads_score_date_alongside_score_and_beat_spy(self, meta_trainer_source):
        """Without ``score_date`` we cannot filter score_performance per
        fold — the SELECT must include it."""
        # Allow any whitespace, including newlines, between "SELECT" and
        # the column list since pd.read_sql_query takes a multi-line string
        pattern = re.compile(
            r"SELECT\s+score,\s+beat_spy_10d,\s+score_date\s+FROM\s+score_performance",
            re.IGNORECASE,
        )
        assert pattern.search(meta_trainer_source), (
            "score_performance load must SELECT score_date so the fold "
            "calibrator can filter to entries dated <= train_end_date."
        )

    def test_research_score_dates_array_is_built(self, meta_trainer_source):
        """The arrays used by the per-fold filter must exist — empty
        np.array fallback is acceptable for the no-data path."""
        assert "research_score_dates" in meta_trainer_source

    def test_fold_calibrator_is_fit_inside_fold_loop(self, meta_trainer_source):
        """A per-fold ResearchCalibrator must be constructed inside the
        ``for i, fold in enumerate(folds):`` loop. Source-text check is the
        cheapest way to lock the structural invariant."""
        # Find the start of the fold loop and verify ResearchCalibrator()
        # appears AFTER it (meta_trainer.py also has a prod_calibrator
        # fit before the loop, so we can't just grep for the constructor).
        loop_start = meta_trainer_source.find("for i, fold in enumerate(folds):")
        assert loop_start != -1, "Walk-forward loop signature not found"
        loop_body_start = loop_start
        # ``fold_calibrator`` is the identifier used inside the loop.
        # Must appear after the loop start.
        cal_in_loop = meta_trainer_source.find("fold_calibrator", loop_body_start)
        assert cal_in_loop != -1 and cal_in_loop > loop_body_start, (
            "fold_calibrator must be constructed inside the walk-forward loop. "
            "Otherwise the per-fold leakage protection isn't actually applied."
        )

    def test_fold_calibrator_filters_by_train_end_date(self, meta_trainer_source):
        """The fold filter must use ``train_end`` from the fold dict —
        anything else (e.g. test_start, full history) reintroduces leakage."""
        # Look for the pattern: research_score_dates <= train_end_np
        # where train_end_np = np.datetime64(fold["train_end"], "D")
        assert re.search(
            r"fold\[\s*[\"']train_end[\"']\s*\]",
            meta_trainer_source,
        ), "Per-fold calibrator must filter score_performance by fold['train_end']"
        assert "research_score_dates <=" in meta_trainer_source, (
            "Per-fold calibrator must use ``research_score_dates <= train_end_np`` "
            "to drop forward outcomes from the fit."
        )

    def test_extract_research_features_called_with_fold_calibrator(self, meta_trainer_source):
        """The row-construction call to ``_extract_research_features`` must
        pass ``fold_calibrator`` (not ``prod_calibrator``). Otherwise the
        fold-local fit isn't actually used."""
        # Grep for the exact call shape; allow whitespace flexibility but
        # forbid prod_calibrator as the third arg.
        bad_pattern = re.compile(
            r"_extract_research_features\([^)]*prod_calibrator[^)]*\)",
            re.DOTALL,
        )
        assert not bad_pattern.search(meta_trainer_source), (
            "_extract_research_features must be called with fold_calibrator "
            "inside the walk-forward loop, not prod_calibrator (leakage)."
        )
        good_pattern = re.compile(
            r"_extract_research_features\([^)]*fold_calibrator[^)]*\)",
            re.DOTALL,
        )
        assert good_pattern.search(meta_trainer_source), (
            "Expected _extract_research_features(..., fold_calibrator) call "
            "inside the walk-forward loop."
        )


class TestProdCalibratorPreserved:
    """The production calibrator (uploaded to S3 as research_calibrator.json)
    must still be fit on full score_performance history — that's what
    inference reads at runtime."""

    def test_prod_calibrator_fit_before_fold_loop(self, meta_trainer_source):
        """``prod_calibrator.fit`` must be called BEFORE the walk-forward
        loop so the production artifact is final by the time S3 upload
        runs at the end of training."""
        prod_fit = meta_trainer_source.find("prod_calibrator.fit(")
        loop_start = meta_trainer_source.find("for i, fold in enumerate(folds):")
        assert prod_fit != -1, "prod_calibrator.fit must still be called"
        assert prod_fit < loop_start, (
            "prod_calibrator must be fit before the walk-forward loop. "
            "If it moved inside the loop, the production artifact won't "
            "have full-history calibration."
        )

    def test_prod_calibrator_uses_full_history(self, meta_trainer_source):
        """The prod calibrator's fit call must use the full unfiltered
        ``research_scores`` / ``research_beat_spy`` arrays."""
        # Look for `prod_calibrator.fit(research_scores, research_beat_spy)`
        assert re.search(
            r"prod_calibrator\.fit\(\s*research_scores\s*,\s*research_beat_spy\s*\)",
            meta_trainer_source,
        ), (
            "prod_calibrator.fit must call .fit(research_scores, research_beat_spy) "
            "with the unfiltered arrays. Filtering would defeat the production "
            "calibrator's purpose (inference reads it at runtime)."
        )
