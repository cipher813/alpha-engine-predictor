"""Tests for the peak-RSS-by-phase instrumentation (config#688).

Origin: the meta-trainer's persisted ``peak_rss_mb`` was previously seeded
by a bare ``peak_rss_mb = _log_rss(...)`` assignment deep in the Step 8
volatility-fit region. Every earlier phase — the Step 3b canonical-alpha
(date×ticker) build, the Step 4 regime-feature build, the Step 4b
macro-array allocation (the 2026-06-01 "OOM hotspot"), and the Step 6
walk-forward loop — ran *before* that init, so a peak in any of them was
invisible to the manifest. config#688 asks to characterize peak-RSS *by
phase*; the fix seeds the running max right after the streaming array
build and threads it as ``max(peak_rss_mb, _log_rss(...))`` through every
phase boundary.

This test pins:
- ``_log_rss`` is a soft-failing int-returning probe (never blocks training).
- ``peak_rss_mb`` is seeded BEFORE the late volatility-fit region, so the
  early-phase peaks contribute to the persisted manifest value.
- The named early phases (Step 3b / Step 4 / Step 4b / Step 6) each carry
  a probe folded into the running max via ``max(peak_rss_mb, _log_rss(...))``.
- The late-phase site no longer *re-initializes* the running max with a
  bare ``peak_rss_mb = _log_rss(...)`` (that was the regression).
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import training.meta_trainer as mt

_SRC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "meta_trainer.py",
)
_SRC = open(_SRC_PATH).read()


def test_log_rss_returns_int_and_soft_fails():
    """``_log_rss`` returns an int and never raises (defensive no-op path)."""
    val = mt._log_rss("unit-test probe")
    assert isinstance(val, int)
    assert val >= 0


def _line_of(needle: str) -> int:
    for i, line in enumerate(_SRC.splitlines(), start=1):
        if needle in line:
            return i
    raise AssertionError(f"marker not found in meta_trainer.py: {needle!r}")


def test_peak_rss_seeded_before_late_volatility_fit():
    """The running max is seeded in the early phases, not at the Step 8 fit.

    Regression guard: if ``peak_rss_mb`` is only first assigned at the
    'before prod_vol (plain) fit' site, the early-phase peaks (canonical
    alpha / regime / macro-array / walk-forward) never reach the manifest.
    """
    seed_line = _line_of('"after streaming array build (Step 1-3)"')
    late_fit_line = _line_of('"before prod_vol (plain) fit"')
    assert seed_line < late_fit_line, (
        "peak_rss_mb must be seeded before the late volatility-fit region"
    )
    # The seed is a plain assignment; every later phase folds into the max.
    assert re.search(
        r"peak_rss_mb\s*=\s*_log_rss\(\s*\"after streaming array build", _SRC
    ), "expected the early seed `peak_rss_mb = _log_rss('after streaming array build...')`"


def test_named_early_phases_each_carry_a_running_max_probe():
    """Each phase the OOM investigation named has a max()-folded RSS probe."""
    for label in (
        "after Step 3b canonical-alpha labels",
        "after Step 4 regime-feature build",
        "after Step 4b macro-array build (OOM hotspot)",
        "after Step 6 walk-forward loop",
    ):
        assert re.search(
            r"peak_rss_mb\s*=\s*max\(\s*peak_rss_mb,\s*_log_rss\(\s*\"" + re.escape(label),
            _SRC,
        ), f"missing running-max RSS probe for phase: {label!r}"


def test_late_fit_site_no_longer_reinitializes_peak():
    """The Step 8 'before prod_vol (plain) fit' probe folds into the max.

    Before config#688 it was a bare `peak_rss_mb = _log_rss(...)` that
    clobbered the early-phase running max.
    """
    assert re.search(
        r"peak_rss_mb\s*=\s*max\(\s*peak_rss_mb,\s*_log_rss\(\s*\"before prod_vol \(plain\) fit",
        _SRC,
    ), "the 'before prod_vol (plain) fit' probe must fold into the running max"
    assert not re.search(
        r"peak_rss_mb\s*=\s*_log_rss\(\s*\"before prod_vol \(plain\) fit",
        _SRC,
    ), "late-phase site must not re-initialize peak_rss_mb (drops early peaks)"
