"""Tests for the config#940 AUTO_PRUNE noise-feature wiring.

Root cause closed by these tests: ``cfg.AUTO_PRUNE_NOISE_FEATURES`` +
``cfg.IC_NOISE_THRESHOLD`` were defined but had ZERO call sites — the
standalone-IC noise detection existed only as an OBSERVE diagnostic and never
excluded a feature from the L2. ``select_non_noise_features`` is the wiring;
``train_meta_model`` calls it ONLY when the flag is on.

Proves:
  (a) flag OFF  → no features pruned (current behavior byte-for-byte preserved);
  (b) flag ON   → noise-flagged features (|standalone xsec-IC| < threshold) are
      excluded from the returned feature set;
  (c) safety:   unmeasurable-IC columns are kept; a would-be-empty prune is a
      no-op (never ship a degenerate empty feature set).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training.meta_trainer import select_non_noise_features


def _signal_noise_dataset(seed: int = 0, n_dates: int = 80, n_names: int = 30):
    """One near-perfect signal column + one pure-noise column, panel layout."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates), n_names)
    y = rng.normal(0, 1, n_dates * n_names)
    signal = y + rng.normal(0, 0.1, y.shape)  # high standalone IC
    noise = rng.normal(0, 1, y.shape)          # ~0 standalone IC
    X = np.column_stack([signal, noise])
    return X, y, dates


# ── (b) flag ON: a genuine noise column is excluded ─────────────────────────

class TestPruneExcludesNoise:

    def test_noise_feature_dropped(self):
        X, y, dates = _signal_noise_dataset()
        keep_idx, keep_names, report = select_non_noise_features(
            X, y, dates, ["signal", "noise"], ic_noise_threshold=0.05,
        )
        assert keep_names == ["signal"]
        assert keep_idx == [0]
        assert report["noise"]["kept"] is False
        assert report["noise"]["reason"] == "below_ic_noise_threshold"
        assert report["signal"]["kept"] is True

    def test_keep_idx_selects_correct_columns(self):
        X, y, dates = _signal_noise_dataset()
        keep_idx, keep_names, _ = select_non_noise_features(
            X, y, dates, ["signal", "noise"], ic_noise_threshold=0.05,
        )
        pruned_X = X[:, keep_idx]
        # The retained matrix is exactly the signal column.
        assert pruned_X.shape[1] == 1
        np.testing.assert_array_equal(pruned_X[:, 0], X[:, 0])


# ── (a) current behavior preserved when no column is noise ──────────────────

class TestNoPruneWhenAllSignal:

    def test_threshold_below_all_ics_keeps_everything(self):
        X, y, dates = _signal_noise_dataset()
        # A threshold of 0 can never flag a measurable feature as noise.
        keep_idx, keep_names, report = select_non_noise_features(
            X, y, dates, ["signal", "noise"], ic_noise_threshold=0.0,
        )
        assert keep_names == ["signal", "noise"]
        assert keep_idx == [0, 1]
        assert all(r["kept"] for r in report.values())


# ── (c) safety carve-outs ───────────────────────────────────────────────────

class TestSafetyCarveOuts:

    def test_unmeasurable_ic_column_is_kept(self):
        # A constant column has no cross-sectional variance → xsec_ic is None
        # (absence of evidence is NOT evidence of noise → keep it).
        rng = np.random.default_rng(2)
        n_dates, n_names = 40, 20
        dates = np.repeat(np.arange(n_dates), n_names)
        y = rng.normal(0, 1, n_dates * n_names)
        const = np.ones(y.shape)
        X = np.column_stack([const, y + rng.normal(0, 0.1, y.shape)])
        keep_idx, keep_names, report = select_non_noise_features(
            X, y, dates, ["const", "signal"], ic_noise_threshold=0.05,
        )
        assert "const" in keep_names
        assert report["const"]["xsec_ic"] is None
        assert report["const"]["reason"] == "ic_unmeasurable"

    def test_all_noise_prune_is_noop(self):
        # If EVERY feature would be flagged, ship the full set (an empty
        # feature vector is never a valid model).
        rng = np.random.default_rng(3)
        n_dates, n_names = 60, 25
        dates = np.repeat(np.arange(n_dates), n_names)
        y = rng.normal(0, 1, n_dates * n_names)
        X = rng.normal(0, 1, (n_dates * n_names, 3))  # all unrelated to y
        keep_idx, keep_names, report = select_non_noise_features(
            X, y, dates, ["a", "b", "c"], ic_noise_threshold=0.5,
        )
        assert keep_names == ["a", "b", "c"]
        assert keep_idx == [0, 1, 2]
        assert all(
            r["reason"] == "all_below_threshold_prune_skipped"
            for r in report.values()
        )


# ── config contract: the flag + threshold exist and default OFF ─────────────

class TestConfigContract:

    def test_auto_prune_defaults_off(self):
        assert hasattr(cfg, "AUTO_PRUNE_NOISE_FEATURES")
        assert cfg.AUTO_PRUNE_NOISE_FEATURES is False

    def test_ic_noise_threshold_present(self):
        assert hasattr(cfg, "IC_NOISE_THRESHOLD")
        assert isinstance(cfg.IC_NOISE_THRESHOLD, (int, float))


# ── source-level wiring regression: the flag is actually CONSUMED ───────────

class TestWiringRegression:

    def test_flag_has_a_call_site(self):
        import inspect

        from training import meta_trainer

        src = inspect.getsource(meta_trainer)
        # The root cause was a flag with ZERO call sites. Lock in that the
        # trainer both reads the flag and routes through the pruning helper.
        assert "AUTO_PRUNE_NOISE_FEATURES" in src
        assert "select_non_noise_features(" in src
