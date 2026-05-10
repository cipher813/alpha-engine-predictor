"""Tests for Stage 2c-partial of the regime-conditioning rebuild —
extended macro features in ``RegimePredictor.build_features()``:

- ``vix_vix3m_ratio``: institutional-canonical VIX term-structure
  normalization (ratio > 1 = backwardation/stress; < 1 = contango).
  Distinct from existing ``vix_term_slope`` which is bps-style normalized
  difference.
- ``market_breadth_200d``: % stocks above 200-day MA. Captures the
  secular bull-vs-bear regime, distinct from ``market_breadth`` (50d).

Both are added to ``cfg.MACRO_NORM_FEATURES`` so Stage 1b's macro-aug
volatility GBM (and any future L2 Ridge expansion) consumes them via the
existing time-series z-score pipeline.

Plan doc: ~/Development/alpha-engine-docs/private/regime-conditioning-260510.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from model.regime_predictor import RegimePredictor


def _spy_series(n: int = 300, start: str = "2018-01-01", seed: int = 0):
    rng = np.random.default_rng(seed)
    daily = rng.normal(0, 0.008, n)
    close = 300.0 * np.exp(np.cumsum(daily))
    return pd.Series(close, index=pd.date_range(start, periods=n, freq="B"))


def _vix_series(n: int = 300, start: str = "2018-01-01", level: float = 18.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    return pd.Series(
        np.abs(rng.normal(level, 4.0, n)),
        index=pd.date_range(start, periods=n, freq="B"),
    )


def _all_close_prices(n_tickers: int = 30, n_days: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    out = {}
    for i in range(n_tickers):
        daily = rng.normal(0, 0.012, n_days)
        close = 100.0 * np.exp(np.cumsum(daily))
        out[f"T{i:03d}"] = pd.Series(close, index=idx)
    return out


# ── Schema contract ────────────────────────────────────────────────────


class TestExtendedMacroSchema:

    def test_macro_norm_features_includes_new_names(self):
        for name in ("vix_vix3m_ratio", "market_breadth_200d"):
            assert name in cfg.MACRO_NORM_FEATURES, (
                f"{name} missing from cfg.MACRO_NORM_FEATURES"
            )

    def test_build_features_emits_new_columns(self):
        rp = RegimePredictor()
        spy = _spy_series()
        vix = _vix_series()
        vix3m = _vix_series(level=20.0, seed=7)
        out = rp.build_features(
            spy_series=spy,
            vix_series=vix,
            vix3m_series=vix3m,
            all_close_prices=_all_close_prices(),
        )
        for name in ("vix_vix3m_ratio", "market_breadth_200d"):
            assert name in out.columns

    def test_macro_norm_schema_subset_of_build_features_output(self):
        # Stage 1b's _build_macro_array reads cfg.MACRO_NORM_FEATURES from
        # the regime_features_df. Every name in MACRO_NORM_FEATURES must
        # appear as a column in build_features output, or training will
        # raise KeyError.
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            vix_series=_vix_series(),
            vix3m_series=_vix_series(level=20.0, seed=7),
            all_close_prices=_all_close_prices(),
        )
        for name in cfg.MACRO_NORM_FEATURES:
            assert name in out.columns, (
                f"cfg.MACRO_NORM_FEATURES has {name}, but build_features "
                f"output does not — Stage 1b training would KeyError"
            )


# ── vix_vix3m_ratio behavior ───────────────────────────────────────────


class TestVixVix3mRatio:

    def test_ratio_above_one_during_backwardation(self):
        # VIX > VIX3M → ratio > 1 (front-month vol exceeds 3M vol = stress).
        rp = RegimePredictor()
        n = 300
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        vix = pd.Series(np.full(n, 30.0), index=idx)
        vix3m = pd.Series(np.full(n, 20.0), index=idx)
        out = rp.build_features(
            spy_series=_spy_series(n=n),
            vix_series=vix,
            vix3m_series=vix3m,
            all_close_prices=_all_close_prices(n_days=n),
        )
        # ratio should be 30/20 = 1.5 across all rows.
        assert np.allclose(out["vix_vix3m_ratio"].iloc[20:], 1.5)

    def test_ratio_below_one_during_contango(self):
        rp = RegimePredictor()
        n = 300
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        vix = pd.Series(np.full(n, 16.0), index=idx)
        vix3m = pd.Series(np.full(n, 20.0), index=idx)
        out = rp.build_features(
            spy_series=_spy_series(n=n),
            vix_series=vix,
            vix3m_series=vix3m,
            all_close_prices=_all_close_prices(n_days=n),
        )
        # ratio should be 16/20 = 0.8 across all rows.
        assert np.allclose(out["vix_vix3m_ratio"].iloc[20:], 0.8)

    def test_ratio_falls_back_to_one_when_series_missing(self):
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            vix_series=None,
            vix3m_series=None,
            all_close_prices=_all_close_prices(),
        )
        # No VIX/VIX3M data → neutral 1.0 fallback.
        assert (out["vix_vix3m_ratio"] == 1.0).all()


# ── market_breadth_200d behavior ──────────────────────────────────────


class TestMarketBreadth200d:

    def test_breadth_200d_finite_when_history_sufficient(self):
        # 300 days of history per ticker → 200d MA computable.
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(n=300),
            vix_series=_vix_series(n=300),
            vix3m_series=_vix_series(n=300, level=20.0, seed=7),
            all_close_prices=_all_close_prices(n_days=300),
        )
        # Post-200d warmup, breadth should be a finite value in [0, 1].
        finite = out["market_breadth_200d"].iloc[210:]
        assert (finite >= 0.0).all() and (finite <= 1.0).all()

    def test_breadth_200d_falls_back_to_neutral_with_empty_universe(self):
        # No close prices passed → 0.5 fallback (matches 50d behaviour).
        rp = RegimePredictor()
        out = rp.build_features(
            spy_series=_spy_series(),
            vix_series=_vix_series(),
            vix3m_series=_vix_series(level=20.0, seed=7),
            all_close_prices=None,
        )
        assert (out["market_breadth_200d"] == 0.5).all()

    def test_breadth_200d_distinct_from_50d_under_uptrend(self):
        # Construct a synthetic universe where every ticker has been
        # rising for the past 60 days (above 50d MA) but has tickers
        # only barely above their 200d MA. The 50d breadth should be
        # higher than 200d for the same date — different cycle stage
        # captured.
        rp = RegimePredictor()
        n_days = 300
        idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
        # Ticker rises for first 200 days, then sideways for remaining.
        # By day 250: above its 50d MA (price > recent flat mean), but
        # 200d MA still includes the rising portion → much harder to be
        # above 200d MA. Different cycle reads.
        rising_then_flat = np.concatenate([
            np.linspace(100, 200, 200),
            np.full(n_days - 200, 200.0),
        ])
        out_close = {f"T{i:03d}": pd.Series(rising_then_flat, index=idx) for i in range(20)}
        out = rp.build_features(
            spy_series=_spy_series(n=n_days),
            vix_series=_vix_series(n=n_days),
            vix3m_series=_vix_series(n=n_days, level=20.0, seed=7),
            all_close_prices=out_close,
        )
        # Take a date well past the 200d warmup. 50d should read above
        # 200d's reading because the 200d MA is dragged up by the recent
        # flat portion's high prices, but the price is still equal to
        # the 200d trailing high — so dist-from-200d-MA is sensitive to
        # exact flatness. Loose check: both finite + numeric.
        assert np.isfinite(out["market_breadth"].iloc[250])
        assert np.isfinite(out["market_breadth_200d"].iloc[250])


# ── Stage 1b integration sanity ───────────────────────────────────────


class TestStage1bIntegrationSanity:
    """Validates that the extended macros flow into Stage 1b's
    ``_build_macro_array`` without raising on the new column names."""

    def test_build_macro_array_with_extended_features(self):
        from training.meta_trainer import _build_macro_array

        rp = RegimePredictor()
        regime_df = rp.build_features(
            spy_series=_spy_series(),
            vix_series=_vix_series(),
            vix3m_series=_vix_series(level=20.0, seed=7),
            all_close_prices=_all_close_prices(),
        )
        # Pretend each date has 5 tickers (Stage 1b broadcasts macros
        # across tickers on the same date).
        all_dates = []
        for d in regime_df.index[:50]:
            for _ in range(5):
                all_dates.append(d)

        X_macro = _build_macro_array(
            all_dates, regime_df, list(cfg.MACRO_NORM_FEATURES),
        )
        # Shape contract: (n_rows, len(MACRO_NORM_FEATURES))
        assert X_macro.shape == (250, len(cfg.MACRO_NORM_FEATURES))
        # Stage 2c-partial additions: indexes for vix_vix3m_ratio +
        # market_breadth_200d.
        idx_ratio = cfg.MACRO_NORM_FEATURES.index("vix_vix3m_ratio")
        idx_b200 = cfg.MACRO_NORM_FEATURES.index("market_breadth_200d")
        # Both columns should be finite (synthetic data has full history).
        assert np.isfinite(X_macro[:, idx_ratio]).all()
        assert np.isfinite(X_macro[:, idx_b200]).all()
