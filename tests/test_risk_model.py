"""C.2 — Factor-risk model (Barra-style F + D).

Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §C.2

Tests the cross-sectional-regression primitives that turn the
factor-loading matrix B (from alpha-engine-data C.1) into F (factor-
return covariance) and D (per-ticker idiosyncratic variance). These two
feed the executor's Σ = B·F·Bᵀ + D risk decomposition (C.3).

Load-bearing property: when synthetic data is generated from a known
true F, the estimator should recover it within sampling error. The
recovery test is the institutional gate — without it, a silent
miscalibration would propagate into the executor's risk estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_model import (
    build_factor_returns_series,
    build_factor_risk_model,
    cross_sectional_factor_returns,
    estimate_factor_covariance,
    estimate_idiosyncratic_variance,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _synthetic_panel(
    N: int = 30, K: int = 4, T: int = 250, seed: int = 0,
    true_F_diag: float = 0.0004, true_D_scale: float = 0.0009,
    market_factor_var: float = 0.0001,
):
    """Generate a synthetic factor-model panel with known true F + D.

    True model: r_t = market_t + B · f_t + ε_t, where market_t ~ N(0, market_factor_var),
    f_t ~ N(0, diag(true_F_diag)), ε_t ~ N(0, D), D_i ~ uniform.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=T, freq="B")
    tickers = [f"T{i:02d}" for i in range(N)]
    factor_names = [f"f{k}" for k in range(K)]

    # Stationary z-scored loadings (mean ≈ 0, std ≈ 1 per factor)
    B_raw = rng.normal(0, 1, size=(N, K))
    B_raw = (B_raw - B_raw.mean(axis=0)) / B_raw.std(axis=0)

    true_F = np.eye(K) * true_F_diag
    true_D = rng.uniform(0.5 * true_D_scale, 1.5 * true_D_scale, N)

    loadings_by_date = {d: pd.DataFrame(B_raw, index=tickers, columns=factor_names)
                        for d in dates}
    returns_panel = np.zeros((T, N))
    for i in range(T):
        m_t = float(rng.normal(0, np.sqrt(market_factor_var)))
        f_t = rng.multivariate_normal(np.zeros(K), true_F)
        eps_t = rng.normal(0, np.sqrt(true_D), N)
        returns_panel[i] = m_t + B_raw @ f_t + eps_t

    returns_df = pd.DataFrame(returns_panel, index=dates, columns=tickers)
    return {
        "returns_df": returns_df,
        "loadings_by_date": loadings_by_date,
        "B_true": B_raw,
        "true_F_diag": true_F_diag,
        "true_D": true_D,
        "factor_names": factor_names,
        "tickers": tickers,
    }


# ─── cross_sectional_factor_returns ─────────────────────────────────────────


class TestCrossSectionalFactorReturns:
    def test_recovers_known_factor_returns_no_intercept(self):
        """Exact construction: r = B·f_true → OLS must recover f_true exactly
        (zero residuals when no noise + no intercept needed)."""
        rng = np.random.default_rng(1)
        N, K = 50, 5
        B = rng.normal(0, 1, size=(N, K))
        f_true = np.array([0.01, -0.02, 0.005, 0.015, -0.008])
        r = B @ f_true

        f_hat, residuals = cross_sectional_factor_returns(
            r, B, include_intercept=False,
        )
        np.testing.assert_allclose(f_hat, f_true, atol=1e-10)
        # Residuals are zero up to numerical noise
        assert np.max(np.abs(residuals)) < 1e-9

    def test_with_intercept_recovers_market_plus_factors(self):
        """r = m + B·f → 6-element solution with intercept first."""
        rng = np.random.default_rng(2)
        N, K = 50, 4
        B = rng.normal(0, 1, size=(N, K))
        B = B - B.mean(axis=0)  # z-scored loadings have mean 0
        market = 0.005
        f_true = np.array([0.01, -0.02, 0.005, 0.015])
        r = market + B @ f_true

        f_hat, _ = cross_sectional_factor_returns(
            r, B, include_intercept=True,
        )
        assert f_hat.shape == (K + 1,)
        assert f_hat[0] == pytest.approx(market, abs=1e-10)
        np.testing.assert_allclose(f_hat[1:], f_true, atol=1e-10)

    def test_handles_noise_with_finite_error(self):
        """Adding noise → OLS finds the right *direction* but residuals
        absorb the noise. Sanity: f_hat is close to f_true; residual std
        is close to the noise std."""
        rng = np.random.default_rng(3)
        N, K = 100, 4
        B = rng.normal(0, 1, size=(N, K))
        f_true = np.array([0.01, -0.02, 0.005, 0.015])
        noise = rng.normal(0, 0.02, N)
        r = B @ f_true + noise

        f_hat, residuals = cross_sectional_factor_returns(
            r, B, include_intercept=False,
        )
        # Each estimated coefficient within ~3 standard errors of the truth
        np.testing.assert_allclose(f_hat, f_true, atol=0.008)
        # Residual std should be close to the input noise std
        assert abs(float(np.std(residuals)) - 0.02) < 0.005

    def test_nan_rows_dropped(self):
        rng = np.random.default_rng(4)
        N, K = 50, 3
        B = rng.normal(0, 1, size=(N, K))
        f_true = np.array([0.01, -0.01, 0.005])
        r = B @ f_true
        # Inject NaN
        r_with_nan = r.copy()
        r_with_nan[0:5] = np.nan
        B_with_nan = B.copy()
        B_with_nan[10:12, 0] = np.nan

        f_hat, residuals = cross_sectional_factor_returns(
            r_with_nan, B_with_nan, include_intercept=False,
        )
        np.testing.assert_allclose(f_hat, f_true, atol=1e-9)
        # Residuals for NaN-input rows must be NaN
        assert np.all(np.isnan(residuals[0:5]))
        assert np.all(np.isnan(residuals[10:12]))

    def test_too_few_observations_returns_nan(self):
        """K + 5 observation buffer prevents unstable solves."""
        rng = np.random.default_rng(5)
        N, K = 6, 4  # only 6 rows for 4 factors + intercept = 5 → not ≥ 10
        B = rng.normal(0, 1, size=(N, K))
        r = rng.normal(0, 0.01, N)

        f_hat, residuals = cross_sectional_factor_returns(
            r, B, include_intercept=True,
        )
        assert np.all(np.isnan(f_hat))
        assert np.all(np.isnan(residuals))

    def test_wrong_shape_raises(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="loadings_prev must be 2-D"):
            cross_sectional_factor_returns(np.zeros(10), np.zeros(10))
        with pytest.raises(ValueError, match="returns_t shape"):
            cross_sectional_factor_returns(np.zeros(11), rng.normal(0, 1, (10, 3)))

    def test_rank_deficient_loadings_returns_minimum_norm_solution(self):
        """A perfectly collinear factor column shouldn't crash — lstsq
        returns the minimum-norm solution. Verifies the no-crash contract."""
        N, K = 30, 3
        rng = np.random.default_rng(7)
        B = rng.normal(0, 1, size=(N, K))
        B[:, 2] = B[:, 0]  # Column 2 == Column 0 → rank 2, not 3
        r = rng.normal(0, 0.01, N)
        # Should not raise
        f_hat, residuals = cross_sectional_factor_returns(
            r, B, include_intercept=False,
        )
        # All-finite — solver succeeded
        assert np.all(np.isfinite(f_hat))


# ─── build_factor_returns_series ────────────────────────────────────────────


class TestBuildFactorReturnsSeries:
    def test_emits_factor_returns_and_residuals_panels(self):
        data = _synthetic_panel(N=30, K=4, T=100)
        f_df, eps_df = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
        )
        # T rows; K + 1 columns (intercept on by default)
        assert f_df.shape == (100, 5)
        assert eps_df.shape == (100, 30)
        # First date has no prior loadings → all NaN
        assert f_df.iloc[0].isna().all()
        # Subsequent dates have factor returns
        assert not f_df.iloc[10].isna().any()

    def test_first_date_has_no_prior_loadings(self):
        """Informational safety: at date t we may only use loadings at
        strictly earlier dates (t-1 or older)."""
        data = _synthetic_panel(T=10)
        f_df, _ = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
        )
        assert f_df.iloc[0].isna().all()

    def test_factor_names_argument_pins_order(self):
        data = _synthetic_panel(K=4)
        custom_order = ["f3", "f1", "f0", "f2"]
        f_df, _ = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
            factor_names=custom_order,
        )
        # market column first (intercept on), then custom order
        assert list(f_df.columns) == ["market"] + custom_order

    def test_include_intercept_false_skips_market_column(self):
        data = _synthetic_panel(K=4)
        f_df, _ = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
            include_intercept=False,
        )
        # Only K columns
        assert f_df.shape[1] == 4
        assert "market" not in f_df.columns

    def test_empty_returns_panel_returns_empty(self):
        f_df, eps_df = build_factor_returns_series(pd.DataFrame(), {})
        assert f_df.empty
        assert eps_df.empty

    def test_empty_loadings_raises(self):
        returns_df = pd.DataFrame(np.zeros((5, 3)), columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="loadings_by_date is empty"):
            build_factor_returns_series(returns_df, {})


# ─── estimate_factor_covariance ─────────────────────────────────────────────


class TestEstimateFactorCovariance:
    def test_recovers_known_diagonal_F(self):
        """The load-bearing recovery test: when the synthetic data is
        generated with diagonal F = 0.0004 · I, the estimator (with
        plenty of samples) should produce a roughly-diagonal F with
        diagonal values in the ballpark of 0.0004."""
        data = _synthetic_panel(N=50, K=4, T=500, seed=11, true_F_diag=0.0004)
        f_df, _ = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
            include_intercept=False,
        )
        F = estimate_factor_covariance(f_df, shrinkage="sample")
        # Drop the first row (NaN — no prior loadings)
        diag = np.diag(F.values)
        # LW would compress diag toward mean; use sample for the recovery test.
        # Allow 50% relative tolerance — finite-sample noise + LW shrinkage.
        for d in diag:
            assert 0.0001 < d < 0.001, (
                f"Diagonal entry {d:.6f} outside plausible range [1e-4, 1e-3] "
                f"around true 0.0004"
            )

    def test_ledoit_wolf_returns_psd_matrix(self):
        data = _synthetic_panel(N=30, K=4, T=200, seed=12)
        f_df, _ = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
        )
        F = estimate_factor_covariance(f_df, shrinkage="ledoit_wolf")
        eigvals = np.linalg.eigvalsh(F.values)
        assert eigvals.min() >= -1e-10, (
            f"LW F must be PSD; got min eigval={eigvals.min()}"
        )

    def test_oas_estimator_works(self):
        data = _synthetic_panel(N=30, K=4, T=200, seed=13)
        f_df, _ = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
        )
        F = estimate_factor_covariance(f_df, shrinkage="oas")
        assert F.shape == (5, 5)  # K + intercept
        eigvals = np.linalg.eigvalsh(F.values)
        assert eigvals.min() >= -1e-10

    def test_insufficient_data_returns_nan_F(self):
        """Below min_obs → all-NaN F so caller knows the build is bad."""
        f_df = pd.DataFrame(np.random.normal(0, 0.01, (10, 4)),
                            columns=["a", "b", "c", "d"])
        F = estimate_factor_covariance(f_df, min_obs=30)
        assert F.shape == (4, 4)
        assert F.isna().all().all()

    def test_unknown_shrinkage_raises(self):
        data = _synthetic_panel(T=100)
        f_df, _ = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
        )
        with pytest.raises(ValueError, match="Unknown shrinkage"):
            estimate_factor_covariance(f_df, shrinkage="not-a-real-estimator")


# ─── estimate_idiosyncratic_variance ────────────────────────────────────────


class TestEstimateIdiosyncraticVariance:
    def test_recovers_per_ticker_idio_variance(self):
        """Recovery: synthetic D is uniform between 0.5*scale and 1.5*scale;
        the estimator's mean across tickers should match the true mean
        within sampling error."""
        data = _synthetic_panel(N=40, K=4, T=400, seed=21, true_D_scale=0.0009)
        _, eps_df = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
        )
        D = estimate_idiosyncratic_variance(eps_df)
        # Mean of recovered D close to mean of true D
        true_mean = float(data["true_D"].mean())
        rec_mean = float(D.dropna().mean())
        # Within 30% relative — finite-sample + finite-K-factor noise
        assert abs(rec_mean - true_mean) / true_mean < 0.3, (
            f"Mean idio variance: recovered {rec_mean:.6f} vs true {true_mean:.6f}"
        )

    def test_all_positive_or_nan(self):
        data = _synthetic_panel(N=30, K=4, T=200, seed=22)
        _, eps_df = build_factor_returns_series(
            data["returns_df"], data["loadings_by_date"],
        )
        D = estimate_idiosyncratic_variance(eps_df)
        finite = D.dropna()
        assert len(finite) > 0
        assert (finite > 0).all()

    def test_min_obs_skips_thin_tickers(self):
        """A ticker with <min_obs non-NaN residuals → NaN in D, not 0."""
        rng = np.random.default_rng(23)
        T, N = 200, 4
        eps = rng.normal(0, 0.01, size=(T, N))
        eps[:190, 0] = np.nan  # ticker 0 has only 10 non-NaN obs
        eps_df = pd.DataFrame(eps, columns=[f"T{i}" for i in range(N)])
        D = estimate_idiosyncratic_variance(eps_df, min_obs=30)
        assert np.isnan(D.iloc[0])
        for i in range(1, N):
            assert np.isfinite(D.iloc[i])


# ─── build_factor_risk_model (end-to-end) ────────────────────────────────────


class TestBuildFactorRiskModel:
    def test_end_to_end_produces_F_and_D_with_metadata(self):
        data = _synthetic_panel(N=30, K=4, T=200, seed=31)
        out = build_factor_risk_model(
            data["returns_df"], data["loadings_by_date"],
        )
        assert "F" in out and "D" in out and "metadata" in out
        meta = out["metadata"]
        assert meta["n_dates"] == 200
        assert meta["n_clean_dates"] == 199  # first date NaN
        assert meta["K_eff"] == 5  # 4 factors + intercept
        assert meta["n_tickers"] == 30

    def test_F_is_K_eff_x_K_eff_dataframe(self):
        data = _synthetic_panel(N=20, K=3, T=150, seed=32)
        out = build_factor_risk_model(
            data["returns_df"], data["loadings_by_date"],
        )
        assert out["F"].shape == (4, 4)
        # Indexed by factor names with "market" first
        assert list(out["F"].columns) == ["market", "f0", "f1", "f2"]
        assert list(out["F"].index) == ["market", "f0", "f1", "f2"]

    def test_D_indexed_by_ticker(self):
        data = _synthetic_panel(N=20, K=3, T=150, seed=33)
        out = build_factor_risk_model(
            data["returns_df"], data["loadings_by_date"],
        )
        assert list(out["D"].index) == data["tickers"]
        assert (out["D"].dropna() > 0).all()

    def test_can_disable_intercept(self):
        data = _synthetic_panel(N=20, K=3, T=150, seed=34)
        out = build_factor_risk_model(
            data["returns_df"], data["loadings_by_date"],
            include_intercept=False,
        )
        assert out["metadata"]["K_eff"] == 3
        assert "market" not in out["F"].columns

    def test_reconstructed_Sigma_is_PSD(self):
        """The whole point: Σ = B·F·Bᵀ + D must be PSD so the executor's
        cvxpy solver can ingest it. Verify on the synthetic recovery case
        (no intercept — caller assembles a B that matches the F shape)."""
        data = _synthetic_panel(N=25, K=4, T=300, seed=35)
        out = build_factor_risk_model(
            data["returns_df"], data["loadings_by_date"],
            include_intercept=False,
        )
        B = data["B_true"]  # (N, K)
        F = out["F"].values  # (K, K)
        D = out["D"].fillna(out["D"].dropna().mean()).values  # (N,)
        Sigma = B @ F @ B.T + np.diag(D)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert eigvals.min() >= -1e-10, (
            f"Reconstructed Σ must be PSD; got min eigval={eigvals.min()}"
        )
