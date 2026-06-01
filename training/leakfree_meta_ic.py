"""Leak-free walk-forward OOS meta IC — ROADMAP L4469 W1.1a (OBSERVE mode).

Why this exists
---------------
The production "meta IC" the promotion gate fires on (``meta_model._val_ic``,
and the manifest's ``meta_model_oos_ic`` = ``horizon_ics["21d"]["spearman"]``)
is **in-sample at the meta level**. The L2 (``BayesianRidge``) is fit on the
FULL stack of walk-forward OOS L1-prediction rows, then its IC is read back on
that SAME stack (``meta_model.predict(meta_X)`` over the rows it was fit on —
see ``meta_trainer.py`` where the variable is literally named
``meta_preds_oos_insample``). The L1 predictions are out-of-sample but the L2
aggregation is not, so the number is inflated (~0.50 on 2026-05-30) vs the
Gu-Kelly-Xiu (2020) realistic monthly-OOS ceiling of IC ≈ 0.03–0.07.

What this computes
------------------
An honest meta IC via a SECOND-level expanding-window walk-forward over the
meta rows: per fold, fit a fresh L2 on the fold's train dates — **purged**
``forward_days`` before the test block (kills the 21d overlapping-label leak)
and **embargoed** ``embargo_days`` after (load-bearing once W1.2 CPCV lands;
a no-op in pure expanding-forward WF where train is always before test) —
predict the held-out test dates, stack the OOS meta predictions, and report
the **cross-sectional rank IC** (per-date Spearman averaged — the
Fama-MacBeth / GKX standard, since the system ranks tickers WITHIN a day to
pick buys) alongside the pooled Spearman, with a date-blocked bootstrap CI.

OBSERVE MODE: callers report the result; they MUST NOT gate promotion on it
yet. W1.4 flips the gate from the in-sample ``_val_ic`` to this number after
2–3 Saturday firings confirm it (and after restating the bar vs 0.03–0.07).
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.stats import spearmanr


def cross_sectional_ic_series(preds, y, dates) -> list[float]:
    """Per-date Spearman rank IC series (one IC per date with ≥2 finite,
    non-degenerate name pairs). This is the "returns" series the W1.3 Deflated
    Sharpe Ratio operates on; its mean is the cross-sectional rank IC."""
    preds = np.asarray(preds, dtype=float)
    y = np.asarray(y, dtype=float)
    dates = np.asarray(dates)
    per_date: list[float] = []
    for d in np.unique(dates):
        m = (dates == d) & np.isfinite(preds) & np.isfinite(y)
        if m.sum() < 2:
            continue
        if np.std(preds[m]) < 1e-12 or np.std(y[m]) < 1e-12:
            continue
        sp = spearmanr(preds[m], y[m]).correlation
        if np.isfinite(sp):
            per_date.append(float(sp))
    return per_date


def cross_sectional_rank_ic(preds, y, dates) -> tuple[float, int]:
    """Mean of the per-date Spearman rank IC (Fama-MacBeth / GKX standard).

    A pooled Spearman across all (name, date) rows conflates time-series with
    cross-sectional skill and inflates the estimate — this is the
    cross-sectional number that matches how the system actually trades (rank
    tickers within a day). Returns ``(mean_ic, n_dates_used)``; ``(nan, 0)``
    if no date is usable.
    """
    series = cross_sectional_ic_series(preds, y, dates)
    if not series:
        return float("nan"), 0
    return float(np.mean(series)), len(series)


def expanding_wf_folds(
    sorted_dates: list,
    *,
    forward_days: int,
    embargo_days: int,
    n_folds: int,
    min_test: int,
):
    """Yield ``(train_dates, test_dates)`` for an expanding-window
    walk-forward with a purge of ``forward_days`` trading dates before each
    test block and an embargo of ``embargo_days`` after.

    The purge is the load-bearing leak control (it removes the train dates
    whose 21d forward label overlaps the test block). The embargo drops
    train dates falling within ``embargo_days`` AFTER the test block — a
    no-op for pure expanding-forward WF (train is always strictly before the
    test), wired here for parity with the W1.2 CPCV path that interleaves
    test groups.
    """
    n = len(sorted_dates)
    # Shrink n_folds if the OOS span is too short to honor min_test per fold.
    if n < (n_folds + 1) * max(min_test, 1):
        n_folds = max(1, n // max(min_test, 1) - 1)
    if n_folds < 1:
        return
    test_size = max(min_test, n // (n_folds + 1))
    for k in range(n_folds):
        test_start = n - (n_folds - k) * test_size
        test_end = min(test_start + test_size, n)
        train_end = test_start - forward_days  # purge
        if train_end <= min_test:
            continue
        train = list(sorted_dates[:train_end])
        if embargo_days > 0:
            emb_hi = min(test_end + embargo_days, n)
            emb = set(sorted_dates[test_end:emb_hi])
            train = [d for d in train if d not in emb]
        test = list(sorted_dates[test_start:test_end])
        if len(train) < min_test or len(test) < 2:
            continue
        yield train, test


def leakfree_meta_oos_ic(
    meta_X,
    meta_y,
    row_dates,
    *,
    fit_predict_fn: Callable,
    forward_days: int,
    embargo_days: int = 0,
    n_folds: int = 5,
    min_test: int = 21,
    bootstrap_fn: Callable | None = None,
) -> dict:
    """Second-level walk-forward OOS meta IC (OBSERVE).

    ``fit_predict_fn(X_train, y_train, X_test) -> preds_test`` is injected so
    the L2 estimator (and tests) stay decoupled from this module. ``row_dates``
    must align row-for-row with ``meta_X`` / ``meta_y``.

    Returns a dict with ``xsec_ic`` (cross-sectional rank IC — the headline),
    ``pooled_ic`` (+ bootstrap CI), fold/sample counts, and a ``status``.
    """
    meta_X = np.asarray(meta_X, dtype=float)
    meta_y = np.asarray(meta_y, dtype=float)
    row_dates = np.asarray(row_dates)
    uniq = sorted({d for d in row_dates.tolist() if d is not None})

    oos_pred: list[float] = []
    oos_true: list[float] = []
    oos_date: list = []
    n_used_folds = 0
    for train_dates, test_dates in expanding_wf_folds(
        uniq, forward_days=forward_days, embargo_days=embargo_days,
        n_folds=n_folds, min_test=min_test,
    ):
        tr_set, te_set = set(train_dates), set(test_dates)
        tr_mask = np.array([d in tr_set for d in row_dates])
        te_mask = np.array([d in te_set for d in row_dates])
        if tr_mask.sum() < 50 or te_mask.sum() < 2:
            continue
        preds = np.asarray(
            fit_predict_fn(meta_X[tr_mask], meta_y[tr_mask], meta_X[te_mask]),
            dtype=float,
        ).ravel()
        oos_pred.extend(preds.tolist())
        oos_true.extend(meta_y[te_mask].tolist())
        oos_date.extend(row_dates[te_mask].tolist())
        n_used_folds += 1

    if n_used_folds == 0 or len(oos_pred) < 10:
        return {
            "status": "insufficient_folds",
            "n_folds": n_used_folds,
            "n": len(oos_pred),
            "xsec_ic": float("nan"),
            "pooled_ic": float("nan"),
            "forward_days": forward_days,
            "embargo_days": embargo_days,
        }

    op = np.asarray(oos_pred, dtype=float)
    ot = np.asarray(oos_true, dtype=float)
    od = np.asarray(oos_date)
    ic_series = cross_sectional_ic_series(op, ot, od)
    xsec_ic = float(np.mean(ic_series)) if ic_series else float("nan")
    n_dates = len(ic_series)
    fin = np.isfinite(op) & np.isfinite(ot)
    pooled = float("nan")
    if fin.sum() >= 10:
        sp = spearmanr(op[fin], ot[fin]).correlation
        pooled = float(sp) if np.isfinite(sp) else float("nan")

    ci_lo = ci_hi = float("nan")
    if bootstrap_fn is not None and fin.sum() >= 30:
        try:
            ci_lo, ci_hi = bootstrap_fn(
                op[fin], ot[fin], [d for d, f in zip(od.tolist(), fin) if f],
            )
        except Exception:  # pragma: no cover - CI is best-effort observability
            ci_lo = ci_hi = float("nan")

    def _r(v):
        return round(float(v), 6) if np.isfinite(v) else float("nan")

    return {
        "status": "ok",
        "n_folds": n_used_folds,
        "n": int(fin.sum()),
        "n_dates": n_dates,
        "xsec_ic": _r(xsec_ic),
        # Per-date cross-sectional IC series — the "returns" the W1.3 Deflated
        # Sharpe Ratio operates on. Rounded for the manifest payload.
        "ic_series": [round(float(x), 6) for x in ic_series],
        "pooled_ic": _r(pooled),
        "pooled_ci_lo": _r(ci_lo),
        "pooled_ci_hi": _r(ci_hi),
        "forward_days": forward_days,
        "embargo_days": embargo_days,
    }


def _contiguous_groups(n: int, n_groups: int) -> list[tuple[int, int]]:
    """Partition index range [0, n) into up to ``n_groups`` contiguous
    inclusive ``(lo, hi)`` index ranges of near-equal size."""
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    return [
        (int(bounds[i]), int(bounds[i + 1] - 1))
        for i in range(n_groups)
        if bounds[i + 1] > bounds[i]
    ]


def cpcv_meta_oos_ic(
    meta_X,
    meta_y,
    row_dates,
    *,
    fit_predict_fn: Callable,
    forward_days: int,
    embargo_days: int = 0,
    n_groups: int = 6,
    k_test: int = 2,
    min_train: int = 50,
) -> dict:
    """Combinatorial purged CV (López de Prado Ch. 12) — W1.2 (L4469, OBSERVE).

    Partition the sorted unique dates into ``n_groups`` contiguous groups; for
    EVERY combination of ``k_test`` test groups, train on the remaining groups
    with each test group **purged** (a train obs at index ``i`` is dropped if
    its forward label ``[i, i+forward_days]`` overlaps a test group, i.e.
    ``i ∈ [a - forward_days, b]``) and **embargoed** (``i ∈ [b, b + embargo_days]``
    after a test group — this is where the embargo becomes load-bearing, since
    CPCV interleaves test groups so train rows can sit just AFTER an interior
    test block). Fit the L2 on the train rows, predict the test rows, and record
    the cross-sectional rank IC. Returns the DISTRIBUTION (per-combination ICs +
    summary stats) — the single-path walk-forward (``leakfree_meta_oos_ic``)
    tests one easily-overfit path; CPCV gives a distribution, which is the input
    to W1.3's Deflated-Sharpe / PBO promotion gate. OBSERVE ONLY.
    """
    meta_X = np.asarray(meta_X, dtype=float)
    meta_y = np.asarray(meta_y, dtype=float)
    row_dates = np.asarray(row_dates)
    uniq = sorted({d for d in row_dates.tolist() if d is not None})
    n = len(uniq)
    groups = _contiguous_groups(n, n_groups)
    if len(groups) < k_test + 1:
        return {
            "status": "insufficient_groups",
            "n_groups": len(groups),
            "ics": [],
            "mean_ic": float("nan"),
        }

    date_to_idx = {d: i for i, d in enumerate(uniq)}
    row_idx = np.array([date_to_idx.get(d, -1) for d in row_dates.tolist()])

    ics: list[float] = []
    n_attempted = 0
    for test_combo in combinations(range(len(groups)), k_test):
        excluded = np.zeros(n, dtype=bool)
        test_idx_set = np.zeros(n, dtype=bool)
        for gi in test_combo:
            a, b = groups[gi]
            test_idx_set[a:b + 1] = True
            lo = max(0, a - forward_days)
            hi = min(n - 1, b + embargo_days)
            excluded[lo:hi + 1] = True
        train_keep = (~test_idx_set) & (~excluded)
        valid = row_idx >= 0
        tr_mask = valid & train_keep[row_idx]
        te_mask = valid & test_idx_set[row_idx]
        n_attempted += 1
        if tr_mask.sum() < min_train or te_mask.sum() < 5:
            continue
        preds = np.asarray(
            fit_predict_fn(meta_X[tr_mask], meta_y[tr_mask], meta_X[te_mask]),
            dtype=float,
        ).ravel()
        ic, _ = cross_sectional_rank_ic(preds, meta_y[te_mask], row_dates[te_mask])
        if np.isfinite(ic):
            ics.append(float(ic))

    if not ics:
        return {
            "status": "no_valid_combos",
            "n_groups": len(groups),
            "n_attempted": n_attempted,
            "ics": [],
            "mean_ic": float("nan"),
        }

    arr = np.asarray(ics, dtype=float)
    # LdP backtest-path count φ = (k / N) · C(N, k).
    n_paths = int(round(k_test / len(groups) * math.comb(len(groups), k_test)))

    def _r(v):
        return round(float(v), 6) if np.isfinite(v) else float("nan")

    return {
        "status": "ok",
        "n_groups": len(groups),
        "k_test": k_test,
        "n_combos": len(ics),
        "n_backtest_paths": n_paths,
        "forward_days": forward_days,
        "embargo_days": embargo_days,
        "mean_ic": _r(arr.mean()),
        "std_ic": _r(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "p05_ic": _r(np.percentile(arr, 5)),
        "p50_ic": _r(np.percentile(arr, 50)),
        "p95_ic": _r(np.percentile(arr, 95)),
        "min_ic": _r(arr.min()),
        "max_ic": _r(arr.max()),
        "frac_positive": _r(float((arr > 0).mean())),
        "ics": [_r(x) for x in ics],
    }


def leakfree_horizon_ic_curve(
    meta_X,
    horizon_to_labels: dict,
    row_dates,
    *,
    fit_predict_fn: Callable,
    embargo_days: int = 0,
    **kwargs,
) -> dict:
    """W3.2 (L4469): leak-free cross-sectional IC at each horizon.

    For each horizon ``h`` (trading days), refit the L2 per fold on that
    horizon's realized label and report the SAME purged+embargoed expanding
    walk-forward read as ``leakfree_meta_oos_ic`` — with the purge set to ``h``
    (the label's own forward span). This is the honest curve that settles
    "which horizon should we target?": where does the leak-free IC peak?

    The pooled / non-overlapping horizon battery (meta_trainer Step 7a) is
    overlapping-Spearman-inflated; this one is leak-free and apples-to-apples
    across horizons (same ``meta_X`` rows / dates, only the label + purge vary).

    Parameters
    ----------
    meta_X : (n_rows, n_meta_features) stacked OOS L1 predictions.
    horizon_to_labels : {h_int: y_array} — realized return at each horizon,
        aligned row-for-row to ``meta_X``. Non-finite rows are masked inside.
    row_dates : per-row dates (for the cross-sectional / purge folds).
    fit_predict_fn : the injected L2 estimator (refit per fold).
    embargo_days : post-test embargo (López de Prado); 0 = current forward-WF.

    Returns ``{f"{h}d": <leakfree_meta_oos_ic dict>}`` — observe-only; gates
    nothing. Extra kwargs pass through to ``leakfree_meta_oos_ic``.
    """
    curve: dict = {}
    dates_arr = np.asarray(row_dates)
    for h, y_h in horizon_to_labels.items():
        y_arr = np.asarray(y_h, dtype=float)
        # Per-horizon finite mask: longer horizons carry more tail-of-history
        # NaN labels (no h-day forward price). leakfree_meta_oos_ic passes y
        # straight into the L2 fit, which rejects NaN — so mask here, dropping
        # NaN-label rows from meta_X / dates together to keep row alignment.
        fin = np.isfinite(y_arr)
        if fin.sum() == 0:
            curve[f"{int(h)}d"] = {"status": "no_finite_labels", "forward_days": int(h)}
            continue
        curve[f"{int(h)}d"] = leakfree_meta_oos_ic(
            meta_X[fin], y_arr[fin], dates_arr[fin],
            fit_predict_fn=fit_predict_fn,
            forward_days=int(h),
            embargo_days=embargo_days,
            **kwargs,
        )
    return curve


def per_feature_standalone_ic(meta_X, meta_y, row_dates, feature_names) -> dict:
    """Standalone cross-sectional rank IC of EACH meta-feature column vs the
    (canonical) realized label — the honest per-input alpha-IC.

    No refit: each column is already an OOS L1 prediction (momentum_score,
    expected_move, research_calibrator_prob, …) or a raw context feature, so its
    per-date Spearman vs realized alpha IS its standalone cross-sectional skill.
    Surfaces the W4 watch-item (L4469): whether ``expected_move`` — the meta's
    dominant coefficient, a VOLATILITY L1 — actually predicts cross-sectional
    ALPHA, or is merely in-sample dominance with ~0 standalone alpha-IC.
    OBSERVE-only; gates nothing. Returns ``{feature: {xsec_ic, n_dates}}``.
    """
    X = np.asarray(meta_X, dtype=float)
    y = np.asarray(meta_y, dtype=float)
    out: dict = {}
    for j, name in enumerate(feature_names):
        series = cross_sectional_ic_series(X[:, j], y, row_dates)
        out[name] = {
            "xsec_ic": round(float(np.mean(series)), 6) if series else None,
            "n_dates": len(series),
        }
    return out


def gbm_blender_fit_predict(**lgb_params) -> Callable:
    """W4.1 (L4469): a NONLINEAR (LightGBM) L2-blender ``fit_predict_fn`` for
    ``leakfree_meta_oos_ic`` — the shadow alternative to the linear BayesianRidge
    meta-learner.

    The W1 finding established the meta is a *linear* stacker; Gu-Kelly-Xiu is
    the canonical evidence that a nonlinear blender materially beats OLS on the
    SAME factor inputs (NN Sharpe 1.35 vs OLS 0.61). This factory lets the
    trainer measure, OBSERVE-only, whether a nonlinear meta recovers more
    leak-free IC from the identical L1 outputs — WITHOUT changing the live L2.
    Because the read is leak-free OOS, overfitting shows up as a LOW OOS IC, so
    the comparison is honest. Conservative defaults bound overfit on the
    ~12-feature meta; a fresh booster is fit per walk-forward fold.
    """
    import lightgbm as lgb

    params = dict(
        n_estimators=200,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=2,
        verbosity=-1,
        random_state=4469,
    )
    params.update(lgb_params)

    def _fit_predict(X_tr, y_tr, X_te):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        return np.asarray(model.predict(X_te), dtype=float).ravel()

    return _fit_predict
