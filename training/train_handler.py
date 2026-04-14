"""
training/train_handler.py — Weekly GBM retraining pipeline for Lambda.

Called by inference/handler.py when event["action"] == "train".

Pipeline:
  1. Download per-ticker OHLCV Parquet files + sector_map.json from S3.
  1b. Refresh any stale parquets with recent yfinance data and upload back to S3.
  2. Build regression arrays (29 features) and apply 70/15/15 time-based split
     with purge gaps of FORWARD_DAYS between train/val and val/test.
  3. Train GBMScorer with Optuna-tuned params from config.GBM_TUNED_PARAMS
     (n_estimators=2000, early_stopping=50).
  4. Evaluate on test set: IC, IC IR, positive-period rate.
  5. Upload dated backup unconditionally; promote to gbm_latest.txt only if IC gate passes.
  5b. Write slim 2-year price cache to predictor/price_cache_slim/ so the daily
      inference Lambda can skip the 2y yfinance fetch and read from S3 instead.
  6. Send training summary email with results.

S3 layout:
  predictor/price_cache/*.parquet       — adjusted OHLCV price history per ticker (10y,
                                          auto_adjust=True; rewritten weekly by this handler)
  predictor/price_cache/sector_map.json — ticker → sector ETF symbol
  predictor/weights/gbm_latest.txt      — (output) active inference weights
  predictor/weights/gbm_{date}.txt      — (output) dated backup

  predictor/daily_closes/{date}.parquet — Backward-split-adjusted OHLCV snapshot per
                                          trading day.  Written by DataPhase1 in
                                          alpha-engine-data/collectors/daily_closes.py
                                          as the first step of the weekday Step Function,
                                          before the predictor inference Lambda runs.
                                          Used as the Mon–Fri delta source by
                                          load_price_data_from_cache(), reducing daily
                                          fetches to only the ~1-2 split tickers per week.
  predictor/price_cache_slim/*.parquet  — 2-year slice of each ticker (written by
                                          write_slim_cache() after each weekly training run).
                                          The inference Lambda downloads this (~9 MB) instead
                                          of re-fetching 2y from yfinance each morning.

Lambda environment variables (same as predictor email):
  EMAIL_SENDER / EMAIL_RECIPIENTS / GMAIL_APP_PASSWORD / AWS_REGION
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import tempfile
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from ssm_secrets import load_secrets

load_secrets()

import numpy as np

log = logging.getLogger(__name__)


# ── S3 data download ───────────────────────────────────────────────────────────

def download_price_cache(bucket: str, prefix: str, local_dir: Path) -> int:
    """
    Download all objects under S3 prefix (Parquets + sector_map.json) to local_dir.
    Returns number of files downloaded.
    """
    import boto3
    s3 = boto3.client("s3")
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]   # strip prefix, keep filename
            if not filename or filename.endswith("/"):
                continue
            local_path = local_dir / filename
            s3.download_file(bucket, key, str(local_path))
            n += 1

    log.info("Downloaded %d files from s3://%s/%s → %s", n, bucket, prefix, local_dir)
    return n


def run_walk_forward(
    X_all: np.ndarray,
    fwd_all: np.ndarray,
    all_dates: list,
    cfg,
) -> dict:
    """
    Expanding-window walk-forward validation.

    Splits data into ~15 folds of WF_TEST_WINDOW_DAYS each, with expanding
    training windows (all data before the fold boundary). Each fold trains a
    fresh GBM and computes IC on the out-of-sample test window.

    Returns dict with fold_ics, median_ic, pct_positive, passes_wf flags,
    and per-fold detail for email reporting.
    """
    from model.gbm_scorer import GBMScorer

    N = len(fwd_all)
    unique_dates = sorted(set(all_dates))
    n_unique = len(unique_dates)
    test_window = cfg.WF_TEST_WINDOW_DAYS
    min_train = cfg.WF_MIN_TRAIN_DAYS
    purge_days = cfg.WF_PURGE_DAYS

    # Build date → index mapping for efficient boundary lookup
    date_to_first_idx: dict = {}
    for i, d in enumerate(all_dates):
        if d not in date_to_first_idx:
            date_to_first_idx[d] = i

    # Generate fold boundaries: expanding train, fixed-size test windows.
    # After the last full fold, include a partial final fold if at least
    # half a test window of data remains — ensures recent data is validated.
    min_partial_days = test_window // 2  # minimum 63 days for partial fold
    folds: list[dict] = []
    fold_start_date_idx = min_train  # index into unique_dates
    while fold_start_date_idx < n_unique:
        remaining = n_unique - fold_start_date_idx
        # Skip if less than half a test window remains
        if remaining < min_partial_days:
            break

        test_start_date = unique_dates[fold_start_date_idx]
        test_end_date_idx = min(fold_start_date_idx + test_window - 1, n_unique - 1)
        test_end_date = unique_dates[test_end_date_idx]

        # Train end: purge_days before test start
        train_end_date_idx = fold_start_date_idx - purge_days
        if train_end_date_idx < min_train // 2:
            fold_start_date_idx += test_window
            continue
        train_end_date = unique_dates[train_end_date_idx]

        # Convert date boundaries to sample indices
        train_mask = np.array([d <= train_end_date for d in all_dates])
        test_mask = np.array([test_start_date <= d <= test_end_date for d in all_dates])

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        if len(train_indices) < 1000 or len(test_indices) < 100:
            fold_start_date_idx += test_window
            continue

        is_partial = remaining < test_window
        folds.append({
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_end_date": str(train_end_date),
            "test_start_date": str(test_start_date),
            "test_end_date": str(test_end_date),
            "partial": is_partial,
        })
        if is_partial:
            log.info("  Partial final fold: %d/%d test days", remaining, test_window)

        fold_start_date_idx += test_window

    log.info("Walk-forward: %d folds generated from %d unique dates", len(folds), n_unique)

    if len(folds) < 3:
        log.warning("Walk-forward: too few folds (%d) — falling back to single split", len(folds))
        return {"folds": [], "median_ic": 0.0, "pct_positive": 0.0, "passes_wf": False,
                "fallback": True}

    tuned_params = getattr(cfg, "GBM_TUNED_PARAMS", None)
    # Walk-forward fast mode: use lighter estimators/early-stopping for fold training
    wf_n_est = getattr(cfg, "WF_N_ESTIMATORS", None) or cfg.GBM_N_ESTIMATORS
    wf_early_stop = getattr(cfg, "WF_EARLY_STOPPING", None) or cfg.GBM_EARLY_STOPPING_ROUNDS
    log.info("Walk-forward training params: n_estimators=%d, early_stopping=%d", wf_n_est, wf_early_stop)
    fold_results: list[dict] = []
    oos_preds_all: list[np.ndarray] = []    # pooled OOS predictions for calibrator
    oos_actuals_all: list[np.ndarray] = []  # pooled OOS forward returns for calibrator

    # CatBoost walk-forward (if enabled)
    catboost_enabled = getattr(cfg, "CATBOOST_ENABLED", False)
    cat_fold_ics: list[float] = []
    cat_oos_preds: list[np.ndarray] = []

    for i, fold in enumerate(folds):
        fold_start = time.time()
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]
        X_train_fold = X_all[train_idx]
        y_train_fold = fwd_all[train_idx]
        X_test_fold = X_all[test_idx]
        y_test_fold = fwd_all[test_idx]

        # Split train into sub-train (85%) + sub-val (15%) for early stopping
        n_sub_train = int(len(y_train_fold) * 0.85)
        X_sub_train = X_train_fold[:n_sub_train]
        y_sub_train = y_train_fold[:n_sub_train]
        X_sub_val = X_train_fold[n_sub_train:]
        y_sub_val = y_train_fold[n_sub_train:]

        scorer = GBMScorer(
            params=tuned_params,
            n_estimators=wf_n_est,
            early_stopping_rounds=wf_early_stop,
        )
        scorer.fit(X_sub_train, y_sub_train, X_sub_val, y_sub_val,
                    feature_names=cfg.GBM_FEATURES)

        test_preds = scorer.predict(X_test_fold)

        # Collect OOS predictions + actuals for calibrator fitting
        oos_preds_all.append(test_preds)
        oos_actuals_all.append(y_test_fold)

        # IC for this fold
        if len(test_preds) > 1 and np.std(test_preds) > 1e-10 and np.std(y_test_fold) > 1e-10:
            fold_ic = float(np.corrcoef(test_preds, y_test_fold)[0, 1])
        else:
            fold_ic = 0.0

        # CatBoost per fold (if enabled)
        cat_fold_ic = None
        if catboost_enabled:
            try:
                from model.catboost_scorer import CatBoostScorer as _CatFold
                cat_scorer_fold = _CatFold(
                    params=getattr(cfg, "CATBOOST_PARAMS", None),
                    n_estimators=wf_n_est,
                    early_stopping_rounds=wf_early_stop,
                )
                cat_scorer_fold.fit(X_sub_train, y_sub_train, X_sub_val, y_sub_val,
                                    feature_names=cfg.GBM_FEATURES)
                cat_preds_fold = cat_scorer_fold.predict(X_test_fold)
                cat_oos_preds.append(cat_preds_fold)
                if len(cat_preds_fold) > 1 and np.std(cat_preds_fold) > 1e-10:
                    cat_fold_ic = float(np.corrcoef(cat_preds_fold, y_test_fold)[0, 1])
                else:
                    cat_fold_ic = 0.0
                cat_fold_ics.append(cat_fold_ic)
            except Exception as _cat_err:
                log.warning("CatBoost fold %d failed: %s", i + 1, _cat_err)
                cat_fold_ics.append(0.0)

        fold_elapsed = time.time() - fold_start
        fold_result = {
            "fold": i + 1,
            "train_end": fold["train_end_date"],
            "test_start": fold["test_start_date"],
            "test_end": fold["test_end_date"],
            "n_train": len(y_train_fold),
            "n_test": len(y_test_fold),
            "ic": round(fold_ic, 6),
            "cat_ic": round(cat_fold_ic, 6) if cat_fold_ic is not None else None,
            "best_iteration": scorer._best_iteration,
            "elapsed_s": round(fold_elapsed, 1),
        }
        fold_results.append(fold_result)
        log.info(
            "  Fold %d/%d: train=%d test=%d  [%s → %s]  IC=%.4f  (%.1fs)",
            i + 1, len(folds), len(y_train_fold), len(y_test_fold),
            fold["test_start_date"], fold["test_end_date"], fold_ic, fold_elapsed,
        )

    fold_ics = np.array([f["ic"] for f in fold_results])
    median_ic = float(np.median(fold_ics))
    pct_positive = float((fold_ics > 0).mean())

    passes_median = median_ic >= cfg.WF_MEDIAN_IC_GATE
    passes_pct = pct_positive >= cfg.WF_MIN_FOLDS_POSITIVE
    passes_wf = passes_median and passes_pct

    log.info(
        "Walk-forward summary: median_IC=%.4f (%s %.4f)  "
        "pct_positive=%.1f%% (%s %.0f%%)  passes=%s",
        median_ic, ">=" if passes_median else "<", cfg.WF_MEDIAN_IC_GATE,
        pct_positive * 100, ">=" if passes_pct else "<", cfg.WF_MIN_FOLDS_POSITIVE * 100,
        passes_wf,
    )

    # Pool OOS predictions + actuals for calibrator fitting
    pooled_oos_preds = np.concatenate(oos_preds_all) if oos_preds_all else np.array([])
    pooled_oos_actuals = np.concatenate(oos_actuals_all) if oos_actuals_all else np.array([])

    # CatBoost walk-forward summary
    cat_wf_summary = None
    best_blend_weight = None
    if catboost_enabled and cat_fold_ics:
        cat_median_ic = float(np.median(cat_fold_ics))
        log.info("CatBoost WF median IC: %.4f  (LGB: %.4f)", cat_median_ic, median_ic)
        cat_wf_summary = {"cat_median_ic": round(cat_median_ic, 6)}

        # Blend optimization: find best w_lgb in [0.3, 0.4, 0.5, 0.6, 0.7]
        if len(cat_oos_preds) == len(oos_preds_all) and len(oos_preds_all) > 0:
            blend_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
            best_blend_ic = -1.0
            for w_lgb in blend_candidates:
                blend_ics = []
                for lgb_p, cat_p, actual in zip(oos_preds_all, cat_oos_preds, oos_actuals_all):
                    from scipy.stats import rankdata as _rd
                    lgb_r = _rd(lgb_p, method="average") / len(lgb_p)
                    cat_r = _rd(cat_p, method="average") / len(cat_p)
                    blend = w_lgb * lgb_r + (1 - w_lgb) * cat_r
                    if np.std(blend) > 1e-10 and np.std(actual) > 1e-10:
                        blend_ics.append(float(np.corrcoef(blend, actual)[0, 1]))
                if blend_ics:
                    avg_blend_ic = float(np.median(blend_ics))
                    if avg_blend_ic > best_blend_ic:
                        best_blend_ic = avg_blend_ic
                        best_blend_weight = w_lgb
            if best_blend_weight is not None:
                log.info("Best blend: w_lgb=%.1f  blend_IC=%.4f", best_blend_weight, best_blend_ic)
                cat_wf_summary["best_blend_weight"] = best_blend_weight
                cat_wf_summary["best_blend_ic"] = round(best_blend_ic, 6)

    return {
        "folds": fold_results,
        "median_ic": round(median_ic, 6),
        "pct_positive": round(pct_positive, 4),
        "passes_median_ic": passes_median,
        "passes_pct_positive": passes_pct,
        "passes_wf": passes_wf,
        "fallback": False,
        "oos_preds": pooled_oos_preds,
        "oos_actuals": pooled_oos_actuals,
        "catboost": cat_wf_summary,
        "best_blend_weight": best_blend_weight,
    }


# ── Training pipeline ──────────────────────────────────────────────────────────

def run_gbm_training(
    data_dir: str,
    bucket: str,
    date_str: str,
    dry_run: bool = False,
) -> dict:
    """
    Build dataset, train GBMScorer, evaluate, and upload to S3.

    If walk-forward validation is enabled (config.WF_ENABLED), runs expanding-
    window cross-validation across multiple regime periods before training the
    final production model. The walk-forward IC summary is included in the
    result dict and email.

    Parameters
    ----------
    data_dir  : Local directory containing *.parquet + sector_map.json.
    bucket    : S3 bucket for model upload.
    date_str  : Training date label (YYYY-MM-DD) for the dated backup key.
    dry_run   : Skip S3 upload if True.

    Returns
    -------
    Result dict with val_ic, test_ic, ic_ir, promoted flag, feature_importance, etc.
    """
    import config as cfg
    from data.dataset import build_regression_arrays
    from model.gbm_scorer import GBMScorer

    start_ts = datetime.now(timezone.utc)

    # Capture git commit hash for experiment tracking
    _code_commit = None
    try:
        import subprocess as _sp
        _code_commit = _sp.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        pass

    # ── Build arrays (no PyTorch dependency) ──────────────────────────────────
    log.info("Building regression arrays from %s ...", data_dir)
    X_all, fwd_all, all_dates = build_regression_arrays(
        data_dir=data_dir,
        config_module=cfg,
        feature_list=cfg.GBM_FEATURES,
    )

    N = len(fwd_all)

    # ── Walk-forward validation (if enabled) ──────────────────────────────────
    wf_result: Optional[dict] = None
    if cfg.WF_ENABLED:
        log.info("Walk-forward validation enabled — running expanding-window CV ...")
        wf_result = run_walk_forward(X_all, fwd_all, all_dates, cfg)
        if wf_result.get("fallback"):
            log.warning("Walk-forward fell back — proceeding with single-split evaluation")
        elif not wf_result["passes_wf"]:
            log.warning(
                "Walk-forward FAILED gates (median_IC=%.4f, pct_positive=%.1f%%) "
                "— training production model anyway but will NOT promote",
                wf_result["median_ic"], wf_result["pct_positive"] * 100,
            )

    # ── Train final production model on all data ──────────────────────────────
    # Use 70/15/15 split for single-split IC metrics (reporting) and early stopping.
    purge_days = cfg.FORWARD_DAYS
    n_train = int(N * cfg.TRAIN_FRAC)
    n_val_raw = int(N * cfg.VAL_FRAC)

    # Purge gap 1: train → val
    train_end_date = all_dates[n_train - 1]
    unique_post_train = sorted(set(d for d in all_dates[n_train:] if d > train_end_date))
    if len(unique_post_train) >= purge_days:
        purge_cutoff_1 = unique_post_train[purge_days - 1]
        val_start = next(i for i in range(n_train, N) if all_dates[i] > purge_cutoff_1)
    else:
        val_start = n_train

    # Val slice
    val_end = min(val_start + n_val_raw, N)

    # Purge gap 2: val → test
    val_end_date = all_dates[min(val_end - 1, N - 1)]
    unique_post_val = sorted(set(d for d in all_dates[val_end:] if d > val_end_date))
    if len(unique_post_val) >= purge_days:
        purge_cutoff_2 = unique_post_val[purge_days - 1]
        test_start = next(i for i in range(val_end, N) if all_dates[i] > purge_cutoff_2)
    else:
        test_start = val_end

    X_train = X_all[:n_train]
    y_train = fwd_all[:n_train]
    X_val   = X_all[val_start:val_end]
    y_val   = fwd_all[val_start:val_end]
    X_test  = X_all[test_start:]
    y_test  = fwd_all[test_start:]

    log.info(
        "Dataset ready: train=%d  val=%d  test=%d  features=%d  "
        "purged=%d samples",
        len(y_train), len(y_val), len(y_test), X_train.shape[1],
        (val_start - n_train) + (test_start - val_end),
    )

    # ── Train with Optuna-tuned hyperparameters ───────────────────────────────
    tuned_params = getattr(cfg, "GBM_TUNED_PARAMS", None)
    ensemble_enabled = getattr(cfg, "GBM_ENSEMBLE_LAMBDARANK", True)
    train_dates_slice = all_dates[:n_train]
    val_dates_slice = all_dates[val_start:val_end]

    log.info(
        "Training GBMScorer (n_estimators=%d, early_stopping=%d, %s%s) ...",
        cfg.GBM_N_ESTIMATORS, cfg.GBM_EARLY_STOPPING_ROUNDS,
        "Optuna-tuned params" if tuned_params else "default params",
        " + lambdarank ensemble" if ensemble_enabled else "",
    )

    # ── MSE model (always trained) ────────────────────────────────────────────
    scorer = GBMScorer(
        params=tuned_params,
        n_estimators=cfg.GBM_N_ESTIMATORS,
        early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
        ranking_objective=False,
    )
    scorer.fit(X_train, y_train, X_val, y_val, feature_names=cfg.GBM_FEATURES)

    # ── Lambdarank model (when ensemble enabled) ──────────────────────────────
    rank_scorer = None
    if ensemble_enabled:
        try:
            rank_scorer = GBMScorer(
                params=tuned_params,
                n_estimators=cfg.GBM_N_ESTIMATORS,
                early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
                ranking_objective=True,
            )
            rank_scorer.fit(
                X_train, y_train, X_val, y_val,
                feature_names=cfg.GBM_FEATURES,
                train_dates=train_dates_slice,
                val_dates=val_dates_slice,
            )
            log.info("Lambdarank model trained: val_IC=%.4f", rank_scorer._val_ic)
        except Exception as e:
            log.warning("Lambdarank training failed — falling back to MSE-only: %s", e)
            rank_scorer = None

    # ── CatBoost training (if enabled) ─────────────────────────────────────
    cat_scorer = None
    if cfg.CATBOOST_ENABLED:
        try:
            from model.catboost_scorer import CatBoostScorer
            cat_scorer = CatBoostScorer(
                params=cfg.CATBOOST_PARAMS,
                n_estimators=cfg.GBM_N_ESTIMATORS,
                early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
            )
            cat_scorer.fit(X_train, y_train, X_val, y_val,
                           feature_names=cfg.GBM_FEATURES)
            log.info("CatBoost model trained: val_IC=%.4f", cat_scorer._val_ic)
        except Exception as e:
            log.warning("CatBoost training failed: %s", e)
            cat_scorer = None

    # ── Evaluate ─────────────────────────────────────────────────────────────
    try:
        from scipy.stats import rankdata
    except ImportError:
        # Numpy-only fallback for rankdata
        def rankdata(a):
            arr = np.asarray(a)
            order = arr.argsort()
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
            return ranks

    test_preds_mse = scorer.predict(X_test)
    mse_ic = float(np.corrcoef(test_preds_mse, y_test)[0, 1])

    # Lambdarank IC + ensemble IC
    rank_ic = 0.0
    ensemble_ic = mse_ic  # default to MSE if no rank model
    if rank_scorer is not None:
        test_preds_rank = rank_scorer.predict(X_test)
        rank_ic = float(np.corrcoef(test_preds_rank, y_test)[0, 1])

        # Ensemble correlation check: if MSE and LambdaRank are too correlated,
        # the ensemble adds no diversification benefit
        pred_corr = float(np.corrcoef(test_preds_mse, test_preds_rank)[0, 1])
        if pred_corr >= 0.85:
            log.warning(
                "MSE-LambdaRank prediction correlation %.4f >= 0.85 — "
                "ensemble adds minimal diversification, preferring single best model",
                pred_corr,
            )
        else:
            log.info("MSE-LambdaRank prediction correlation: %.4f (good diversity)", pred_corr)

        # Ensemble: rank-normalize each model's predictions, then average
        mse_ranked = rankdata(test_preds_mse).astype(np.float32)
        rank_ranked = rankdata(test_preds_rank).astype(np.float32)
        ensemble_preds = 0.5 * mse_ranked + 0.5 * rank_ranked
        ensemble_ic = float(np.corrcoef(ensemble_preds, y_test)[0, 1])
        log.info(
            "Ensemble ICs: MSE=%.4f  Lambdarank=%.4f  Ensemble=%.4f  corr=%.4f",
            mse_ic, rank_ic, ensemble_ic, pred_corr,
        )

    # CatBoost IC + LGB-Cat blend
    cat_ic = 0.0
    lgb_cat_blend_ic = 0.0
    blend_weights = None
    if cat_scorer is not None:
        test_preds_cat = cat_scorer.predict(X_test)
        cat_ic = float(np.corrcoef(test_preds_cat, y_test)[0, 1])

        # Blend optimization using WF-optimized weight or grid search on test set
        w_lgb = 0.5
        if wf_result and wf_result.get("best_blend_weight") is not None:
            w_lgb = wf_result["best_blend_weight"]
        mse_ranked_b = rankdata(test_preds_mse).astype(np.float32)
        cat_ranked_b = rankdata(test_preds_cat).astype(np.float32)
        blend_preds = w_lgb * mse_ranked_b + (1 - w_lgb) * cat_ranked_b
        lgb_cat_blend_ic = float(np.corrcoef(blend_preds, y_test)[0, 1])
        blend_weights = {"lgb": round(w_lgb, 2), "cat": round(1 - w_lgb, 2)}
        log.info(
            "CatBoost IC: %.4f  LGB-Cat blend IC: %.4f (w_lgb=%.1f)",
            cat_ic, lgb_cat_blend_ic, w_lgb,
        )

    # Pick the best IC among all candidates for the gate
    if rank_scorer is not None:
        candidates = {"mse": mse_ic, "ensemble": ensemble_ic}
        candidates["rank"] = rank_ic
    else:
        candidates = {"mse": mse_ic}
    if cat_scorer is not None:
        candidates["catboost"] = cat_ic
        candidates["lgb_cat_blend"] = lgb_cat_blend_ic
    best_mode = max(candidates, key=candidates.get)
    best_ic = candidates[best_mode]
    test_ic = best_ic
    test_preds = test_preds_mse  # for chunk IC and backwards compat
    log.info(
        "Best IC selection: %s (%.4f) from candidates %s",
        best_mode, best_ic,
        {k: round(v, 4) for k, v in candidates.items()},
    )

    n_chunks   = 20
    chunk_size = len(test_preds) // n_chunks
    chunk_ics  = np.array([
        float(np.corrcoef(
            test_preds[i*chunk_size:(i+1)*chunk_size],
            y_test[i*chunk_size:(i+1)*chunk_size],
        )[0, 1])
        for i in range(n_chunks)
    ])
    ic_ir = float(chunk_ics.mean() / (chunk_ics.std() + 1e-8))

    importance = scorer.feature_importance(importance_type="gain")
    top10      = sorted(importance.items(), key=lambda x: -x[1])[:10]

    # SHAP feature importance (more reliable than gain-based) — computed on MSE model
    shap_importance = None
    try:
        import shap
        explainer = shap.TreeExplainer(scorer._booster)
        shap_values = explainer.shap_values(X_test[:500])  # cap at 500 rows for speed
        shap_importance = dict(zip(
            cfg.GBM_FEATURES,
            [round(float(v), 4) for v in np.abs(shap_values).mean(axis=0)]
        ))
        shap_top10 = sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
        log.info("SHAP top 5: %s", shap_top10[:5])
    except Exception as e:
        log.warning("SHAP computation failed (non-blocking): %s", e)

    # ── Per-feature IC tracking ────────────────────────────────────────────────
    feature_ics = {}
    for i, fname in enumerate(cfg.GBM_FEATURES):
        feat_vals = X_test[:, i]
        if np.std(feat_vals) > 1e-10:
            fic = float(np.corrcoef(feat_vals, y_test)[0, 1])
        else:
            fic = 0.0
        feature_ics[fname] = round(fic, 6)

    sorted_by_abs_ic = sorted(feature_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    log.info("Per-feature IC — top 5: %s", sorted_by_abs_ic[:5])
    log.info("Per-feature IC — bottom 5: %s", sorted_by_abs_ic[-5:])

    # ── SHAP-based noise feature detection ─────────────────────────────────────
    noise_candidates = []
    if shap_importance:
        max_shap = max(shap_importance.values()) if shap_importance else 0.0
        shap_noise_thresh = max_shap * (cfg.SHAP_NOISE_THRESHOLD_PCT / 100.0)
        ic_noise_thresh = cfg.IC_NOISE_THRESHOLD
        for fname in cfg.GBM_FEATURES:
            shap_val = shap_importance.get(fname, 0.0)
            ic_val = abs(feature_ics.get(fname, 0.0))
            if shap_val < shap_noise_thresh and ic_val < ic_noise_thresh:
                noise_candidates.append(fname)
        if noise_candidates:
            log.info(
                "Noise feature candidates (%d): %s (SHAP < %.4f AND |IC| < %.4f)",
                len(noise_candidates), noise_candidates, shap_noise_thresh, ic_noise_thresh,
            )
            # Auto-prune: retrain production model without noise features
            if cfg.AUTO_PRUNE_NOISE_FEATURES and len(noise_candidates) < len(cfg.GBM_FEATURES) // 2:
                pruned_features = [f for f in cfg.GBM_FEATURES if f not in noise_candidates]
                pruned_indices = [cfg.GBM_FEATURES.index(f) for f in pruned_features]
                log.info("Auto-pruning: retraining with %d features (dropped %d noise)",
                         len(pruned_features), len(noise_candidates))
                _pruned_scorer = GBMScorer(
                    params=tuned_params,
                    n_estimators=cfg.GBM_N_ESTIMATORS,
                    early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
                )
                _pruned_scorer.fit(
                    X_train[:, pruned_indices], y_train,
                    X_val[:, pruned_indices], y_val,
                    feature_names=pruned_features,
                )
                _pruned_ic = float(np.corrcoef(
                    _pruned_scorer.predict(X_test[:, pruned_indices]), y_test
                )[0, 1])
                log.info("Pruned model IC: %.4f (vs full: %.4f)", _pruned_ic, mse_ic)
                if _pruned_ic >= mse_ic * 0.95:  # accept if within 5% of full model
                    scorer = _pruned_scorer
                    mse_ic = _pruned_ic
                    test_ic = _pruned_ic
                    log.info("Pruned model accepted — using %d features", len(pruned_features))
                else:
                    log.info("Pruned model rejected (IC dropped >5%%) — keeping full model")
        else:
            log.info("No noise feature candidates detected")

    # ── Promotion decision ────────────────────────────────────────────────────
    # If walk-forward enabled and passed: use WF gate for promotion.
    # If walk-forward enabled but failed: do NOT promote regardless of single-split IC.
    # If walk-forward disabled: fall back to single-split IC gate.
    passes_single_ic = float(test_ic) >= cfg.MIN_IC
    passes_ic_ir = ic_ir >= cfg.GBM_IC_IR_GATE

    # Hit rate gate: predicted direction must match actual direction often enough
    test_hit_rate = float((np.sign(test_preds) == np.sign(y_test)).mean())
    passes_hit_rate = test_hit_rate >= cfg.MIN_HIT_RATE

    if wf_result and not wf_result.get("fallback"):
        passes_ic = wf_result["passes_wf"]
    else:
        # Single-split fallback: also require chunk-level consistency
        pct_chunks_positive = float((chunk_ics > 0).mean())
        if passes_single_ic and pct_chunks_positive < 0.50:
            log.warning(
                "Single-split IC passes (%.4f) but chunk consistency fails "
                "(%.0f%% positive < 50%%) — NOT promoting",
                float(test_ic), pct_chunks_positive * 100,
            )
        passes_ic = passes_single_ic and (pct_chunks_positive >= 0.50)

    if passes_ic and not passes_hit_rate:
        log.warning(
            "IC gate passes but hit rate %.2f%% < MIN_HIT_RATE %.2f%% — NOT promoting",
            test_hit_rate * 100, cfg.MIN_HIT_RATE * 100,
        )
    passes_ic = passes_ic and passes_hit_rate

    elapsed_s = (datetime.now(timezone.utc) - start_ts).total_seconds()
    model_version = f"GBM-v{scorer._best_iteration}"

    log.info(
        "Training complete: val_IC=%.4f  test_IC=%.4f  IC_IR=%.3f  "
        "hit_rate=%.2f%%  passes_ic=%s  elapsed=%.0fs",
        scorer._val_ic, test_ic, ic_ir, test_hit_rate * 100, passes_ic, elapsed_s,
    )

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    promoted = False
    if not dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            booster_path = Path(tmp) / "gbm_model.txt"
            scorer.save(booster_path)

            import boto3
            s3 = boto3.client("s3")

            # Always save dated backups
            dated_key = f"predictor/weights/gbm_{date_str}.txt"
            s3.upload_file(str(booster_path), bucket, dated_key)
            log.info("Uploaded dated backup (MSE): s3://%s/%s", bucket, dated_key)

            # Also save as gbm_mse_{date}.txt
            dated_mse_key = f"predictor/weights/gbm_mse_{date_str}.txt"
            s3.upload_file(str(booster_path), bucket, dated_mse_key)

            # Save lambdarank model if trained
            rank_booster_path = None
            if rank_scorer is not None:
                rank_booster_path = Path(tmp) / "gbm_rank_model.txt"
                rank_scorer.save(rank_booster_path)
                dated_rank_key = f"predictor/weights/gbm_rank_{date_str}.txt"
                s3.upload_file(str(rank_booster_path), bucket, dated_rank_key)
                log.info("Uploaded dated backup (lambdarank): s3://%s/%s", bucket, dated_rank_key)

            # Save CatBoost model (if trained)
            cat_booster_path = None
            if cat_scorer is not None:
                cat_booster_path = Path(tmp) / "catboost_model.cbm"
                cat_scorer.save(cat_booster_path)
                dated_cat_key = f"predictor/weights/catboost_{date_str}.cbm"
                s3.upload_file(str(cat_booster_path), bucket, dated_cat_key)
                log.info("Uploaded dated backup (CatBoost): s3://%s/%s", bucket, dated_cat_key)

            if passes_ic:
                # Always upload BOTH models — inference runs both side by side.
                # MSE → predicted_alpha (calibrated returns)
                # Lambdarank → model_rank (cross-sectional ranking)
                s3.upload_file(str(booster_path), bucket, cfg.GBM_MSE_WEIGHTS_KEY)
                log.info("Uploaded MSE model: s3://%s/%s", bucket, cfg.GBM_MSE_WEIGHTS_KEY)
                if rank_booster_path is not None:
                    s3.upload_file(str(rank_booster_path), bucket, cfg.GBM_RANK_WEIGHTS_KEY)
                    log.info("Uploaded lambdarank model: s3://%s/%s", bucket, cfg.GBM_RANK_WEIGHTS_KEY)
                if cat_booster_path is not None:
                    s3.upload_file(str(cat_booster_path), bucket, cfg.CATBOOST_WEIGHTS_KEY)
                    s3.upload_file(str(cat_booster_path) + ".meta.json", bucket, cfg.CATBOOST_WEIGHTS_META_KEY)
                    log.info("Uploaded CatBoost model: s3://%s/%s", bucket, cfg.CATBOOST_WEIGHTS_KEY)

                # Promote best_mode to gbm_latest (for model_version tracking)
                if best_mode == "mse":
                    s3.upload_file(str(booster_path), bucket, cfg.GBM_WEIGHTS_KEY)
                elif best_mode == "rank" and rank_booster_path is not None:
                    s3.upload_file(str(rank_booster_path), bucket, cfg.GBM_WEIGHTS_KEY)
                elif best_mode in ("ensemble", "lgb_cat_blend", "catboost"):
                    s3.upload_file(str(booster_path), bucket, cfg.GBM_WEIGHTS_KEY)
                log.info("Promoted %s to active weights: s3://%s/%s", best_mode, bucket, cfg.GBM_WEIGHTS_KEY)

                # Write gbm_mode.json to S3
                mode_data = {"mode": best_mode}
                if blend_weights is not None:
                    mode_data["blend_weights"] = blend_weights
                mode_payload = json.dumps(mode_data, indent=2).encode()
                s3.put_object(
                    Bucket=bucket,
                    Key=cfg.GBM_MODE_KEY,
                    Body=mode_payload,
                    ContentType="application/json",
                )
                log.info("Wrote gbm_mode.json: mode=%s", best_mode)

                # ── Append to mode_history.json on S3 ────────────────────
                try:
                    MODE_HISTORY_KEY = "predictor/metrics/mode_history.json"
                    try:
                        hist_obj = s3.get_object(Bucket=bucket, Key=MODE_HISTORY_KEY)
                        mode_history = json.loads(hist_obj["Body"].read())
                        if not isinstance(mode_history, list):
                            mode_history = []
                    except Exception:
                        mode_history = []

                    mode_history_entry = {
                        "date": date_str,
                        "mse_ic": round(float(mse_ic), 6),
                        "rank_ic": round(float(rank_ic), 6) if rank_scorer is not None else None,
                        "ensemble_ic": round(float(ensemble_ic), 6) if rank_scorer is not None else None,
                        "best_mode": best_mode,
                        "promoted": True,
                        "ic_delta_rank_vs_mse": round(float(rank_ic - mse_ic), 6) if rank_scorer is not None else None,
                        "ic_delta_ens_vs_mse": round(float(ensemble_ic - mse_ic), 6) if rank_scorer is not None else None,
                        "n_train": int(len(y_train)),
                        "ic_ir": round(ic_ir, 4),
                    }

                    # Deduplicate by date (replace if same date already exists)
                    mode_history = [e for e in mode_history if e.get("date") != date_str]
                    mode_history.append(mode_history_entry)
                    # Cap at 52 entries (1 year of weekly runs)
                    mode_history = mode_history[-52:]

                    s3.put_object(
                        Bucket=bucket,
                        Key=MODE_HISTORY_KEY,
                        Body=json.dumps(mode_history, indent=2).encode(),
                        ContentType="application/json",
                    )
                    log.info("Appended mode history entry for %s (mode=%s)", date_str, best_mode)
                except Exception as e:
                    log.warning("Failed to update mode_history.json: %s", e)

                meta = {
                    "model_version":  model_version,
                    "val_ic":         round(float(scorer._val_ic), 6),
                    "test_ic":        round(float(test_ic), 6),
                    "mse_ic":         round(float(mse_ic), 6),
                    "ic_ir":          round(ic_ir, 4),
                    "trained_date":   date_str,
                    "best_iteration": scorer._best_iteration,
                    "n_train":        int(len(y_train)),
                    "rank_normalized": True,
                    "feature_list":   cfg.GBM_FEATURES,
                    "n_features":     len(cfg.GBM_FEATURES),
                    "gain_importance": dict(importance),
                    "shap_importance": shap_importance,
                    "feature_ics": feature_ics,
                    "noise_candidates": noise_candidates,
                    "ensemble_enabled": rank_scorer is not None,
                    "promoted_mode":  best_mode,
                }
                if rank_scorer is not None:
                    meta["rank_ic"] = round(float(rank_ic), 6)
                    meta["ensemble_ic"] = round(float(ensemble_ic), 6)
                if wf_result and not wf_result.get("fallback"):
                    meta["walk_forward"] = {
                        "median_ic": wf_result["median_ic"],
                        "pct_positive": wf_result["pct_positive"],
                        "n_folds": len(wf_result["folds"]),
                    }
                s3.put_object(
                    Bucket=bucket,
                    Key=cfg.GBM_WEIGHTS_META_KEY,
                    Body=json.dumps(meta, indent=2).encode(),
                    ContentType="application/json",
                )

                promoted = True
            else:
                reason = "walk-forward" if (wf_result and not wf_result.get("fallback")) else "single-split IC"
                log.warning(
                    "IC gate FAILED (%s) — backup saved, NOT promoted",
                    reason,
                )

            # ── Feature importance time series (saved every retrain, promoted or not) ──
            try:
                fi_payload = {
                    "date": date_str,
                    "model_version": cfg.GBM_VERSION,
                    "promoted": promoted,
                    "promoted_mode": best_mode if promoted else None,
                    "best_iteration": scorer._best_iteration,
                    "gain_importance": dict(importance),
                    "shap_importance": shap_importance,
                    "feature_ics": feature_ics,
                    "n_train": int(len(y_train)),
                }
                fi_body = json.dumps(fi_payload, indent=2).encode()
                s3.put_object(
                    Bucket=bucket,
                    Key=f"predictor/metrics/feature_importance_{date_str}.json",
                    Body=fi_body,
                    ContentType="application/json",
                )
                # Write latest pointer so dashboard can read without knowing the date
                s3.put_object(
                    Bucket=bucket,
                    Key="predictor/metrics/feature_importance_latest.json",
                    Body=fi_body,
                    ContentType="application/json",
                )
                log.info("Feature importance time series written for %s (+ latest)", date_str)
            except Exception as _fi_err:
                log.warning("Feature importance write failed (non-blocking): %s", _fi_err)

    # ── Fit and upload calibrator (Platt scaling) ──────────────────────────────
    calibration_metrics = None
    if (cfg.CALIBRATION_ENABLED and wf_result and not wf_result.get("fallback")
            and len(wf_result.get("oos_preds", [])) >= 100 and not dry_run):
        try:
            from model.calibrator import PlattCalibrator

            oos_preds_raw = np.clip(wf_result["oos_preds"], -cfg.LABEL_CLIP, cfg.LABEL_CLIP)
            oos_up_labels = (wf_result["oos_actuals"] > 0).astype(np.int32)

            calibrator = PlattCalibrator(method=cfg.CALIBRATION_METHOD)
            calibrator.fit(oos_preds_raw, oos_up_labels, label_clip=cfg.LABEL_CLIP)

            if calibrator.is_fitted and promoted:
                with tempfile.TemporaryDirectory() as cal_tmp:
                    cal_path = Path(cal_tmp) / "calibrator.pkl"
                    calibrator.save(cal_path)

                    import boto3 as _boto3_cal
                    _s3_cal = _boto3_cal.client("s3")
                    _s3_cal.upload_file(str(cal_path), bucket, cfg.CALIBRATOR_WEIGHTS_KEY)
                    _s3_cal.upload_file(
                        str(cal_path) + ".meta.json", bucket, cfg.CALIBRATOR_WEIGHTS_META_KEY,
                    )
                    # Dated backup
                    _s3_cal.upload_file(
                        str(cal_path), bucket,
                        f"predictor/weights/calibrator_{date_str}.pkl",
                    )
                log.info(
                    "Calibrator uploaded: ECE %.4f → %.4f (%s)",
                    calibrator._ece_before, calibrator._ece_after, cfg.CALIBRATION_METHOD,
                )

            calibration_metrics = calibrator.metrics()
        except Exception as _cal_err:
            log.warning("Calibrator fitting/upload failed (non-blocking): %s", _cal_err)

    # ── Write comprehensive training summary to S3 ───────────────────────────
    # Persists all training metrics so dashboard/debugging don't depend on email.
    if not dry_run:
        try:
            import boto3 as _boto3_summary
            _s3_sum = _boto3_summary.client("s3")
            training_summary = {
                "date": date_str,
                "model_version": model_version,
                "code_commit": _code_commit,
                "promoted": promoted,
                "promoted_mode": best_mode if promoted else None,
                "elapsed_s": round(elapsed_s, 1),
                "hyperparameters": getattr(cfg, "GBM_TUNED_PARAMS", None),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "n_features": len(cfg.GBM_FEATURES),
                # IC metrics
                "mse_ic": round(float(mse_ic), 6),
                "rank_ic": round(float(rank_ic), 6) if rank_scorer is not None else None,
                "ensemble_ic": round(float(ensemble_ic), 6) if rank_scorer is not None else None,
                "test_ic": round(float(test_ic), 6),
                "ic_ir": round(ic_ir, 4),
                "test_hit_rate": round(test_hit_rate, 4),
                # CatBoost
                "catboost_enabled": cat_scorer is not None,
                "catboost_ic": round(float(cat_ic), 6) if cat_scorer is not None else None,
                "lgb_cat_blend_ic": round(float(lgb_cat_blend_ic), 6) if cat_scorer is not None else None,
                "blend_weights": blend_weights,
                # Calibration
                "calibration": calibration_metrics,
                # Walk-forward
                "walk_forward": {
                    "median_ic": wf_result["median_ic"],
                    "pct_positive": wf_result["pct_positive"],
                    "passes_wf": wf_result["passes_wf"],
                    "n_folds": len(wf_result["folds"]),
                    "catboost_median_ic": wf_result.get("catboost", {}).get("cat_median_ic") if wf_result.get("catboost") else None,
                } if wf_result and not wf_result.get("fallback") else None,
                # Features
                "feature_ics": feature_ics,
                "noise_candidates": noise_candidates,
                "feature_importance_top10": [
                    {"feature": f, "gain": round(s, 2)} for f, s in
                    sorted(importance.items(), key=lambda x: -x[1])[:10]
                ],
                "shap_top10": [
                    {"feature": f, "shap": round(s, 4)} for f, s in
                    sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
                ] if shap_importance else [],
                # Gates
                "gates": {
                    "min_ic": cfg.MIN_IC,
                    "min_hit_rate": cfg.MIN_HIT_RATE,
                    "wf_median_ic_gate": cfg.WF_MEDIAN_IC_GATE,
                    "wf_min_folds_positive": cfg.WF_MIN_FOLDS_POSITIVE,
                },
            }
            _sum_body = json.dumps(training_summary, indent=2, default=str).encode()
            _s3_sum.put_object(
                Bucket=bucket,
                Key=f"predictor/metrics/training_summary_{date_str}.json",
                Body=_sum_body,
                ContentType="application/json",
            )
            _s3_sum.put_object(
                Bucket=bucket,
                Key="predictor/metrics/training_summary_latest.json",
                Body=_sum_body,
                ContentType="application/json",
            )
            log.info("Training summary written to S3 (dated + latest)")

            # Append to training history log (JSONL — one line per run)
            try:
                _history_entry = json.dumps({
                    "date": date_str,
                    "promoted": promoted,
                    "test_ic": round(float(test_ic), 6),
                    "ensemble_ic": round(float(ensemble_ic), 6) if rank_scorer is not None else None,
                    "test_hit_rate": round(test_hit_rate, 4),
                    "elapsed_s": round(elapsed_s, 1),
                    "code_commit": _code_commit,
                    "model_version": model_version,
                }, default=str) + "\n"
                _history_key = "predictor/training_logs/history.jsonl"
                _existing = b""
                try:
                    _existing = _s3_sum.get_object(Bucket=bucket, Key=_history_key)["Body"].read()
                except Exception:
                    pass  # First run — no history yet
                _s3_sum.put_object(
                    Bucket=bucket,
                    Key=_history_key,
                    Body=_existing + _history_entry.encode(),
                    ContentType="application/jsonlines",
                )
                log.info("Training history appended to %s", _history_key)
            except Exception as _hist_err:
                log.warning("Training history append failed (non-blocking): %s", _hist_err)
        except Exception as _sum_err:
            log.warning("Training summary write failed (non-blocking): %s", _sum_err)

    # ── Write training feature stats for drift detection ───────────────────
    if not dry_run:
        try:
            import boto3 as _boto3_stats
            _s3_stats = _boto3_stats.client("s3")
            _feat_means = X_train.mean(axis=0).tolist()
            _feat_stds = X_train.std(axis=0).tolist()
            _feat_stats = {
                "date": date_str,
                "features": list(cfg.GBM_FEATURES),
                "mean": _feat_means,
                "std": _feat_stds,
                "n_train_samples": int(len(y_train)),
            }
            _s3_stats.put_object(
                Bucket=bucket,
                Key="predictor/metrics/training_feature_stats.json",
                Body=json.dumps(_feat_stats, indent=2).encode(),
                ContentType="application/json",
            )
            log.info("Training feature stats written for drift detection")
        except Exception as _fs_err:
            log.warning("Training feature stats write failed (non-blocking): %s", _fs_err)

    result = {
        "model_version":         model_version,
        "best_iteration":        scorer._best_iteration,
        "val_ic":                round(float(scorer._val_ic), 6),
        "test_ic":               round(float(test_ic), 6),
        "mse_ic":                round(float(mse_ic), 6),
        "rank_ic":               round(float(rank_ic), 6) if rank_scorer is not None else None,
        "ensemble_ic":           round(float(ensemble_ic), 6) if rank_scorer is not None else None,
        "ensemble_enabled":      rank_scorer is not None,
        "catboost_enabled":      cat_scorer is not None,
        "catboost_ic":           round(float(cat_ic), 6) if cat_scorer is not None else None,
        "lgb_cat_blend_ic":      round(float(lgb_cat_blend_ic), 6) if cat_scorer is not None else None,
        "blend_weights":         blend_weights,
        "promoted_mode":         best_mode if promoted else None,
        "test_ic_p":             0.0,  # p-value omitted (numpy-only path)
        "ic_ir":                 round(ic_ir, 4),
        "ic_positive_20":        int((chunk_ics > 0).sum()),
        "passes_ic_gate":        passes_ic,
        "passes_ic_ir_gate":     passes_ic_ir,
        "promoted":              promoted,
        "n_train":               int(len(y_train)),
        "n_val":                 int(len(y_val)),
        "n_test":                int(len(y_test)),
        "elapsed_s":             round(elapsed_s, 1),
        "feature_importance_top10": [
            {"feature": f, "gain": round(s, 2)} for f, s in top10
        ],
        "feature_importance_shap_top10": [
            {"feature": f, "shap": round(s, 4)} for f, s in
            sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
        ] if shap_importance else [],
        "feature_ics": feature_ics,
        "noise_candidates": noise_candidates,
        "calibration": calibration_metrics,
    }

    # Attach walk-forward detail for email reporting
    if wf_result and not wf_result.get("fallback"):
        # Don't include large numpy arrays in serialized result
        wf_for_result = {k: v for k, v in wf_result.items()
                         if k not in ("oos_preds", "oos_actuals")}
        result["walk_forward"] = wf_for_result

    # ── SHAP history: store dated SHAP and compute drift ──────────────────
    if shap_importance and not dry_run:
        try:
            import boto3
            s3_shap = boto3.client("s3")

            # Write current SHAP to dated S3 key
            shap_key = f"predictor/metrics/shap_{date_str}.json"
            s3_shap.put_object(
                Bucket=bucket,
                Key=shap_key,
                Body=json.dumps(shap_importance, indent=2).encode(),
                ContentType="application/json",
            )
            log.info("SHAP importance saved to s3://%s/%s", bucket, shap_key)

            # Load previous week's SHAP for drift detection
            from datetime import timedelta
            prev_date = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=7)
            prev_key = f"predictor/metrics/shap_{prev_date.strftime('%Y-%m-%d')}.json"
            try:
                prev_obj = s3_shap.get_object(Bucket=bucket, Key=prev_key)
                prev_shap = json.loads(prev_obj["Body"].read().decode())

                # Compute Spearman rank correlation between current and previous SHAP
                common_features = [f for f in cfg.GBM_FEATURES
                                   if f in shap_importance and f in prev_shap]
                if len(common_features) >= 5:
                    from scipy.stats import spearmanr
                    curr_vals = [shap_importance[f] for f in common_features]
                    prev_vals = [prev_shap[f] for f in common_features]
                    rho, _ = spearmanr(curr_vals, prev_vals)
                    rho = round(float(rho), 4)
                    result["shap_rank_stability"] = rho
                    log.info("SHAP rank stability (Spearman): %.4f", rho)
                    if rho < 0.80:
                        log.warning(
                            "SHAP feature drift detected: rank correlation %.4f < 0.80",
                            rho,
                        )
                else:
                    log.info("Not enough common features for SHAP drift (%d)", len(common_features))
            except s3_shap.exceptions.NoSuchKey:
                log.info("No previous SHAP found at %s — skipping drift check", prev_key)
            except Exception as e:
                log.warning("SHAP drift check failed (non-blocking): %s", e)
        except Exception as e:
            log.warning("SHAP history storage failed (non-blocking): %s", e)

    return result


# ── Training email ─────────────────────────────────────────────────────────────

def _build_ic_table_html(result, is_meta, ic_color, ic_label, promoted_mode,
                         promo_color, promo_label, val_ic, mse_ic, test_ic,
                         rank_ic, ensemble_ic, ensemble_on, ic_ir, ic_pos):
    """Build the IC metrics table HTML for the training email."""
    _bg_style = ' style="background:#f9f9f9;"'
    _row = lambda label, value, bg=False: (
        f'<tr{_bg_style if bg else ""}>'
        f'<td style="padding:5px 10px; color:#555; width:160px;">{label}</td>'
        f'<td style="padding:5px 10px; font-family:monospace; font-weight:bold;">{value}</td></tr>'
    )

    rows = '<table style="border-collapse:collapse; width:100%; margin-bottom:12px;">'

    if is_meta:
        meta_ic = result.get("meta_model_ic", test_ic)
        mom_ic = result.get("momentum_test_ic", mse_ic)
        vol_ic = result.get("volatility_test_ic", 0)
        regime_acc = result.get("regime_accuracy", 0)
        rows += _row("Meta-Model IC", f'{meta_ic:.4f}', bg=True)
        rows += _row("Momentum IC", f'{mom_ic:.4f}')
        rows += _row("Volatility IC", f'{vol_ic:.4f}', bg=True)
        rows += _row("Regime Accuracy", f'{regime_acc*100:.1f}%')
        coefs = result.get("meta_coefficients", {})
        if coefs:
            coef_str = " | ".join(
                f'{k}={v:+.3f}' for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1]))
                if k != "intercept" and abs(v) > 0.0001
            )
            rows += _row("Meta Coefficients", coef_str, bg=True)
    else:
        rows += _row("Val IC", f'{val_ic:.4f}', bg=True)
        rows += _row("MSE Model IC",
                      f'{mse_ic:.4f}{f" — {ic_label}" if promoted_mode == "mse" else ""}')
        if ensemble_on and rank_ic is not None:
            rows += _row("Lambdarank IC",
                          f'{rank_ic:.4f}{f" — {ic_label}" if promoted_mode == "rank" else ""}', bg=True)
            rows += _row("Ensemble IC",
                          f'{ensemble_ic:.4f}{f" — {ic_label}" if promoted_mode == "ensemble" else ""}')
        else:
            rows += _row("Test IC", f'<span style="color:{ic_color};">{test_ic:.4f} — {ic_label}</span>')
        rows += _row("IC IR", f'{ic_ir:.3f} ({ic_pos}/20 positive)', bg=True)

    rows += (
        f'<tr style="background:#f9f9f9;">'
        f'<td style="padding:5px 10px; color:#555; font-weight:bold;">Promotion</td>'
        f'<td style="padding:5px 10px; font-weight:bold; color:{promo_color};">{promo_label}</td></tr>'
    )
    rows += '</table>'
    return rows


def send_training_email(result: dict, date_str: str) -> bool:
    """
    Send GBM training summary email via Gmail SMTP (primary) or SES (fallback).
    Returns True on success. Never raises.
    """
    import config as cfg

    sender     = cfg.EMAIL_SENDER
    recipients = cfg.EMAIL_RECIPIENTS

    if not sender or not recipients:
        log.info("Training email skipped — EMAIL_SENDER/EMAIL_RECIPIENTS not set")
        return False

    promoted     = result.get("promoted", False)
    promoted_mode = result.get("promoted_mode")
    passes_ic    = result.get("passes_ic_gate", False)
    val_ic       = result.get("val_ic", 0.0)
    test_ic      = result.get("test_ic", 0.0)
    mse_ic       = result.get("mse_ic", test_ic)
    rank_ic      = result.get("rank_ic")
    ensemble_ic  = result.get("ensemble_ic")
    ensemble_on  = result.get("ensemble_enabled", False)
    ic_ir        = result.get("ic_ir", 0.0)
    version      = result.get("model_version", "unknown")
    is_meta      = "meta" in str(version).lower()
    elapsed_s    = result.get("elapsed_s", 0)
    n_train      = result.get("n_train", 0)
    ic_pos       = result.get("ic_positive_20", 0)
    top10        = result.get("feature_importance_top10", [])

    ic_color    = "#2e7d32" if passes_ic else "#c62828"
    ic_label    = "PASS ✓" if passes_ic else "FAIL ✗"
    promo_color = "#2e7d32" if promoted else "#c62828"

    # Build the promotion label. Previously this was hardcoded to
    # "NOT promoted (IC gate failed ✗)" for all non-promotion cases, which
    # was actively misleading: the 2026-04-11 v3.0-meta run produced
    # Meta-Model IC 0.0525 (well above the 0.03 gate) but was blocked by
    # the walk-forward validation on the momentum base model. The email
    # said "IC gate failed" when the IC gate actually passed.
    def _build_failure_reason() -> str:
        wf = result.get("walk_forward") or {}
        if not wf:
            return "IC gate failed"  # fallback — no wf info available
        mom = wf.get("momentum_median_ic")
        vol = wf.get("volatility_median_ic")
        reasons: list[str] = []
        if mom is not None and mom <= 0:
            reasons.append(f"momentum median IC {mom:+.4f}")
        if vol is not None and vol <= 0:
            reasons.append(f"volatility median IC {vol:+.4f}")
        if reasons:
            return "walk-forward failed: " + ", ".join(reasons)
        # wf section present but no negative medians — must be the
        # in-sample IC gate that failed
        return "IC gate failed"

    promo_label = (
        f"Promoted → weights/meta/ ✓" if promoted and is_meta
        else f"Promoted → gbm_latest ({promoted_mode}) ✓" if promoted
        else f"NOT promoted ({_build_failure_reason()}) ✗"
    )
    status_str  = "PASS" if passes_ic else "FAIL"

    # CatBoost metrics for email
    cat_enabled  = result.get("catboost_enabled", False)
    cat_ic_val   = result.get("catboost_ic")
    blend_ic_val = result.get("lgb_cat_blend_ic")
    blend_wts    = result.get("blend_weights")
    cal_metrics  = result.get("calibration")
    mh_data      = result.get("multi_horizon")

    subject = (
        f"Alpha Engine Training | {date_str} | "
        f"{'Meta IC' if is_meta else 'IC'} {test_ic:.4f} {status_str} | "
        f"{'Promoted (' + promoted_mode + ')' if promoted else 'Not promoted'}"
    )

    # Feature importance bar chart (top 5)
    max_gain  = max((r["gain"] for r in top10), default=1)
    feat_rows = "".join(
        f'<tr>'
        f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{r["feature"]}</td>'
        f'<td style="padding:2px 8px; width:130px;">'
        f'<div style="background:#1976d2; height:10px; width:{int(r["gain"]/max_gain*100)}%;"></div>'
        f'</td>'
        f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{r["gain"]:.1f}</td>'
        f'</tr>'
        for r in top10[:5]
    )

    # Walk-forward section for email
    wf_data = result.get("walk_forward")
    wf_html = ""
    wf_plain = ""
    is_meta = "meta" in str(result.get("model_version", "")).lower()
    if wf_data and wf_data.get("folds"):
        # Meta-model uses per-model median ICs; v2.0 uses single median_ic
        wf_median = wf_data.get("median_ic") or wf_data.get("momentum_median_ic", 0.0)
        wf_pct = wf_data.get("pct_positive", 0.0)
        wf_pass = wf_data.get("passes_wf", False)
        wf_color = "#2e7d32" if wf_pass else "#c62828"
        wf_label = "PASS ✓" if wf_pass else "FAIL ✗"

        if is_meta:
            mom_median = wf_data.get("momentum_median_ic", "n/a")
            vol_median = wf_data.get("volatility_median_ic", "n/a")
            fold_rows = "".join(
                f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["fold"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_start"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_end"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
                f'color:{"#2e7d32" if f.get("mom_ic", 0) > 0 else "#c62828"};">{f.get("mom_ic", 0):.4f}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
                f'color:#2e7d32;">{f.get("vol_ic", 0):.4f}</td>'
                f'</tr>'
                for i, f in enumerate(wf_data["folds"])
            )
            wf_html = (
                f'<h3 style="margin-top:16px; margin-bottom:4px;">Walk-Forward Validation '
                f'({len(wf_data["folds"])} folds)</h3>'
                f'<p style="font-size:12px; margin:2px 0;">Momentum median IC: <b>{mom_median}</b> '
                f'&nbsp;|&nbsp; Volatility median IC: <b>{vol_median}</b> '
                f'&nbsp;|&nbsp; Status: <b style="color:{wf_color};">{wf_label}</b></p>'
                f'<table style="border-collapse:collapse; width:100%; font-size:11px;">'
                f'<tr style="background:#e0e0e0;">'
                f'<th style="padding:3px 6px;">Fold</th>'
                f'<th style="padding:3px 6px;">Test Start</th>'
                f'<th style="padding:3px 6px;">Test End</th>'
                f'<th style="padding:3px 6px;">Mom IC</th>'
                f'<th style="padding:3px 6px;">Vol IC</th></tr>'
                f'{fold_rows}</table>'
            )
            wf_plain = (
                f"\n--- Walk-Forward ({len(wf_data['folds'])} folds) ---"
                f"\nMomentum median IC: {mom_median}  |  Volatility median IC: {vol_median}"
                f"\nStatus: {wf_label}\n"
                + "\n".join(
                    f"  Fold {f['fold']}: [{f['test_start']} → {f['test_end']}] "
                    f"mom={f.get('mom_ic', 0):.4f}  vol={f.get('vol_ic', 0):.4f}"
                    for f in wf_data["folds"]
                ) + "\n"
            )
        else:
            fold_rows = "".join(
                f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["fold"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_start"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["test_end"]}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px;">{f["n_train"]:,}</td>'
                f'<td style="padding:2px 6px; font-family:monospace; font-size:11px; '
                f'color:{"#2e7d32" if f["ic"] > 0 else "#c62828"};">{f["ic"]:.4f}</td>'
                f'</tr>'
                for i, f in enumerate(wf_data["folds"])
            )
            wf_html = (
                f'<h3 style="margin-top:16px; margin-bottom:4px;">Walk-Forward Validation '
                f'({len(wf_data["folds"])} folds)</h3>'
                f'<p style="font-size:12px; margin:2px 0;">Median IC: <b style="color:{wf_color};">'
                f'{wf_median:.4f}</b> — {wf_label} &nbsp;|&nbsp; '
                f'Positive folds: <b>{wf_pct*100:.0f}%</b></p>'
                f'<table style="border-collapse:collapse; width:100%; font-size:11px;">'
                f'<tr style="background:#e0e0e0;">'
                f'<th style="padding:3px 6px;">Fold</th>'
                f'<th style="padding:3px 6px;">Test Start</th>'
                f'<th style="padding:3px 6px;">Test End</th>'
                f'<th style="padding:3px 6px;">Train N</th>'
                f'<th style="padding:3px 6px;">IC</th></tr>'
                f'{fold_rows}</table>'
            )
            wf_plain = (
                f"\n--- Walk-Forward ({len(wf_data['folds'])} folds) ---"
                f"\nMedian IC: {wf_median:.4f} — {wf_label}"
                f"\nPositive folds: {wf_pct*100:.0f}%\n"
                + "\n".join(
                    f"  Fold {f['fold']}: [{f['test_start']} → {f['test_end']}] "
                    f"train={f['n_train']:,}  IC={f['ic']:.4f}"
                    for f in wf_data["folds"]
                ) + "\n"
            )

    # ── SHAP vs Gain comparison section ──────────────────────────────────────
    shap_top10    = result.get("feature_importance_shap_top10", [])
    shap_stability = result.get("shap_rank_stability")

    shap_html = ""
    shap_plain = ""
    if shap_top10 and top10:
        # Build rank lookup: feature → rank (1-based)
        gain_rank = {r["feature"]: i + 1 for i, r in enumerate(top10)}
        shap_rank = {r["feature"]: i + 1 for i, r in enumerate(shap_top10)}
        all_features = list(dict.fromkeys(
            [r["feature"] for r in top10] + [r["feature"] for r in shap_top10]
        ))

        comparison_rows = ""
        comparison_plain_lines = []
        for feat in all_features[:10]:
            g_rank = gain_rank.get(feat, "-")
            s_rank = shap_rank.get(feat, "-")
            divergence = ""
            if isinstance(g_rank, int) and isinstance(s_rank, int):
                diff = abs(g_rank - s_rank)
                if diff > 3:
                    divergence = f' style="color:#c62828; font-weight:bold;"'
            comparison_rows += (
                f'<tr>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px;">{feat}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px; text-align:center;">{g_rank}</td>'
                f'<td style="padding:2px 8px; font-family:monospace; font-size:12px; text-align:center;"'
                f'{divergence}>{s_rank}</td>'
                f'</tr>'
            )
            flag = " ***" if isinstance(g_rank, int) and isinstance(s_rank, int) and abs(g_rank - s_rank) > 3 else ""
            comparison_plain_lines.append(f"  {feat:<22} Gain:{g_rank}  SHAP:{s_rank}{flag}")

        stability_note = ""
        stability_plain = ""
        if shap_stability is not None:
            stab_color = "#2e7d32" if shap_stability >= 0.80 else "#c62828"
            stab_label = "stable" if shap_stability >= 0.80 else "DRIFT WARNING"
            stability_note = (
                f'<p style="font-size:12px; margin:4px 0;">SHAP rank stability (vs last week): '
                f'<b style="color:{stab_color};">rho={shap_stability:.4f} — {stab_label}</b></p>'
            )
            stability_plain = f"\nSHAP rank stability: rho={shap_stability:.4f} — {stab_label}"

        shap_html = (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Feature Importance: Gain vs SHAP</h3>'
            f'{stability_note}'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Feature</th>'
            f'<th style="padding:3px 8px;">Gain Rank</th>'
            f'<th style="padding:3px 8px;">SHAP Rank</th></tr>'
            f'{comparison_rows}</table>'
            f'<p style="font-size:10px; color:#888;">Features with rank divergence &gt;3 highlighted in red.</p>'
        )
        shap_plain = (
            "\n--- Gain vs SHAP Rank ---"
            + stability_plain
            + "\n" + "\n".join(comparison_plain_lines)
            + "\n  (*** = rank divergence > 3)\n"
        )

    # ── Feature Health section (per-feature IC + noise detection) ──────────
    feat_ics = result.get("feature_ics", {})
    noise_cands = result.get("noise_candidates", [])
    feat_health_html = ""
    feat_health_plain = ""
    if feat_ics:
        sorted_ics = sorted(feat_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        ic_rows = "".join(
            f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
            f'<td style="padding:2px 8px; font-family:monospace; font-size:11px;">{fname}</td>'
            f'<td style="padding:2px 8px; font-family:monospace; font-size:11px; '
            f'color:{"#2e7d32" if fic > 0 else "#c62828"};">{fic:.4f}</td>'
            f'</tr>'
            for i, (fname, fic) in enumerate(sorted_ics[:10])
        )
        noise_note = ""
        if noise_cands:
            noise_note = (
                f'<p style="font-size:11px; color:#c62828; margin:4px 0;">'
                f'Noise candidates ({len(noise_cands)}): {", ".join(noise_cands)}</p>'
            )
        feat_health_html = (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Feature Health</h3>'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Feature</th>'
            f'<th style="padding:3px 8px;">IC vs Forward</th></tr>'
            f'{ic_rows}</table>'
            f'{noise_note}'
        )
        feat_health_plain = (
            "\n--- Feature Health (top 10 by |IC|) ---\n"
            + "\n".join(f"  {fname:<22} IC={fic:.4f}" for fname, fic in sorted_ics[:10])
            + (f"\nNoise candidates: {', '.join(noise_cands)}" if noise_cands else "")
            + "\n"
        )

    html_body = (
        f'<html><body style="font-family:sans-serif; font-size:13px; color:#222; max-width:600px;">'
        f'<h2 style="margin-bottom:4px;">Alpha Engine Training — {date_str}</h2>'
        f'<p style="color:#555; font-size:12px; margin-top:0;">'
        f'Model: <b>{version}</b> &nbsp;|&nbsp;'
        f'Training samples: <b>{n_train:,}</b> &nbsp;|&nbsp;'
        f'Elapsed: <b>{elapsed_s:.0f}s ({elapsed_s/60:.1f} min)</b></p>'

        + _build_ic_table_html(result, is_meta, ic_color, ic_label, promoted_mode,
                               promo_color, promo_label, val_ic, mse_ic, test_ic,
                               rank_ic, ensemble_ic, ensemble_on, ic_ir, ic_pos) +

        f'{wf_html}'

        + (
            f'<h3 style="margin-bottom:4px;">Top 5 Features by Gain</h3>'
            f'<table>{feat_rows}</table>'
            if feat_rows else ""
        ) +

        f'{shap_html}'

        f'{feat_health_html}'

        + (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Confidence Calibration</h3>'
            f'<p style="font-size:12px;">Method: <b>{cal_metrics["method"]}</b> &nbsp;|&nbsp; '
            f'Samples: <b>{cal_metrics["n_samples"]:,}</b> &nbsp;|&nbsp; '
            f'ECE: <b>{cal_metrics["ece_before"]:.4f} → {cal_metrics["ece_after"]:.4f}</b> '
            f'({(1 - cal_metrics["ece_after"] / max(cal_metrics["ece_before"], 1e-8)) * 100:.0f}% reduction)</p>'
            if cal_metrics and cal_metrics.get("fitted") else ""
        )
        + (
            f'<h3 style="margin-top:16px; margin-bottom:4px;">Multi-Horizon Models</h3>'
            f'<table style="border-collapse:collapse; font-size:11px;">'
            f'<tr style="background:#e0e0e0;">'
            f'<th style="padding:3px 8px;">Horizon</th>'
            f'<th style="padding:3px 8px;">IC</th>'
            f'<th style="padding:3px 8px;">Promoted</th></tr>'
            + "".join(
                f'<tr><td style="padding:2px 8px; font-family:monospace;">{h}d</td>'
                f'<td style="padding:2px 8px; font-family:monospace;">{v.get("test_ic", "err")}</td>'
                f'<td style="padding:2px 8px; font-family:monospace;">{v.get("promoted", False)}</td></tr>'
                for h, v in mh_data["auxiliary"].items() if isinstance(v, dict) and "error" not in v
            )
            + f'</table>'
            if mh_data and mh_data.get("auxiliary") else ""
        ) +

        f'<p style="font-size:11px; color:#aaa; margin-top:20px;">'
        # Meta (v3.0) trainer uses a simpler walk-forward gate:
        # both base models must have strictly positive median IC.
        # See meta_trainer.py:483. The v2 single-model path uses
        # cfg.WF_MEDIAN_IC_GATE + WF_MIN_FOLDS_POSITIVE. Describe
        # whichever one actually gates this run.
        + (
            f'IC gate: ≥{cfg.MIN_IC:.2f} meta IC to promote &nbsp;|&nbsp; '
            f'Walk-forward: momentum &amp; volatility median IC both &gt; 0</p>'
            if is_meta else
            f'IC gate: ≥{cfg.MIN_IC:.2f} to promote &nbsp;|&nbsp; '
            f'Walk-forward: median IC ≥{cfg.WF_MEDIAN_IC_GATE:.2f}, '
            f'{cfg.WF_MIN_FOLDS_POSITIVE*100:.0f}%+ positive folds</p>'
        )
        + f'</body></html>'
    )

    if is_meta:
        meta_ic = result.get("meta_model_ic", test_ic)
        mom_ic = result.get("momentum_test_ic", mse_ic)
        vol_ic_val = result.get("volatility_test_ic", 0)
        regime_acc = result.get("regime_accuracy", 0)
        plain_body = (
            f"Alpha Engine Training — {date_str}\n"
            f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
            f"\nMeta-Model IC:      {meta_ic:.4f} — {ic_label}"
            f"\nMomentum IC:        {mom_ic:.4f}"
            f"\nVolatility IC:      {vol_ic_val:.4f}"
            f"\nRegime Accuracy:    {regime_acc*100:.1f}%"
            f"\nPromotion:          {promo_label}\n"
            f"{wf_plain}"
        )
        coefs = result.get("meta_coefficients", {})
        if coefs:
            plain_body += "\nMeta coefficients:\n" + "\n".join(
                f"  {k:<28} {v:+.4f}"
                for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1]))
                if k != "intercept" and abs(v) > 0.0001
            ) + "\n"
    else:
        _mse_mark  = " ✓" if promoted_mode == "mse" else ""
        _rank_mark = " ✓" if promoted_mode == "rank" else ""
        _ens_mark  = " ✓" if promoted_mode == "ensemble" else ""
        plain_body = (
            f"Alpha Engine Training — {date_str}\n"
            f"Model: {version}  Samples: {n_train:,}  Elapsed: {elapsed_s:.0f}s\n"
            f"\nVal IC:             {val_ic:.4f}"
            f"\nMSE Model IC:       {mse_ic:.4f}{' — ' + ic_label if promoted_mode == 'mse' else ''}{_mse_mark}"
            + (
                f"\nLambdarank Model IC: {rank_ic:.4f}{' — ' + ic_label if promoted_mode == 'rank' else ''}{_rank_mark}"
                f"\nEnsemble IC:        {ensemble_ic:.4f}{' — ' + ic_label if promoted_mode == 'ensemble' else ''}{_ens_mark}"
                if ensemble_on and rank_ic is not None and ensemble_ic is not None else
                f"\nTest IC:            {test_ic:.4f} — {ic_label}"
            )
            + (
                f"\nCatBoost IC:        {cat_ic_val:.4f}"
                f"\nLGB-Cat Blend IC:   {blend_ic_val:.4f} (w_lgb={blend_wts['lgb']:.1f})"
                if cat_enabled and cat_ic_val is not None else ""
            ) +
            f"\nPromoted:           {promoted_mode if promoted else 'none'}"
            f"\nIC IR:              {ic_ir:.3f} ({ic_pos}/20 positive)"
            f"\nPromotion:          {promo_label}\n"
            f"{wf_plain}"
            f"\nTop features: " + ", ".join(r["feature"] for r in top10[:5])
            + f"\n{shap_plain}"
            + f"{feat_health_plain}\n"
        )

    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()

    if app_password:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body,  "html",  "utf-8"))
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender, app_password.replace(" ", ""))
                server.sendmail(sender, recipients, msg.as_string())
            log.info("Training email sent via Gmail SMTP: '%s'", subject)
            return True
        except Exception as exc:
            log.warning("Gmail SMTP failed (%s) — trying SES fallback", exc)

    # SES fallback
    try:
        import boto3
        ses = boto3.client("ses", region_name=cfg.AWS_REGION)
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body,  "Charset": "UTF-8"},
                },
            },
        )
        log.info("Training email sent via SES: '%s'", subject)
        return True
    except Exception as exc:
        log.warning("SES failed: %s — training email not delivered", exc)
        return False


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main(
    bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Entry point called from inference/handler.py for the "train" action.

    1.  Download price cache from S3 to /tmp.
    1b. Refresh any stale parquets with recent yfinance data → upload back to S3.
    2.  Run GBM training pipeline on the now-current local cache.
    2b. Write a 2-year slim cache to S3 (predictor/price_cache_slim/).
        The inference Lambda uses this slim cache + daily_closes delta instead
        of fetching 2 years from yfinance every morning, reducing daily yfinance
        API calls by ~125×.
    3.  Send training summary email.

    Returns the result dict from run_gbm_training().
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    fd = None

    log.info("GBM training run: date=%s  bucket=%s  dry_run=%s", date_str, bucket, dry_run)

    # Step 1: Load price cache — try ArcticDB first, fall back to S3 parquets
    tmp_cache = Path(tempfile.mkdtemp()) / "cache"
    n_files = 0

    try:
        from store.arctic_reader import download_from_arctic
        log.info("[data_source=arcticdb] Loading universe from ArcticDB...")
        n_files = download_from_arctic(bucket=bucket, local_dir=tmp_cache)
    except Exception as exc:
        log.warning("[data_source=arcticdb] ArcticDB load failed — falling back to S3 parquets: %s", exc)

    if n_files == 0:
        log.info("[data_source=legacy] Downloading price cache from S3...")
        n_files = download_price_cache(
            bucket=bucket,
            prefix="predictor/price_cache/",
            local_dir=tmp_cache,
        )

    if n_files == 0:
        raise RuntimeError(
            f"No files found from ArcticDB or s3://{bucket}/predictor/price_cache/ — "
            "run builders/backfill.py or bootstrap_fetcher.py first."
        )

    # Step 1b: Price cache refresh now handled by alpha-engine-data (Phase 1).
    # The data repo runs weekly_collector.py --phase 1 before training,
    # so S3 parquets are already current.
    log.info("Price cache refresh: skipped (handled by alpha-engine-data)")

    # Step 2: Train + upload (v3 meta-model only — v2 single-GBM and
    # multi-horizon dispatch branches removed 2026-04-13; v2 machinery
    # itself still in-tree, full rip-out tracked in ROADMAP).
    from training.meta_trainer import run_meta_training
    result = run_meta_training(
        data_dir=str(tmp_cache),
        bucket=bucket,
        date_str=date_str,
        dry_run=dry_run,
    )

    # Step 2b: Slim cache write now handled by alpha-engine-data (Phase 1).
    # The data repo writes price_cache_slim/ from the full cache after refresh.
    log.info("Slim cache write: skipped (handled by alpha-engine-data)")

    # Feature store registry upload removed — alpha-engine-data handles this now.

    # Step 2d: Write training summary to S3 (works for all modes)
    if not dry_run:
        try:
            import boto3 as _b3_sum
            _s3_sum = _b3_sum.client("s3")
            _sum_body = json.dumps(result, indent=2, default=str).encode()
            _s3_sum.put_object(
                Bucket=bucket,
                Key=f"predictor/metrics/training_summary_{date_str}.json",
                Body=_sum_body, ContentType="application/json",
            )
            _s3_sum.put_object(
                Bucket=bucket,
                Key="predictor/metrics/training_summary_latest.json",
                Body=_sum_body, ContentType="application/json",
            )
            log.info("Training summary written to S3 (dated + latest)")
        except Exception as _sum_err:
            log.warning("Training summary write failed (non-blocking): %s", _sum_err)

    # Step 3: Email
    if not dry_run:
        send_training_email(result, date_str)

    # Step 4: Health status
    if not dry_run:
        try:
            from health_status import write_health
            write_health(
                bucket=bucket,
                module_name="predictor_training",
                status="ok",
                run_date=date_str,
                duration_seconds=result.get("train_time_seconds", 0),
                summary={
                    "promoted": result.get("promoted", False),
                    "ic_30d": result.get("ic_30d"),
                    "n_train": result.get("n_train"),
                    "slim_cache_tickers": result.get("slim_cache_tickers"),
                    "slim_cache_failed": result.get("slim_cache_failed", 0),
                },
            )
        except Exception as _he:
            log.warning("Health status write failed: %s", _he)

        # Data manifest
        try:
            from health_status import write_data_manifest
            write_data_manifest(
                bucket=bucket,
                module_name="predictor_training",
                run_date=date_str,
                manifest={
                    "promoted": result.get("promoted", False),
                    "promoted_mode": result.get("promoted_mode"),
                    "test_ic": result.get("test_ic"),
                    "n_train": result.get("n_train"),
                    "n_test": result.get("n_test"),
                    "slim_cache_tickers": result.get("slim_cache_tickers"),
                    "slim_cache_failed": result.get("slim_cache_failed", 0),
                },
            )
        except Exception as _me:
            log.warning("Data manifest write failed: %s", _me)

    return result


# ── Multi-horizon training ────────────────────────────────────────────────────

def run_multi_horizon_training(
    data_dir: str,
    bucket: str,
    date_str: str,
    dry_run: bool = False,
) -> dict:
    """
    Train separate GBM models for each configured horizon (1d, 5d, 10d, 20d).

    The 5d model uses the standard run_gbm_training() pipeline (with full
    walk-forward, calibrator, CatBoost, etc.). Other horizons train simpler
    MSE-only models with basic IC validation.

    Returns dict with per-horizon results and a horizon_agreement summary.
    """
    import config as cfg

    horizons = cfg.MULTI_HORIZON_LIST
    log.info("Multi-horizon training: horizons=%s", horizons)

    # Train the primary model using the full pipeline
    primary_result = run_gbm_training(data_dir, bucket, date_str, dry_run=dry_run)

    # Train auxiliary horizon models
    aux_results = {}
    for h in horizons:
        if h == cfg.FORWARD_DAYS:
            continue  # already trained as primary

        log.info("Training auxiliary %dd model...", h)
        try:
            aux_result = _train_auxiliary_horizon(
                data_dir, bucket, date_str, h, dry_run=dry_run,
            )
            aux_results[h] = aux_result
            log.info(
                "Auxiliary %dd model: IC=%.4f  promoted=%s",
                h, aux_result.get("test_ic", 0.0), aux_result.get("promoted", False),
            )
        except Exception as exc:
            log.warning("Auxiliary %dd training failed: %s", h, exc)
            aux_results[h] = {"error": str(exc)}

    return {
        "primary": primary_result,
        "auxiliary": aux_results,
        "horizons": horizons,
    }


def _train_auxiliary_horizon(
    data_dir: str,
    bucket: str,
    date_str: str,
    horizon_days: int,
    dry_run: bool = False,
) -> dict:
    """Train a single auxiliary horizon model (MSE only, no WF)."""
    import config as cfg
    from data.dataset import build_regression_arrays
    from model.gbm_scorer import GBMScorer

    # Temporarily override FORWARD_DAYS to build arrays for this horizon
    _orig_fwd = cfg.FORWARD_DAYS
    try:
        cfg.FORWARD_DAYS = horizon_days
        X_all, fwd_all, all_dates = build_regression_arrays(
            data_dir=data_dir,
            config_module=cfg,
            feature_list=cfg.GBM_FEATURES,
        )
    finally:
        cfg.FORWARD_DAYS = _orig_fwd

    N = len(fwd_all)
    n_train = int(N * cfg.TRAIN_FRAC)
    n_val = int(N * cfg.VAL_FRAC)
    val_end = min(n_train + n_val, N)

    X_train, y_train = X_all[:n_train], fwd_all[:n_train]
    X_val, y_val = X_all[n_train:val_end], fwd_all[n_train:val_end]
    X_test, y_test = X_all[val_end:], fwd_all[val_end:]

    scorer = GBMScorer(
        params=cfg.GBM_TUNED_PARAMS,
        n_estimators=cfg.GBM_N_ESTIMATORS,
        early_stopping_rounds=cfg.GBM_EARLY_STOPPING_ROUNDS,
    )
    scorer.fit(X_train, y_train, X_val, y_val, feature_names=cfg.GBM_FEATURES)

    test_preds = scorer.predict(X_test)
    test_ic = float(np.corrcoef(test_preds, y_test)[0, 1]) if len(test_preds) > 1 else 0.0

    promoted = test_ic >= cfg.MIN_IC
    if promoted and not dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"gbm_mse_{horizon_days}d.txt"
            scorer.save(path)
            import boto3
            s3 = boto3.client("s3")
            s3.upload_file(str(path), bucket, f"predictor/weights/gbm_mse_{horizon_days}d_latest.txt")
            s3.upload_file(
                str(path) + ".meta.json", bucket,
                f"predictor/weights/gbm_mse_{horizon_days}d_latest.txt.meta.json",
            )
            s3.upload_file(str(path), bucket, f"predictor/weights/gbm_mse_{horizon_days}d_{date_str}.txt")
            log.info("Auxiliary %dd model uploaded (IC=%.4f)", horizon_days, test_ic)

    return {
        "horizon": horizon_days,
        "test_ic": round(test_ic, 6),
        "promoted": promoted,
        "n_train": len(y_train),
        "best_iteration": scorer._best_iteration,
    }
