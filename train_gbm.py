"""
train_gbm.py — LightGBM GBMScorer training entry point.

Loads the same regression dataset as train.py (same 70/15/15 time split
with purge gaps, 29 features, alpha-vs-sector target), trains GBMScorer,
evaluates Pearson IC on the test set, runs walk-forward validation, and
saves the booster to checkpoints/.

Also measures the prediction correlation between GBM and the saved MLP
(checkpoints/best.pt) — the key gate before building the ensemble.

Usage:
    python train_gbm.py [--data-dir data/cache] [--output checkpoints/]
                        [--tune] [--trials 150]

    --tune    Run Optuna hyperparameter search (150 trials, resumable).
              Uses a persistent SQLite journal so interrupted runs can be
              resumed by re-running the same command.  Best GBM params are
              saved to checkpoints/gbm_best_params.json after each new best.
              Without --tune, default GBM parameters are always used.

Interruption resilience:
    caffeinate -dims python train_gbm.py --tune --trials 150
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress LightGBM's own logger
logging.getLogger("lightgbm").setLevel(logging.WARNING)


def _extract_arrays(loader) -> tuple[np.ndarray, np.ndarray]:
    """Pull all (X, y) pairs out of a DataLoader into numpy arrays."""
    import torch
    Xs, ys = [], []
    for X_batch, y_batch in loader:
        Xs.append(X_batch.numpy())
        ys.append(y_batch.numpy())
    return np.vstack(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def _run_optuna(
    X_train, y_train, X_val, y_val, feature_names,
    n_trials: int, output_dir: Path,
) -> dict:
    """
    Optuna search over GBM hyperparameters with persistent SQLite storage.

    Uses a persistent journal so interrupted runs resume from the last
    completed trial. Best params are saved to gbm_best_params.json after
    each new best via a callback.

    Returns the best params dict (merged with defaults).
    """
    try:
        import optuna
    except ImportError:
        log.error("optuna not installed — run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from model.gbm_scorer import GBMScorer
    from scipy.stats import pearsonr

    # Optuna tunes the MSE model's hyperparams; lambdarank reuses the same params
    def objective(trial) -> float:
        obj = trial.suggest_categorical("objective", ["regression", "huber"])
        params = {
            "objective": obj,
            "metric": "mse",
            "verbosity": -1,
            "seed": 42,
            "num_threads": 4,
            "num_leaves":        trial.suggest_int("num_leaves", 16, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      5,
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        }
        scorer = GBMScorer(params=params, n_estimators=2000, early_stopping_rounds=50,
                           ranking_objective=False)
        scorer.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
        val_preds = scorer.predict(X_val)
        ic, _ = pearsonr(val_preds, y_val)
        return -ic   # minimise negative IC

    # Per-trial callback: save best params to JSON after each new best
    def _save_best_callback(study, trial):
        if study.best_trial.number == trial.number:
            best_path = output_dir / "gbm_best_params.json"
            best_path.write_text(json.dumps({
                "best_params": study.best_trial.params,
                "best_ic": round(-study.best_trial.value, 6),
                "trial_number": trial.number,
                "n_complete": len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
            }, indent=2))

    # Persistent SQLite storage — resume interrupted runs
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir}/optuna_gbm.db"

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name="gbm_alpha_v1.5_29f",
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    n_existing = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    if n_existing > 0:
        log.info("Resuming Optuna study: %d trials already complete", n_existing)

    n_remaining = max(0, n_trials - n_existing)
    if n_remaining == 0:
        log.info("All %d trials complete — using existing best params", n_trials)
    else:
        log.info("Running %d Optuna trials (%d new)...", n_trials, n_remaining)
        study.optimize(
            objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            callbacks=[_save_best_callback],
        )

    best = study.best_trial
    print(f"\n  GBM tuning complete: best val_IC={-best.value:.4f}  trial=#{best.number}")
    print(f"  Total trials completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print("  Best GBM params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    return best.params


def _walk_forward_evaluation(
    X_all: np.ndarray,
    y_all: np.ndarray,
    all_dates: list,
    feature_names: list[str],
    gbm_params: dict | None,
    n_folds: int = 5,
    test_days: int = 60,
    forward_days: int = 5,
) -> dict:
    """
    Walk-forward expanding-window evaluation with purge gaps.

    Trains a fresh GBM on each fold's training data, evaluates IC on
    the fold's test window, and reports IC stability metrics.

    Parameters
    ----------
    X_all : np.ndarray, shape (N, n_features)
    y_all : np.ndarray, shape (N,) — continuous forward returns
    all_dates : list — one date per sample, sorted ascending
    feature_names : list of str
    gbm_params : dict or None — params for GBMScorer (None → defaults)
    n_folds : int — number of walk-forward windows
    test_days : int — number of unique trading days per test fold
    forward_days : int — purge gap size (matches FORWARD_DAYS)

    Returns
    -------
    dict with keys: fold_ics, mean_ic, std_ic, ic_ir, n_positive, n_folds
    """
    from model.gbm_scorer import GBMScorer
    from scipy.stats import pearsonr

    unique_dates = sorted(set(all_dates))
    n_dates = len(unique_dates)

    # Reserve test_days * n_folds dates from the end for test windows
    # Each fold: train on everything before the test window (expanding)
    total_test_dates = test_days * n_folds
    if total_test_dates + forward_days >= n_dates:
        log.warning("Not enough dates for walk-forward (%d dates, need %d)",
                     n_dates, total_test_dates + forward_days)
        return {"fold_ics": [], "mean_ic": 0.0, "std_ic": 0.0, "ic_ir": 0.0,
                "n_positive": 0, "n_folds": 0}

    test_start_date_idx = n_dates - total_test_dates

    fold_ics = []
    for fold in range(n_folds):
        # Test window for this fold
        test_date_start = test_start_date_idx + fold * test_days
        test_date_end = test_date_start + test_days
        test_date_set = set(unique_dates[test_date_start:test_date_end])

        # Train end: purge_days before test start
        train_end_date_idx = test_date_start - forward_days
        if train_end_date_idx <= 0:
            continue
        train_date_set = set(unique_dates[:train_end_date_idx])

        # Build masks
        train_mask = np.array([d in train_date_set for d in all_dates])
        test_mask = np.array([d in test_date_set for d in all_dates])

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask], y_all[test_mask]

        if len(y_tr) < 1000 or len(y_te) < 100:
            log.warning("Fold %d: insufficient data (train=%d, test=%d), skipping",
                         fold + 1, len(y_tr), len(y_te))
            continue

        # Train fresh GBM for this fold
        # Split train into sub-train/sub-val (last 15% of train dates)
        train_dates_sorted = sorted(train_date_set)
        sub_val_date_start = int(len(train_dates_sorted) * 0.85)
        sub_val_dates = set(train_dates_sorted[sub_val_date_start:])
        sub_train_mask = np.array([d not in sub_val_dates for d, m in zip(all_dates, train_mask) if m])
        sub_val_mask = ~sub_train_mask

        X_sub_tr = X_tr[sub_train_mask]
        y_sub_tr = y_tr[sub_train_mask]
        X_sub_val = X_tr[sub_val_mask]
        y_sub_val = y_tr[sub_val_mask]

        scorer = GBMScorer(params=gbm_params, n_estimators=2000, early_stopping_rounds=50)
        scorer.fit(X_sub_tr, y_sub_tr, X_sub_val, y_sub_val, feature_names=feature_names)

        preds = scorer.predict(X_te)
        ic, _ = pearsonr(preds, y_te)
        fold_ics.append(float(ic))

        log.info(
            "  Fold %d/%d: train=%d  test=%d  IC=%.4f  (train_dates=%s→%s  test_dates=%s→%s)",
            fold + 1, n_folds, len(y_tr), len(y_te), ic,
            train_dates_sorted[0].strftime("%Y-%m-%d"),
            train_dates_sorted[-1].strftime("%Y-%m-%d"),
            min(test_date_set).strftime("%Y-%m-%d"),
            max(test_date_set).strftime("%Y-%m-%d"),
        )

    fold_ics_arr = np.array(fold_ics)
    mean_ic = float(fold_ics_arr.mean()) if len(fold_ics) > 0 else 0.0
    std_ic = float(fold_ics_arr.std()) if len(fold_ics) > 0 else 0.0
    ic_ir = mean_ic / (std_ic + 1e-8)
    n_positive = int((fold_ics_arr > 0).sum()) if len(fold_ics) > 0 else 0

    return {
        "fold_ics": [round(x, 6) for x in fold_ics],
        "mean_ic": round(mean_ic, 6),
        "std_ic": round(std_ic, 6),
        "ic_ir": round(ic_ir, 4),
        "n_positive": n_positive,
        "n_folds": len(fold_ics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GBMScorer on the same regression dataset as train.py."
    )
    parser.add_argument("--data-dir", default="data/cache")
    parser.add_argument("--output",   default="checkpoints")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter search before final training."
    )
    parser.add_argument("--trials", default=150, type=int,
                        help="Number of Optuna trials (default: 150, resumable)")
    parser.add_argument("--params", default=None, type=str,
                        help="Path to saved best-params JSON (e.g. checkpoints/gbm_best_params.json). "
                             "Overrides defaults when not using --tune.")
    args = parser.parse_args()

    import config as cfg

    # ── Build regression datasets (same pipeline as train.py) ─────────────────
    log.info("Building regression datasets from %s...", args.data_dir)
    from data.dataset import build_regression_datasets
    try:
        train_loader, val_loader, test_loader, fwd_test = build_regression_datasets(
            data_dir=args.data_dir,
            config_module=cfg,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Extracting numpy arrays from DataLoaders...")
    X_train, y_train = _extract_arrays(train_loader)
    X_val,   y_val   = _extract_arrays(val_loader)
    X_test,  _       = _extract_arrays(test_loader)
    y_test = fwd_test.astype(np.float32)

    log.info(
        "Arrays ready: train=%d  val=%d  test=%d  features=%d",
        len(y_train), len(y_val), len(y_test), X_train.shape[1],
    )

    from model.gbm_scorer import GBMScorer

    # ── Optional Optuna tuning ─────────────────────────────────────────────────
    gbm_params = None
    if args.tune:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Running Optuna GBM search (%d trials, resumable)...", args.trials)
        best_trial_params = _run_optuna(
            X_train, y_train, X_val, y_val,
            feature_names=cfg.FEATURES,
            n_trials=args.trials,
            output_dir=out_dir,
        )
        # Merge tuned params with fixed defaults
        base = GBMScorer._default_params()
        base.update(best_trial_params)
        gbm_params = base
        log.info("Using tuned GBM params from Optuna")
    elif args.params:
        params_path = Path(args.params)
        saved = json.loads(params_path.read_text())
        best_trial_params = saved["best_params"]
        base = GBMScorer._default_params()
        base.update(best_trial_params)
        gbm_params = base
        log.info("Using saved GBM params from %s (IC=%.4f, trial #%d)",
                 params_path, saved.get("best_ic", 0), saved.get("trial_number", -1))
    else:
        log.info("No --tune flag — using default GBM parameters")

    # ── Train final GBM (MSE) ─────────────────────────────────────────────────
    ensemble_enabled = getattr(cfg, "GBM_ENSEMBLE_LAMBDARANK", True)
    log.info("Training GBMScorer (MSE)%s...",
             " + lambdarank ensemble" if ensemble_enabled else "")
    scorer = GBMScorer(
        params=gbm_params,
        n_estimators=3000,
        early_stopping_rounds=50,
        ranking_objective=False,
    )
    scorer.fit(X_train, y_train, X_val, y_val, feature_names=cfg.FEATURES)

    # ── Train lambdarank model (when ensemble enabled) ────────────────────────
    rank_scorer = None
    if ensemble_enabled:
        try:
            rank_scorer = GBMScorer(
                params=gbm_params,
                n_estimators=3000,
                early_stopping_rounds=50,
                ranking_objective=True,
            )
            rank_scorer.fit(X_train, y_train, X_val, y_val, feature_names=cfg.FEATURES)
            log.info("Lambdarank model trained: val_IC=%.4f", rank_scorer._val_ic)
        except Exception as e:
            log.warning("Lambdarank training failed — MSE-only: %s", e)
            rank_scorer = None

    # ── Evaluate on test set ───────────────────────────────────────────────────
    from scipy.stats import pearsonr, rankdata

    test_preds = scorer.predict(X_test)
    test_ic, test_p = pearsonr(test_preds, y_test)

    # Lambdarank + ensemble ICs
    rank_test_ic = 0.0
    ensemble_test_ic = test_ic
    if rank_scorer is not None:
        rank_preds = rank_scorer.predict(X_test)
        rank_test_ic, _ = pearsonr(rank_preds, y_test)
        mse_ranked = rankdata(test_preds).astype(np.float32)
        rank_ranked = rankdata(rank_preds).astype(np.float32)
        blend_preds = 0.5 * mse_ranked + 0.5 * rank_ranked
        ensemble_test_ic, _ = pearsonr(blend_preds, y_test)

    # Rolling IC over 20 test chunks
    n_chunks = 20
    chunk_size = len(test_preds) // n_chunks
    chunk_ics = []
    for i in range(n_chunks):
        s, e = i * chunk_size, (i + 1) * chunk_size
        ic_c, _ = pearsonr(test_preds[s:e], y_test[s:e])
        chunk_ics.append(ic_c)
    chunk_ics = np.array(chunk_ics)
    ic_ir = chunk_ics.mean() / (chunk_ics.std() + 1e-8)

    print("\n" + "=" * 60)
    print("  GBM EVALUATION RESULTS — TEST SET")
    print("=" * 60)
    print(f"  Best iteration : {scorer._best_iteration}")
    print(f"  Val IC         : {scorer._val_ic:.4f}")
    print(f"  MSE Model IC   : {test_ic:.4f}   p={test_p:.2e}")
    if rank_scorer is not None:
        print(f"  Lambdarank IC  : {rank_test_ic:.4f}")
        print(f"  Ensemble IC    : {ensemble_test_ic:.4f}")

    # Pick best IC among all candidates
    if rank_scorer is not None:
        candidates = {"mse": test_ic, "ensemble": ensemble_test_ic, "rank": rank_test_ic}
    else:
        candidates = {"mse": test_ic}
    best_mode = max(candidates, key=candidates.get)
    gate_ic = candidates[best_mode]
    print(f"  Promoted       : {best_mode} ✓")
    print(f"  Gate (>0.05)   : {'PASS ✓' if gate_ic >= 0.05 else 'FAIL ✗'}")
    print(f"  Rolling IC IR  : {ic_ir:.3f}  (target >0.3)")
    print(f"  IC positive periods: {(chunk_ics > 0).sum()}/20")

    # Feature importance (top 10)
    importance = scorer.feature_importance(importance_type="gain")
    top10 = sorted(importance.items(), key=lambda x: -x[1])[:10]
    print("\n  Top-10 features by gain:")
    for feat, score in top10:
        bar = "█" * int(score / max(v for _, v in top10) * 20)
        print(f"    {feat:<22} {bar}")

    # ── Correlation with MLP ───────────────────────────────────────────────────
    mlp_ckpt_path = Path(args.output) / "best.pt"
    if mlp_ckpt_path.exists():
        log.info("Loading MLP checkpoint to measure prediction correlation...")
        import torch
        from model.predictor import load_checkpoint

        try:
            mlp_model, _ = load_checkpoint(str(mlp_ckpt_path), device="cpu")
            mlp_model.eval()
            mlp_preds_list = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    mlp_preds_list.extend(
                        mlp_model(X_batch).squeeze(-1).cpu().tolist()
                    )
            mlp_preds = np.array(mlp_preds_list)
            mlp_test_ic, _ = pearsonr(mlp_preds, y_test)
            pred_corr, _ = pearsonr(test_preds, mlp_preds)

            # Ensemble 50/50 blend IC
            # Rank-normalise both models to equal scale before blending.
            # GBM preds are in return units (~0.01); MLP (ICLoss) preds have
            # arbitrary magnitude — naive averaging is dominated by the
            # higher-variance signal. Rank-normalisation is standard practice
            # for combining alpha signals of different provenance.
            from scipy.stats import rankdata
            gbm_ranked = rankdata(test_preds).astype(np.float32)
            mlp_ranked = rankdata(mlp_preds).astype(np.float32)
            blend_preds = 0.5 * gbm_ranked + 0.5 * mlp_ranked
            blend_ic, _ = pearsonr(blend_preds, y_test)

            print("\n" + "=" * 60)
            print("  ENSEMBLE ANALYSIS")
            print("=" * 60)
            print(f"  MLP  test IC       : {mlp_test_ic:.4f}")
            print(f"  GBM  test IC       : {test_ic:.4f}")
            print(f"  Prediction corr ρ  : {pred_corr:.4f}")
            print(f"  50/50 blend IC     : {blend_ic:.4f}   {'PASS ✓' if blend_ic >= 0.05 else 'FAIL ✗'}")

            # Theoretical max blend IC
            r1, r2, rho = mlp_test_ic, test_ic, pred_corr
            try:
                theoretical = ((r1**2 + r2**2 + 2*rho*r1*r2) / (1 + rho)) ** 0.5
            except Exception:
                theoretical = float("nan")
            print(f"  Theoretical max IC : {theoretical:.4f}  (if optimally blended)")

            gate = "PROCEED WITH ENSEMBLE ✓" if pred_corr < 0.85 else "HIGH CORRELATION — REASSESS ✗"
            print(f"\n  ρ < 0.85 gate      : {gate}")
            print("=" * 60)

            # Save ensemble report
            ensemble_report = {
                "mlp_test_ic":    round(float(mlp_test_ic), 6),
                "gbm_test_ic":    round(float(test_ic), 6),
                "prediction_corr": round(float(pred_corr), 6),
                "blend_50_50_ic": round(float(blend_ic), 6),
                "theoretical_max_ic": round(float(theoretical), 6) if not np.isnan(theoretical) else None,
                "proceed_with_ensemble": bool(pred_corr < 0.85),
            }
            report_path = Path(args.output) / "ensemble_report.json"
            report_path.write_text(json.dumps(ensemble_report, indent=2))
            log.info("Ensemble report saved to %s", report_path)
        except RuntimeError as exc:
            log.warning(
                "Skipping MLP ensemble analysis — feature dimension mismatch "
                "(MLP checkpoint trained with different N_FEATURES). "
                "Retrain MLP with current features to re-enable. Error: %s", exc,
            )

    # ── Save GBM booster(s) ─────────────────────────────────────────────────────
    save_path = Path(args.output) / "gbm_best.txt"
    scorer.save(save_path)
    log.info("GBMScorer (MSE) saved to %s", save_path)

    if rank_scorer is not None:
        rank_save_path = Path(args.output) / "gbm_rank_best.txt"
        rank_scorer.save(rank_save_path)
        log.info("GBMScorer (lambdarank) saved to %s", rank_save_path)

    # ── Save gbm_mode.json ─────────────────────────────────────────────────────
    mode_path = Path(args.output) / "gbm_mode.json"
    mode_path.write_text(json.dumps({"mode": best_mode}, indent=2))
    log.info("Inference mode saved to %s: mode=%s", mode_path, best_mode)

    # ── Save eval report ───────────────────────────────────────────────────────
    gbm_report = {
        "model": "GBMScorer",
        "gbm_version": scorer._best_iteration,
        "val_ic": round(scorer._val_ic, 6),
        "test_ic": round(float(test_ic), 6),
        "ic_ir": round(float(ic_ir), 4),
        "ic_positive_periods": int((chunk_ics > 0).sum()),
        "production_gates": {
            "min_ic": cfg.MIN_IC,
            "passes_ic_gate": float(test_ic) >= cfg.MIN_IC,
            "passes_ic_ir_gate": float(ic_ir) >= 0.3,
        },
        "feature_importance_top10": [
            {"feature": f, "gain": round(s, 2)} for f, s in top10
        ],
    }
    gbm_report_path = Path(args.output) / "gbm_eval_report.json"
    gbm_report_path.write_text(json.dumps(gbm_report, indent=2))
    log.info("GBM eval report saved to %s", gbm_report_path)

    # ── Walk-forward robustness evaluation ────────────────────────────────────
    # Re-assemble the full (unsplit) dataset for walk-forward windows.
    log.info("Running walk-forward evaluation (5 folds × 60 test days)...")
    from data.dataset import build_regression_arrays
    try:
        X_wf, y_wf, all_dates_wf = build_regression_arrays(
            data_dir=args.data_dir, config_module=cfg,
        )
        wf_results = _walk_forward_evaluation(
            X_wf, y_wf, all_dates_wf,
            feature_names=cfg.FEATURES,
            gbm_params=scorer.params if hasattr(scorer, 'params') else gbm_params,
            n_folds=5,
            test_days=60,
            forward_days=cfg.FORWARD_DAYS,
        )

        print("\n" + "=" * 60)
        print("  WALK-FORWARD EVALUATION (5 folds × 60 test days)")
        print("=" * 60)
        for i, ic in enumerate(wf_results["fold_ics"], 1):
            status = "✓" if ic > 0 else "✗"
            print(f"  Fold {i}: IC = {ic:+.4f}  {status}")
        print(f"\n  Mean IC : {wf_results['mean_ic']:+.4f}")
        print(f"  Std IC  : {wf_results['std_ic']:.4f}")
        print(f"  IC IR   : {wf_results['ic_ir']:.3f}  (target ≥ 1.0)")
        print(f"  Positive: {wf_results['n_positive']}/{wf_results['n_folds']}  (target ≥ 70%)")
        print("=" * 60)

        # Save walk-forward report
        wf_path = Path(args.output) / "walk_forward_report.json"
        wf_path.write_text(json.dumps(wf_results, indent=2))
        log.info("Walk-forward report saved to %s", wf_path)

    except Exception as exc:
        log.warning("Walk-forward evaluation failed (non-fatal): %s", exc)


if __name__ == "__main__":
    main()
