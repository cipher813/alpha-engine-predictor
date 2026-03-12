"""
train_gbm.py — LightGBM GBMScorer training entry point.

Loads the same regression dataset as train.py (same 70/15/15 time split,
same 17 features, same alpha-vs-SPY target), trains GBMScorer, evaluates
Pearson IC on the test set, and saves the booster to checkpoints/.

Also measures the prediction correlation between GBM and the saved MLP
(checkpoints/best.pt) — the key gate before building the ensemble.

Usage:
    python train_gbm.py [--data-dir data/cache] [--output checkpoints/]
                        [--tune] [--trials 30]

    --tune    Run a quick Optuna search (30 trials) before final training.
              Saves best GBM params to checkpoints/gbm_best_params.json.
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


def _run_optuna(X_train, y_train, X_val, y_val, feature_names, n_trials: int) -> dict:
    """
    Quick Optuna search over the most impactful GBM hyperparameters.
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

    def objective(trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1,
            "seed": 42,
            "num_threads": 4,
            "num_leaves":        trial.suggest_int("num_leaves", 16, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      5,
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-3, 1.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-3, 1.0, log=True),
        }
        scorer = GBMScorer(params=params, n_estimators=500, early_stopping_rounds=30)
        scorer.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
        val_preds = scorer.predict(X_val)
        ic, _ = pearsonr(val_preds, y_val)
        return -ic   # minimise negative IC

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  GBM tuning complete: best val_IC={-best.value:.4f}  trial=#{best.number}")
    print("  Best GBM params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    return best.params


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
    parser.add_argument("--trials", default=30, type=int,
                        help="Number of Optuna trials (default: 30)")
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
        log.info("Running Optuna GBM search (%d trials)...", args.trials)
        best_trial_params = _run_optuna(
            X_train, y_train, X_val, y_val,
            feature_names=cfg.FEATURES,
            n_trials=args.trials,
        )
        # Merge tuned params with fixed defaults
        base = GBMScorer._default_params()
        base.update(best_trial_params)
        gbm_params = base

        # Save for reference
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        params_path = out_dir / "gbm_best_params.json"
        params_path.write_text(json.dumps({"best_params": gbm_params}, indent=2))
        log.info("GBM tuned params saved to %s", params_path)
    else:
        # Check for previously saved GBM params
        params_path = Path(args.output) / "gbm_best_params.json"
        if params_path.exists():
            saved = json.loads(params_path.read_text())
            gbm_params = saved.get("best_params")
            log.info("Loaded GBM params from %s", params_path)
        else:
            log.info("No gbm_best_params.json found — using defaults")

    # ── Train final GBM ────────────────────────────────────────────────────────
    log.info("Training GBMScorer...")
    scorer = GBMScorer(
        params=gbm_params,
        n_estimators=3000,
        early_stopping_rounds=50,
    )
    scorer.fit(X_train, y_train, X_val, y_val, feature_names=cfg.FEATURES)

    # ── Evaluate on test set ───────────────────────────────────────────────────
    from scipy.stats import pearsonr

    test_preds = scorer.predict(X_test)
    test_ic, test_p = pearsonr(test_preds, y_test)

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
    print(f"  Test IC        : {test_ic:.4f}   p={test_p:.2e}")
    print(f"  Gate (>0.05)   : {'PASS ✓' if test_ic >= 0.05 else 'FAIL ✗'}")
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
        blend_preds = 0.5 * test_preds + 0.5 * mlp_preds
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

    # ── Save GBM booster ───────────────────────────────────────────────────────
    save_path = Path(args.output) / "gbm_best.txt"
    scorer.save(save_path)
    log.info("GBMScorer saved to %s", save_path)

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


if __name__ == "__main__":
    main()
