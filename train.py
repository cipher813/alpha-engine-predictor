"""
train.py — Root-level training entry point.

Loads datasets from a local parquet cache, trains the DirectionPredictor MLP,
evaluates on the held-out test set, and prints final metrics. The best
checkpoint is saved under the specified output directory.

Usage:
    # Regression mode with ICLoss (scale-invariant, optimizes Pearson IC directly):
    python train.py --mode regression [--data-dir data/cache] [--device cpu]

    # Regression mode with MSE loss (allows macro features to provide cross-date
    # gradients; better when DateGroupedSampler batches would make macro features
    # constant-within-batch, producing zero ICLoss gradient):
    python train.py --mode regression --loss mse [--data-dir data/cache]

    # Classification mode (3-class cross-entropy):
    python train.py --mode classification [--data-dir data/cache]

After training, use inference/daily_predict.py --local to run predictions
with the trained model, or upload checkpoints/best.pt to S3 for Lambda use.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the DirectionPredictor MLP on bootstrapped OHLCV data."
    )
    parser.add_argument(
        "--data-dir",
        default="data/cache",
        help="Directory containing per-ticker parquet files (default: data/cache)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device to train on (default: cpu)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--norm-stats",
        default="data/norm_stats.json",
        help="Path to save normalization statistics (default: data/norm_stats.json)",
    )
    parser.add_argument(
        "--mode",
        default="regression",
        choices=["regression", "classification"],
        help=(
            "Training mode: 'regression' (optimizes return prediction) or "
            "'classification' (CrossEntropyLoss, 3-class UP/FLAT/DOWN). "
            "Default: regression"
        ),
    )
    parser.add_argument(
        "--loss",
        default="ic",
        choices=["ic", "mse"],
        help=(
            "Regression loss function: 'ic' (ICLoss — directly optimizes Pearson IC, "
            "scale-invariant but macro features have zero gradient within date-group "
            "batches) or 'mse' (MSELoss — allows macro features to drive gradients "
            "across dates, better when macro regime features dominate). "
            "Only used in regression mode. Default: ic"
        ),
    )
    args = parser.parse_args()

    import config as cfg
    from model.predictor import DirectionPredictor
    from model.trainer import train
    from model.evaluator import evaluate, compute_ic

    # ── Load tuned hyperparameters if available ───────────────────────────────
    best_params_path = Path(args.output) / "best_params.json"
    if best_params_path.exists():
        try:
            with open(best_params_path) as f:
                tuned = json.load(f).get("best_params", {})
            for key, val in tuned.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
            log.info(
                "Loaded tuned hyperparameters from %s: %s",
                best_params_path,
                {k: v for k, v in tuned.items()},
            )
        except Exception as exc:
            log.warning("Could not load best_params.json (%s) — using config.py defaults", exc)
    else:
        log.info("No best_params.json found — using config.py defaults")

    # ── Validate device ───────────────────────────────────────────────────────
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but not available — falling back to CPU")
        args.device = "cpu"

    log.info("Device: %s  Mode: %s  Loss: %s", args.device, args.mode, args.loss)

    # ── Build datasets ────────────────────────────────────────────────────────
    log.info("Building datasets from %s (mode=%s)...", args.data_dir, args.mode)

    regression_mode = args.mode == "regression"

    try:
        if regression_mode:
            from data.dataset import build_regression_datasets, load_norm_stats
            train_loader, val_loader, test_loader, test_forward_returns = build_regression_datasets(
                data_dir=args.data_dir,
                config_module=cfg,
                norm_stats_path=args.norm_stats,
            )
        else:
            from data.dataset import build_datasets, load_norm_stats
            train_loader, val_loader, test_loader, test_forward_returns = build_datasets(
                data_dir=args.data_dir,
                config_module=cfg,
                norm_stats_path=args.norm_stats,
            )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("Run: python data/bootstrap_fetcher.py --output-dir %s", args.data_dir)
        sys.exit(1)

    from data.dataset import load_norm_stats
    norm_mean, norm_std = load_norm_stats(args.norm_stats)
    norm_stats = {
        "mean": norm_mean.tolist(),
        "std": norm_std.tolist(),
        "features": cfg.FEATURES,
        "mode": args.mode,
    }

    # ── Instantiate model ─────────────────────────────────────────────────────
    n_classes = 1 if regression_mode else cfg.N_CLASSES
    model = DirectionPredictor(
        n_features=cfg.N_FEATURES,
        hidden_1=cfg.HIDDEN_1,
        hidden_2=cfg.HIDDEN_2,
        n_classes=n_classes,
        dropout_1=cfg.DROPOUT_1,
        dropout_2=cfg.DROPOUT_2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Model: DirectionPredictor  n_features=%d  n_classes=%d  parameters=%d",
        cfg.N_FEATURES, n_classes, total_params,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training (max_epochs=%d, patience=%d)...",
             cfg.MAX_EPOCHS, cfg.EARLY_STOPPING_PATIENCE)

    class_weights = None
    if not regression_mode:
        from model.trainer import compute_class_weights
        class_weights = compute_class_weights(train_loader, n_classes=cfg.N_CLASSES)
        log.info(
            "Computed class weights from train set: DOWN=%.3f  FLAT=%.3f  UP=%.3f",
            class_weights[0].item(), class_weights[1].item(), class_weights[2].item(),
        )

    train_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=args.device,
        checkpoint_dir=args.output,
        norm_stats=norm_stats,
        class_weights=class_weights,
        regression_loss=args.loss,
    )

    log.info(
        "Training complete: best_epoch=%d  best_val_loss=%.6f  time=%.1fs",
        train_results["best_epoch"] + 1,
        train_results["best_val_loss"],
        train_results["total_time_s"],
    )

    # ── Load best checkpoint for evaluation ───────────────────────────────────
    from model.predictor import load_checkpoint
    best_path = Path(args.output) / "best.pt"
    best_model, best_checkpoint = load_checkpoint(str(best_path), device=args.device)
    log.info("Loaded best checkpoint from %s", best_path)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    log.info("Evaluating on test set...")

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS — TEST SET")
    print("=" * 60)

    best_model.eval()

    if regression_mode:
        # ── Regression evaluation ─────────────────────────────────────────────
        all_preds: list[float] = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(args.device)
                output = best_model(X_batch).squeeze(-1)
                all_preds.extend(output.cpu().tolist())

        import numpy as np
        preds_arr = np.array(all_preds, dtype=np.float64)
        actuals_arr = test_forward_returns.astype(np.float64)

        mae = float(np.abs(preds_arr - actuals_arr).mean())
        rmse = float(np.sqrt(((preds_arr - actuals_arr) ** 2).mean()))

        print(f"  Mode             : regression (ICLoss)")
        print(f"  Samples          : {len(preds_arr)}")
        print(f"  MAE              : {mae:.5f}  ({mae*100:.3f}%)")
        print(f"  RMSE             : {rmse:.5f}  ({rmse*100:.3f}%)")
        print()

        # IC: direct Pearson r between predicted and actual return
        from scipy.stats import pearsonr
        ic, ic_pval = pearsonr(preds_arr, actuals_arr)
        ic = round(float(ic), 6)

        print("  Pearson IC (test set):")
        print(f"    IC = {ic:.4f}  p={ic_pval:.2e}  (production gate: >{cfg.MIN_IC:.2f})")
        if ic >= cfg.MIN_IC:
            print("  ✓ IC meets production gate")
        else:
            delta = cfg.MIN_IC - ic
            print(f"  ✗ IC below production gate by {delta:.4f}")

        # Directional accuracy from regression predictions
        up_mask = actuals_arr > cfg.UP_THRESHOLD
        down_mask = actuals_arr < cfg.DOWN_THRESHOLD
        pred_up = preds_arr > cfg.UP_THRESHOLD
        pred_down = preds_arr < cfg.DOWN_THRESHOLD
        dir_correct = ((up_mask & pred_up) | (down_mask & pred_down)).sum()
        dir_total = (up_mask | down_mask).sum()
        dir_acc = dir_correct / max(dir_total, 1)
        print()
        print(f"  Directional accuracy (UP/DOWN only): {dir_acc:.4f}  ({dir_acc*100:.1f}%)")
        print(f"    (gate: >{cfg.MIN_HIT_RATE:.0%})")
        if dir_acc >= cfg.MIN_HIT_RATE:
            print("  ✓ Directional accuracy meets production gate")
        else:
            print("  ✗ Directional accuracy below production gate")

        print("=" * 60)

        eval_results_summary = {
            "mode": "regression",
            "n_samples": len(preds_arr),
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "directional_accuracy": round(dir_acc, 4),
        }
        passes_accuracy = dir_acc >= cfg.MIN_HIT_RATE

    else:
        # ── Classification evaluation ─────────────────────────────────────────
        eval_results = evaluate(best_model, test_loader, device=args.device)

        print(f"  Mode             : classification (CrossEntropyLoss)")
        print(f"  Overall accuracy : {eval_results['accuracy']:.4f}  ({eval_results['accuracy'] * 100:.1f}%)")
        print(f"  Samples          : {eval_results['n_samples']}")
        print()
        print("  Per-class accuracy:")
        for label, acc in eval_results["per_class_accuracy"].items():
            gate_marker = ""
            if label in ("UP", "DOWN") and acc >= cfg.MIN_HIT_RATE:
                gate_marker = "  ✓ (meets MIN_HIT_RATE)"
            print(f"    {label:5s}: {acc:.4f}{gate_marker}")

        print()
        print("  Confusion matrix (rows=actual, cols=predicted):")
        print("               DOWN   FLAT   UP")
        labels = ["DOWN", "FLAT", "UP"]
        for i, row_label in enumerate(labels):
            row = eval_results["confusion_matrix"][i]
            print(f"    {row_label:5s}:  {row[0]:6d} {row[1]:6d} {row[2]:6d}")

        print()
        if eval_results["accuracy"] >= cfg.MIN_HIT_RATE:
            print(f"  ✓ Accuracy {eval_results['accuracy']:.1%} meets production gate (>{cfg.MIN_HIT_RATE:.0%})")
        else:
            print(f"  ✗ Accuracy {eval_results['accuracy']:.1%} below production gate (>{cfg.MIN_HIT_RATE:.0%})")
        print("=" * 60)

        # Compute IC from p_up - p_down
        all_p_up: list[float] = []
        all_p_down: list[float] = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(args.device)
                logits = best_model(X_batch)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_p_down.extend(probs[:, 0].tolist())
                all_p_up.extend(probs[:, 2].tolist())

        predictions_for_ic = [
            {"p_up": p_up, "p_down": p_down}
            for p_up, p_down in zip(all_p_up, all_p_down)
        ]
        ic = compute_ic(predictions_for_ic, test_forward_returns.tolist())

        print()
        print("  Pearson IC (test set):")
        print(f"    IC = {ic:.4f}  (production gate: >{cfg.MIN_IC:.2f})")
        if ic >= cfg.MIN_IC:
            print("  ✓ IC meets production gate")
        else:
            print(f"  ✗ IC below production gate — signal too weak for production")

        eval_results_summary = eval_results
        passes_accuracy = eval_results["accuracy"] >= cfg.MIN_HIT_RATE

    # ── Save evaluation report ────────────────────────────────────────────────
    report = {
        "mode": args.mode,
        "train": {
            "best_epoch": train_results["best_epoch"],
            "best_val_loss": train_results["best_val_loss"],
            "total_time_s": train_results["total_time_s"],
        },
        "test": eval_results_summary,
        "ic": ic,
        "production_gates": {
            "min_hit_rate": cfg.MIN_HIT_RATE,
            "min_ic": cfg.MIN_IC,
            "passes_accuracy_gate": bool(passes_accuracy),
            "passes_ic_gate": bool(ic >= cfg.MIN_IC),
            "passes_all_gates": bool(passes_accuracy and ic >= cfg.MIN_IC),
        },
    }
    report_path = Path(args.output) / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Evaluation report saved to %s", report_path)


if __name__ == "__main__":
    main()
