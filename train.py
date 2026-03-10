"""
train.py — Root-level training entry point.

Loads datasets from a local parquet cache, trains the DirectionPredictor MLP,
evaluates on the held-out test set, and prints final metrics. The best
checkpoint is saved under the specified output directory.

Usage:
    python train.py [--data-dir data/cache] [--device cpu|cuda] [--output checkpoints/]

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
    args = parser.parse_args()

    import config as cfg
    from data.dataset import build_datasets, load_norm_stats
    from model.predictor import DirectionPredictor
    from model.trainer import train
    from model.evaluator import evaluate

    # ── Load tuned hyperparameters if available ───────────────────────────────
    # tune.py writes checkpoints/best_params.json after a search run.
    # If present, those values override the defaults in config.py for this run.
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

    log.info("Device: %s", args.device)

    # ── Build datasets ────────────────────────────────────────────────────────
    log.info("Building datasets from %s...", args.data_dir)
    try:
        train_loader, val_loader, test_loader = build_datasets(
            data_dir=args.data_dir,
            config_module=cfg,
            norm_stats_path=args.norm_stats,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("Run: python data/bootstrap_fetcher.py --output-dir %s", args.data_dir)
        sys.exit(1)

    # Load norm stats so they can be embedded in the checkpoint
    norm_mean, norm_std = load_norm_stats(args.norm_stats)
    norm_stats = {
        "mean": norm_mean.tolist(),
        "std": norm_std.tolist(),
        "features": cfg.FEATURES,
    }

    # ── Instantiate model ─────────────────────────────────────────────────────
    model = DirectionPredictor(
        n_features=cfg.N_FEATURES,
        hidden_1=cfg.HIDDEN_1,
        hidden_2=cfg.HIDDEN_2,
        n_classes=cfg.N_CLASSES,
        dropout_1=cfg.DROPOUT_1,
        dropout_2=cfg.DROPOUT_2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: DirectionPredictor  parameters=%d", total_params)

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training (max_epochs=%d, patience=%d)...",
             cfg.MAX_EPOCHS, cfg.EARLY_STOPPING_PATIENCE)

    train_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=args.device,
        checkpoint_dir=args.output,
        norm_stats=norm_stats,
    )

    log.info(
        "Training complete: best_epoch=%d  best_val_loss=%.4f  time=%.1fs",
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
    eval_results = evaluate(best_model, test_loader, device=args.device)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS — TEST SET")
    print("=" * 60)
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
        print("    Model not ready for production. Consider:")
        print("    - Tuning hyperparameters in config.py")
        print("    - Rerunning bootstrap with more history (--period 5y)")
        print("    - Reviewing label thresholds (UP_THRESHOLD / DOWN_THRESHOLD)")
    print("=" * 60)

    # ── Save evaluation report ────────────────────────────────────────────────
    report = {
        "train": {
            "best_epoch": train_results["best_epoch"],
            "best_val_loss": train_results["best_val_loss"],
            "total_time_s": train_results["total_time_s"],
        },
        "test": eval_results,
        "production_gates": {
            "min_hit_rate": cfg.MIN_HIT_RATE,
            "passes_accuracy_gate": eval_results["accuracy"] >= cfg.MIN_HIT_RATE,
        },
    }
    report_path = Path(args.output) / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Evaluation report saved to %s", report_path)


if __name__ == "__main__":
    main()
