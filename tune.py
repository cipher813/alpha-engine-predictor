"""
tune.py — Optuna-based hyperparameter search for DirectionPredictor.

Runs N trials of reduced-epoch training, each with sampled hyperparameters.
Saves the best parameters to checkpoints/best_params.json, which train.py
picks up automatically on the next run.

The tuner optimizes for validation loss using Optuna's TPE sampler with
MedianPruner to kill unpromising trials early.

Usage:
    # Regression mode (default — aligns with train.py --mode regression):
    python tune.py --mode regression [--data-dir data/cache] [--trials 50]

    # Classification mode:
    python tune.py --mode classification [--data-dir data/cache] [--trials 50]

The --epochs flag controls the per-trial budget (default 30, vs. 100 for full
training). Each trial trains a fresh model from scratch.

After tuning, run:
    python train.py --mode regression  # automatically loads checkpoints/best_params.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress Optuna's per-trial INFO noise — show only warnings and the summary.
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _suggest_params(trial) -> dict[str, Any]:
    """
    Define the hyperparameter search space.

    Label thresholds are the highest-impact knob: they control class balance
    across the full training set. Architecture and regularization parameters
    are secondary.
    """
    return {
        # Architecture
        "HIDDEN_1":  trial.suggest_categorical("HIDDEN_1",  [32, 64, 128, 256]),
        "HIDDEN_2":  trial.suggest_categorical("HIDDEN_2",  [16, 32, 64, 128]),
        "DROPOUT_1": trial.suggest_float("DROPOUT_1", 0.1, 0.5, step=0.1),
        "DROPOUT_2": trial.suggest_float("DROPOUT_2", 0.0, 0.4, step=0.1),

        # Optimization
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1e-4, 5e-3, log=True),
        "WEIGHT_DECAY":  trial.suggest_float("WEIGHT_DECAY",  1e-5, 1e-3, log=True),

        # Label thresholds (symmetric search — UP threshold drives class balance)
        "UP_THRESHOLD": trial.suggest_categorical(
            "UP_THRESHOLD", [0.005, 0.01, 0.015, 0.02]
        ),
    }


def _params_to_config_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """Expand sampled params into a full config override dict."""
    return {
        "HIDDEN_1":      params["HIDDEN_1"],
        "HIDDEN_2":      params["HIDDEN_2"],
        "DROPOUT_1":     params["DROPOUT_1"],
        "DROPOUT_2":     params["DROPOUT_2"],
        "LEARNING_RATE": params["LEARNING_RATE"],
        "WEIGHT_DECAY":  params["WEIGHT_DECAY"],
        "UP_THRESHOLD":  params["UP_THRESHOLD"],
        "DOWN_THRESHOLD": -params["UP_THRESHOLD"],  # symmetric
    }


# ---------------------------------------------------------------------------
# Per-trial objective
# ---------------------------------------------------------------------------

def _make_objective(
    train_loader, val_loader, cfg, device: str, trial_epochs: int,
    class_weights=None, regression_mode: bool = True,
):
    """
    Return an Optuna objective function that trains a model with the sampled
    params and returns the best validation loss achieved.

    In regression_mode=True: HuberLoss(delta=0.01), n_classes=1.
    In regression_mode=False: CrossEntropyLoss (± class_weights), n_classes=3.
    """
    import torch
    import torch.nn as nn
    from model.predictor import DirectionPredictor

    def objective(trial) -> float:
        params = _suggest_params(trial)
        overrides = _params_to_config_overrides(params)

        n_classes = 1 if regression_mode else cfg.N_CLASSES
        model = DirectionPredictor(
            n_features=cfg.N_FEATURES,
            hidden_1=overrides["HIDDEN_1"],
            hidden_2=overrides["HIDDEN_2"],
            n_classes=n_classes,
            dropout_1=overrides["DROPOUT_1"],
            dropout_2=overrides["DROPOUT_2"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=overrides["LEARNING_RATE"],
            weight_decay=overrides["WEIGHT_DECAY"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )

        if regression_mode:
            from model.trainer import ICLoss
            criterion = ICLoss()
        elif class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience = 5  # shorter patience for trial runs
        patience_counter = 0

        for epoch in range(trial_epochs):
            # Notify date-grouped sampler of new epoch for shuffling
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                train_loader.batch_sampler.set_epoch(epoch)

            # Training pass
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                if regression_mode:
                    loss = criterion(output.squeeze(-1), y_batch)
                else:
                    loss = criterion(output, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation pass
            model.eval()
            val_loss_sum = 0.0
            n_batches = 0
            all_val_preds = []
            all_val_targets = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    if regression_mode:
                        all_val_preds.append(output.squeeze(-1).cpu())
                        all_val_targets.append(y_batch.cpu())
                    else:
                        val_loss_sum += criterion(output, y_batch).item()
                        n_batches += 1
            if regression_mode:
                # Full-set IC: stable cross-sectional Pearson across all val samples
                import torch as _torch
                all_p = _torch.cat(all_val_preds)
                all_t = _torch.cat(all_val_targets)
                avg_val_loss = criterion(all_p, all_t).item()
            else:
                avg_val_loss = val_loss_sum / max(n_batches, 1)

            scheduler.step(avg_val_loss)

            # Optuna pruning — kill unpromising trials early
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise __import__("optuna").exceptions.TrialPruned()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_val_loss

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for DirectionPredictor using Optuna."
    )
    parser.add_argument("--data-dir",  default="data/cache",   help="Parquet cache directory")
    parser.add_argument("--trials",    default=50, type=int,   help="Number of Optuna trials (default: 50)")
    parser.add_argument("--epochs",    default=30, type=int,   help="Max epochs per trial (default: 30)")
    parser.add_argument("--device",    default="cpu",          choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output",    default="checkpoints",  help="Directory to save best_params.json")
    parser.add_argument("--sampler",   default="tpe",          choices=["tpe", "random"],
                        help="Optuna sampler: tpe (default) or random")
    parser.add_argument("--study-name", default="direction_predictor_reg", help="Optuna study name")
    parser.add_argument("--storage",   default=None,
                        help="Optuna storage URL (e.g. sqlite:///tune.db). Optional — enables resume.")
    parser.add_argument(
        "--mode",
        default="regression",
        choices=["regression", "classification"],
        help="Training mode matching train.py (default: regression)",
    )
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        log.error("optuna is not installed. Run: pip install optuna")
        sys.exit(1)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS not available — falling back to CPU")
        args.device = "cpu"

    import config as cfg

    regression_mode = args.mode == "regression"
    log.info("Mode: %s", args.mode)

    log.info("Building datasets from %s...", args.data_dir)
    try:
        if regression_mode:
            from data.dataset import build_regression_datasets
            train_loader, val_loader, *_ = build_regression_datasets(
                data_dir=args.data_dir,
                config_module=cfg,
            )
        else:
            from data.dataset import build_datasets
            train_loader, val_loader, *_ = build_datasets(
                data_dir=args.data_dir,
                config_module=cfg,
            )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("Run: python data/bootstrap_fetcher.py --output-dir %s", args.data_dir)
        sys.exit(1)

    _class_weights = None
    if not regression_mode:
        from model.trainer import compute_class_weights
        _class_weights = compute_class_weights(train_loader, n_classes=3)
        log.info(
            "Class weights: DOWN=%.3f  FLAT=%.3f  UP=%.3f",
            _class_weights[0].item(), _class_weights[1].item(), _class_weights[2].item(),
        )

    log.info(
        "Starting Optuna search: trials=%d  epochs_per_trial=%d  sampler=%s  device=%s  mode=%s",
        args.trials, args.epochs, args.sampler, args.device, args.mode,
    )

    # Build sampler
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=42)
    else:
        sampler = optuna.samplers.RandomSampler(seed=42)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,   # don't prune the first 10 trials
        n_warmup_steps=5,      # don't prune before epoch 5 within a trial
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,    # resume if storage is set and study exists
    )

    objective = _make_objective(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=args.device,
        trial_epochs=args.epochs,
        class_weights=_class_weights,
        regression_mode=regression_mode,
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best_trial = study.best_trial
    best_params = _params_to_config_overrides(_suggest_params.__wrapped__
                                               if hasattr(_suggest_params, "__wrapped__")
                                               else best_trial.params)

    # best_trial.params already has the flat sampled dict — expand it
    best_params = _params_to_config_overrides(best_trial.params)

    print("\n" + "=" * 60)
    print("  OPTUNA SEARCH RESULTS")
    print("=" * 60)
    print(f"  Trials completed : {len(study.trials)}")
    pruned = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  Trials pruned    : {pruned}")
    print(f"  Best trial       : #{best_trial.number}")
    print(f"  Best val_loss    : {best_trial.value:.6f}")
    print()
    print("  Best hyperparameters:")
    for k, v in best_params.items():
        current = getattr(cfg, k, "N/A")
        changed = " ← changed" if v != current else ""
        print(f"    {k:<22} {v}{changed}")
    print("=" * 60)

    # ── Save best params ──────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "best_params.json"

    save_data = {
        "best_params": best_params,
        "best_val_loss": best_trial.value,
        "best_trial_number": best_trial.number,
        "n_trials": len(study.trials),
        "n_pruned": pruned,
        "search_space": {
            "HIDDEN_1": [32, 64, 128, 256],
            "HIDDEN_2": [16, 32, 64, 128],
            "DROPOUT_1": "float [0.1, 0.5]",
            "DROPOUT_2": "float [0.0, 0.4]",
            "LEARNING_RATE": "log-float [1e-4, 5e-3]",
            "WEIGHT_DECAY": "log-float [1e-5, 1e-3]",
            "UP_THRESHOLD": [0.005, 0.01, 0.015, 0.02],
        },
    }

    params_path.write_text(json.dumps(save_data, indent=2))
    log.info("Best params saved to %s", params_path)

    # ── Top 5 trials summary ──────────────────────────────────────────────────
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value)[:5]
    if top5:
        print("\n  Top 5 trials by val_loss:")
        print(f"  {'Trial':>6}  {'val_loss':>10}  {'HIDDEN_1':>8}  {'HIDDEN_2':>8}  "
              f"{'DROPOUT_1':>9}  {'LR':>10}  {'UP_THR':>8}")
        for t in top5:
            p = t.params
            print(
                f"  #{t.number:5d}  {t.value:10.6f}  {p.get('HIDDEN_1', '-'):>8}  "
                f"{p.get('HIDDEN_2', '-'):>8}  {p.get('DROPOUT_1', '-'):>9.1f}  "
                f"{p.get('LEARNING_RATE', 0):>10.2e}  {p.get('UP_THRESHOLD', '-'):>8}"
            )

    print(f"\nNext step: python train.py --mode {args.mode}  (will load {params_path})\n")


if __name__ == "__main__":
    main()
