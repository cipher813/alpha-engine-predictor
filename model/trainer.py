"""
model/trainer.py — Training loop with early stopping and LR scheduling.

Uses AdamW optimizer and ReduceLROnPlateau scheduler. Training halts early
if validation loss does not improve for `patience` consecutive epochs.
The best checkpoint (lowest val_loss) is saved to checkpoints/best.pt.

Supports two modes:
  Classification (model.n_classes == 3): CrossEntropyLoss + accuracy tracking.
  Regression     (model.n_classes == 1): Direct IC loss (negative batch Pearson
    correlation) + MAE tracking.

    IC loss = −mean( z_score(pred) * z_score(actual) )

    This directly optimises what we evaluate (Pearson IC), unlike MSE/Huber
    which optimise absolute prediction magnitude — irrelevant for ranking.
    A model predicting [1x, 2x, 3x] and [100x, 200x, 300x] have identical IC
    but very different Huber loss; IC loss treats them identically (correct).

Usage (from train.py or directly):
    from model.trainer import train
    results = train(model, train_loader, val_loader, config, device='cpu')
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class ICLoss(nn.Module):
    """
    Negative batch Pearson correlation between predictions and actual returns.

    This is the standard quant objective for alpha signal optimisation —
    it directly maximises the Information Coefficient (IC) used in evaluation.

    Unlike MSE/Huber, IC loss is scale-invariant: a model that ranks stocks
    correctly gets the same gradient regardless of prediction magnitude.
    This matches how alpha signals are actually used in production (cross-
    sectional ranking / z-scoring before portfolio construction).

    Loss = −mean( z(pred) · z(actual) )  where z(x) = (x − μ) / (σ + ε)

    Returns values in [−1, +1]; a perfect ranker scores −1 (minimised to +1 IC).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] < 2:
            # Degenerate batch — fall back to zero loss (no gradient)
            return pred.sum() * 0.0
        pred_z   = (pred   - pred.mean())   / (pred.std()   + self.eps)
        actual_z = (actual - actual.mean()) / (actual.std() + self.eps)
        return -(pred_z * actual_z).mean()  # negative IC (minimise → maximise IC)


def compute_class_weights(train_loader: DataLoader, n_classes: int = 3) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training DataLoader.

    Returns a float32 tensor of shape (n_classes,) where each weight equals
    total_samples / (n_classes * class_count).  Weights are normalised so they
    sum to n_classes, keeping the effective learning rate roughly unchanged.
    """
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for _, y_batch in train_loader:
        for cls in range(n_classes):
            counts[cls] += (y_batch == cls).sum().float()
    total = counts.sum()
    weights = total / (n_classes * counts.clamp(min=1.0))
    return weights


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints",
    norm_stats: dict | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Full training loop with early stopping.

    Automatically detects regression vs classification from model.n_classes:
      n_classes == 1  → regression mode: HuberLoss(delta=0.01), tracks MAE.
      n_classes >= 2  → classification mode: CrossEntropyLoss with optional
                        class weights, tracks accuracy.

    Parameters
    ----------
    model :          DirectionPredictor (or any nn.Module with compatible output).
    train_loader :   Training DataLoader.
    val_loader :     Validation DataLoader.
    config :         Config module with LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS,
                     EARLY_STOPPING_PATIENCE attributes.
    device :         Torch device string.
    checkpoint_dir : Directory where best.pt will be saved.
    norm_stats :     Normalization statistics dict to embed in checkpoint.
    class_weights :  Optional float32 tensor of shape (n_classes,) for
                     CrossEntropyLoss (classification mode only).

    Returns
    -------
    dict with keys:
        best_epoch    (int)   : Epoch index of best validation loss (0-indexed).
        best_val_loss (float) : Best validation loss achieved.
        history       (list)  : Per-epoch dicts with train_loss, val_loss,
                                and val_accuracy (classification) or val_mae (regression).
        total_time_s  (float) : Wall-clock training time in seconds.
    """
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False
        log.warning("tqdm not installed — progress bars disabled")

    model = model.to(device)

    # Detect mode from model output dimension
    regression_mode: bool = getattr(model, "n_classes", 3) == 1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",       # minimize val_loss
        factor=0.5,       # halve LR on plateau
        patience=3,       # 3 epochs without improvement triggers LR reduction
        min_lr=1e-6,
    )

    if regression_mode:
        # IC loss: directly optimises Pearson IC (scale-invariant ranking objective)
        criterion = ICLoss()
        log.info("Regression mode: ICLoss (direct IC optimisation)")
    elif class_weights is not None:
        w = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        log.info(
            "Class weights: DOWN=%.3f  FLAT=%.3f  UP=%.3f",
            w[0].item(), w[1].item(), w[2].item(),
        )
    else:
        criterion = nn.CrossEntropyLoss()
        log.warning("No class weights supplied — class collapse risk if labels are imbalanced")

    checkpoint_path = Path(checkpoint_dir) / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    t_start = time.time()

    log.info(
        "Training: max_epochs=%d  patience=%d  lr=%.1e  device=%s",
        config.MAX_EPOCHS,
        config.EARLY_STOPPING_PATIENCE,
        config.LEARNING_RATE,
        device,
    )

    for epoch in range(config.MAX_EPOCHS):
        # Notify date-grouped sampler of new epoch for shuffling
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        if _has_tqdm:
            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1:3d}/{config.MAX_EPOCHS} [train]",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            train_iter = train_loader

        for X_batch, y_batch in train_iter:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)

            if regression_mode:
                # output shape: (batch, 1) → squeeze to (batch,) for loss
                loss = criterion(output.squeeze(-1), y_batch)
            else:
                loss = criterion(output, y_batch)

            loss.backward()

            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss_sum / max(n_train_batches, 1)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        # Classification trackers
        n_correct = 0
        n_total = 0
        # Regression trackers: collect all preds+targets for full-set IC
        abs_error_sum = 0.0
        all_val_preds: list[torch.Tensor] = []
        all_val_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(X_batch)

                if regression_mode:
                    pred = output.squeeze(-1)
                    abs_error_sum += (pred - y_batch).abs().sum().item()
                    n_total += y_batch.size(0)
                    all_val_preds.append(pred.cpu())
                    all_val_targets.append(y_batch.cpu())
                else:
                    loss = criterion(output, y_batch)
                    preds = output.argmax(dim=-1)
                    n_correct += (preds == y_batch).sum().item()
                    n_total += y_batch.size(0)
                    val_loss_sum += loss.item()
                    n_val_batches += 1

        if regression_mode:
            # Compute IC on the full val set — stable cross-sectional Pearson
            all_preds_t = torch.cat(all_val_preds)
            all_targets_t = torch.cat(all_val_targets)
            avg_val_loss = criterion(all_preds_t, all_targets_t).item()
        else:
            avg_val_loss = val_loss_sum / max(n_val_batches, 1)

        if regression_mode:
            val_mae = abs_error_sum / max(n_total, 1)
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            log.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  val_mae=%.5f  lr=%.1e",
                epoch + 1, config.MAX_EPOCHS, avg_train_loss, avg_val_loss, val_mae, current_lr,
            )
            history.append({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 8),
                "val_loss": round(avg_val_loss, 8),
                "val_mae": round(val_mae, 6),
                "lr": current_lr,
            })
        else:
            val_accuracy = n_correct / max(n_total, 1)
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            log.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  lr=%.1e",
                epoch + 1, config.MAX_EPOCHS, avg_train_loss, avg_val_loss, val_accuracy, current_lr,
            )
            history.append({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "val_accuracy": round(val_accuracy, 4),
                "lr": current_lr,
            })

        # ── Checkpoint if best ────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            from model.predictor import save_checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=avg_val_loss,
                norm_stats=norm_stats or {},
                path=checkpoint_path,
            )
            log.info("  ✓ New best val_loss=%.4f — checkpoint saved", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    config.EARLY_STOPPING_PATIENCE,
                )
                break

    total_time = time.time() - t_start
    log.info(
        "Training complete: best_epoch=%d  best_val_loss=%.4f  time=%.1fs",
        best_epoch + 1,
        best_val_loss,
        total_time,
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "history": history,
        "total_time_s": round(total_time, 1),
        "checkpoint_path": str(checkpoint_path),
    }
