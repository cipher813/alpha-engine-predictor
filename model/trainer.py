"""
model/trainer.py — Training loop with early stopping and LR scheduling.

Uses AdamW optimizer and ReduceLROnPlateau scheduler. Training halts early
if validation loss does not improve for `patience` consecutive epochs.
The best checkpoint (lowest val_loss) is saved to checkpoints/best.pt.

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


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints",
    norm_stats: dict | None = None,
) -> dict[str, Any]:
    """
    Full training loop with early stopping.

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

    Returns
    -------
    dict with keys:
        best_epoch    (int)   : Epoch index of best validation loss (0-indexed).
        best_val_loss (float) : Best validation loss achieved.
        history       (list)  : Per-epoch dicts with train_loss, val_loss, val_accuracy.
        total_time_s  (float) : Wall-clock training time in seconds.
    """
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False
        log.warning("tqdm not installed — progress bars disabled")

    model = model.to(device)

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

    criterion = nn.CrossEntropyLoss()

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
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
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
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item()
                n_val_batches += 1

                preds = logits.argmax(dim=-1)
                n_correct += (preds == y_batch).sum().item()
                n_total += y_batch.size(0)

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)
        val_accuracy = n_correct / max(n_total, 1)

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  lr=%.1e",
            epoch + 1,
            config.MAX_EPOCHS,
            avg_train_loss,
            avg_val_loss,
            val_accuracy,
            current_lr,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "val_accuracy": round(val_accuracy, 4),
                "lr": current_lr,
            }
        )

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
