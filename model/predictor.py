"""
model/predictor.py — MLP predictor (classification or regression).

Architecture:
    Input(N) → Linear(H1) → BatchNorm1d → ReLU → Dropout(d1)
             → Linear(H2) → BatchNorm1d → ReLU → Dropout(d2)
             → Linear(n_classes)

Classification mode (n_classes=3):
    Output: raw logits for [DOWN, FLAT, UP].  No softmax in forward() —
    use torch.nn.CrossEntropyLoss (includes log-softmax).
    At inference apply F.softmax(logits, dim=-1) to get probabilities.

Regression mode (n_classes=1):
    Output: single scalar (predicted 5-day return relative to SPY).
    Training loss: HuberLoss(delta=0.01).  IC = corr(pred, actual_return).
    At inference the scalar output is directly the return prediction.
    The signal used downstream is pred.squeeze(-1) directly.

Model version string is embedded in checkpoints so inference can log it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bump this when architecture changes — stored in every checkpoint.
MODEL_VERSION = "v1.2.0"  # v1.2: 17 features; regression mode (n_classes=1) supported


class DirectionPredictor(nn.Module):
    """
    Multi-layer perceptron for 3-class direction prediction.

    Parameters
    ----------
    n_features : int
        Number of input features. Default 8 (matches config.N_FEATURES).
    hidden_1 : int
        Width of the first hidden layer. Default 64 (config.HIDDEN_1).
    hidden_2 : int
        Width of the second hidden layer. Default 32 (config.HIDDEN_2).
    n_classes : int
        Number of output classes. Default 3 (DOWN / FLAT / UP).
    dropout_1 : float
        Dropout probability after first hidden layer. Default 0.3.
    dropout_2 : float
        Dropout probability after second hidden layer. Default 0.2.
    """

    def __init__(
        self,
        n_features: int = 8,
        hidden_1: int = 64,
        hidden_2: int = 32,
        n_classes: int = 3,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────────────
            nn.Linear(n_features, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_1),
            # ── Block 2 ──────────────────────────────────────────────────────
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_2),
            # ── Output ───────────────────────────────────────────────────────
            nn.Linear(hidden_2, n_classes),
            # No softmax — CrossEntropyLoss includes log-softmax internally.
        )

        # Store constructor args in the model so checkpoints are self-describing
        self.n_features = n_features
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.n_classes = n_classes
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, n_features)

        Returns
        -------
        torch.Tensor, shape (batch_size, n_classes)
            Raw logits. Apply softmax to get probabilities.
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference and return softmax probabilities.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, n_features) — already normalized.

        Returns
        -------
        torch.Tensor, shape (batch_size, n_classes)
            Class probabilities summing to 1.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def save_checkpoint(
    model: DirectionPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    norm_stats: dict,
    path: str | Path,
) -> None:
    """
    Save a full training checkpoint to disk.

    The checkpoint stores everything needed to resume training or run
    inference: model weights, optimizer state, epoch, validation loss,
    normalization statistics, and the model version string.

    Parameters
    ----------
    model :       Trained DirectionPredictor instance.
    optimizer :   Optimizer at the time of saving.
    epoch :       Current training epoch (0-indexed).
    val_loss :    Validation loss at this checkpoint.
    norm_stats :  Dict with 'mean' and 'std' lists (from build_datasets).
    path :        File path to write (e.g., 'checkpoints/best.pt').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "norm_stats": norm_stats,       # {"mean": [...], "std": [...], "features": [...]}
        "model_version": MODEL_VERSION,
        "model_config": {
            "n_features": model.n_features,
            "hidden_1": model.hidden_1,
            "hidden_2": model.hidden_2,
            "n_classes": model.n_classes,
            "dropout_1": model.dropout_1,
            "dropout_2": model.dropout_2,
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[DirectionPredictor, dict]:
    """
    Load a model checkpoint from disk and reconstruct the model.

    Parameters
    ----------
    path :   Path to the .pt checkpoint file.
    device : Torch device string ('cpu' or 'cuda').

    Returns
    -------
    (model, checkpoint_dict)
        model             — DirectionPredictor loaded with saved weights.
        checkpoint_dict   — Full checkpoint dict (norm_stats, epoch, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct model from saved config (handles architecture changes)
    model_config = checkpoint.get("model_config", {})
    model = DirectionPredictor(
        n_features=model_config.get("n_features", 8),
        hidden_1=model_config.get("hidden_1", 64),
        hidden_2=model_config.get("hidden_2", 32),
        n_classes=model_config.get("n_classes", 3),
        dropout_1=model_config.get("dropout_1", 0.3),
        dropout_2=model_config.get("dropout_2", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint
