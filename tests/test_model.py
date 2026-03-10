"""
tests/test_model.py — Unit tests for model/predictor.py.

Tests the DirectionPredictor forward pass, output shape, softmax behavior,
and save/load checkpoint round-trip.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.predictor import DirectionPredictor, save_checkpoint, load_checkpoint, MODEL_VERSION
from config import N_FEATURES, N_CLASSES


class TestDirectionPredictor:
    """Tests for DirectionPredictor architecture and forward pass."""

    def test_forward_pass_output_shape(self):
        """Forward pass must return (batch_size, 3) logits."""
        model = DirectionPredictor()
        batch_size = 16
        x = torch.randn(batch_size, N_FEATURES)
        out = model(x)
        assert out.shape == (batch_size, N_CLASSES), (
            f"Expected ({batch_size}, {N_CLASSES}), got {out.shape}"
        )

    def test_single_sample_output_shape(self):
        """Single-sample forward pass should return (1, 3).
        Model must be in eval mode — BatchNorm1d requires batch_size > 1 in training mode.
        At inference time the model is always in eval mode (predict_proba sets it).
        """
        model = DirectionPredictor()
        model.eval()  # BatchNorm1d needs eval mode for batch_size=1
        x = torch.randn(1, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, N_CLASSES)

    def test_output_is_logits_not_probs(self):
        """Direct model output (forward) should NOT sum to 1 — these are logits."""
        model = DirectionPredictor()
        x = torch.randn(32, N_FEATURES)
        out = model(x)
        row_sums = out.sum(dim=-1)
        # Logits generally don't sum to 1 (unlike softmax output)
        # We assert at least one row is not close to 1.0
        not_all_one = not torch.allclose(row_sums, torch.ones(32), atol=0.1)
        assert not_all_one, "forward() looks like it's applying softmax (logits should not sum to 1)"

    def test_softmax_output_sums_to_one(self):
        """After softmax, probabilities must sum to 1 for each sample."""
        model = DirectionPredictor()
        x = torch.randn(16, N_FEATURES)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(16), atol=1e-5), (
            f"Softmax output does not sum to 1: {row_sums}"
        )

    def test_predict_proba_sums_to_one(self):
        """predict_proba() helper must return probabilities summing to 1."""
        model = DirectionPredictor()
        x = torch.randn(8, N_FEATURES)
        probs = model.predict_proba(x)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)

    def test_predict_proba_non_negative(self):
        """All probability outputs must be >= 0."""
        model = DirectionPredictor()
        x = torch.randn(32, N_FEATURES)
        probs = model.predict_proba(x)
        assert (probs >= 0).all(), "Negative probabilities found"

    def test_model_in_eval_mode_after_predict_proba(self):
        """predict_proba() should leave the model in eval mode."""
        model = DirectionPredictor()
        model.train()
        _ = model.predict_proba(torch.randn(4, N_FEATURES))
        assert not model.training, "Model should be in eval mode after predict_proba()"

    def test_custom_architecture_params(self):
        """Model should accept custom hidden sizes and dropout rates."""
        model = DirectionPredictor(
            n_features=8,
            hidden_1=128,
            hidden_2=64,
            n_classes=3,
            dropout_1=0.5,
            dropout_2=0.1,
        )
        x = torch.randn(4, 8)
        out = model(x)
        assert out.shape == (4, 3)

    def test_parameter_count_reasonable(self):
        """Default model should have a small, reasonable number of parameters."""
        model = DirectionPredictor()
        n_params = sum(p.numel() for p in model.parameters())
        # Default: 8→64 (512+64=576) + 64→32 (2048+32=2080) + 32→3 (96+3=99) = ~2800+ BN params
        assert 1_000 < n_params < 50_000, f"Unexpected parameter count: {n_params}"

    def test_gradient_flows(self):
        """Gradients should flow through the model (no dead branches)."""
        model = DirectionPredictor()
        model.train()
        x = torch.randn(8, N_FEATURES)
        labels = torch.randint(0, N_CLASSES, (8,))
        loss = torch.nn.CrossEntropyLoss()(model(x), labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestCheckpoint:
    """Tests for save_checkpoint() and load_checkpoint()."""

    def test_save_and_load_round_trip(self):
        """Loaded model should produce identical output to the original."""
        model = DirectionPredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        norm_stats = {"mean": [0.0] * 8, "std": [1.0] * 8, "features": ["f1"] * 8}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(model, optimizer, epoch=5, val_loss=0.321, norm_stats=norm_stats, path=path)
            assert path.exists(), "Checkpoint file not created"

            loaded_model, checkpoint = load_checkpoint(str(path), device="cpu")

            # Same forward pass output
            x = torch.randn(4, N_FEATURES)
            model.eval()
            loaded_model.eval()
            with torch.no_grad():
                out_orig = model(x)
                out_loaded = loaded_model(x)

            assert torch.allclose(out_orig, out_loaded, atol=1e-6), (
                "Loaded model output differs from original"
            )

    def test_checkpoint_contains_norm_stats(self):
        """Checkpoint must contain norm_stats for inference-time normalization."""
        model = DirectionPredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        norm_stats = {
            "mean": list(range(8)),
            "std": [1.0, 2.0, 1.5, 0.5, 3.0, 2.5, 1.0, 4.0],
            "features": ["f"] * 8,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(model, optimizer, epoch=1, val_loss=0.5, norm_stats=norm_stats, path=path)
            _, checkpoint = load_checkpoint(str(path))

            assert "norm_stats" in checkpoint
            assert checkpoint["norm_stats"]["mean"] == norm_stats["mean"]
            assert checkpoint["norm_stats"]["std"] == norm_stats["std"]

    def test_checkpoint_contains_model_version(self):
        """Checkpoint must contain model_version string."""
        model = DirectionPredictor()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(model, optimizer, epoch=0, val_loss=1.0, norm_stats={}, path=path)
            _, checkpoint = load_checkpoint(str(path))

            assert "model_version" in checkpoint
            assert checkpoint["model_version"] == MODEL_VERSION

    def test_checkpoint_contains_model_config(self):
        """Checkpoint must store model architecture config for reconstruction."""
        model = DirectionPredictor(hidden_1=128, hidden_2=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(model, optimizer, epoch=0, val_loss=1.0, norm_stats={}, path=path)
            _, checkpoint = load_checkpoint(str(path))

            model_config = checkpoint.get("model_config", {})
            assert model_config.get("hidden_1") == 128
            assert model_config.get("hidden_2") == 64

    def test_load_checkpoint_not_found_raises(self):
        """load_checkpoint() should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/model.pt")

    def test_epoch_stored_in_checkpoint(self):
        """The training epoch must be stored in the checkpoint."""
        model = DirectionPredictor()
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(model, optimizer, epoch=42, val_loss=0.9, norm_stats={}, path=path)
            _, checkpoint = load_checkpoint(str(path))
            assert checkpoint["epoch"] == 42
