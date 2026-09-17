# Team Astra — Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
utils.py — General helper functions for reproducibility, device management,
           checkpoint I/O, and plotting utilities.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    extra: dict = None,
) -> None:
    """Save a training checkpoint to disk.
    
    Args:
        model: The model whose state_dict to save.
        optimizer: The optimizer whose state_dict to save.
        epoch: Current epoch number.
        loss: Current loss value.
        path: File path for the checkpoint (.pth).
        extra: Optional dictionary of additional data to save.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved → {path} (epoch={epoch}, loss={loss:.6f})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
) -> dict:
    """Load a training checkpoint from disk.
    
    Args:
        path: File path to the checkpoint (.pth).
        model: The model to load state_dict into.
        optimizer: Optional optimizer to load state_dict into.
        device: Device to map the checkpoint to.
    
    Returns:
        The full checkpoint dictionary.
    """
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded ← {path} (epoch={checkpoint.get('epoch', '?')})")
    return checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Plotting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: list,
    val_losses: list = None,
    title: str = "Training Loss",
    save_path: str = None,
    components: dict = None,
) -> None:
    """Plot training (and optionally validation) loss curves.
    
    Args:
        train_losses: List of training loss values per epoch.
        val_losses: Optional list of validation loss values per epoch.
        title: Plot title.
        save_path: Optional path to save the figure.
        components: Optional dict of {name: [values]} for loss component curves.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss")
    if val_losses is not None:
        ax.plot(epochs, val_losses, "r--", linewidth=2, label="Val Loss")

    if components is not None:
        colors = ["#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]
        for i, (name, values) in enumerate(components.items()):
            ax.plot(
                range(1, len(values) + 1),
                values,
                linestyle=":",
                linewidth=1.5,
                color=colors[i % len(colors)],
                label=name,
            )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()
    plt.close(fig)


def plot_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    n: int = 8,
    title: str = "Reconstructions",
    save_path: str = None,
) -> None:
    """Plot original vs reconstructed images side by side.
    
    Args:
        originals: Tensor of shape (N, 1, H, W).
        reconstructions: Tensor of shape (N, 1, H, W).
        n: Number of image pairs to display (capped at batch size).
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    n = min(n, originals.size(0))
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Original
        axes[0, i].imshow(
            originals[i].squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[0, i].set_title("Original", fontsize=8)
        axes[0, i].axis("off")

        # Reconstruction
        axes[1, i].imshow(
            reconstructions[i].squeeze().cpu().detach().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, i].set_title("Reconstructed", fontsize=8)
        axes[1, i].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()
    plt.close(fig)


def ensure_dirs() -> None:
    """Create all required output directories."""
    dirs = [
        "outputs/models",
        "outputs/latent_vectors",
        "outputs/plots",
        "outputs/scores",
        "outputs/report",
        "data/images",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("[Setup] Output directories created.")
