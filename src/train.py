# Team Astra — Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
train.py — Training loop utilities with early stopping, LR scheduling,
           checkpoint saving, and β-annealing support for VAE models.
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CombinedLoss
from .utils import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """Run one training epoch.
    
    Args:
        model: The autoencoder model.
        loader: Training data loader.
        criterion: Combined loss function.
        optimizer: Optimizer instance.
        device: Training device.
        grad_clip: Maximum gradient norm for clipping (0 to disable).
    
    Returns:
        Dictionary with average loss components over the epoch.
    """
    model.train()
    running = {"total": 0.0, "mse": 0.0, "ssim": 0.0, "kl": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        images = batch[0].to(device)
        optimizer.zero_grad()

        # Forward pass — handle VAE and non-VAE models
        if model.is_vae:
            x_hat, mu, logvar = model(images)
            loss, loss_dict = criterion(x_hat, images, mu, logvar)
        else:
            x_hat = model(images)
            loss, loss_dict = criterion(x_hat, images)

        # Backward pass
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Accumulate
        for k in running:
            running[k] += loss_dict[k]
        n_batches += 1

    # Average
    return {k: v / n_batches for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    """Run validation pass.
    
    Args:
        model: The autoencoder model.
        loader: Validation data loader.
        criterion: Combined loss function.
        device: Validation device.
    
    Returns:
        Dictionary with average loss components over the validation set.
    """
    model.eval()
    running = {"total": 0.0, "mse": 0.0, "ssim": 0.0, "kl": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        images = batch[0].to(device)

        if model.is_vae:
            x_hat, mu, logvar = model(images)
            loss, loss_dict = criterion(x_hat, images, mu, logvar)
        else:
            x_hat = model(images)
            loss, loss_dict = criterion(x_hat, images)

        for k in running:
            running[k] += loss_dict[k]
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    version: str = "v1",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    patience: int = 10,
    grad_clip: float = 1.0,
    beta_annealing: bool = False,
    beta_max: float = 0.001,
    save_dir: str = "outputs/models",
) -> dict:
    """Complete training pipeline with early stopping and optional β-annealing.
    
    Args:
        model: Autoencoder model (any version).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Training device.
        version: Model version string (for checkpoint naming).
        epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        weight_decay: L2 regularization weight.
        alpha: MSE loss weight.
        beta: SSIM loss weight.
        gamma: KL divergence weight (0 for non-VAE).
        patience: Early stopping patience (epochs without improvement).
        grad_clip: Max gradient norm (0 to disable).
        beta_annealing: If True, linearly anneal gamma from 0 to beta_max.
        beta_max: Maximum gamma value when using β-annealing.
        save_dir: Directory for saving checkpoints.
    
    Returns:
        Dictionary containing training history (per-epoch losses).
    """
    model = model.to(device)

    # Set up loss, optimizer, scheduler
    criterion = CombinedLoss(alpha=alpha, beta=beta, gamma=gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # History tracking
    history = {
        "train_total": [], "train_mse": [], "train_ssim": [], "train_kl": [],
        "val_total": [], "val_mse": [], "val_ssim": [], "val_kl": [],
    }

    # Early stopping state
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_path = str(Path(save_dir) / f"best_{version}.pth")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Training {version.upper()} | Epochs: {epochs} | LR: {lr}")
    print(f" Loss weights — α(MSE): {alpha} | β(SSIM): {beta} | γ(KL): {gamma}")
    print(f" β-annealing: {beta_annealing} (max={beta_max})")
    print(f" Device: {device} | Patience: {patience}")
    print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # β-annealing: linearly increase gamma over epochs
        if beta_annealing and model.is_vae:
            current_gamma = beta_max * min(1.0, epoch / (epochs * 0.5))
            criterion.gamma = current_gamma

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Record history
        for k in ["total", "mse", "ssim", "kl"]:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])

        # LR scheduling
        scheduler.step(val_metrics["total"])

        # Logging
        gamma_str = f" | γ={criterion.gamma:.6f}" if beta_annealing else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} │ "
            f"Train: {train_metrics['total']:.6f} │ "
            f"Val: {val_metrics['total']:.6f} │ "
            f"MSE: {val_metrics['mse']:.6f} │ "
            f"SSIM: {val_metrics['ssim']:.6f}"
            f"{gamma_str}"
        )

        # Early stopping check
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics["total"],
                best_path,
                extra={"history": history, "version": version},
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  ⚡ Early stopping at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - start_time
    print(f"\n  ✓ Training complete in {elapsed/60:.1f} min")
    print(f"  ✓ Best val loss: {best_val_loss:.6f} → {best_path}\n")

    # Save final checkpoint
    final_path = str(Path(save_dir) / f"final_{version}.pth")
    save_checkpoint(
        model, optimizer, epoch, val_metrics["total"],
        final_path,
        extra={"history": history, "version": version},
    )

    return history
