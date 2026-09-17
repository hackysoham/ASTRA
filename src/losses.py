# Team Astra — Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
losses.py — Loss functions for autoencoder training.

Implements:
    - MSE (Mean Squared Error) for pixel-level fidelity
    - SSIM (Structural Similarity Index) for perceptual quality
    - KL Divergence for VAE latent regularization
    - CombinedLoss: weighted aggregation of all components

Reference:
    SSIM: Wang et al., "Image Quality Assessment: From Error Visibility
    to Structural Similarity," IEEE TIP, 2004.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity Index (SSIM) loss.
    
    Computes SSIM between two image tensors using Gaussian-weighted windows.
    Returns (1 - SSIM) so that minimizing this loss maximizes SSIM.
    
    Args:
        window_size: Size of the Gaussian kernel window (must be odd).
        sigma: Standard deviation for the Gaussian kernel.
        channels: Number of image channels (1 for grayscale).
    """

    def __init__(
        self, window_size: int = 11, sigma: float = 1.5, channels: int = 1
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels

        # Precompute Gaussian kernel
        kernel_1d = self._gaussian_kernel_1d(window_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)  # outer product
        kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer("window", kernel_2d)

        # SSIM constants (for [0, 1] range)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
        """Create a 1D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM loss = 1 - SSIM(prediction, target).
        
        Args:
            prediction: Reconstructed images, shape (B, C, H, W).
            target: Original images, shape (B, C, H, W).
        
        Returns:
            Scalar tensor: mean (1 - SSIM) over the batch.
        """
        pad = self.window_size // 2

        # Compute local means
        mu_x = F.conv2d(prediction, self.window, padding=pad, groups=self.channels)
        mu_y = F.conv2d(target, self.window, padding=pad, groups=self.channels)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        # Compute local variances and covariance
        sigma_x_sq = (
            F.conv2d(prediction ** 2, self.window, padding=pad, groups=self.channels)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(target ** 2, self.window, padding=pad, groups=self.channels)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(prediction * target, self.window, padding=pad, groups=self.channels)
            - mu_xy
        )

        # SSIM formula
        numerator = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        ssim_map = numerator / denominator

        return 1.0 - ssim_map.mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between N(mu, sigma²) and N(0, 1).
    
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    
    Args:
        mu: Mean of the approximate posterior, shape (B, latent_dim).
        logvar: Log-variance of the approximate posterior, shape (B, latent_dim).
    
    Returns:
        Scalar tensor: mean KL divergence over the batch.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class CombinedLoss(nn.Module):
    """Combined reconstruction + regularization loss.
    
    L_total = alpha * MSE + beta * (1 - SSIM) + gamma * KL
    
    For standard autoencoders, set gamma=0.
    For VAEs, gamma > 0 activates KL regularization.
    
    Args:
        alpha: Weight for MSE loss.
        beta: Weight for SSIM loss.
        gamma: Weight for KL divergence loss (0 for non-VAE).
        window_size: SSIM kernel window size.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        window_size: int = 11,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(window_size=window_size)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor = None,
        logvar: torch.Tensor = None,
    ) -> tuple:
        """Compute the combined loss.
        
        Args:
            prediction: Reconstructed images, shape (B, 1, 227, 227).
            target: Original images, shape (B, 1, 227, 227).
            mu: VAE mean (None for standard AE).
            logvar: VAE log-variance (None for standard AE).
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual component values for logging.
        """
        loss_mse = self.mse(prediction, target)
        loss_ssim = self.ssim(prediction, target)

        total = self.alpha * loss_mse + self.beta * loss_ssim

        loss_dict = {
            "mse": loss_mse.item(),
            "ssim": loss_ssim.item(),
            "kl": 0.0,
            "total": 0.0,
        }

        if self.gamma > 0 and mu is not None and logvar is not None:
            loss_kl = kl_divergence(mu, logvar)
            total = total + self.gamma * loss_kl
            loss_dict["kl"] = loss_kl.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
