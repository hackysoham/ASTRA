# Team Astra -- Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
models.py -- Five evolutionary autoencoder architectures (v1-v5) and
             companion loss functions, all built **from scratch** in PyTorch.

Architecture quick-reference
----------------------------
v1  Baseline 4-layer CAE           (ReLU, latent=128, MSE loss)
v2  Structural-Loss CAE            (same arch, MSE + SSIM)
v3  Capacity-Optimised CAE         (5 layers, latent=256, LeakyReLU, BN)
v4  Variational Autoencoder        (reparam trick, KL divergence)
v5  Multi-Scale Edge-Aware VAE/CAE (skip connections, Sobel edge, MS-SSIM)

Stride / padding maths (verified)
----------------------------------
4-layer encoder (v1, v2):
    227 -> 113 -> 56 -> 28 -> 14     (Conv2d k=4 s=2 p=1)
    flatten 256*14*14 = 50176

4-layer decoder (v1, v2):
    14 -> 28 -> 56 -> 113 -> 227     (ConvTranspose2d k=4 s=2 p=1)
    output_padding=1 on last two layers for the odd-dim recovery

5-layer encoder (v3, v4, v5):
    227 -> 113 -> 56 -> 28 -> 14 -> 7   (Conv2d k=4 s=2 p=1)
    flatten 512*7*7 = 25088

5-layer decoder (v3, v4, v5):
    7 -> 14 -> 28 -> 56 -> 113 -> 227   (ConvTranspose2d k=4 s=2 p=1)
    output_padding=1 on last two layers
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
#   LOSS  UTILITIES  (all from scratch, pure PyTorch)                      #
# ====================================================================== #


class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity Index loss (1 - SSIM).

    Computed over Gaussian-weighted local windows following Wang et al.
    (IEEE TIP, 2004).  Minimising this loss maximises SSIM.

    Args:
        window_size: Gaussian kernel side length (odd).
        sigma:       Gaussian standard deviation.
        channel:     Number of image channels (1 for greyscale).
    """

    def __init__(
        self, window_size: int = 11, sigma: float = 1.5, channel: int = 1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        # Constants for [0, 1] dynamic range
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2

        # Build 2-D Gaussian kernel
        kernel_1d = self._gaussian_1d(window_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
        self.register_buffer("window", window)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _gaussian_1d(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    # ------------------------------------------------------------------ #
    def _ssim_map(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-pixel SSIM map (not yet averaged)."""
        pad = self.window_size // 2
        mu_x = F.conv2d(x, self.window, padding=pad, groups=self.channel)
        mu_y = F.conv2d(y, self.window, padding=pad, groups=self.channel)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(x * x, self.window, padding=pad, groups=self.channel) - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y * y, self.window, padding=pad, groups=self.channel) - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, self.window, padding=pad, groups=self.channel) - mu_xy
        )

        num = (2.0 * mu_xy + self.C1) * (2.0 * sigma_xy + self.C2)
        den = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        return num / den

    # ------------------------------------------------------------------ #
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar ``1 - mean(SSIM)``."""
        return 1.0 - self._ssim_map(pred, target).mean()

    def ssim_value(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the mean SSIM value (higher is better)."""
        return self._ssim_map(pred, target).mean()


# ---------------------------------------------------------------------- #

class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM loss (1 - MS-SSIM).

    Computes SSIM contrast/structure components at ``n_scales`` progressively
    down-sampled resolutions, and the luminance component only at the
    coarsest scale, following Wang et al. (2003).

    Args:
        window_size: Gaussian window per scale.
        n_scales:    Number of resolution levels (max 5 for 227x227).
        sigma:       Gaussian sigma.
        channel:     Image channels.
    """

    # Default weights from the MS-SSIM paper (5 scales)
    _DEFAULT_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def __init__(
        self,
        window_size: int = 11,
        n_scales: int = 5,
        sigma: float = 1.5,
        channel: int = 1,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.channel = channel
        weights = self._DEFAULT_WEIGHTS[:n_scales]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]  # re-normalise
        self.register_buffer(
            "scale_weights", torch.tensor(weights, dtype=torch.float32),
        )

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        kernel_1d = SSIMLoss._gaussian_1d(window_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
        self.register_buffer("window", window)
        self.pad = window_size // 2

    # ------------------------------------------------------------------ #
    def _components(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (luminance * contrast_structure, contrast_structure)."""
        mu_x = F.conv2d(x, self.window, padding=self.pad, groups=self.channel)
        mu_y = F.conv2d(y, self.window, padding=self.pad, groups=self.channel)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(x * x, self.window, padding=self.pad, groups=self.channel)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y * y, self.window, padding=self.pad, groups=self.channel)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, self.window, padding=self.pad, groups=self.channel)
            - mu_xy
        )

        luminance = (2.0 * mu_xy + self.C1) / (mu_x_sq + mu_y_sq + self.C1)
        cs = (2.0 * sigma_xy + self.C2) / (sigma_x_sq + sigma_y_sq + self.C2)
        return luminance.mean(), cs.mean()

    # ------------------------------------------------------------------ #
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return ``1 - MS_SSIM``."""
        ms_components: list = []
        x, y = pred, target

        for s in range(self.n_scales):
            lum, cs = self._components(x, y)
            if s < self.n_scales - 1:
                # Only contrast-structure at intermediate scales
                ms_components.append(cs)
                # Downsample by 2 for next scale
                x = F.avg_pool2d(x, kernel_size=2)
                y = F.avg_pool2d(y, kernel_size=2)
            else:
                # At the coarsest scale, include luminance
                ms_components.append(lum * cs)

        # Weighted product
        ms_ssim_val = torch.ones(1, device=pred.device, dtype=pred.dtype)
        for comp, w in zip(ms_components, self.scale_weights):
            # Clamp for numerical safety
            ms_ssim_val = ms_ssim_val * torch.clamp(comp, min=1e-8) ** w

        return 1.0 - ms_ssim_val.squeeze()


# ---------------------------------------------------------------------- #

class SobelEdgeLoss(nn.Module):
    """Differentiable Sobel directional-gradient loss.

    Applies fixed Sobel-x and Sobel-y kernels to both the prediction and
    the target, then penalises the L1 difference of the resulting edge
    maps.  This forces the decoder to preserve sharp boundaries.

    Forward returns a scalar loss.
    """

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
        ).reshape(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
        ).reshape(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edges(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gx = F.conv2d(img, self.kx, padding=1)
        gy = F.conv2d(img, self.ky, padding=1)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean L1 error between Sobel-x/y maps of *pred* vs *target*."""
        px, py = self._edges(pred)
        tx, ty = self._edges(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


# ---------------------------------------------------------------------- #

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log(var) - mu^2 - var)."""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


# ====================================================================== #
#  v1 -- BASELINE 4-LAYER CONVOLUTIONAL AUTOENCODER                        #
# ====================================================================== #
#                                                                          #
#  Encoder spatial dims:                                                   #
#    227 -> 113 -> 56 -> 28 -> 14    (k=4, s=2, p=1 throughout)           #
#  Flatten: 256 * 14 * 14 = 50 176 -> Linear -> 128                       #
#                                                                          #
#  Decoder spatial dims:                                                   #
#    14 -> 28 -> 56 -> 113 -> 227    (ConvTranspose2d k=4, s=2, p=1)      #
#    output_padding=1 on the last two layers to recover odd dims           #
# ====================================================================== #

class AEv1(nn.Module):
    """v1 -- Baseline 4-layer Convolutional Autoencoder.

    * Activation : ReLU
    * Norm       : none
    * Latent dim : 128
    * Loss       : MSE
    """

    FLATTEN_DIM = 256 * 14 * 14  # 50 176

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.is_vae = False

        # ---------- encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, 4, 2, 1),  nn.ReLU(True),   # 227->113
            nn.Conv2d(32, 64, 4, 2, 1),  nn.ReLU(True),   # 113->56
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),   # 56->28
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),  # 28->14
        )
        self.fc_enc = nn.Linear(self.FLATTEN_DIM, latent_dim)

        # ---------- decoder ----------
        self.fc_dec = nn.Linear(latent_dim, self.FLATTEN_DIM)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),                nn.ReLU(True),  # 14->28
            nn.ConvTranspose2d(128,  64, 4, 2, 1),                nn.ReLU(True),  # 28->56
            nn.ConvTranspose2d( 64,  32, 4, 2, 1, output_padding=1), nn.ReLU(True),  # 56->113
            nn.ConvTranspose2d( 32,   1, 4, 2, 1, output_padding=1), nn.Sigmoid(),   # 113->227
        )

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                       # (B, 256, 14, 14)
        return self.fc_enc(h.view(h.size(0), -1))  # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 256, 14, 14)
        return self.decoder(h)                     # (B, 1, 227, 227)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ====================================================================== #
#  v2 -- STRUCTURAL-LOSS AUTOENCODER  (same arch, MSE + SSIM)             #
# ====================================================================== #

class AEv2(nn.Module):
    """v2 -- Structural-Loss 4-layer CAE.

    Architecture identical to v1.  The difference is at loss-computation
    time: the training loop should use ``SSIMLoss`` alongside MSE.

    * Activation : ReLU
    * Norm       : none
    * Latent dim : 128
    * Loss       : MSE + SSIM (implemented from scratch above)
    """

    FLATTEN_DIM = 256 * 14 * 14

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.is_vae = False

        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, 4, 2, 1),  nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),
        )
        self.fc_enc = nn.Linear(self.FLATTEN_DIM, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.FLATTEN_DIM)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),                nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1),                nn.ReLU(True),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d( 32,   1, 4, 2, 1, output_padding=1), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.fc_enc(h.view(h.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 256, 14, 14)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ====================================================================== #
#  v3 -- CAPACITY-OPTIMISED AUTOENCODER                                    #
#        (5 layers, latent=256, LeakyReLU, BatchNorm)                      #
# ====================================================================== #
#                                                                          #
#  Encoder spatial dims:                                                   #
#    227 -> 113 -> 56 -> 28 -> 14 -> 7    (k=4, s=2, p=1)                 #
#  Flatten: 512 * 7 * 7 = 25 088 -> Linear -> 256                         #
#                                                                          #
#  Decoder spatial dims:                                                   #
#    7 -> 14 -> 28 -> 56 -> 113 -> 227    (ConvTranspose2d)               #
#    output_padding=1 on the last two layers                               #
# ====================================================================== #

class AEv3(nn.Module):
    """v3 -- Capacity-Optimised 5-layer CAE.

    * Activation : LeakyReLU(0.2)
    * Norm       : BatchNorm2d after every conv/deconv (except final)
    * Latent dim : 256
    * Loss       : MSE + SSIM (external)
    """

    FLATTEN_DIM = 512 * 7 * 7  # 25 088

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.is_vae = False

        lrelu = dict(negative_slope=0.2, inplace=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(1,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.LeakyReLU(**lrelu),  # 227->113
            nn.Conv2d(32,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(**lrelu),  # 113->56
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(**lrelu),  # 56->28
            nn.Conv2d(128,256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(**lrelu),  # 28->14
            nn.Conv2d(256,512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(**lrelu),  # 14->7
        )
        self.fc_enc = nn.Linear(self.FLATTEN_DIM, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.FLATTEN_DIM)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(**lrelu),  # 7->14
            nn.ConvTranspose2d(256,128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(**lrelu),  # 14->28
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(**lrelu),  # 28->56
            nn.ConvTranspose2d( 64, 32, 4, 2, 1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(**lrelu),  # 56->113
            nn.ConvTranspose2d( 32,  1, 4, 2, 1, output_padding=1), nn.Sigmoid(),              # 113->227
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.fc_enc(h.view(h.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 512, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ====================================================================== #
#  v4 -- VARIATIONAL AUTOENCODER                                           #
#        (reparam trick, KL divergence)                                    #
# ====================================================================== #

class AEv4(nn.Module):
    """v4 -- Variational Autoencoder.

    Based on the v3 backbone with a stochastic latent space:
        fc_mu, fc_logvar  ->  reparameterize  ->  decoder

    * Activation : LeakyReLU(0.2)
    * Norm       : BatchNorm2d
    * Latent dim : 256
    * Loss       : MSE + SSIM + beta * KL
    """

    FLATTEN_DIM = 512 * 7 * 7

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.is_vae = True

        lrelu = dict(negative_slope=0.2, inplace=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(1,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.LeakyReLU(**lrelu),
            nn.Conv2d(32,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(**lrelu),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(**lrelu),
            nn.Conv2d(128,256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(**lrelu),
            nn.Conv2d(256,512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(**lrelu),
        )

        # Two parallel heads for the variational bottleneck
        self.fc_mu     = nn.Linear(self.FLATTEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(self.FLATTEN_DIM, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.FLATTEN_DIM)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(**lrelu),
            nn.ConvTranspose2d(256,128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(**lrelu),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(**lrelu),
            nn.ConvTranspose2d( 64, 32, 4, 2, 1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(**lrelu),
            nn.ConvTranspose2d( 32,  1, 4, 2, 1, output_padding=1), nn.Sigmoid(),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        """z = mu + std * eps    (eps ~ N(0,I) only during training)."""
        if training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z, mu, logvar)."""
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar, self.training)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 512, 7, 7)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_hat, mu, logvar)."""
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


# ====================================================================== #
#  v5 -- MULTI-SCALE EDGE-AWARE COMPOUND VAE/CAE                          #
#        (U-Net skip connections + Sobel + MS-SSIM)                        #
# ====================================================================== #

class AEv5(nn.Module):
    """v5 -- Multi-Scale Edge-Aware Compound VAE / CAE.

    Builds on the v4 VAE backbone with two major additions:

    1. **U-Net skip connections** between encoder and decoder at four
       matching spatial resolutions (113, 56, 28, 14).  This gives the
       decoder a direct pathway for high-frequency detail, which is
       otherwise lost through the stochastic bottleneck.

    2. **Edge-aware compound loss** -- the recommended training loss
       combines MSE + MS-SSIM + beta*KL + Sobel-edge L1.  The companion
       loss classes ``MSSSIMLoss`` and ``SobelEdgeLoss`` are defined
       above.

    * Activation  : LeakyReLU(0.2)
    * Norm        : BatchNorm2d
    * Latent dim  : 256
    * Init        : Kaiming (tuned for LeakyReLU a=0.2)
    * Loss        : MSE + MS-SSIM + beta*KL + Sobel-edge
    """

    FLATTEN_DIM = 512 * 7 * 7

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.is_vae = True

        lrelu = dict(negative_slope=0.2, inplace=True)

        # ---- Encoder (individual blocks for skip tapping) ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(1,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.LeakyReLU(**lrelu),  # 227->113
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(**lrelu),  # 113->56
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(**lrelu),  # 56->28
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128,256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(**lrelu),  # 28->14
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256,512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(**lrelu),  # 14->7
        )

        # VAE heads
        self.fc_mu     = nn.Linear(self.FLATTEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(self.FLATTEN_DIM, latent_dim)

        # ---- Decoder (channels doubled at cat points) ----
        self.fc_dec = nn.Linear(latent_dim, self.FLATTEN_DIM)

        # dec5 output: 256 ch @ 14x14 -> cat with enc4 (256) => 512 ch
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(**lrelu),
        )
        # dec4 input: 512 ch -> output 128 @ 28x28 -> cat with enc3 (128) => 256
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(**lrelu),
        )
        # dec3 input: 256 ch -> output 64 @ 56x56 -> cat with enc2 (64) => 128
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(**lrelu),
        )
        # dec2 input: 128 ch -> output 32 @ 113x113 -> cat with enc1 (32) => 64
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1, output_padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(**lrelu),
        )
        # dec1 input: 64 ch -> output 1 @ 227x227
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

        # Kaiming init for LeakyReLU
        self._kaiming_init()

    # ------------------------------------------------------------------ #
    def _kaiming_init(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def encode(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Return (z, mu, logvar, [e1, e2, e3, e4])."""
        e1 = self.enc1(x)    # (B,  32, 113, 113)
        e2 = self.enc2(e1)   # (B,  64,  56,  56)
        e3 = self.enc3(e2)   # (B, 128,  28,  28)
        e4 = self.enc4(e3)   # (B, 256,  14,  14)
        e5 = self.enc5(e4)   # (B, 512,   7,   7)

        flat = e5.view(x.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = AEv4.reparameterize(mu, logvar, self.training)
        return z, mu, logvar, [e1, e2, e3, e4]

    # ------------------------------------------------------------------ #
    def decode(self, z: torch.Tensor, skips: list) -> torch.Tensor:
        """Decode with skip connections [e1, e2, e3, e4]."""
        h = self.fc_dec(z).view(-1, 512, 7, 7)

        d5 = self.dec5(h)                             # (B, 256, 14, 14)
        d5 = torch.cat([d5, skips[3]], dim=1)         # (B, 512, 14, 14)

        d4 = self.dec4(d5)                             # (B, 128, 28, 28)
        d4 = torch.cat([d4, skips[2]], dim=1)         # (B, 256, 28, 28)

        d3 = self.dec3(d4)                             # (B,  64, 56, 56)
        d3 = torch.cat([d3, skips[1]], dim=1)         # (B, 128, 56, 56)

        d2 = self.dec2(d3)                             # (B,  32, 113, 113)
        d2 = torch.cat([d2, skips[0]], dim=1)         # (B,  64, 113, 113)

        return self.dec1(d2)                           # (B,   1, 227, 227)

    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_hat, mu, logvar)."""
        z, mu, logvar, skips = self.encode(x)
        x_hat = self.decode(z, skips)
        return x_hat, mu, logvar


# ====================================================================== #
#   MODEL  REGISTRY  &  FACTORY                                            #
# ====================================================================== #

MODEL_REGISTRY: Dict[str, type] = {
    "v1": AEv1,
    "v2": AEv2,
    "v3": AEv3,
    "v4": AEv4,
    "v5": AEv5,
}


def get_model(version: str, **kwargs) -> nn.Module:
    """Instantiate a model by version tag.

    Args:
        version: One of ``'v1'`` .. ``'v5'``.
        **kwargs: Forwarded to the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If *version* is not in the registry.
    """
    if version not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown version '{version}'. Choose from {list(MODEL_REGISTRY.keys())}"
        )
    model = MODEL_REGISTRY[version](**kwargs)
    n = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {version.upper()} | {model.__class__.__name__} | "
        f"latent={model.latent_dim} | params={n:,} | VAE={model.is_vae}"
    )
    return model
