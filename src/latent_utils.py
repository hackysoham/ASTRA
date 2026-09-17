# Team Astra — Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
latent_utils.py — Latent vector extraction and dimensionality reduction
                  visualization using t-SNE and UMAP.
"""

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def extract_latents(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """Extract latent vectors for the entire dataset.
    
    Args:
        model: Trained autoencoder model (any version).
        loader: DataLoader over the full dataset (no shuffle).
        device: Computation device.
    
    Returns:
        Tuple of (latents, filenames) where:
            latents: numpy array of shape (N, latent_dim)
            filenames: list of N filename strings
    """
    model.eval()
    all_latents = []
    all_filenames = []

    with torch.no_grad():
        for batch in loader:
            images, fnames = batch
            images = images.to(device)

            if model.is_vae:
                # For VAE models, use mu as the latent representation
                # (deterministic point estimate)
                if hasattr(model, 'enc1'):
                    # v4, v5 with skip connections
                    z, mu, logvar, _ = model.encode(images)
                    latent = mu  # Use mean for stable representation
                else:
                    # v3 without skip connections
                    z, mu, logvar = model.encode(images)
                    latent = mu
            else:
                # Standard AE
                latent = model.encode(images)

            all_latents.append(latent.cpu().numpy())
            all_filenames.extend(fnames)

    latents = np.concatenate(all_latents, axis=0)
    print(f"[Latent] Extracted {latents.shape[0]} vectors of dim {latents.shape[1]}")
    return latents, all_filenames


def save_latents(
    latents: np.ndarray,
    filenames: List[str],
    save_dir: str = "outputs/latent_vectors",
    version: str = "v1",
) -> None:
    """Save latent vectors and filenames to disk.
    
    Args:
        latents: Numpy array of shape (N, latent_dim).
        filenames: List of N filename strings.
        save_dir: Directory to save files.
        version: Model version string for naming.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(save_dir) / f"latents_{version}.npy", latents)
    np.save(Path(save_dir) / f"filenames_{version}.npy", np.array(filenames))
    print(f"[Latent] Saved → {save_dir}/latents_{version}.npy")


def load_latents(
    save_dir: str = "outputs/latent_vectors",
    version: str = "v1",
) -> Tuple[np.ndarray, List[str]]:
    """Load saved latent vectors and filenames from disk."""
    latents = np.load(Path(save_dir) / f"latents_{version}.npy")
    filenames = np.load(
        Path(save_dir) / f"filenames_{version}.npy", allow_pickle=True
    ).tolist()
    print(f"[Latent] Loaded {latents.shape[0]} vectors of dim {latents.shape[1]}")
    return latents, filenames


def plot_tsne(
    latents: np.ndarray,
    perplexity: int = 30,
    title: str = "t-SNE Latent Space",
    scores: np.ndarray = None,
    save_path: str = None,
    random_state: int = 42,
) -> np.ndarray:
    """Compute and plot t-SNE embedding of latent vectors.
    
    Args:
        latents: Numpy array of shape (N, latent_dim).
        perplexity: t-SNE perplexity parameter.
        title: Plot title.
        scores: Optional novelty scores for color-coding (N,).
        save_path: Optional path to save the figure.
        random_state: Random seed for reproducibility.
    
    Returns:
        2D t-SNE embedding of shape (N, 2).
    """
    print(f"[t-SNE] Computing with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    embedding = tsne.fit_transform(latents)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if scores is not None:
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=scores,
            cmap="RdYlBu_r",
            s=5,
            alpha=0.6,
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Novelty Score", fontsize=11)
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c="#3498db",
            s=5,
            alpha=0.5,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()
    plt.close(fig)

    return embedding


def plot_umap(
    latents: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP Latent Space",
    scores: np.ndarray = None,
    save_path: str = None,
    random_state: int = 42,
) -> np.ndarray:
    """Compute and plot UMAP embedding of latent vectors.
    
    Args:
        latents: Numpy array of shape (N, latent_dim).
        n_neighbors: Number of nearest neighbors for UMAP.
        min_dist: Minimum distance parameter for UMAP.
        title: Plot title.
        scores: Optional novelty scores for color-coding.
        save_path: Optional path to save the figure.
        random_state: Random seed.
    
    Returns:
        2D UMAP embedding of shape (N, 2).
    """
    try:
        import umap
    except ImportError:
        print("[UMAP] umap-learn not installed. Skipping UMAP plot.")
        return np.zeros((latents.shape[0], 2))

    print(f"[UMAP] Computing with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(latents)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if scores is not None:
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=scores,
            cmap="RdYlBu_r",
            s=5,
            alpha=0.6,
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Novelty Score", fontsize=11)
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c="#e74c3c",
            s=5,
            alpha=0.5,
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()
    plt.close(fig)

    return embedding
