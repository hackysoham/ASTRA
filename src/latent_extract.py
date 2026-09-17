# Team Astra -- Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
latent_extract.py -- Extract 1-D latent vectors from a trained autoencoder
                     (default: Model v5) and project them into 2-D with
                     t-SNE and UMAP for qualitative assessment.

Public API
----------
extract_latents(model, loader, device)  -> (latents, filenames)
save_latents(latents, filenames, ...)   -> None
load_latents(save_dir, version)         -> (latents, filenames)
run_tsne(latents, ...)                  -> embedding_2d
run_umap(latents, ...)                  -> embedding_2d
plot_projection(embedding, ...)         -> None
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")  # non-interactive backend safe for any env
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ====================================================================== #
#   LATENT  EXTRACTION                                                     #
# ====================================================================== #

@torch.no_grad()
def extract_latents(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """Run *model* in eval mode over *loader* and collect latent vectors.

    For **VAE** models (``model.is_vae == True``) the *mu* vector is used
    as the deterministic point-estimate of the latent representation --
    this avoids sampling noise and gives a stable embedding for
    downstream Isolation-Forest scoring.

    For **CAE** models the output of ``model.encode()`` is used directly.

    Args:
        model:  Trained autoencoder (any version).
        loader: DataLoader covering the **full** dataset (no shuffle).
        device: Computation device (cpu / cuda).

    Returns:
        latents   -- ``np.ndarray`` of shape ``(N, latent_dim)``
        filenames -- ``list[str]`` of length N (preserves loader order)
    """
    model.eval()
    all_latents: list = []
    all_fnames: list = []

    for batch in loader:
        images = batch[0].to(device)
        fnames = batch[1]  # dataset returns (tensor, filename, source_id)

        if model.is_vae:
            enc_out = model.encode(images)
            # v5 returns (z, mu, logvar, skips); v4 returns (z, mu, logvar)
            mu = enc_out[1]
            all_latents.append(mu.cpu().numpy())
        else:
            z = model.encode(images)
            all_latents.append(z.cpu().numpy())

        # fnames may be a tuple or list from the collate_fn
        if isinstance(fnames, (list, tuple)):
            all_fnames.extend(fnames)
        else:
            all_fnames.extend(list(fnames))

    latents = np.concatenate(all_latents, axis=0)
    print(
        f"[latent_extract] Extracted {latents.shape[0]} vectors, "
        f"dim = {latents.shape[1]}"
    )
    return latents, all_fnames


# ====================================================================== #
#   SAVE / LOAD                                                            #
# ====================================================================== #

def save_latents(
    latents: np.ndarray,
    filenames: List[str],
    save_dir: str = "outputs/latent_vectors",
    version: str = "v5",
) -> None:
    """Persist latent vectors and the corresponding filename list."""
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"latents_{version}.npy", latents)
    np.save(d / f"filenames_{version}.npy", np.array(filenames, dtype=object))
    print(f"[latent_extract] Saved -> {d / f'latents_{version}.npy'}")


def load_latents(
    save_dir: str = "outputs/latent_vectors",
    version: str = "v5",
) -> Tuple[np.ndarray, List[str]]:
    """Load previously saved latents."""
    d = Path(save_dir)
    latents = np.load(d / f"latents_{version}.npy")
    filenames = np.load(
        d / f"filenames_{version}.npy", allow_pickle=True
    ).tolist()
    print(
        f"[latent_extract] Loaded {latents.shape[0]} vectors, "
        f"dim = {latents.shape[1]}"
    )
    return latents, filenames


# ====================================================================== #
#   t-SNE                                                                  #
# ====================================================================== #

def run_tsne(
    latents: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Compute a 2-D t-SNE embedding.

    Args:
        latents:    (N, D) feature matrix.
        perplexity: t-SNE perplexity (typical 5-50).
        n_iter:     Optimisation iterations.
        seed:       Random state for reproducibility.

    Returns:
        (N, 2) embedding array.
    """
    print(f"[t-SNE] Running with perplexity={perplexity}, n_iter={n_iter} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    emb = tsne.fit_transform(latents)
    print(f"[t-SNE] Done.  KL divergence = {tsne.kl_divergence_:.4f}")
    return emb


# ====================================================================== #
#   UMAP                                                                   #
# ====================================================================== #

def run_umap(
    latents: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Compute a 2-D UMAP embedding.

    Falls back to t-SNE if ``umap-learn`` is not installed, printing a
    warning rather than crashing.

    Args:
        latents:     (N, D) feature matrix.
        n_neighbors: UMAP neighbour count.
        min_dist:    Minimum embedding distance.
        seed:        Random state.

    Returns:
        (N, 2) embedding array.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        print(
            "[UMAP] WARNING: umap-learn is not installed.  "
            "Falling back to t-SNE with perplexity=15."
        )
        return run_tsne(latents, perplexity=15, seed=seed)

    print(
        f"[UMAP] Running with n_neighbors={n_neighbors}, "
        f"min_dist={min_dist} ..."
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    emb = reducer.fit_transform(latents)
    print("[UMAP] Done.")
    return emb


# ====================================================================== #
#   PLOTTING                                                               #
# ====================================================================== #

def plot_projection(
    embedding: np.ndarray,
    title: str = "Latent Projection",
    scores: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    point_size: float = 4,
) -> None:
    """Scatter-plot a 2-D embedding, optionally coloured by *scores*.

    Args:
        embedding:  (N, 2) array.
        title:      Plot title.
        scores:     (N,) novelty scores for colour mapping.
        save_path:  If given, save figure here.
        cmap:       Matplotlib colourmap name.
        point_size: Marker size.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if scores is not None:
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=scores, cmap=cmap, s=point_size, alpha=0.6,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Novelty Score", fontsize=11)
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c="#3498db", s=point_size, alpha=0.5,
        )

    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {save_path}")

    plt.close(fig)
