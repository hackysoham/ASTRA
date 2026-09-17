# Team Astra -- Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
heatmaps.py -- Phase 3: Reconstruction Interpretability.

Filters the top-K anomalies strictly exceeding the anomaly threshold,
computes per-pixel absolute reconstruction errors, and renders them
as an overlay (Original | Reconstruction | Error Overlay) using the
'inferno' colormap.

Also includes utilities for automated geological hypothesis generation
based on spatial error patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ====================================================================== #
#   HEATMAP  RENDERING                                                     #
# ====================================================================== #

def plot_anomaly_panel(
    original: np.ndarray,
    reconstruction: np.ndarray,
    error_map: np.ndarray,
    filename: str,
    score: float,
    rank: int,
    save_path: str,
) -> None:
    """Render a 3-panel figure: Original | Reconstruction | Error Overlay.

    The error map is overlaid on the original image using the 'inferno'
    colormap with alpha blending to highlight the anomalous regions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ensure inputs are 2D
    orig_2d = original.squeeze()
    recon_2d = reconstruction.squeeze()
    err_2d = error_map.squeeze()

    # 1. Original Image
    axes[0].imshow(orig_2d, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original Crop\n({filename})", fontsize=12)
    axes[0].axis("off")

    # 2. Reconstruction
    axes[1].imshow(recon_2d, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Autoencoder\nReconstruction", fontsize=12)
    axes[1].axis("off")

    # 3. Error Overlay ('inferno' colormap)
    axes[2].imshow(orig_2d, cmap="gray", vmin=0, vmax=1)
    im3 = axes[2].imshow(err_2d, cmap="inferno", alpha=0.5)
    axes[2].set_title(f"Absolute Error Overlay\n(Score: {score:.4f} | Rank: {rank})", fontsize=12)
    axes[2].axis("off")
    
    # Add a colorbar for the error map
    cbar = fig.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar.set_label("Absolute Pixel Error", fontsize=10)

    fig.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ====================================================================== #
#   TOP-K  EXTRACTION                                                      #
# ====================================================================== #

@torch.no_grad()
def top_k_anomaly_panels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scores: np.ndarray,
    filenames: List[str],
    threshold: float,
    k: int = 5,
    save_dir: str = "outputs/heatmaps",
) -> List[Dict]:
    """Identify the top-K anomalies strictly exceeding the threshold,
    generate their heatmaps, and return diagnostic metadata.

    Args:
        model:     Trained autoencoder.
        loader:    DataLoader over the full dataset (no shuffle).
        device:    Computation device.
        scores:    (N,) novelty scores aligned with the loader.
        filenames: (N,) list of filenames aligned with the loader.
        threshold: The KDE knee statistical threshold.
        k:         Number of top anomalies to extract (default 5).
        save_dir:  Directory to save the heatmap PNGs.

    Returns:
        List of dictionaries containing analysis data for each anomaly.
    """
    model.eval()
    
    # Filter indices strictly exceeding threshold
    valid_mask = scores > threshold
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("[heatmaps] No anomalies strictly exceeded the threshold.")
        return []
        
    # Sort the valid indices by score (descending)
    valid_scores = scores[valid_indices]
    sorted_relative = np.argsort(valid_scores)[::-1]
    
    # Take top K
    top_k_relative = sorted_relative[:k]
    top_k_absolute = valid_indices[top_k_relative]
    
    target_indices = set(top_k_absolute)
    
    results = []
    
    # Single pass over the dataloader to grab the needed images
    current_idx = 0
    for batch_idx, batch in enumerate(loader):
        images = batch[0]
        batch_size = images.size(0)
        
        for i in range(batch_size):
            global_idx = current_idx + i
            if global_idx in target_indices:
                rank = list(top_k_absolute).index(global_idx) + 1
                fname = filenames[global_idx]
                score = float(scores[global_idx])
                
                # Forward pass for this single image
                img_tensor = images[i:i+1].to(device)
                
                if model.is_vae:
                    recon = model(img_tensor)[0]
                else:
                    recon = model(img_tensor)
                    
                orig_np = img_tensor.cpu().numpy()[0, 0]
                recon_np = recon.cpu().numpy()[0, 0]
                err_np = np.abs(orig_np - recon_np)
                
                save_path = f"{save_dir}/anomaly_rank{rank:02d}_{fname}.png"
                plot_anomaly_panel(
                    original=orig_np,
                    reconstruction=recon_np,
                    error_map=err_np,
                    filename=fname,
                    score=score,
                    rank=rank,
                    save_path=save_path,
                )
                
                # Basic pattern analysis for geological hypothesis
                mean_err = float(err_np.mean())
                max_err = float(err_np.max())
                
                # Simple heuristic classification based on spatial distribution
                # In a full implementation, we'd use FFTs, row/col variance, etc.
                err_std = float(err_np.std())
                if err_std > mean_err * 1.5:
                    pattern = "Sharp, localized"
                else:
                    pattern = "Diffuse, uniform"
                
                results.append({
                    "rank": rank,
                    "filename": fname,
                    "score": score,
                    "mean_error": mean_err,
                    "max_error": max_err,
                    "pattern": pattern,
                })
                
        current_idx += batch_size
        if len(results) == len(top_k_absolute):
            break  # Found all top-K

    # Sort results by rank before returning
    results = sorted(results, key=lambda x: x["rank"])
    return results


# ====================================================================== #
#   GEOLOGICAL  REPORT  GENERATOR                                          #
# ====================================================================== #

def analyze_error_pattern(result_dict: Dict) -> str:
    """Map a measured error pattern to a physical geological hypothesis."""
    pattern = result_dict.get("pattern", "")
    
    if "Sharp, localized" in pattern:
        return (
            "The error is highly localized with sharp boundaries. This strongly suggests "
            "a spliced boundary artifact where different orbital passes have been stitched "
            "together, or a discrete terrestrial interference injection. The model successfully "
            "reconstructs the natural Martian terrain but fails to replicate the discontinuous edge."
        )
    elif "Diffuse, uniform" in pattern:
        return (
            "The error is diffusely spread across the entire crop with elevated mean residuals. "
            "This suggests a significant domain shift — possibly a sensor calibration change, "
            "extreme contrast/exposure anomaly, or a terrain type completely absent from the "
            "training distribution."
        )
    else:
        return (
            "Complex error topology detected. Could indicate periodic sensor readout glitches "
            "(hardware anomalies) or multifaceted terrain distortions."
        )


def generate_geological_report(
    anomaly_results: List[Dict],
    save_path: str = "report/Geological_Analysis.md"
) -> str:
    """Generate the markdown report for Phase 3.2."""
    
    lines = [
        "# Phase 3.2: Geological Analysis of Top Anomalies",
        "",
        "**Team Astra** | NSSC 2026 | IIT Kharagpur",
        "",
        "This report details the physical hypotheses for the top-ranked anomalies strictly "
        "exceeding the KDE knee threshold. The analysis correlates spatial error distributions "
        "with potential Martian terrain features, sensor artifacts, or synthetic data injections.",
        "",
        "---",
        ""
    ]
    
    if not anomaly_results:
        lines.append("*No anomalies exceeded the defined threshold during this run.*")
    
    for res in anomaly_results:
        hypothesis = analyze_error_pattern(res)
        
        lines.extend([
            f"## Rank {res['rank']} — Image: `{res['filename']}`",
            f"- **Novelty Score:** {res['score']:.4f}",
            f"- **Max Absolute Error:** {res['max_error']:.4f}",
            f"- **Error Pattern:** {res['pattern']}",
            "",
            "### Physical Hypothesis",
            f"{hypothesis}",
            "",
            "---",
            ""
        ])
        
    markdown_content = "\n".join(lines)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"[heatmaps] Geological report saved to {save_path}")
    return markdown_content
