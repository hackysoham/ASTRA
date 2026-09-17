# Team Astra -- Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
isolation_engine.py -- Isolation-Forest novelty scoring with a
                       mathematically justified KDE-knee threshold.

Design constraints (competition rules)
---------------------------------------
* Higher Novelty Score = greater anomalousness.
* sklearn's ``decision_function`` uses the opposite convention
  (more negative = more anomalous), so we **invert** it.
* **No** assumed contamination fraction.
* **No** arbitrary fixed-count cutoffs.
* Threshold is derived from the score distribution itself using
  non-parametric Kernel Density Estimation (KDE) knee detection.

Public API
----------
fit_isolation_forest(features, ...)  -> novelty_scores
kde_knee_threshold(scores, ...)      -> (threshold, info_dict)
plot_score_distribution(scores, ...) -> None
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.ensemble import IsolationForest


# ====================================================================== #
#   ISOLATION  FOREST                                                      #
# ====================================================================== #

def fit_isolation_forest(
    features: np.ndarray,
    n_estimators: int = 300,
    max_samples: str = "auto",
    random_state: int = 42,
) -> np.ndarray:
    """Fit an Isolation Forest and return **inverted** novelty scores.

    sklearn convention:  decision_function < 0  -->  anomaly
    Our convention:      higher score           -->  greater anomaly

    Inversion:  ``novelty = -1 * decision_function(X)``

    Args:
        features:     (N, D) feature matrix (latents or fused vectors).
        n_estimators: Number of isolation trees.
        max_samples:  Samples drawn per tree.
        random_state: Seed.

    Returns:
        (N,) array of novelty scores (higher = more anomalous).
    """
    print(
        f"[IsolationForest] Fitting {n_estimators} trees on "
        f"{features.shape[0]} samples x {features.shape[1]} features ..."
    )
    iso = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        # NOTE: no contamination parameter -- threshold is statistical
    )
    iso.fit(features)

    raw = iso.decision_function(features)
    novelty = -1.0 * raw  # INVERT so higher = more anomalous

    print(
        f"[IsolationForest] Score range: "
        f"[{novelty.min():.6f}, {novelty.max():.6f}]  "
        f"mean={novelty.mean():.6f}  std={novelty.std():.6f}"
    )
    return novelty


# ====================================================================== #
#   KDE  KNEE  THRESHOLD                                                   #
# ====================================================================== #

def kde_knee_threshold(
    scores: np.ndarray,
    n_grid: int = 2000,
    bw_method: str = "scott",
) -> Tuple[float, Dict]:
    """Derive an anomaly threshold from the KDE *knee* of the score
    distribution -- the point where density drops off sharply on the
    right tail.

    Algorithm
    ---------
    1. Fit a Gaussian KDE to the novelty scores.
    2. Evaluate the density on a fine grid spanning [min, max].
    3. Locate the **mode** (global density peak).
    4. Restrict attention to the **right tail** (scores > mode).
    5. Compute the **second derivative** of log-density w.r.t. score.
    6. The knee is the grid point where the second derivative is most
       negative (maximum downward curvature), skipping a small region
       near the peak to avoid false detection.

    Mathematical justification
    --------------------------
    Under the null hypothesis that the bulk of the data is "normal",
    the score density is concentrated around its mode.  Anomalous
    scores populate a sparse right tail.  The knee marks the boundary
    where the density *regime changes* from the dense normal cluster
    to the sparse anomalous tail.  This is analogous to the elbow /
    knee heuristic used in scree plots but applied to a continuous
    density estimate.

    Args:
        scores:    (N,) novelty score array.
        n_grid:    Number of evaluation points for the KDE.
        bw_method: Bandwidth selection rule for ``gaussian_kde``.

    Returns:
        threshold: The score value at the detected knee.
        info:      Dictionary with diagnostic data (for plotting /
                   reporting).
    """
    kde = gaussian_kde(scores, bw_method=bw_method)
    x = np.linspace(scores.min(), scores.max(), n_grid)
    density = kde(x)

    # --- locate mode ---
    mode_idx = int(np.argmax(density))
    mode_val = x[mode_idx]

    # --- right tail only ---
    right_x = x[mode_idx:]
    right_density = density[mode_idx:]

    if len(right_x) < 20:
        # Degenerate case: fall back to mean + 3*std
        fallback = float(scores.mean() + 3.0 * scores.std())
        print(
            f"[KDE-knee] WARNING: right tail too short "
            f"({len(right_x)} pts).  Falling back to mean+3*std = "
            f"{fallback:.6f}"
        )
        return fallback, {
            "method": "KDE-knee (fallback: mean+3*std)",
            "threshold": fallback,
            "n_flagged": int(np.sum(scores > fallback)),
        }

    # --- second derivative of log-density ---
    log_d = np.log(right_density + 1e-30)
    dx = right_x[1] - right_x[0]
    d2 = np.gradient(np.gradient(log_d, dx), dx)

    # Skip the first 10 % near the mode to avoid edge artefacts
    skip = max(1, len(d2) // 10)
    search_d2 = d2[skip:]
    search_x = right_x[skip:]

    if len(search_d2) == 0:
        fallback = float(scores.mean() + 3.0 * scores.std())
        return fallback, {
            "method": "KDE-knee (fallback: mean+3*std)",
            "threshold": fallback,
            "n_flagged": int(np.sum(scores > fallback)),
        }

    knee_local_idx = int(np.argmin(search_d2))  # most negative curvature
    threshold = float(search_x[knee_local_idx])
    n_flagged = int(np.sum(scores > threshold))

    info: Dict = {
        "method": "KDE-knee (non-parametric)",
        "threshold": threshold,
        "mode": float(mode_val),
        "n_flagged": n_flagged,
        "pct_flagged": 100.0 * n_flagged / len(scores),
        "grid_x": x,
        "grid_density": density,
        "knee_x": threshold,
    }

    print(
        f"[KDE-knee] mode = {mode_val:.6f}  |  "
        f"threshold = {threshold:.6f}  |  "
        f"flagged = {n_flagged} ({info['pct_flagged']:.2f}%)"
    )
    return threshold, info


# ====================================================================== #
#   VISUALISATION                                                          #
# ====================================================================== #

def plot_score_distribution(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    info: Optional[Dict] = None,
    title: str = "Novelty Score Distribution",
    save_path: Optional[str] = None,
) -> None:
    """Plot a histogram + KDE of the novelty scores with the threshold
    line, plus a box-plot panel.

    Args:
        scores:    (N,) novelty scores.
        threshold: Vertical line marking the anomaly boundary.
        info:      If given (from ``kde_knee_threshold``), overlays the
                   full KDE curve.
        title:     Plot title.
        save_path: If given, save figure here.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # ---- left panel: histogram + KDE ----
    ax1.hist(
        scores, bins=120, density=True,
        color="#3498db", alpha=0.6, edgecolor="white", linewidth=0.3,
    )

    if info is not None and "grid_x" in info:
        ax1.plot(
            info["grid_x"], info["grid_density"],
            "k-", linewidth=2, label="KDE",
        )

    if threshold is not None:
        ax1.axvline(
            threshold, color="#e74c3c", linestyle="--", linewidth=2,
            label=f"Knee threshold: {threshold:.4f}",
        )

    ax1.set_xlabel("Novelty Score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(f"{title} -- Histogram + KDE", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ---- right panel: box-plot ----
    bp = ax2.boxplot(
        scores, vert=True, patch_artist=True,
        boxprops=dict(facecolor="#3498db", alpha=0.5),
        flierprops=dict(
            marker="o", markerfacecolor="#e74c3c",
            markersize=3, alpha=0.5,
        ),
    )

    if threshold is not None:
        ax2.axhline(
            threshold, color="#e74c3c", linestyle="--", linewidth=2,
            label=f"Knee: {threshold:.4f}",
        )
        ax2.legend(fontsize=9)

    ax2.set_ylabel("Novelty Score", fontsize=12)
    ax2.set_title(f"{title} -- Box Plot", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {save_path}")

    plt.close(fig)


def plot_ranked_scores(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Ranked Novelty Scores",
    save_path: Optional[str] = None,
) -> None:
    """Plot scores sorted descending to visualise the tail break.

    Args:
        scores:    (N,) novelty scores.
        threshold: Horizontal line at the knee.
        title:     Plot title.
        save_path: Path to save figure.
    """
    ranked = np.sort(scores)[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ranked, color="#2c3e50", linewidth=1.2)

    if threshold is not None:
        ax.axhline(
            threshold, color="#e74c3c", linestyle="--", linewidth=2,
            label=f"Threshold: {threshold:.4f}",
        )
        n_above = int(np.sum(scores > threshold))
        ax.axvline(
            n_above, color="#e74c3c", linestyle=":", linewidth=1,
            alpha=0.5,
        )
        ax.legend(fontsize=10)

    ax.set_xlabel("Rank (1 = most anomalous)", fontsize=12)
    ax.set_ylabel("Novelty Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved -> {save_path}")

    plt.close(fig)
