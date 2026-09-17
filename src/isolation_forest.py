# Team Astra — Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
isolation_forest.py — Isolation Forest novelty scoring with metadata fusion
                      and three independent statistical thresholding methods.

Threshold Methods:
    1. Generalized Pareto Distribution (GPD) — Extreme Value Theory
    2. KDE Knee Detection — Density-based inflection point
    3. Median Absolute Deviation (MAD) — Robust dispersion bound

Final anomaly set = consensus (flagged by ≥2 of 3 methods).
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats
from scipy.signal import argrelextrema


# ═════════════════════════════════════════════════════════════════════════════
# METADATA FUSION
# ═════════════════════════════════════════════════════════════════════════════

def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load and return the source image metadata CSV.
    
    Expected columns: source_image_id, latitude, longitude,
                      sun_angle, season, resolution.
    """
    df = pd.read_csv(csv_path)
    print(f"[Metadata] Loaded {len(df)} rows from {csv_path}")
    print(f"[Metadata] Columns: {list(df.columns)}")
    return df


def extract_source_id(filename: str) -> str:
    """Extract source_image_id from an image filename.
    
    Attempts to parse the source ID from the filename by removing
    the file extension and any crop-index suffixes.
    
    Args:
        filename: Image filename (e.g., 'ESP_012345_1234_crop_001.png').
    
    Returns:
        Source image ID string.
    """
    # Remove extension
    name = Path(filename).stem
    # Try to extract the base source ID (before '_crop' or similar suffix)
    # Adjust this parsing logic based on actual filename conventions
    parts = name.split("_crop")
    return parts[0] if len(parts) > 1 else name


def build_features(
    latents: np.ndarray,
    filenames: List[str],
    metadata_path: str = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build combined feature vectors by fusing latent embeddings with metadata.
    
    If metadata is available, concatenates:
        [latent_vector | normalized_sun_angle | encoded_season]
    Otherwise, uses only the latent vectors.
    
    Args:
        latents: Numpy array of shape (N, latent_dim).
        filenames: List of N filenames.
        metadata_path: Optional path to source_image_metadata.csv.
    
    Returns:
        Tuple of (features, info_df) where:
            features: Combined feature array of shape (N, D)
            info_df: DataFrame with filename, source_id, and metadata columns
    """
    # Create base info DataFrame
    info_df = pd.DataFrame({"filename": filenames})
    info_df["source_image_id"] = info_df["filename"].apply(extract_source_id)

    if metadata_path is not None and Path(metadata_path).exists():
        meta_df = load_metadata(metadata_path)

        # Join on source_image_id
        info_df = info_df.merge(meta_df, on="source_image_id", how="left")
        print(
            f"[Fusion] Matched {info_df['sun_angle'].notna().sum()}/{len(info_df)} "
            f"images to metadata"
        )

        # Encode metadata features
        meta_features = []

        # Normalize sun_angle to [0, 1]
        if "sun_angle" in info_df.columns:
            sun_angle = info_df["sun_angle"].fillna(info_df["sun_angle"].median())
            scaler = MinMaxScaler()
            sun_angle_norm = scaler.fit_transform(
                sun_angle.values.reshape(-1, 1)
            )
            meta_features.append(sun_angle_norm)
            print(f"[Fusion] sun_angle range: [{sun_angle.min():.1f}, {sun_angle.max():.1f}]")

        # Cyclically encode season (spring=0, summer=π/2, fall=π, winter=3π/2)
        if "season" in info_df.columns:
            season_map = {
                "spring": 0.0, "summer": np.pi / 2,
                "fall": np.pi, "autumn": np.pi,
                "winter": 3 * np.pi / 2,
            }
            season_vals = info_df["season"].str.lower().map(season_map)
            season_vals = season_vals.fillna(0.0)
            season_sin = np.sin(season_vals.values).reshape(-1, 1)
            season_cos = np.cos(season_vals.values).reshape(-1, 1)
            meta_features.append(season_sin)
            meta_features.append(season_cos)
            print(f"[Fusion] season encoded cyclically (sin, cos)")

        if meta_features:
            meta_array = np.hstack(meta_features)
            features = np.hstack([latents, meta_array])
            print(
                f"[Fusion] Combined feature dim: {latents.shape[1]} (latent) + "
                f"{meta_array.shape[1]} (metadata) = {features.shape[1]}"
            )
        else:
            features = latents
    else:
        print("[Fusion] No metadata file found — using latent vectors only")
        features = latents

    return features, info_df


# ═════════════════════════════════════════════════════════════════════════════
# ISOLATION FOREST
# ═════════════════════════════════════════════════════════════════════════════

def run_isolation_forest(
    features: np.ndarray,
    n_estimators: int = 300,
    max_samples: str = "auto",
    random_state: int = 42,
) -> np.ndarray:
    """Fit Isolation Forest and return inverted novelty scores.
    
    Scikit-learn's decision_function returns scores where more negative
    means more anomalous. We invert so that HIGHER = more anomalous,
    as required by the competition rules.
    
    Args:
        features: Feature array of shape (N, D).
        n_estimators: Number of isolation trees.
        max_samples: Number of samples per tree.
        random_state: Random seed.
    
    Returns:
        Novelty scores array of shape (N,) where higher = more anomalous.
    """
    print(f"[IF] Fitting Isolation Forest (n_estimators={n_estimators})...")
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        # No contamination parameter — we derive threshold statistically
    )
    iso_forest.fit(features)

    # Get raw decision scores (more negative = more anomalous)
    raw_scores = iso_forest.decision_function(features)

    # INVERT: higher = more anomalous
    novelty_scores = -1.0 * raw_scores

    print(
        f"[IF] Score range: [{novelty_scores.min():.4f}, {novelty_scores.max():.4f}] "
        f"(mean={novelty_scores.mean():.4f}, std={novelty_scores.std():.4f})"
    )

    return novelty_scores


# ═════════════════════════════════════════════════════════════════════════════
# THRESHOLD METHOD 1: GENERALIZED PARETO DISTRIBUTION (GPD)
# ═════════════════════════════════════════════════════════════════════════════

def compute_threshold_gpd(
    scores: np.ndarray,
    tail_percentile: float = 90.0,
    p_value: float = 0.01,
) -> Tuple[float, dict]:
    """Compute anomaly threshold using Extreme Value Theory (GPD).
    
    Fits a Generalized Pareto Distribution to scores exceeding the
    specified tail percentile. The threshold is set at the GPD quantile
    corresponding to the given p-value.
    
    Mathematical justification:
        Under the Pickands–Balkema–De Haan theorem, exceedances over
        a sufficiently high threshold follow a GPD. Anomalies are defined
        as observations whose scores exceed the (1-p) quantile of this
        fitted GPD.
    
    Args:
        scores: Novelty scores array of shape (N,).
        tail_percentile: Percentile for defining the tail (default 90th).
        p_value: Tail probability for anomaly cutoff.
    
    Returns:
        Tuple of (threshold_value, info_dict).
    """
    u = np.percentile(scores, tail_percentile)  # Initial threshold
    exceedances = scores[scores > u] - u  # Excess above threshold

    if len(exceedances) < 10:
        print("[GPD] Warning: Too few exceedances. Falling back to u directly.")
        return u, {"method": "GPD", "threshold": u, "n_exceedances": len(exceedances)}

    # Fit GPD to exceedances
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

    # Compute the quantile: P(X > threshold | X > u) = p_value
    n_total = len(scores)
    n_exceed = len(exceedances)
    excess_rate = n_exceed / n_total

    # Threshold = u + GPD quantile at adjusted probability
    adjusted_p = p_value / excess_rate
    adjusted_p = min(adjusted_p, 1.0 - 1e-10)  # Clamp for numerical stability
    gpd_quantile = stats.genpareto.ppf(1 - adjusted_p, shape, loc=0, scale=scale)
    threshold = u + gpd_quantile

    info = {
        "method": "GPD (Extreme Value Theory)",
        "threshold": threshold,
        "initial_u": u,
        "shape": shape,
        "scale": scale,
        "n_exceedances": n_exceed,
        "p_value": p_value,
        "n_flagged": int(np.sum(scores > threshold)),
    }
    print(f"[GPD] Threshold = {threshold:.4f} → {info['n_flagged']} anomalies")
    return threshold, info


# ═════════════════════════════════════════════════════════════════════════════
# THRESHOLD METHOD 2: KDE KNEE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_threshold_kde_knee(
    scores: np.ndarray,
    n_points: int = 1000,
) -> Tuple[float, dict]:
    """Compute anomaly threshold via KDE knee detection.
    
    Estimates the probability density function of novelty scores using
    Gaussian KDE, then finds the "knee point" where density drops sharply.
    This is detected via the second derivative (curvature analysis).
    
    Mathematical justification:
        The distribution of normal scores is concentrated around the mode,
        while anomalous scores populate the sparse right tail. The knee
        point marks the transition from the dense "normal" region to the
        sparse "anomalous" tail.
    
    Args:
        scores: Novelty scores array of shape (N,).
        n_points: Number of evaluation points for KDE.
    
    Returns:
        Tuple of (threshold_value, info_dict).
    """
    # Fit KDE
    kde = stats.gaussian_kde(scores, bw_method="scott")
    x = np.linspace(scores.min(), scores.max(), n_points)
    density = kde(x)

    # Find the mode (peak of density)
    mode_idx = np.argmax(density)

    # Only look at the right tail (after the mode)
    right_x = x[mode_idx:]
    right_density = density[mode_idx:]

    if len(right_x) < 10:
        # Fallback: use mode + 2 std
        threshold = np.mean(scores) + 2 * np.std(scores)
        return threshold, {"method": "KDE-Knee (fallback)", "threshold": threshold}

    # Compute second derivative of log-density for curvature
    log_density = np.log(right_density + 1e-30)
    d2 = np.gradient(np.gradient(log_density, right_x), right_x)

    # The knee is where the curvature is most negative (sharpest drop)
    # Skip the first few points near the mode
    start_search = max(1, len(d2) // 10)
    search_d2 = d2[start_search:]
    search_x = right_x[start_search:]

    if len(search_d2) > 0:
        knee_idx = np.argmin(search_d2)
        threshold = search_x[knee_idx]
    else:
        threshold = np.mean(scores) + 2 * np.std(scores)

    info = {
        "method": "KDE Knee Detection",
        "threshold": threshold,
        "mode": x[mode_idx],
        "n_flagged": int(np.sum(scores > threshold)),
    }
    print(f"[KDE] Threshold = {threshold:.4f} → {info['n_flagged']} anomalies")
    return threshold, info


# ═════════════════════════════════════════════════════════════════════════════
# THRESHOLD METHOD 3: MEDIAN ABSOLUTE DEVIATION (MAD)
# ═════════════════════════════════════════════════════════════════════════════

def compute_threshold_mad(
    scores: np.ndarray,
    k: float = 3.5,
) -> Tuple[float, dict]:
    """Compute anomaly threshold using the Modified Z-score (MAD) method.
    
    The MAD (Median Absolute Deviation) is a robust measure of dispersion
    that is resistant to outliers (unlike standard deviation).
    
    Mathematical justification:
        Modified Z-score = 0.6745 * (x - median) / MAD
        An observation is considered anomalous if its Modified Z-score
        exceeds k (default: 3.5, per Iglewicz & Hoaglin, 1993).
        
        Equivalently: threshold = median + k * MAD / 0.6745
    
    Args:
        scores: Novelty scores array of shape (N,).
        k: Modified Z-score cutoff (default 3.5 per Iglewicz & Hoaglin).
    
    Returns:
        Tuple of (threshold_value, info_dict).
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))

    if mad < 1e-10:
        # Fallback if MAD is essentially zero
        threshold = median + k * np.std(scores)
        method_note = "MAD (fallback to std)"
    else:
        threshold = median + k * mad / 0.6745
        method_note = "MAD (Modified Z-score)"

    info = {
        "method": method_note,
        "threshold": threshold,
        "median": median,
        "mad": mad,
        "k": k,
        "n_flagged": int(np.sum(scores > threshold)),
    }
    print(f"[MAD] Threshold = {threshold:.4f} → {info['n_flagged']} anomalies")
    return threshold, info


# ═════════════════════════════════════════════════════════════════════════════
# CONSENSUS THRESHOLDING
# ═════════════════════════════════════════════════════════════════════════════

def consensus_threshold(
    scores: np.ndarray,
    min_votes: int = 2,
) -> Tuple[np.ndarray, float, Dict]:
    """Apply all three threshold methods and flag by consensus vote.
    
    An image is flagged as anomalous if it is identified by at least
    `min_votes` of the three independent methods.
    
    Args:
        scores: Novelty scores array of shape (N,).
        min_votes: Minimum number of methods that must agree (default 2).
    
    Returns:
        Tuple of (is_anomaly, effective_threshold, all_info) where:
            is_anomaly: Boolean array of shape (N,)
            effective_threshold: Approximate effective threshold
            all_info: Dict with details from all three methods
    """
    # Run all three methods
    t_gpd, info_gpd = compute_threshold_gpd(scores)
    t_kde, info_kde = compute_threshold_kde_knee(scores)
    t_mad, info_mad = compute_threshold_mad(scores)

    # Vote matrix
    votes = np.zeros(len(scores), dtype=int)
    votes += (scores > t_gpd).astype(int)
    votes += (scores > t_kde).astype(int)
    votes += (scores > t_mad).astype(int)

    is_anomaly = votes >= min_votes

    # Effective threshold = minimum score among flagged (if any)
    if is_anomaly.any():
        effective_threshold = scores[is_anomaly].min()
    else:
        effective_threshold = max(t_gpd, t_kde, t_mad)

    n_flagged = int(is_anomaly.sum())

    all_info = {
        "gpd": info_gpd,
        "kde": info_kde,
        "mad": info_mad,
        "consensus_min_votes": min_votes,
        "n_flagged": n_flagged,
        "effective_threshold": effective_threshold,
    }

    print(f"\n{'─'*50}")
    print(f"  CONSENSUS THRESHOLD (min_votes={min_votes})")
    print(f"  GPD:  {t_gpd:.4f} → {info_gpd['n_flagged']} flagged")
    print(f"  KDE:  {t_kde:.4f} → {info_kde['n_flagged']} flagged")
    print(f"  MAD:  {t_mad:.4f} → {info_mad['n_flagged']} flagged")
    print(f"  ────────────────────────────────")
    print(f"  FINAL: {n_flagged} anomalies (threshold ≈ {effective_threshold:.4f})")
    print(f"{'─'*50}\n")

    return is_anomaly, effective_threshold, all_info


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_score_distribution(
    scores: np.ndarray,
    thresholds: Dict[str, float] = None,
    title: str = "Novelty Score Distribution",
    save_path: str = None,
) -> None:
    """Plot the distribution of novelty scores with threshold lines.
    
    Args:
        scores: Novelty scores array of shape (N,).
        thresholds: Optional dict of {method_name: threshold_value}.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Histogram
    ax1.hist(scores, bins=100, color="#3498db", alpha=0.7, edgecolor="white",
             density=True)
    ax1.set_xlabel("Novelty Score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(f"{title} — Histogram", fontsize=13, fontweight="bold")

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(scores, bw_method="scott")
    x = np.linspace(scores.min(), scores.max(), 500)
    ax1.plot(x, kde(x), "k-", linewidth=2, label="KDE")

    # Threshold lines
    colors = {"GPD": "#e74c3c", "KDE": "#2ecc71", "MAD": "#f39c12"}
    if thresholds is not None:
        for name, val in thresholds.items():
            c = colors.get(name.split()[0], "#9b59b6")
            ax1.axvline(val, color=c, linestyle="--", linewidth=2,
                        label=f"{name}: {val:.3f}")

    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Box plot
    ax2.boxplot(scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#3498db", alpha=0.5),
                flierprops=dict(marker="o", markerfacecolor="#e74c3c",
                                markersize=3, alpha=0.5))
    ax2.set_ylabel("Novelty Score", fontsize=12)
    ax2.set_title(f"{title} — Box Plot", fontsize=13, fontweight="bold")

    if thresholds is not None:
        for name, val in thresholds.items():
            c = colors.get(name.split()[0], "#9b59b6")
            ax2.axhline(val, color=c, linestyle="--", linewidth=2, label=name)
        ax2.legend(fontsize=9)

    ax2.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()
    plt.close(fig)


def save_scores(
    scores: np.ndarray,
    filenames: List[str],
    is_anomaly: np.ndarray,
    info_df: pd.DataFrame = None,
    save_path: str = "outputs/scores/novelty_scores.csv",
    version: str = "v1",
) -> pd.DataFrame:
    """Save novelty scores to CSV.
    
    Args:
        scores: Novelty scores array (N,).
        filenames: List of N filenames.
        is_anomaly: Boolean anomaly flags (N,).
        info_df: Optional metadata DataFrame to merge.
        save_path: Path for the output CSV.
        version: Model version string.
    
    Returns:
        DataFrame with scores and anomaly flags.
    """
    df = pd.DataFrame({
        "filename": filenames,
        "novelty_score": scores,
        "is_anomaly": is_anomaly,
        "model_version": version,
    })

    if info_df is not None:
        # Merge metadata columns
        meta_cols = [c for c in info_df.columns if c != "filename"]
        if meta_cols:
            df = df.merge(
                info_df[["filename"] + meta_cols],
                on="filename",
                how="left",
            )

    # Sort by novelty score (most anomalous first)
    df = df.sort_values("novelty_score", ascending=False).reset_index(drop=True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Scores] Saved {len(df)} entries → {save_path}")
    print(f"[Scores] {df['is_anomaly'].sum()} flagged as anomalous")

    return df
