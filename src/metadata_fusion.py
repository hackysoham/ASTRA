# Team Astra -- Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
metadata_fusion.py -- Load, merge, encode, and fuse crop-level metadata
                      with image latent vectors for the Isolation-Forest
                      novelty scoring pipeline.

Workflow
--------
1. ``crop_metadata_index.csv``   (filename, source_image_id)
2. ``source_image_metadata.csv`` (source_image_id, latitude, longitude,
                                  sun_angle, season, resolution)
3. Inner-join on ``source_image_id``.
4. Normalise ``sun_angle`` to [0, 1] via min-max scaling.
5. Cyclically encode ``season`` as (sin(theta), cos(theta)).
6. Concatenate  [image_latent | sun_angle | season_sin | season_cos]
   to form the fused feature vector.
7. Run Isolation Forest on (a) image-only latents and (b) fused vectors.
8. Output a side-by-side comparison table.

Public API
----------
load_and_merge_metadata(crop_csv, source_csv)
    -> merged DataFrame

encode_metadata(merged_df)
    -> (N, K) numpy array of encoded metadata columns

build_fused_features(latents, filenames, crop_csv, source_csv)
    -> (fused_features, meta_features, merged_df)

compare_image_vs_fused(latents, fused, filenames, threshold_fn, ...)
    -> comparison DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .isolation_engine import fit_isolation_forest, kde_knee_threshold


# ====================================================================== #
#   CSV  LOADING  &  MERGING                                               #
# ====================================================================== #

def load_and_merge_metadata(
    crop_csv: str,
    source_csv: str,
) -> pd.DataFrame:
    """Load the two metadata CSVs and inner-join on ``source_image_id``.

    Args:
        crop_csv:   Path to ``crop_metadata_index.csv``
                    (columns: filename, source_image_id).
        source_csv: Path to ``source_image_metadata.csv``
                    (columns: source_image_id, latitude, longitude,
                    sun_angle, season, resolution).

    Returns:
        Merged DataFrame with one row per crop.

    Raises:
        FileNotFoundError: If either CSV is missing.
        KeyError:          If required columns are absent.
    """
    for p, label in [(crop_csv, "crop_metadata_index"), (source_csv, "source_image_metadata")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    crop_df = pd.read_csv(crop_csv)
    source_df = pd.read_csv(source_csv)

    # Validate required columns
    for col in ("filename", "source_image_id"):
        if col not in crop_df.columns:
            raise KeyError(f"'{col}' missing from {crop_csv}")
    if "source_image_id" not in source_df.columns:
        raise KeyError(f"'source_image_id' missing from {source_csv}")

    merged = crop_df.merge(source_df, on="source_image_id", how="left")

    n_matched = merged["source_image_id"].notna().sum()
    print(
        f"[metadata_fusion] Crops: {len(crop_df)}  |  "
        f"Source images: {len(source_df)}  |  "
        f"Joined rows: {len(merged)}  |  "
        f"Metadata hits: {n_matched}"
    )
    return merged


# ====================================================================== #
#   FEATURE  ENCODING                                                      #
# ====================================================================== #

_SEASON_ANGLE_MAP = {
    "spring": 0.0,
    "summer": np.pi / 2.0,
    "fall":   np.pi,
    "autumn": np.pi,          # alias
    "winter": 3.0 * np.pi / 2.0,
}


def encode_metadata(merged_df: pd.DataFrame) -> np.ndarray:
    """Encode ``sun_angle`` and ``season`` into a compact numeric vector.

    Encoding scheme
    ---------------
    * ``sun_angle``:  min-max normalised to [0, 1].
    * ``season``:     mapped to an angle theta, then encoded as
                      ``(sin(theta), cos(theta))`` for cyclical
                      continuity (winter is close to spring, not
                      maximally distant).

    Args:
        merged_df: DataFrame produced by ``load_and_merge_metadata``.

    Returns:
        (N, 3) array  --  columns: [sun_angle_norm, season_sin, season_cos]
        If a column is missing, its entries default to 0.0.
    """
    N = len(merged_df)
    out = np.zeros((N, 3), dtype=np.float32)

    # ---- sun_angle (col 0) ----
    if "sun_angle" in merged_df.columns:
        sa = merged_df["sun_angle"].astype(float)
        sa = sa.fillna(sa.median())
        lo, hi = sa.min(), sa.max()
        if hi - lo > 0:
            out[:, 0] = ((sa - lo) / (hi - lo)).values
        print(
            f"[metadata_fusion] sun_angle range: "
            f"[{lo:.2f}, {hi:.2f}] -> normalised to [0, 1]"
        )
    else:
        print("[metadata_fusion] WARNING: 'sun_angle' column not found.")

    # ---- season (cols 1-2) ----
    if "season" in merged_df.columns:
        angles = (
            merged_df["season"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_SEASON_ANGLE_MAP)
        )
        angles = angles.fillna(0.0).values
        out[:, 1] = np.sin(angles)
        out[:, 2] = np.cos(angles)
        unique_seasons = merged_df["season"].dropna().unique()
        print(
            f"[metadata_fusion] season values: {list(unique_seasons)} "
            f"-> cyclically encoded (sin, cos)"
        )
    else:
        print("[metadata_fusion] WARNING: 'season' column not found.")

    return out


# ====================================================================== #
#   FUSION  BUILDER                                                        #
# ====================================================================== #

def build_fused_features(
    latents: np.ndarray,
    filenames: List[str],
    crop_csv: str,
    source_csv: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the fused feature vector: [image_latent | encoded_metadata].

    Steps
    -----
    1. Merge metadata CSVs.
    2. Align the merged DataFrame to the order of *filenames* so that
       row *i* of the metadata corresponds to ``latents[i]``.
    3. Encode ``sun_angle`` and ``season``.
    4. Horizontally stack ``(N, latent_dim)`` with ``(N, 3)`` to produce
       ``(N, latent_dim + 3)`` fused features.

    Args:
        latents:    (N, D) image latent vectors.
        filenames:  Length-N list of crop filenames (same order as latents).
        crop_csv:   Path to crop_metadata_index.csv.
        source_csv: Path to source_image_metadata.csv.

    Returns:
        fused_features: (N, D+3) fused array.
        meta_features:  (N, 3)   encoded metadata only.
        merged_df:      Aligned DataFrame (same row order as latents).
    """
    merged_raw = load_and_merge_metadata(crop_csv, source_csv)

    # Build a lookup from filename -> row-index in merged_raw
    fname_lookup = {
        fn: idx for idx, fn in enumerate(merged_raw["filename"].values)
    }

    # Align to latent order
    aligned_rows = []
    for fn in filenames:
        if fn in fname_lookup:
            aligned_rows.append(merged_raw.iloc[fname_lookup[fn]])
        else:
            # Missing metadata: create a row with NaNs
            row = pd.Series(dtype=object)
            row["filename"] = fn
            aligned_rows.append(row)

    merged_df = pd.DataFrame(aligned_rows).reset_index(drop=True)

    # Encode
    meta_features = encode_metadata(merged_df)

    # Fuse
    fused = np.hstack([latents, meta_features])
    print(
        f"[metadata_fusion] Fused feature dim: "
        f"{latents.shape[1]} (latent) + {meta_features.shape[1]} (meta) "
        f"= {fused.shape[1]}"
    )
    return fused, meta_features, merged_df


# ====================================================================== #
#   COMPARISON  TABLE                                                      #
# ====================================================================== #

def compare_image_vs_fused(
    latents: np.ndarray,
    fused_features: np.ndarray,
    filenames: List[str],
    n_estimators: int = 300,
    random_state: int = 42,
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Run Isolation Forest on (a) image-only latents and (b) fused
    vectors, then produce a side-by-side comparison table.

    For each branch, the KDE-knee threshold is computed independently.

    Args:
        latents:         (N, D)   image-only latent vectors.
        fused_features:  (N, D+K) fused vectors.
        filenames:       (N,)     crop filenames.
        n_estimators:    Trees per forest.
        random_state:    Seed.
        save_csv:        If given, save the comparison to this path.

    Returns:
        DataFrame with columns:
            filename,
            score_image, rank_image, anomaly_image,
            score_fused, rank_fused, anomaly_fused,
            rank_change
    """
    # ---- image-only branch ----
    print("\n--- Image-Only Isolation Forest ---")
    scores_img = fit_isolation_forest(
        latents, n_estimators=n_estimators, random_state=random_state,
    )
    thresh_img, info_img = kde_knee_threshold(scores_img)

    # ---- fused branch ----
    print("\n--- Fused (Image + Metadata) Isolation Forest ---")
    scores_fused = fit_isolation_forest(
        fused_features, n_estimators=n_estimators, random_state=random_state,
    )
    thresh_fused, info_fused = kde_knee_threshold(scores_fused)

    # ---- build comparison DataFrame ----
    df = pd.DataFrame({"filename": filenames})

    df["score_image"] = scores_img
    df["rank_image"] = df["score_image"].rank(ascending=False, method="min").astype(int)
    df["anomaly_image"] = scores_img > thresh_img

    df["score_fused"] = scores_fused
    df["rank_fused"] = df["score_fused"].rank(ascending=False, method="min").astype(int)
    df["anomaly_fused"] = scores_fused > thresh_fused

    df["rank_change"] = df["rank_image"] - df["rank_fused"]

    # Sort by fused score descending
    df = df.sort_values("score_fused", ascending=False).reset_index(drop=True)

    # ---- summary ----
    n_img = df["anomaly_image"].sum()
    n_fused = df["anomaly_fused"].sum()
    overlap = (df["anomaly_image"] & df["anomaly_fused"]).sum()

    print("\n" + "=" * 72)
    print("  COMPARISON: Image-Only vs. Fused (Image + Metadata)")
    print("=" * 72)
    print(f"  Image-only threshold : {thresh_img:.6f}   -> {n_img} anomalies")
    print(f"  Fused threshold      : {thresh_fused:.6f}   -> {n_fused} anomalies")
    print(f"  Overlap              : {overlap} flagged by both")
    print(f"  Unique to fused      : {n_fused - overlap}")
    print(f"  Unique to image-only : {n_img - overlap}")
    print("-" * 72)
    print("  Top-10 by Fused Score:")
    cols_show = [
        "filename", "score_image", "rank_image",
        "score_fused", "rank_fused", "rank_change",
    ]
    print(df[cols_show].head(10).to_string(index=False))
    print("=" * 72)

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"[metadata_fusion] Comparison saved -> {save_csv}")

    return df
