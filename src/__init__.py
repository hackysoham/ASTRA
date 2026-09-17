# Team Astra — Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur
# 
"""
src package — Core modules for the anomaly detection pipeline.

Modules:
    dataset       - PyTorch Dataset for Mars HiRISE imagery
    models        - Autoencoder architectures v1–v5
    losses        - MSE, SSIM, and combined loss functions
    train         - Training loop utilities
    latent_utils  - Latent extraction and visualization (t-SNE, UMAP)
    isolation_forest - Isolation Forest scoring and statistical thresholding
    heatmaps      - Per-pixel reconstruction error heatmaps
    utils         - General helpers (seeding, device, checkpointing)
"""
