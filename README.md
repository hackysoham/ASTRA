# Mars HiRISE Unsupervised Anomaly Detection

**Team Astra** | National Students' Space Challenge (NSSC 2026), IIT Kharagpur

End-to-end unsupervised anomaly detection for Mars HiRISE imagery, built from scratch in PyTorch without pretrained weights or ground-truth labels.

## Pipeline

**Phase 1 — Deep Latent Compression**
Five autoencoder iterations (v1–v5), culminating in a **256-D Multi-Scale Edge-Aware VAE** with U-Net skip connections, LeakyReLU, MS-SSIM, Sobel loss, and exact 227×227 reconstruction.

**Phase 2 — Novelty Detection**
VAE `μ` vectors are combined with normalized sun-angle and cyclic season metadata and scored using **Isolation Forest**. Anomaly thresholds are derived using **GPD, KDE-knee, and MAD**, with ≥2/3 consensus.

**Phase 3 — Interpretability**
Top anomalies are analyzed using per-pixel reconstruction-error heatmaps. Spatial concentration, directionality, and periodicity are used to generate candidate physical hypotheses.

**Phase 4 — Design Journal**
Documents the engineering evolution from baseline CAE (v1) to the final v5 architecture.

## Directory Structure

```text
ASTRA/
├── data/
│   ├── crop_metadata_index.csv
│   └── source_image_metadata.csv
├── notebooks/
│   ├── 01_Phase1_Deep_Latent_Compression.ipynb
│   ├── 02_Phase2_Isolation_Forest_Novelty.ipynb
│   ├── 03_Phase3_Reconstruction_Interpretability.ipynb
│   └── 04_Phase4_Architecture_Journal.ipynb
├── outputs/
├── report/
│   ├── Engineering_Changelog.md
│   ├── Geological_Analysis.md
│   ├── Team_Astra_Final_Report.md
│   └── Team_Astra_Technical_Report.md
├── src/
│   ├── dataset.py
│   ├── heatmaps.py
│   ├── isolation_engine.py
│   ├── latent_extract.py
│   ├── losses.py
│   ├── metadata_fusion.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── verify_shapes.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Reports

Detailed methodology, geological analysis, final results, and architecture evolution are available in the `report/` directory.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks sequentially from Phase 1 through Phase 4.

---

**Team Astra** | NSSC 2026 | IIT Kharagpur
