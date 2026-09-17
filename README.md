# Mars HiRISE Unsupervised Anomaly Detection

**Team Astra** | National Students' Space Challenge (NSSC 2026), IIT Kharagpur  
This repository houses the end-to-end unsupervised anomaly detection pipeline for Mars HiRISE orbital imagery. Built from scratch in PyTorch, the system utilizes a custom Multi-Scale Edge-Aware Variational Autoencoder combined with mathematically thresholded Isolation Forests.

## Directory Structure

```text
ASTRA/
├── data/                       # Contains all 227x227 crop images (not tracked in git)
│   ├── crop_metadata_index.csv # Crop-level metadata
│   └── source_image_metadata.csv # Image-level metadata
├── notebooks/                  # Jupyter notebooks for interactive analysis
│   ├── 01_Phase1_Deep_Latent_Compression.ipynb
│   ├── 02_Phase2_Isolation_Forest_Novelty.ipynb
│   ├── 03_Phase3_Reconstruction_Interpretability.ipynb
│   └── 04_Phase4_Architecture_Journal.ipynb
├── outputs/                    # Generated models, plots, heatmaps, and scores
├── report/                     # Markdown and PDF technical reports
│   ├── Engineering_Changelog.md
│   ├── Geological_Analysis.md
│   ├── Team_Astra_Final_Report.md
│   └── Team_Astra_Technical_Report.md
├── src/                        # Core Python pipeline modules
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset and DataLoader
│   ├── heatmaps.py             # Anomaly visualization and hypothesis logic
│   ├── isolation_engine.py     # Isolation Forest and KDE-knee logic
│   ├── latent_extract.py       # Latent extraction, t-SNE, UMAP
│   ├── losses.py               # Custom loss functions
│   ├── metadata_fusion.py      # Metadata merging and cyclic encoding
│   ├── models.py               # All 5 architecture iterations (v1 - v5)
│   ├── train.py                # Training loops
│   └── utils.py                # Helper utilities
├── tests/                      # Verification scripts
│   └── verify_shapes.py        # Validates exact (1, 227, 227) output shapes
├── README.md                   # This file
├── requirements.txt            # Dependency list
└── .gitignore                  # Git ignore file
```

## Adding Collaborators to the GitHub Repository

As per the NSSC 2026 submission guidelines, the following mandatory collaborators must be granted access to the private repository.

**Required Handles:**
*   `SarthakXSingh09`
*   `R15HV`
*   `Shivam3473`
*   `aditohates-bugs`
*   `Dhairya646`

**Instructions for the Repository Owner:**
1. Navigate to your private repository on [GitHub](https://github.com/).
2. Click on the **Settings** tab located near the top right of the repository page.
3. In the left sidebar, click on **Collaborators** (under the "Access" section). You may be prompted to enter your GitHub password or use two-factor authentication.
4. Click the green **Add people** button.
5. In the search box, copy and paste each of the required handles one by one.
6. Select the correct user from the dropdown and ensure their role is set to `Write` or `Maintain` (depending on competition requirements).
7. Click **Add [Username] to this repository**.
8. Repeat steps 4-7 until all five collaborators have been added. They will receive an email invitation which they must accept to gain access.

---
*Developed for NSSC 2026. No external pretrained weights were utilized.*
