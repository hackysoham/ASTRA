# Phase 4: Engineering Changelog

**Team Astra** | NSSC 2026 | IIT Kharagpur  
This document chronicles the step-by-step engineering of our unsupervised anomaly detection pipeline, tracing the evolution of our deep compression module from a simple convolutional autoencoder to a multi-scale edge-aware compound VAE.

---

## Iteration 1: v1 — Baseline Convolutional Autoencoder (CAE)

### Symptom
No prior architecture existed. The objective was to establish a minimum viable model to gauge baseline reconstruction quality and define the dimensions of the bottleneck, particularly addressing the non-standard 227x227 crop size.

### Diagnosis
A standard convolutional encoder-decoder network without normalization or advanced regularizations was required as a starting point.

### Fix
Built a 4-layer Convolutional Autoencoder (Encoder: 1->32->64->128->256, bottleneck latent dim = 128) using standard `ReLU` activations. Addressed the 227x227 dimensionality by applying `output_padding=1` on the final two layers of the `ConvTranspose2d` decoder to perfectly recreate the (1, 227, 227) shape. Loss function was strictly Mean Squared Error (MSE).

### Supporting Evidence & Outcome
*   **Result:** The model converged but trained slowly and exhibited unstable gradients in early epochs.
*   **Evaluation:** Reconstructions were heavily blurred, discarding high-frequency Martian surface textures. The latent space (visualized via t-SNE) was highly deterministic and disjointed, limiting its utility for anomaly clustering.

---

## Iteration 2: v2 — Structural Loss Autoencoder

### Symptom
v1 produced overly smooth/blurry images. Identical MSE loss values could correspond to vastly different perceptual qualities, meaning the model wasn't properly penalizing structural distortions.

### Diagnosis
MSE treats all pixels independently. Without a structural loss component, the model lacked the incentive to preserve edges. Furthermore, the absence of batch normalization led to internal covariate shift, slowing convergence.

### Fix
Implemented a from-scratch Differentiable Structural Similarity Index (SSIM) loss using 11x11 Gaussian-weighted local windows. The new loss function became an equal weighting of `MSE + SSIM`. We explicitly opted *not* to add Batch Normalization yet to isolate the impact of the structural loss.

### Supporting Evidence & Outcome
*   **Result:** The perceptual quality of reconstructions improved significantly. Craters and dune ripples were more sharply defined compared to v1.
*   **Evaluation:** Despite sharper edges, the latent space remained unregularized. Isolation Forest scoring was noisy because structurally similar inputs were sometimes embedded far apart.

---

## Iteration 3: v3 — Capacity-Optimized Autoencoder

### Symptom
While v2 had better edges, convergence was still sub-optimal and the model struggled to generalize across diverse terrain types in the 10,000-image dataset. 

### Diagnosis
The 128-dimensional latent space was a bottleneck, forcing too much information compression. Additionally, ReLU activations in the deep layers were causing "dead neurons" (zero gradients for negative values), and internal covariate shift was bottlenecking training speed.

### Fix
Increased depth to 5 layers (Encoder: 1->32->64->128->256->512) and expanded the `latent_dim` to 256. Replaced `ReLU` with `LeakyReLU(0.2)` to preserve gradients. Added `BatchNorm2d` after every convolutional layer.

### Supporting Evidence & Outcome
*   **Result:** Convergence speed tripled. The dead-neuron problem was eliminated.
*   **Evaluation:** The network successfully compressed and reconstructed the full diversity of the dataset with lower overall MSE. However, the latent space was still a deterministic pointwise mapping, lacking the continuous probability distribution required for robust Isolation Forest novelty detection.

---

## Iteration 4: v4 — Variational Autoencoder (VAE)

### Symptom
t-SNE and UMAP projections of v3’s 256-dimensional latent space showed disconnected clusters. Interpolating between latent vectors produced artifacts rather than smooth geological transitions.

### Diagnosis
A deterministic bottleneck provides no guarantee of continuity or completeness. Points close in latent space were decoding into vastly different outputs, causing the Isolation Forest to trigger false positives on minor interpolative variations.

### Fix
Replaced the deterministic bottleneck with a Variational Autoencoder framework. Split the final encoder layer into two parallel heads (`fc_mu` and `fc_logvar`). Implemented the reparameterization trick ($z = \mu + \sigma \cdot \epsilon$) and added Kullback-Leibler (KL) divergence to the loss function to regularize the latent space toward a standard normal distribution $\mathcal{N}(0, I)$.

### Supporting Evidence & Outcome
*   **Result:** The latent space smoothed out significantly. t-SNE projections showed continuous gradients of terrain types.
*   **Evaluation:** Isolation Forest performance drastically improved due to the structured embedding space. However, the stochastic nature of the sampling bottleneck caused a slight loss in high-frequency detail during reconstruction compared to v3.

---

## Iteration 5: v5 — Multi-Scale Edge-Aware Compound VAE/CAE

### Symptom
v4 suffered a slight degradation in edge fidelity due to the VAE bottleneck. Furthermore, some localized anomalies (e.g., sharp splicing artifacts) were being smoothed over by the decoder, making them harder to detect via pixel-wise residual heatmaps.

### Diagnosis
The VAE bottleneck inherently acts as a low-pass filter. The decoder lacked a direct pathway to access high-frequency spatial information from the original image. The loss function also lacked a multi-scale perceptual metric.

### Fix
1.  **Architecture:** Added U-Net-style skip connections between the encoder and decoder at matching spatial resolutions (113x113, 56x56, 28x28, 14x14).
2.  **Loss:** Upgraded the SSIM to a Multi-Scale SSIM (MS-SSIM) implemented from scratch.
3.  **Loss:** Added a custom `SobelEdgeLoss` (L1 distance of Sobel directional gradients) to heavily penalize blurry reconstructions.
4.  **Initialization:** Applied Kaiming Normal initialization tuned specifically for `LeakyReLU` (`a=0.2`).

### Supporting Evidence & Outcome
*   **Result:** v5 successfully achieved the "best of both worlds." The VAE's `mu` vector provided a beautifully continuous, regularized embedding space for the Isolation Forest, while the skip connections and Sobel/MS-SSIM loss ensured razor-sharp, edge-aware reconstructions.
*   **Evaluation:** This final architecture provided the most accurate spatial error heatmaps, allowing clear differentiation between sharp spliced boundary artifacts and diffuse domain shifts.
