# Team Astra: Mars HiRISE Anomaly Detection Report
**Author:** , BE Group 2  
**Event:** National Students' Space Challenge (NSSC 2026), IIT Kharagpur

---

## Abstract
This report details the end-to-end unsupervised anomaly detection pipeline developed by Team Astra for the Mars HiRISE dataset. Operating strictly without ground-truth labels or external pretrained weights, the system leverages a custom-built Multi-Scale Edge-Aware Variational Autoencoder (v5) to compress 10,000 grayscale 227x227 crops into a regularized 256-dimensional continuous latent space. An Isolation Forest algorithm, mathematically thresholded using non-parametric Kernel Density Estimation (KDE), scores these latents for novelty. Additionally, crop-level metadata (sun angle and cyclically encoded season) is fused with the latent vectors to boost detection accuracy. Spatial error overlays generated via the `inferno` colormap automatically map residuals to physical hypotheses, effectively distinguishing between terrestrial data-splicing injections and genuine domain shifts.

---

## Methodology Overview

The pipeline is segregated into three functional phases:

1.  **Deep Latent Compression:** 
    We engineered 5 progressive model architectures from scratch using PyTorch. The culminating architecture (v5) is a Variational Autoencoder (VAE) featuring U-Net style skip connections, `LeakyReLU` activations, and Kaiming initialization. It maps the non-standard 227x227 image shape to a 256-dimensional bottleneck, handling the exact dimensional recovery using `output_padding=1` on the transposed convolutions.
2.  **Isolation Forest Novelty Engine:**
    The deterministic $\mu$ vectors from the VAE bottleneck are extracted and scored using an Isolation Forest. The standard `decision_function` output is inverted (multiplied by -1) ensuring a higher score equates to greater anomalousness. Metadata features (`sun_angle` and `season`) are normalized and cyclically encoded (using sine/cosine mapping) prior to concatenation with the latent vector.
3.  **Reconstruction Interpretability:**
    Anomalies strictly exceeding a mathematical threshold are fed back through the VAE. The absolute spatial difference between the input crop and the reconstruction forms an error heatmap, overlaying the original image to physically diagnose the anomaly type.

---

## Mathematical Justifications

### 1. Loss Functions

Our final objective function for v5 combines four distinct metrics to balance structural fidelity with latent space regularization:

$$ L = \alpha \cdot \text{MSE} + \beta \cdot (1 - \text{MS-SSIM}) + \gamma \cdot \text{KL} + \delta \cdot \text{Sobel-L1} $$

*   **Mean Squared Error (MSE):** Captures pixel-wise intensity differences.
*   **MS-SSIM:** Multi-Scale Structural Similarity Index. Applied over 5 resolution scales using 11x11 Gaussian-weighted windows. It evaluates luminance at the coarsest scale and contrast/structure across all scales, punishing structural deviations that MSE ignores.
*   **Kullback-Leibler (KL) Divergence:** Regularizes the VAE latent space $q(z|x)$ to approximate a standard normal distribution $p(z) \sim \mathcal{N}(0, I)$.
    $$ D_{KL} = -0.5 \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right) $$
*   **Sobel Edge Loss:** Computes the L1 distance between directional gradients (horizontal and vertical Sobel filters) of the prediction and target, explicitly forcing the decoder to maintain sharp topographical boundaries.

### 2. KDE Knee Threshold Boundary

Rather than relying on arbitrary contamination fractions, we derive the anomaly threshold dynamically from the novelty score distribution:

1.  We fit a non-parametric Gaussian Kernel Density Estimate (KDE) to the novelty scores.
2.  Assuming the bulk of the data is nominal, the global maximum (mode) represents normal images.
3.  We restrict our analysis to the right tail (scores > mode).
4.  The "knee" is mathematically defined as the point of maximum downward curvature. We locate this by finding the minimum of the second derivative of the log-density:
    $$ \text{Threshold} = \arg\min_x \frac{d^2}{dx^2} \log(\text{KDE}(x)) $$
This identifies the exact regime change where the dense cluster of normal data transitions into the sparse tail of true anomalies.

---

## Architecture Iteration Results

| Iteration | Architecture Type | Latent Dim | Key Enhancements | Optimization Metric |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | Baseline CAE | 128 | 4-layer CAE, standard ReLU | Baseline MSE |
| **v2** | Structural CAE | 128 | Added custom SSIM loss | Better perceptual edges |
| **v3** | Capacity CAE | 256 | 5-layer, LeakyReLU, BatchNorm | Faster convergence, 0 dead neurons |
| **v4** | VAE | 256 | Reparameterization, KL Loss | Continuous latent clustering |
| **v5** | Compound VAE/CAE | 256 | Skip connections, Sobel, MS-SSIM | Razor-sharp detail, optimal IF scoring |
