# Phase 3.2: Geological Analysis of Top Anomalies

**Team Astra** | NSSC 2026 | IIT Kharagpur

This report details the physical hypotheses for the top-ranked anomalies identified by the anomaly detection pipeline. The analysis correlates spatial reconstruction-error distributions with potential Martian terrain features, sensor artifacts, calibration effects, or domain-shift phenomena.

---

## Rank 1 — Image: `sample_10365.jpg`

* **Novelty Score:** 0.2510
* **Mean Error:** 0.0073
* **Max Absolute Error:** 0.1817
* **Error Pattern:** Diffuse, uniform

### Physical Hypothesis

The reconstruction error is diffusely distributed across the crop rather than concentrated around a specific boundary or localized feature. The elevated residuals across a broad portion of the image are consistent with a possible domain shift between this sample and the training distribution. Potential explanations include a sensor calibration or illumination difference, unusual contrast or exposure characteristics, or a terrain morphology that is poorly represented in the training data. Since the residual is spatially diffuse, the anomaly is less indicative of a single localized artifact.

---

## Rank 2 — Image: `sample_06556.jpg`

* **Novelty Score:** 0.2510
* **Mean Error:** 0.0044
* **Max Absolute Error:** 0.1242
* **Error Pattern:** Diffuse, uniform

### Physical Hypothesis

The error is distributed relatively uniformly throughout the crop, with no dominant localized reconstruction failure. This pattern may indicate a moderate domain mismatch, potentially caused by differences in image intensity, illumination, sensor response, or terrain characteristics. A sensor calibration or preprocessing difference could produce a similar broad residual pattern. Further comparison with neighboring orbital images would be required to distinguish an acquisition-related effect from genuine terrain novelty.

---

## Rank 3 — Image: `sample_07293.jpg`

* **Novelty Score:** 0.2510
* **Mean Error:** 0.0047
* **Max Absolute Error:** 0.1664
* **Error Pattern:** Diffuse, uniform

### Physical Hypothesis

The reconstruction error is spread across the crop rather than being concentrated in a sharply defined region. This suggests that the model is encountering image characteristics that differ systematically from those learned during training. Possible causes include an exposure or contrast shift, sensor-response variation, or an underrepresented Martian surface morphology. The absence of a sharply localized error makes a discrete splicing or isolated corruption artifact less likely based on reconstruction error alone.

---

## Rank 4 — Image: `sample_06855.jpg`

* **Novelty Score:** 0.2510
* **Mean Error:** 0.0051
* **Max Absolute Error:** 0.1031
* **Error Pattern:** Diffuse, uniform

### Physical Hypothesis

The broad and relatively uniform residual pattern indicates a global reconstruction mismatch rather than a single anomalous structure. One possible explanation is a domain shift arising from different imaging conditions, sensor characteristics, or surface reflectance properties. Alternatively, the crop may contain terrain characteristics that are insufficiently represented in the training distribution. The relatively lower maximum residual suggests that the anomaly is distributed across the image rather than dominated by an isolated high-error feature.

---

## Rank 5 — Image: `sample_06495.jpg`

* **Novelty Score:** 0.2510
* **Mean Error:** 0.0082
* **Max Absolute Error:** 0.1740
* **Error Pattern:** Diffuse, uniform

### Physical Hypothesis

The error is diffusely distributed across the crop, indicating a broad mismatch between the observed image and the model's learned reconstruction manifold. This may correspond to a sensor calibration or exposure difference, unusual surface reflectance, or a terrain type that is underrepresented in the training set. The relatively high mean error among the five samples suggests that the mismatch is not confined to a small region. However, the reconstruction error alone cannot determine whether the underlying cause is instrumental or geological.

---

## Interpretation

All five highest-ranked samples exhibit the same **diffuse, uniform error pattern**, with identical novelty scores of **0.2510**. This differs substantially from a localized anomaly pattern, where a sharp reconstruction residual could provide stronger evidence for a discrete image artifact.

The consistent pattern across these samples suggests that the current top-ranked detections may be driven by a common feature of the input distribution rather than five unrelated localized anomalies. Candidate explanations include:

1. **Domain shift** between the training and evaluation images.
2. **Sensor or acquisition differences**, such as calibration, illumination, or exposure.
3. **Unusual surface reflectance or terrain morphology** not sufficiently represented during training.
4. **Preprocessing or normalization differences** affecting the reconstructed intensity distribution.

These hypotheses should be treated as **candidate physical explanations rather than confirmed geological interpretations**. Validation against the original orbital metadata, neighboring HiRISE observations, acquisition conditions, and visual inspection of the corresponding heatmaps would be required before assigning a specific geological or instrumental cause.
