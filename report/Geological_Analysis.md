# Phase 3.2: Geological Analysis of Top Anomalies

**Team Astra** | NSSC 2026 | IIT Kharagpur

This report details the physical hypotheses for the top-ranked anomalies strictly exceeding the KDE knee threshold. The analysis correlates spatial error distributions with potential Martian terrain features, sensor artifacts, or synthetic data injections.

---

## Rank 1 — Image: `ESP_011261_1435_RED_crop_042.png`
- **Novelty Score:** 0.8924
- **Max Absolute Error:** 0.9412
- **Error Pattern:** Sharp, localized

### Physical Hypothesis
The error is highly localized with sharp boundaries. This strongly suggests a spliced boundary artifact where different orbital passes have been stitched together, or a discrete terrestrial interference injection. The model successfully reconstructs the natural Martian terrain but fails to replicate the discontinuous edge.

---

## Rank 2 — Image: `ESP_011261_1435_RED_crop_108.png`
- **Novelty Score:** 0.8512
- **Max Absolute Error:** 0.9105
- **Error Pattern:** Sharp, localized

### Physical Hypothesis
The error is highly localized with sharp boundaries. This strongly suggests a spliced boundary artifact where different orbital passes have been stitched together, or a discrete terrestrial interference injection. The model successfully reconstructs the natural Martian terrain but fails to replicate the discontinuous edge.

---

## Rank 3 — Image: `ESP_022415_1750_RED_crop_015.png`
- **Novelty Score:** 0.8101
- **Max Absolute Error:** 0.7654
- **Error Pattern:** Diffuse, uniform

### Physical Hypothesis
The error is diffusely spread across the entire crop with elevated mean residuals. This suggests a significant domain shift — possibly a sensor calibration change, extreme contrast/exposure anomaly, or a terrain type completely absent from the training distribution.

---

## Rank 4 — Image: `ESP_022415_1750_RED_crop_088.png`
- **Novelty Score:** 0.7933
- **Max Absolute Error:** 0.8841
- **Error Pattern:** Complex, periodic

### Physical Hypothesis
Complex error topology detected. Could indicate periodic sensor readout glitches (hardware anomalies) or multifaceted terrain distortions.

---

## Rank 5 — Image: `ESP_033100_1525_RED_crop_002.png`
- **Novelty Score:** 0.7820
- **Max Absolute Error:** 0.7102
- **Error Pattern:** Diffuse, uniform

### Physical Hypothesis
The error is diffusely spread across the entire crop with elevated mean residuals. This suggests a significant domain shift — possibly a sensor calibration change, extreme contrast/exposure anomaly, or a terrain type completely absent from the training distribution.

---
