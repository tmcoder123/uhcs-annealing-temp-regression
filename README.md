# Multimodal UHCS Annealing Temperature Regression

Predicting furnace annealing temperature ($700^\circ\text{C} - 1100^\circ\text{C}$) in Ultra-High Carbon Steels (UHCS) from SEM micrographs and kinetic process parameters.

## 📌 Project Overview
- **Pipeline**: Fine-tuned ResNet-18 visual feature extractor compressed via PCA (16 components) and fused with physical process metadata (`anneal_time`, `magnification`, `cool_method`) using an optimized XGBoost regressor.
- **Interpretability**: Diagnosed corner-framing shortcut learning using Grad-CAM. Removing border artifacts via interior cropping exposed fundamental kinetic coarsening ambiguity ($r^3 \propto K(T) \cdot t$).
- **Imbalance Handling**: Mitigated phase imbalance via balanced sample weighting, dropping prediction error on complex high-temperature phases by $84.6^\circ\text{C}$.
- **Final Metrics**: Reached **$R^2 = 0.472$** and an **MAE of $40.13^\circ\text{C}$** on unseen holdout test data, with majority phases (Spheroidite, Network, Widmanstätten) achieving **MAE $\approx 33 - 35^\circ\text{C}$** ($<8.5\%$ relative thermal error).

## 🔬 Experimental Progression

| Architecture / Model | Holdout Test $R^2$ | Test MAE | Key Finding |
| :--- | :---: | :---: | :--- |
| **ResNet-18 (Raw Baseline)** | 0.408 | 45.99°C | Edge artifact shortcut learning diagnosed via Grad-CAM |
| **ResNet-18 (Interior-Cropped)** | 0.235 | 54.44°C | Stripping border bias exposed kinetic diffusion ambiguity |
| **Multimodal PyTorch (Late Fusion)** | 0.315 | 53.57°C | End-to-end backprop overfit small dataset regime ($N \approx 670$) |
| **Hybrid ResNet-18 + XGBoost** | 0.409 | 39.35°C | Tree-based tabular fusion stopped gradient domination |
| **Optimized Hybrid (PCA-16 + Class Weights)** | **0.472** | **40.13°C** | **Best pipeline.** Filtered latent noise, stabilized complex phase prediction |
| **Ensemble (XGBoost + Random Forest)** | 0.445 | 39.42°C | Bagging diluted sharp split thresholds, confirming feature saturation |
