# Acoustic Bird Biodiversity Monitoring Under Realistic Environmental Noise

## Fixed Research Question
> **“How does realistic environmental noise affect bird species classification, and to what extent can noise-aware training improve the robustness and reliability of acoustic biodiversity monitoring?”**

*Note: This research question and direction are strictly fixed and govern all subsequent phases of this project.*

---

## 1. Executive Summary & Problem Formulation
Passive Acoustic Monitoring (PAM) is a vital tool for automated biodiversity tracking in complex biomes (e.g., tropical and temperate forest reserves). However, field-deployed autonomous recording units (ARUs) capture vocalizations under severe, non-stationary environmental degradation:
- Weather artifacts: wind turbulence, tropical rain downpours, thunder.
- Biophonic interference: high-frequency insect stridulation, overlapping anuran and mammal calls.
- Anthropogenic noise: vehicle engines, aircraft flyovers, chainsaws, human presence.
- Geophonic disturbances: flowing rivers, waterfalls, foliage rustling.

Standard supervised deep audio classifiers trained primarily on curated, high-SNR focal audio fail catastrophically when evaluated against real-world continuous soundscapes containing low Signal-to-Noise Ratios (SNR).

This research systematically assesses:
1. **Performance Degradation Profiling**: Quantifying macro and species-level degradation curves as a function of specific environmental noise classes and SNR regimes (from -15 dB to +15 dB).
2. **Noise-Aware Training Strategies**: Comparing baseline empirical risk minimization (ERM) with realistic synthetic noise injection, domain-adversarial representation learning, and multi-condition training.
3. **Generalization & Calibration**: Evaluating classification accuracy, macro ROC-AUC, false discovery rate (FDR), and out-of-distribution calibration on uncurated soundscapes from BirdCLEF 2024.

---

## 2. Research Hypotheses
- **$H_1$ (Degradation Asymmetry)**: Environmental noise degrades high-frequency vocalizing species disproportionately during insect stridulation and wind gusts, shifting decision boundaries and causing systematic false positives in silent/low-amplitude windows.
- **$H_2$ (Noise-Aware Robustness)**: Injecting calibrated realistic noise profiles during training preserves multi-label ROC-AUC under severe noise ($\text{SNR} \le 0\text{ dB}$) with $\le 3\%$ degradation compared to clean audio, outperforming standard generic data augmentations (e.g., simple Gaussian noise or SpecAugment alone).
- **$H_3$ (Uncertainty & Calibration)**: Noise-aware models demonstrate superior temperature-calibrated Brier scores and expected calibration error (ECE) under corrupted acoustic channels, preventing high-confidence hallucinations on out-of-sample ambient noise.

---

## 3. Dataset Configuration & Governance
- **Dataset**: BirdCLEF 2024 (Western Ghats & associated biodiversity hotspots)
- **External Dataset Location**: `C:/Datasets/birdclef-2024`
- **Data Governance Policy**:
  - The raw dataset is strictly immutable and read-only.
  - No raw audio or metadata shall be modified, moved, copied into the repository, or re-downloaded.
  - Manifests, split indices, feature caches, and benchmark subsets will be saved inside `data/manifests/` and `data/processed/`.

---

## 4. Controlled Initial Baseline Parameters
All acoustic processing conforms to the following fixed initial research settings:
- **Random Seed**: `42`
- **Sample Rate**: `32,000 Hz` (preserving frequencies up to 16 kHz Nyquist)
- **Clip Duration**: `5.0 seconds` (160,000 samples)
- **FFT Window Length (`n_fft`)**: `2048` (~64 ms window)
- **Hop Length**: `512` (~16 ms stride, 75% overlap)
- **Mel Filterbanks (`n_mels`)**: `128`
- **Frequency Range**: `fmin = 50 Hz`, `fmax = 16,000 Hz`

*These values are strictly maintained as initial baseline parameters. Any variation must be executed through controlled comparative ablation experiments.*

---

## 5. Phased Research Roadmap
1. **Phase 1: Environment & Architecture Initialization** *(Current Phase)*
   - Directory structure, configuration schema, basic import & configuration tests, documentation.
2. **Phase 2: Dataset Auditing & Manifest Creation**
   - Headless manifest generation, species distribution auditing, metadata integrity checks.
3. **Phase 3: Acoustic Preprocessing & Baseline Pipeline**
   - Mel-spectrogram transformation pipeline, deterministic caching, clean baseline classifier.
4. **Phase 4: Realistic Noise Profiling & Simulation Engine**
   - Environmental noise decomposition, calibrated SNR mixing engine (-15 dB to +15 dB).
5. **Phase 5: Noise-Aware Modeling & Comparative Experiments**
   - Clean baseline vs. generic augmentation vs. realistic noise-aware training.
6. **Phase 6: Evaluation, Calibration & Final Reporting**
   - Macro ROC-AUC, confusion matrices across noise types, reliability diagrams, biodiversity monitoring deployment recommendations.
