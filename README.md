# Acoustic Bird Biodiversity Research

> **A Research-Oriented Acoustic Bird Biodiversity Monitoring System Using BirdCLEF 2024**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Fixed Research Question

> **“How does realistic environmental noise affect bird species classification, and to what extent can noise-aware training improve the robustness and reliability of acoustic biodiversity monitoring?”**

*This research question and directional focus are strictly fixed and define all experimental baselines, data processing steps, and ablation studies conducted within this repository.*

---

## 2. Project Architecture & Directory Layout

```text
bird-biodiversity-research/
├── configs/             # Central research configuration files (config.yaml)
├── src/                 # Core modular ML and audio source code
│   ├── utils/           # Configuration management, seeding, and logging
│   ├── audio/           # Mel-spectrogram processing and feature pipelines
│   ├── data/            # Manifest creation and PyTorch Dataset abstractions
│   ├── models/          # Deep learning backbones and classification heads
│   └── evaluation/      # Multi-label ROC-AUC, calibration, and SNR metrics
├── notebooks/           # Exploratory acoustic and data analysis notebooks
├── scripts/             # Standalone operational CLI scripts
├── experiments/         # Experiment run configurations and tracking records
├── reports/             # Generated research reports, figures, and benchmark metrics
│   ├── figures/         # Spectrogram figures, SNR degradation curves
│   └── results/         # Quantitative results and metric CSVs
├── docs/                # In-depth architectural and research specifications
│   ├── research_plan.md # Formal research plan and hypotheses
│   └── ...
├── tests/               # Unit and regression test suite
├── models/              # Checkpoint weights and model exports
├── app/                 # Biodiversity monitoring interface / dashboard
├── data/                # Local data storage
│   ├── raw/             # Read-only reference stubs
│   ├── processed/       # Cached Mel-spectrogram features
│   └── manifests/       # Partitioned CSV manifests
├── README.md            # Project overview and setup instructions
├── requirements.txt     # Bounded Python dependencies
├── .gitignore           # Version control filter rules
└── PROJECT_ARCHITECTURE.md # Architectural design details
```

For a comprehensive explanation of every subsystem, refer to [`PROJECT_ARCHITECTURE.md`](file:///C:/projects/bird-biodiversity-research/PROJECT_ARCHITECTURE.md).

---

## 3. Dataset Configuration & Governance

- **Dataset**: BirdCLEF 2024 bioacoustic dataset.
- **External Dataset Location**: `C:/Datasets/birdclef-2024`
- **Data Governance Policy**:
  - The raw dataset is strictly read-only and immutable.
  - Raw audio files must **NOT** be copied into this repository.
  - Raw audio files must **NOT** be modified.
  - Secondary copies of the dataset must **NOT** be downloaded.
  - All dataset references are resolved dynamically via [`configs/config.yaml`](file:///C:/projects/bird-biodiversity-research/configs/config.yaml).

---

## 4. Controlled Initial Baseline Parameters

The initial research configuration is established in [`configs/config.yaml`](file:///C:/projects/bird-biodiversity-research/configs/config.yaml) with the following parameters:

| Parameter | Baseline Value | Description |
| :--- | :--- | :--- |
| **Random Seed** | `42` | Deterministic seed for Python, NumPy, and PyTorch |
| **Sample Rate** | `32,000 Hz` | Target audio sampling rate (16 kHz Nyquist) |
| **Clip Duration** | `5.0 seconds` | Fixed duration of acoustic analysis window (160,000 samples) |
| **FFT Size (`n_fft`)** | `2048` | Short-Time Fourier Transform window length (~64 ms) |
| **Hop Length** | `512` | STFT hop length / frame stride (~16 ms, 75% overlap) |
| **Mel Filterbanks (`n_mels`)** | `128` | Number of Mel frequency bins |
| **fmin** | `50 Hz` | Minimum frequency cutoff (filters low-frequency rumble) |
| **fmax** | `16,000 Hz` | Maximum frequency cutoff (Nyquist upper bound) |

> **IMPORTANT RESEARCH NOTICE**:
> These parameters represent **initial research configuration values** providing a controlled, standardized baseline. They can only be modified through controlled ablation experiments where performance impacts are rigorously recorded and benchmarked against the baseline.

---

## 5. Environment Setup & Installation

### Prerequisites
- Python 3.10 or higher (Python 3.13 supported)
- Windows / Linux / macOS

### Installation Steps

1. Clone or navigate to the repository directory:
   ```bash
   cd C:/projects/bird-biodiversity-research
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Running Tests & Configuration Verification

Run the test suite using Python's standard `unittest` framework:

```bash
# From repository root:
python -m unittest discover tests -v
```

Or using `pytest`:

```bash
pytest tests/ -v
```

The test suite validates:
1. Integrity and syntax of [`configs/config.yaml`](file:///C:/projects/bird-biodiversity-research/configs/config.yaml).
2. Presence and validity of all initial baseline reproducibility parameters (`random_seed = 42`, `sample_rate = 32000`, etc.).
3. Correctness of the external dataset root path (`C:/Datasets/birdclef-2024`).
4. Determinism of the cross-platform seeding mechanism.
