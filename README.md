# Spectrum Rhapsody

*A personal project by Ankita S - PES1UG24CS066 ; BTech CSE Student @ PES University *

## Overview

Spectrum Rhapsody is an AI-powered system that analyzes speech patterns — such as tone, pitch, and pauses — to assist in the early detection of Autism Spectrum Disorder (ASD). The project combines machine learning and audio processing techniques to identify vocal biomarkers that may indicate early signs of ASD, with the goal of supporting timely screening and intervention.

## Motivation

Early detection of ASD can significantly improve outcomes through timely intervention. Speech and vocal patterns — including intonation, rhythm, pitch variability, and pause frequency — have been linked to early markers of ASD in research literature. Spectrum Rhapsody explores how these acoustic features can be captured and analyzed computationally to support (not replace) clinical screening processes.

## Features

- **Audio Feature Extraction** — Extracts key vocal characteristics such as pitch, tone, pause duration, and speech rhythm from audio recordings.
- **Machine Learning Analysis** — Applies ML models to detect patterns in speech that may correlate with early ASD indicators.
- **Signal Processing Pipeline** — Preprocesses raw audio (noise reduction, normalization, segmentation) before feature extraction.
- **Result Interpretation** — Presents analysis outputs in a way intended to assist, not diagnose.

## Tech Stack

> _This is a suggested stack for a project of this kind — adjust to match your actual implementation._

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Audio Processing** | Librosa, PyAudio, SoundFile, Praat-Parselmouth (pitch/tone extraction) |
| **Signal Processing** | SciPy, NumPy |
| **Machine Learning** | scikit-learn (classical models), TensorFlow / PyTorch (deep learning models) |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Experiment Tracking** | MLflow / Weights & Biases (optional) |
| **API / Interface** | FastAPI or Flask (if exposing a web service) |
| **Notebook Environment** | Jupyter Notebook |
| **Version Control** | Git, GitHub |
| **Environment Management** | venv / conda, `requirements.txt` |

## Folder Structure

```
spectrum-rhapsody/
│
├── data/
│   ├── raw/                # Original, unprocessed audio recordings
│   ├── interim/            # Intermediate/cleaned audio
│   └── processed/          # Final feature-extracted datasets
│
├── notebooks/
│   ├── 01_exploration.ipynb        # Data exploration and audio visualization
│   ├── 02_feature_extraction.ipynb # Pitch, tone, pause feature engineering
│   └── 03_model_training.ipynb     # Model experimentation
│
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Noise reduction, normalization, segmentation
│   │   └── feature_extraction.py  # Pitch, tone, pause, rhythm extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Model training pipeline
│   │   ├── predict.py         # Inference on new audio samples
│   │   └── evaluate.py        # Metrics and evaluation
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── run_pipeline.py    # End-to-end pipeline orchestration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration and constants
│       └── logger.py          # Logging utilities
│
├── models/
│   └── saved_models/          # Trained model checkpoints/artifacts
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_models.py
│
├── configs/
│   └── config.yaml            # Hyperparameters, paths, and settings
│
├── app/                        # Optional web/API interface
│   ├── main.py
│   └── templates/
│
├── docs/
│   └── methodology.md          # Notes on approach, features used, references
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

### Key Directory Notes

- **`data/`** — Kept out of version control (see `.gitignore`); raw audio should never be committed, especially given the sensitivity of the data involved.
- **`src/audio/`** — Core signal processing: converts raw `.wav`/`.mp3` recordings into cleaned, analyzable audio.
- **`src/models/`** — Houses training, inference, and evaluation logic, kept separate from feature extraction for modularity.
- **`configs/`** — Centralizes tunable parameters (sample rate, window size, model hyperparameters) so experiments stay reproducible.
- **`app/`** — Optional layer if the project exposes results through a web interface or API rather than notebooks alone.

## Project Status

🚧 This project is under active development as a personal research/learning initiative.

## Disclaimer

Spectrum Rhapsody is intended as an assistive research and exploratory tool only. It is **not** a diagnostic tool and should not be used as a substitute for professional medical or clinical evaluation. Any concerns about ASD should be discussed with a qualified healthcare provider.

## Getting Started

> _Add installation and usage instructions here once the implementation is ready, e.g.:_

```bash
git clone <repo-url>
cd spectrum-rhapsody
pip install -r requirements.txt
```

## Author

**Ankita S**
*Personal Project*

## License

> _Add your chosen license here (e.g. MIT, Apache 2.0)._
