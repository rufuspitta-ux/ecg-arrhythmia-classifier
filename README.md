# ECG Arrhythmia Classifier

A research-oriented Python/Tkinter application for exploring heartbeat classification on MIT-BIH Arrhythmia Database records. The current repository contains a GUI implementation in `ecg_arrhythmia_classifier.py` and a legacy alternate script named `ecg arthymines.py`.

> **Research prototype:** This software is not a medical device and must not be used for autonomous diagnosis or clinical decision-making. Clinical use would require independent validation, quality management, regulatory review, and physician oversight.

## Current implementation

The checked-in GUI loads a scikit-learn-compatible model and feature dataset from local pickle files, reads MIT-BIH records with WFDB, extracts 14 time-domain features from 180-sample beat windows, displays a waveform, generates a confusion matrix, provides optional speech alerts, and creates a PDF summary.

The supported labels in the implementation are:

| Label | Meaning | Risk display |
|---|---|---|
| `N` | Normal beat | Low |
| `V` | Ventricular ectopic beat | Critical |
| `A` | Atrial premature beat | Moderate |
| `L` | Left bundle branch beat | High |
| `R` | Right bundle branch beat | High |

The model and dataset artifacts are not committed to the repository. You must provide them locally under `model/`, or set the `MIT_BIH_PATH` and `ECG_MODEL_PATH` environment variables to the appropriate directories.

## Requirements

Use Python 3.8 or newer. Install the declared dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The application also requires a local MIT-BIH record set and two model artifacts:

```text
model/ecg_model.pkl
model/ecg_data.pkl
```

`ecg_data.pkl` must contain a tuple `(X, y)`, and the model must expose a compatible `predict` method. Pickle files should only be loaded when their provenance is trusted because pickle deserialization can execute code.

## Running the GUI

Download the MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) and place record files in a local data directory. Then run:

```bash
python ecg_arrhythmia_classifier.py
```

To use custom locations:

```bash
export MIT_BIH_PATH=/path/to/mit-bih
export ECG_MODEL_PATH=/path/to/model
python ecg_arrhythmia_classifier.py
```

Select a record and beat number, then use **CLASSIFY**, **RANDOM**, **VOICE**, or **PDF REPORT**. The application expects record files such as `100.dat`, `100.hea`, and `100.atr` in the configured data directory.

## Evaluation and reproducibility

The repository currently does not contain the training pipeline or model artifacts. Therefore, the advertised accuracy cannot be reproduced from a clean clone alone. Any serious evaluation should use patient-disjoint train/validation/test partitions, fixed random seeds, per-class sensitivity and specificity, confidence intervals, and an explicit account of excluded beats and preprocessing.

The current feature extractor validates that each beat is a finite numeric array of exactly 180 samples. Lightweight regression tests can be run with:

```bash
pytest -q
```

## Known limitations

The model artifact is loaded with pickle, dependency versions are not fully locked, and the GUI is designed for local demonstration rather than service deployment. The PDF report is a software-generated summary, not a clinical report. Accuracy and latency claims should be treated as project targets until supported by a reproducible evaluation artifact.

## Development priorities

The next priorities are to publish the training and evaluation pipeline, document the exact dataset mapping, add patient-disjoint validation, provide model checksums and version metadata, expand test coverage, and separate runtime dependencies from training and development dependencies.

## License

MIT License. See [LICENSE](LICENSE).
