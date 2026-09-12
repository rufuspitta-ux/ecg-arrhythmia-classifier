# ECG Arrhythmia Classifier

A research-oriented Python/Tkinter application for exploring **heartbeat classification with the MIT-BIH Arrhythmia Database**.

The project combines ECG record loading with WFDB, beat-level feature extraction, a scikit-learn-compatible classifier, waveform visualization, optional speech output, confusion-matrix generation, and PDF summaries.

> **Research / portfolio prototype:** this software is not a medical device and must not be used for autonomous diagnosis or clinical decision-making. Clinical use would require independent validation, quality management, regulatory review, and qualified clinical oversight.

## Why this project matters

This project sits at the intersection of **biomedical signal processing + machine learning + desktop healthcare software**. It demonstrates a complete prototype path from an ECG waveform to an interpretable software classification workflow.

```text
MIT-BIH ECG record
       ↓
Beat selection / 180-sample window
       ↓
14 time-domain features
       ↓
ML classifier
       ↓
Heartbeat class
       ↓
Waveform + optional speech + PDF summary
```

## What is implemented

- WFDB-based ECG record loading
- 180-sample beat windows
- 14 time-domain features
- Scikit-learn classifier integration
- Tkinter desktop GUI
- ECG waveform visualization
- Optional speech output
- Confusion-matrix plotting support
- PDF summary generation
- Core feature-validation tests with pytest

The main application is `ecg_arrhythmia_classifier.py`.

## Supported labels

| Label | Class description |
|---|---|
| `N` | Normal beat |
| `V` | Ventricular ectopic beat |
| `A` | Atrial premature beat |
| `L` | Left bundle branch beat |
| `R` | Right bundle branch beat |

These labels are **dataset/model classes**, not clinical risk assessments.

## Dataset

The project uses the **MIT-BIH Arrhythmia Database** available through PhysioNet. The dataset itself is not redistributed in this repository.

Download the required records from PhysioNet and keep them in a local directory containing the appropriate `.dat`, `.hea`, and `.atr` files.

## Requirements

Python 3.8+ is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

The application also expects local model/data artifacts:

```text
model/ecg_model.pkl
model/ecg_data.pkl
```

`ecg_data.pkl` is expected to contain `(X, y)`, and the classifier must expose a compatible `predict()` method.

### Security note

Only load pickle files from trusted sources. Python pickle deserialization can execute arbitrary code.

## Running the application

```bash
python ecg_arrhythmia_classifier.py
```

Optional Windows PowerShell configuration:

```powershell
$env:MIT_BIH_PATH = "C:\path\to\mit-bih"
$env:ECG_MODEL_PATH = "C:\path\to\model"
python ecg_arrhythmia_classifier.py
```

The GUI includes actions such as **CLASSIFY**, **RANDOM**, **VOICE**, and **PDF REPORT** for local experimentation.

## Evaluation and reproducibility

The current repository does **not** include the original training pipeline or model artifacts. Therefore, historical accuracy figures should not be presented as independently reproducible results from a clean clone.

A rigorous evaluation should include:

- Patient-disjoint train/validation/test splits
- Fixed random seeds
- Per-class precision and recall
- Sensitivity and specificity where appropriate
- Confusion matrix
- Explicit preprocessing and beat-selection rules
- Exact feature definitions and dataset mapping
- Versioned model metadata/checksums

This boundary is intentional: medical ML projects should distinguish reproducible evidence from historical development results.

## Testing

Run the automated tests with:

```bash
pytest -q
```

The current tests cover core feature extraction behavior, including expected input shape, finite numeric values, and documented class metadata.

## Repository structure

```text
ecg-arrhythmia-classifier/
├── ecg_arrhythmia_classifier.py  # Main GUI/application
├── ecg arthymines.py             # Legacy alternate script
├── test_ecg_core.py              # Core feature tests
├── requirements.txt              # Dependencies
├── CHANGELOG.md                  # Project history
├── SECURITY.md                   # Security/safety notes
└── LICENSE                       # MIT license
```

## Current limitations

- Training/model artifacts are not currently committed
- The training pipeline is not packaged as a reproducible end-to-end workflow
- Dependency versions are not completely locked
- Pickle is used for local model/data loading
- The GUI is intended for local exploration rather than production deployment
- PDF output is a software-generated summary, not a clinical report
- Model classes are not validated clinical decisions

## Next engineering steps

1. Publish the training/evaluation pipeline.
2. Add patient-disjoint benchmark splits.
3. Add a reproducible metrics report and saved confusion matrix.
4. Version model artifacts and feature definitions.
5. Expand automated tests around preprocessing and feature extraction.
6. Separate training/development dependencies from runtime dependencies.

## License

MIT License. See [LICENSE](LICENSE).
