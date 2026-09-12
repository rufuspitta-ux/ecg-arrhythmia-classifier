# ECG Arrhythmia Classifier

A research-oriented Python/Tkinter application for exploring **heartbeat classification using the MIT-BIH Arrhythmia Database**.

The project combines ECG record loading with WFDB, beat-level feature extraction, a scikit-learn-compatible classifier, waveform visualization, optional speech alerts, confusion-matrix visualization, and PDF reporting.

> **Research / portfolio prototype:** This software is not a medical device and must not be used for autonomous diagnosis or clinical decision-making. Clinical use would require independent validation, quality management, regulatory review, and qualified clinical oversight.

## What it demonstrates

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
Waveform + optional alert + PDF summary
```

The current GUI implementation is in `ecg_arrhythmia_classifier.py`. A legacy alternate script, `ecg arthymines.py`, is also present.

## Supported labels

| Label | Meaning | Demo risk display |
|---|---|---|
| `N` | Normal beat | Low |
| `V` | Ventricular ectopic beat | Critical |
| `A` | Atrial premature beat | Moderate |
| `L` | Left bundle branch beat | High |
| `R` | Right bundle branch beat | High |

The risk wording is a **software demonstration label**, not a medical risk assessment.

## Dataset

The project uses the **MIT-BIH Arrhythmia Database** available through PhysioNet. The database itself is not redistributed in this repository.

Download the dataset from PhysioNet and place the required record files (`.dat`, `.hea`, `.atr`) in a local directory.

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

The application also requires local model/data artifacts:

```text
model/ecg_model.pkl
model/ecg_data.pkl
```

`ecg_data.pkl` is expected to contain `(X, y)`, while the classifier must provide a compatible `predict()` method.

### Security note

Only load pickle files from trusted sources. Python pickle deserialization can execute arbitrary code.

## Running the application

```bash
python ecg_arrhythmia_classifier.py
```

Optional environment variables can be used to point the application at custom data/model locations:

```bash
# Windows PowerShell
$env:MIT_BIH_PATH = "C:\\path\\to\\mit-bih"
$env:ECG_MODEL_PATH = "C:\\path\\to\\model"
python ecg_arrhythmia_classifier.py
```

The GUI supports record/beat exploration and actions such as **CLASSIFY**, **RANDOM**, **VOICE**, and **PDF REPORT**.

## Evaluation & reproducibility

The current repository does **not** include the original training pipeline or model artifacts. Therefore, previously reported accuracy figures should not be treated as independently reproducible from a clean clone.

A rigorous evaluation should include:

- Patient-disjoint train/validation/test splits
- Fixed random seeds
- Per-class precision, recall, sensitivity, and specificity
- Confusion matrix
- Confidence intervals where appropriate
- Explicit preprocessing and excluded-beat rules
- Exact feature definitions and dataset mapping
- Model version/checksum metadata

This distinction is intentional: a medical ML project should make it clear which results are reproducible and which are historical/project-level results.

## Testing

If the repository test suite is installed:

```bash
pytest -q
```

The feature extraction logic validates that a beat is finite numeric data with the expected 180-sample length.

## Known limitations

- Training/model artifacts are not currently committed
- The training pipeline is not yet packaged as a reproducible workflow
- Dependency versions are not completely locked
- Pickle is used for local model/data loading
- The GUI is intended for local exploration rather than production deployment
- PDF output is a software-generated summary, not a clinical report
- Classification labels and displayed risk wording are not validated clinical decisions

## Recommended next improvements

1. Publish the training and evaluation pipeline.
2. Add patient-disjoint benchmark splits.
3. Add a reproducible metrics report and saved confusion matrix.
4. Version model artifacts and feature definitions.
5. Expand automated tests for preprocessing and feature extraction.
6. Separate training/development dependencies from runtime dependencies.

## License

MIT License. See [LICENSE](LICENSE).
