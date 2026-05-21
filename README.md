<div align="center">

# 🫀 ECG Arrhythmia Classifier

### Real-time heartbeat classification powered by the MIT-BIH Arrhythmia Database

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen) ![Dataset](https://img.shields.io/badge/Dataset-MIT--BIH-orange)

**Built by [Rufus Pitta](https://github.com/rufuspitta-ux)**

*A desktop application that loads real clinical ECG data, classifies heartbeats in milliseconds using a trained ML model, renders live waveforms, speaks voice alerts for dangerous arrhythmias, and generates professional PDF reports.*

</div>

---

## 🎬 Demo
![ECG Demo](demo.gif)

---

## 🔬 Beat Classifications

| Symbol | Beat Type | Clinical Name | Risk Level | Action |
|--------|-----------|---------------|------------|--------|
| `N` | Normal Beat | Normal Sinus Rhythm (NSR) | 🟢 Low | Routine monitoring |
| `V` | Ventricular Ectopic | Premature Ventricular Contraction (PVC) | 🔴 Critical | Urgent cardiology referral |
| `A` | Atrial Premature | Premature Atrial Contraction (PAC) | 🟡 Moderate | Schedule consultation |
| `L` | Left Bundle Branch | Left Bundle Branch Block (LBBB) | 🟣 High | Full cardiac workup |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Live ECG Waveform** | Real MIT-BIH signal rendered with Matplotlib in dark theme |
| 🤖 **Real-time ML Classification** | Sklearn model with inference time shown in milliseconds |
| 🔊 **Voice Alerts** | Automatic spoken warning for dangerous PVC beats via `pyttsx3` |
| 📄 **PDF Clinical Reports** | One-click export with full stats, diagnosis, and recommendations |
| 🎲 **Random Beat Explorer** | Jump to any record and beat instantly for quick testing |
| ⚡ **Beat Caching** | Records loaded once and cached — no repeated disk reads |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 90%+ |
| Dataset | MIT-BIH (92,781 beats) |
| Training Split | 80/20 |
| Inference Time | <50ms per beat |

---

## 🗂 Project Structure

```
ecg-arrhythmia-classifier/
│
├── ecg_arrhythmia_classifier.py    # Main application — GUI + ML pipeline
├── requirements.txt                 # Python dependencies
├── README.md
├── .github/
│   └── topics.txt                   # Repository topics
│
└── model/
    ├── ecg_model.pkl                # Trained scikit-learn classifier
    └── ecg_data.pkl                 # Extracted beat features + labels
```

---

## ⚙️ Setup & Installation

**1. Clone the repo**

```bash
git clone https://github.com/rufuspitta-ux/ecg-arrhythmia-classifier.git
cd ecg-arrhythmia-classifier
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the MIT-BIH Arrhythmia Database**

Get it free from PhysioNet: https://physionet.org/content/mitdb/1.0.0/

**4. Configure your paths**

Set environment variables for data and model paths:

```bash
# Windows
set MIT_BIH_PATH=C:\path\to\mit-bih-arrhythmia-database-1.0.0
set ECG_MODEL_PATH=C:\path\to\model

# macOS/Linux
export MIT_BIH_PATH=/path/to/mit-bih-arrhythmia-database-1.0.0
export ECG_MODEL_PATH=/path/to/model
```

Alternatively, place data in `./data/` and model in `./model/` directories relative to the script.

**5. Run**

```bash
python ecg_arrhythmia_classifier.py
```

---

## 🧪 Feature Extraction

Each heartbeat is a 180-sample window centred on the R-peak. Fourteen time-domain features are extracted per beat:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Mean amplitude | Average signal level |
| 2 | Standard deviation | Signal variability |
| 3 | Maximum value | Peak amplitude |
| 4 | Minimum value | Trough amplitude |
| 5 | Range | Max − Min |
| 6 | Signal energy | Sum of squared samples |
| 7 | R-peak index | Sample index of highest point |
| 8 | Normalised R-peak position | R-peak index / 180 |
| 9 | Skewness | Asymmetry of amplitude distribution |
| 10 | Kurtosis | Sharpness / tail weight |
| 11 | Zero-crossing rate | Number of sign changes in signal |
| 12 | Energy ratio | Pre-R vs post-R energy balance |
| 13 | Median | Central amplitude value |
| 14 | IQR | Interquartile range (Q3 − Q1) |

---

## 📄 PDF Report

Click **PDF REPORT** after any classification to export a clinical-style report containing the record ID, beat classification, risk level, signal statistics, inference time, and AI clinical assessment with personalized recommendations.

Reports are saved as `ECG_Report_<record>_<YYYYMMDD_HHMMSS>.pdf` in the working directory.

---

## 📋 Requirements

```
numpy
matplotlib
wfdb
pyttsx3
reportlab
scikit-learn
scipy
seaborn
```

> `tkinter` is bundled with standard Python — no separate install needed.

---

## 📚 Dataset

**MIT-BIH Arrhythmia Database** — Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 2001;20(3):45–50.

Available at: https://physionet.org/content/mitdb/1.0.0/

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not a certified medical device and must not be used for clinical diagnosis or patient care. Always consult a qualified cardiologist for medical decisions. The accuracy of 90%+ reflects performance on the MIT-BIH database under controlled conditions and may vary with different ECG signals.

---

<div align="center">

Made with ❤️ and Python &nbsp;·&nbsp; MIT-BIH Arrhythmia Database &nbsp;·&nbsp; Accuracy 90%+

</div>
