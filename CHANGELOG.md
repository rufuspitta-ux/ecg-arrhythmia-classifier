# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Changed
- Clarified that classifier outputs are research/portfolio classifications, not clinical diagnoses.
- Removed unsupported clinical-grade, HIPAA-compliance, and validated-performance claims from the project history.
- Removed the unverified `<50ms` inference claim.
- Documented the current reproducibility boundary: the training pipeline and model artifacts are not included in the repository.

### Planned
- Publish a reproducible training/evaluation pipeline.
- Add patient-disjoint benchmark splits and per-class metrics.
- Version model artifacts and feature definitions.

## [1.0.0] - 2026-05-20

### Added
- Initial release of ECG Arrhythmia Classifier.
- Multi-class heartbeat classification prototype using MIT-BIH ECG data.
- ECG record exploration with WFDB.
- 14-feature time-domain signal analysis.
- Tkinter GUI with waveform visualization.
- Optional speech alerts and PDF summary generation.
- Automated tests for core feature extraction validation.

### Technical Scope
- **Primary ML stack:** scikit-learn
- **Python:** 3.8+
- **Dataset:** MIT-BIH Arrhythmia Database
- **Current status:** Research/portfolio prototype

> Historical project descriptions may have contained broader performance or compliance claims. Those claims are intentionally not treated as validated results unless supported by a reproducible experiment in the repository.

## Development Roadmap

- Reproducible training pipeline
- Patient-disjoint evaluation
- Per-class sensitivity, specificity, precision and recall
- Versioned model metadata/checksums
- Expanded preprocessing and feature tests
- Runtime/development dependency separation

---

**Maintained by Rufus Pitta**
