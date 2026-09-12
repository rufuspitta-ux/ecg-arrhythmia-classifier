# Security Policy

## Scope

This repository is a research/portfolio prototype for ECG signal exploration and heartbeat classification. It is **not a certified medical device** and is not approved for autonomous clinical decision-making.

## Reporting a Security Issue

Please do not publish sensitive security details in a public issue. Report a suspected vulnerability privately through the contact method listed in the repository profile.

When reporting an issue, include:
- A short description
- Affected file or component
- Reproduction steps, if safe to provide
- Potential impact
- Suggested mitigation, if known

## Security Considerations

### Pickle model files

The application loads local Python pickle artifacts. **Only load model/data files from trusted sources.** Pickle deserialization can execute arbitrary Python code.

### Local data

The application is designed for local exploration. Users should avoid placing identifiable patient information into repositories, screenshots, logs, sample files, or generated reports.

### Dependencies

Keep the Python environment isolated and review dependency updates before installing them. The repository separates the documented research scope from claims of regulatory or compliance certification.

## Medical Safety Boundary

Predictions and labels produced by this project are software outputs for research and demonstration. They must not be interpreted as a diagnosis, treatment recommendation, or emergency instruction.

Clinical deployment would require appropriate clinical validation, quality management, security controls, regulatory assessment, and qualified professional oversight.

## Responsible Disclosure

Please allow reasonable time for investigation and remediation before publicly disclosing a vulnerability.
