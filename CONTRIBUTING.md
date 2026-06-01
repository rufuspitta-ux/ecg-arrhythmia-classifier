# Contributing to ECG Arrhythmia Classifier

Thank you for your interest in contributing to this project! We welcome contributions from the community to help improve the ECG Arrhythmia Classifier.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### 1. Report Bugs
- Check existing [issues](https://github.com/rufuspitta-ux/ecg-arrhythmia-classifier/issues) first
- Provide clear description with steps to reproduce
- Include Python version, OS, and relevant error messages
- Add sample data if possible

### 2. Suggest Enhancements
- Clearly describe the enhancement and expected behavior
- Provide use cases and benefits
- Link to relevant research or references
- Check if similar requests exist

### 3. Submit Code Changes

#### Setup Development Environment
```bash
git clone https://github.com/rufuspitta-ux/ecg-arrhythmia-classifier.git
cd ecg-arrhythmia-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"  # Install dev dependencies
```

#### Make Your Changes
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes following code standards (see below)
3. Add/update tests for your changes
4. Update documentation if needed
5. Run tests: `pytest tests/ -v`

#### Code Standards
- **Style:** PEP 8 compliant
- **Type Hints:** Use type annotations where possible
- **Docstrings:** Include docstrings for functions and classes
- **Testing:** Aim for >80% code coverage
- **Naming:** Clear, descriptive variable/function names

#### Example Commit Message
```
feat: add ensemble model support

- Implement voting classifier with multiple models
- Add configuration for model weights
- Update documentation with ensemble usage
- Add tests for ensemble predictions
```

#### Pull Request Process
1. Push your branch to GitHub
2. Create a Pull Request with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Any breaking changes noted
3. Address review feedback
4. Ensure CI/CD checks pass

## Development Workflow

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# With coverage report
pytest --cov=src --cov-report=html tests/
```

### Code Quality Checks
```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

### Building Documentation
```bash
cd docs/
sphinx-build -b html . _build/
```

## Areas for Contribution

We're actively seeking contributions in:

- [ ] **LSTM Architecture:** Implement LSTM models for temporal pattern recognition
- [ ] **Ensemble Methods:** Implement voting or stacking ensembles
- [ ] **Real-time Streaming:** Add support for continuous ECG stream processing
- [ ] **Web Interface:** Create a web dashboard using Flask/FastAPI
- [ ] **Mobile Integration:** Develop iOS/Android applications
- [ ] **Multi-lead Support:** Extend to handle 12-lead ECG data
- [ ] **Performance Optimization:** GPU acceleration, model pruning
- [ ] **International Languages:** Add support for multiple languages in reports
- [ ] **Cloud Deployment:** Templates for AWS/GCP/Azure deployment
- [ ] **Documentation:** Improve guides, tutorials, and API docs

## Reporting Security Issues

⚠️ **Do not open public issues for security vulnerabilities.**

Please email security concerns to: rufuspitta@gmail.com

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions or Need Help?

- 📧 Email: rufuspitta@gmail.com
- 💬 Discussions: [GitHub Discussions](https://github.com/rufuspitta-ux/ecg-arrhythmia-classifier/discussions)
- 📚 Documentation: Check [ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

**Thank you for making this project better!** ❤️
