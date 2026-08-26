import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name('ecg_arrhythmia_classifier.py')
spec = importlib.util.spec_from_file_location('ecg_classifier', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extract_features_shape_and_finiteness():
    beat = np.sin(np.linspace(0, 4 * np.pi, 180)).reshape(1, -1)
    features = module.extract_features(beat)
    assert features.shape == (1, 14)
    assert np.isfinite(features).all()


def test_extract_features_rejects_wrong_length():
    with pytest.raises(ValueError, match='shape'):
        module.extract_features(np.zeros((1, 179)))


def test_extract_features_rejects_non_finite_values():
    beat = np.zeros((1, 180))
    beat[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        module.extract_features(beat)


def test_all_documented_classes_have_metadata():
    assert set('NVALR').issubset(module.LABEL_INFO)
