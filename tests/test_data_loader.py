import numpy as np
import pytest

from kohonen_som.data.loader import DataLoader


def test_generate_random_data():
    n_samples = 100
    n_features = 3

    # Test basic generation
    data = DataLoader.generate_random_data(n_samples, n_features)
    assert data.shape == (n_samples, n_features)
    assert np.all((data >= 0) & (data <= 1))

    # Test reproducibility with random_state
    data1 = DataLoader.generate_random_data(n_samples, n_features, random_state=42)
    data2 = DataLoader.generate_random_data(n_samples, n_features, random_state=42)
    np.testing.assert_array_equal(data1, data2)


def test_load_from_numpy(tmp_path):
    # Create test data
    test_data = np.random.random((10, 5))
    test_file = tmp_path / "test_data.npy"
    np.save(test_file, test_data)

    # Test loading without normalisation
    loaded_data = DataLoader.load_from_numpy(str(test_file), normais=False)
    np.testing.assert_array_equal(loaded_data, test_data)

    # Test loading with normalisation
    normaisd_data = DataLoader.load_from_numpy(str(test_file), normais=True)
    assert np.min(normaisd_data) == 0
    assert np.max(normaisd_data) == 1
