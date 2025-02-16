import numpy as np
import pytest

from kohonen_som.models.som import SelfOrganisingMap


def test_som_initialisation():
    width, height = 10, 10
    input_dim = 3
    som = SelfOrganisingMap(width, height, input_dim, random_state=42)

    assert som.weights.shape == (width, height, input_dim)
    assert som.x_coords.shape == (width, height)
    assert som.y_coords.shape == (width, height)


def test_som_fit():
    # Create test data
    input_data = np.random.random((100, 3))
    som = SelfOrganisingMap(5, 5, 3, random_state=42)

    # Test basic fitting
    fitted_som = som.fit(input_data, n_iterations=10)
    assert fitted_som is som  # Check if returns self

    # Test batch fitting
    som.fit(input_data, n_iterations=10, batch_size=32)

    # Test input validation
    with pytest.raises(ValueError):
        wrong_dim_data = np.random.random((100, 4))
        som.fit(wrong_dim_data, n_iterations=10)


def test_som_transform():
    input_data = np.random.random((100, 3))
    som = SelfOrganisingMap(5, 5, 3, random_state=42)
    som.fit(input_data, n_iterations=10)

    # Test transform
    coordinates = som.transform(input_data)
    assert coordinates.shape == (100, 2)
    assert np.all(coordinates[:, 0] < som.width)
    assert np.all(coordinates[:, 1] < som.height)
