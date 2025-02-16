import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return np.random.random((100, 3))


@pytest.fixture
def trained_som():
    """Create a pre-trained SOM for testing."""
    from kohonen_som.models.som import SelfOrganisingMap

    som = SelfOrganisingMap(5, 5, 3, random_state=42)
    data = np.random.random((100, 3))
    som.fit(data, n_iterations=10)
    return som
