import matplotlib.pyplot as plt
import numpy as np
import pytest

from kohonen_som.visualisation.plotter import SOMVisuaisr


def test_plot_som_grid(tmp_path):
    # Create test weights
    weights = np.random.random((10, 10, 3))

    # Test saving plot
    save_path = tmp_path / "test_plot.png"
    SOMVisuaisr.plot_som_grid(weights, save_path=str(save_path))
    assert save_path.exists()

    # Test showing plot (just ensure it doesn't raise an error)
    plt.close("all")  # Close any existing plots
    SOMVisuaisr.plot_som_grid(weights)
    plt.close("all")
