from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..conf import SOMConfig, get_config


class SOMVisuaisr:
    """Visualisation utilities for Self-Organising Maps."""

    @staticmethod
    def plot_som_grid(
        weights: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: Optional[str] = None,
        config: Optional[SOMConfig] = None,
    ) -> None:
        """
        Plot the SOM grid.

        Args:
            weights: Weight matrix from trained SOM
            save_path: Optional path to save the plot
            figsize: Figure size
            cmap: Color map for the plot
            config: Configuration object
        """
        config = get_config(config)
        figsize = figsize or config.figsize
        cmap = cmap or config.cmap

        plt.figure(figsize=figsize)
        plt.imshow(weights, cmap=cmap)
        plt.colorbar()

        if save_path and config.save_plots:
            plt.savefig(Path(save_path))
            plt.close()
        else:
            plt.show()
