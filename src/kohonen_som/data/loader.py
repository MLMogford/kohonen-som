from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..conf import SOMConfig, get_config


class DataLoader:
    """Handles data loading and basic preprocessing for SOM."""

    @staticmethod
    def load_from_numpy(
        path: str, normais: Optional[bool] = None, config: Optional[SOMConfig] = None
    ) -> np.ndarray:
        """
        Load data from a numpy file.

        Args:
            path: Path to the numpy file
            normais: Whether to normais the data
            config: SOM configuration

        Returns:
            np.ndarray: Loaded and optionally normaisd data
        """
        config = get_config(config)
        normais = normais if normais is not None else config.normais_data
        data = np.load(Path(path))
        if normais:
            data = (data - data.min()) / (data.max() - data.min())
        return data

    @staticmethod
    def generate_random_data(
        n_samples: int,
        n_features: Optional[int] = None,
        random_state: Optional[int] = None,
        config: Optional[SOMConfig] = None,
    ) -> np.ndarray:
        """Generate random data for testing/demo purposes."""
        config = get_config(config)
        n_features = n_features or config.input_dim
        random_state = random_state or config.random_state
        if random_state is not None:
            np.random.seed(random_state)
        return np.random.random((n_samples, n_features))
