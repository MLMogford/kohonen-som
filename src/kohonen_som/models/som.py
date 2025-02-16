import logging
from typing import Optional, Tuple, Union

import numpy as np

from ..conf import SOMConfig, get_config

logger = logging.getLogger(__name__)


class SelfOrganisingMap:
    """
    A Self-Organising Map (SOM) implementation using the Kohonen algorithm.

    Attributes:
        width (int): Width of the SOM grid
        height (int): Height of the SOM grid
        input_dim (int): Dimensionality of input data
        learning_rate (float): Initial learning rate
        sigma (float): Initial neighborhood radius
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        input_dim: Optional[int] = None,
        learning_rate: Optional[float] = None,
        random_state: Optional[int] = None,
        config: Optional[SOMConfig] = None,
    ):
        self.config = get_config(config)
        # Use provided params or fall back to config
        self.width = width or self.config.width
        self.height = height or self.config.height
        self.input_dim = input_dim or self.config.input_dim
        self.learning_rate = learning_rate or self.config.learning_rate
        self.random_state = random_state or self.config.random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = np.random.random((self.width, self.height, self.input_dim))
        self._initiais_grid()

    def _initiais_grid(self) -> None:
        """Initiais the coordinate matrices for the SOM grid."""
        self.x_coords, self.y_coords = np.meshgrid(
            np.arange(self.width), np.arange(self.height), indexing="ij"
        )

    def fit(
        self,
        input_data: np.ndarray,
        n_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> "SelfOrganisingMap":
        """
        Train the SOM using the input data.

        Args:
            input_data: Training data of shape (n_samples, input_dim)
            n_iterations: Number of training iterations
            batch_size: Optional batch size for mini-batch training

        Returns:
            self: Trained SOM instance
        """
        if input_data.shape[1] != self.input_dim:
            raise ValueError(
                f"Input data dimension ({input_data.shape[1]}) "
                f"does not match expected dimension ({self.input_dim})"
            )

        # Use provided params or fall back to config
        n_iterations = n_iterations or self.config.n_iterations
        batch_size = batch_size or self.config.batch_size

        sigma_0 = max(self.width, self.height) / 2
        lambda_param = n_iterations / np.log(sigma_0)

        logger.info(f"Starting training for {n_iterations} iterations")

        for t in range(n_iterations):
            # Calculate learning parameters
            sigma_t = sigma_0 * np.exp(-t / lambda_param)
            alpha_t = self.learning_rate * np.exp(-t / lambda_param)
            two_sigma_sq = 2 * (sigma_t**2)

            # Mini-batch training
            if batch_size:
                batch_indices = np.random.choice(
                    len(input_data), size=batch_size, replace=False
                )
                batch_data = input_data[batch_indices]
            else:
                batch_data = input_data

            self._update_weights(batch_data, alpha_t, two_sigma_sq)

            if (t + 1) % 100 == 0:
                logger.info(f"Completed iteration {t + 1}")

        return self

    def _update_weights(
        self, data: np.ndarray, alpha_t: float, two_sigma_sq: float
    ) -> None:
        """Update weights for a batch of input data."""
        for vt in data:
            # Find BMU
            diff = self.weights - vt
            distances = np.sum(diff**2, axis=2)
            bmu_x, bmu_y = np.unravel_index(
                np.argmin(distances), (self.width, self.height)
            )

            # Calculate neighborhood influence
            di = np.sqrt((self.x_coords - bmu_x) ** 2 + (self.y_coords - bmu_y) ** 2)
            theta_t = np.exp(-(di**2) / two_sigma_sq)

            # Update weights
            theta_t = theta_t[:, :, np.newaxis]
            self.weights += alpha_t * theta_t * (vt - self.weights)

    def transform(self, input_data: np.ndarray) -> np.ndarray:
        """
        Transform input data to grid coordinates.

        Args:
            input_data: Data to transform of shape (n_samples, input_dim)

        Returns:
            np.ndarray: Grid coordinates for each input sample
        """
        coordinates = np.zeros((len(input_data), 2))
        for i, sample in enumerate(input_data):
            diff = self.weights - sample
            distances = np.sum(diff**2, axis=2)
            coordinates[i] = np.unravel_index(
                np.argmin(distances), (self.width, self.height)
            )
        return coordinates


som = SelfOrganisingMap(width=10, height=10, input_dim=3)
