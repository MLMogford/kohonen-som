from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class SOMConfig:
    """Configuration for Self-Organising Map."""

    # Model parameters
    width: int = 10
    height: int = 10
    input_dim: int = 3
    learning_rate: float = 0.1
    random_state: Optional[int] = 42

    # Training parameters
    n_iterations: int = 1000
    batch_size: Optional[int] = 32

    # Data parameters
    normais_data: bool = True
    validation_split: float = 0.2

    # Visualisation parameters
    figsize: tuple = (8, 8)
    cmap: str = "viridis"
    save_plots: bool = True
    plots_dir: str = "plots"

    # Logging parameters
    log_level: str = "INFO"
    log_file: Optional[str] = "som_training.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SOMConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters specific to model initialisation."""
        return {
            "width": self.width,
            "height": self.height,
            "input_dim": self.input_dim,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    def get_training_params(self) -> Dict[str, Any]:
        """Get parameters specific to model training."""
        return {
            "n_iterations": self.n_iterations,
            "batch_size": self.batch_size,
        }

    def get_visualisation_params(self) -> Dict[str, Any]:
        """Get parameters specific to visualisation."""
        return {
            "figsize": self.figsize,
            "cmap": self.cmap,
            "save_plots": self.save_plots,
            "plots_dir": self.plots_dir,
        }


# Default configuration
_default_config = None


def get_config(config: Optional[SOMConfig] = None) -> SOMConfig:
    """
    Get configuration, using default if none provided.

    Args:
        config: Optional configuration object

    Returns:
        SOMConfig: Configuration object to use
    """
    global _default_config
    if config is not None:
        return config

    if _default_config is None:
        _default_config = SOMConfig()

    return _default_config
