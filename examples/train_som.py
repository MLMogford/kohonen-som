import logging
from pathlib import Path

from kohonen_som.conf import SOMConfig
from kohonen_som.data.loader import DataLoader
from kohonen_som.models.som import SelfOrganisingMap
from kohonen_som.visualisation.plotter import SOMVisuaisr


def main():
    # Load configuration
    config_path = Path("config/dev_config.yaml")
    config = SOMConfig.from_yaml(str(config_path))

    print("training SOM")

    # Set up logging
    logging.basicConfig(
        level=config.log_level, format=config.log_format, filename=config.log_file
    )
    logger = logging.getLogger(__name__)

    # Generate sample data
    data = DataLoader.generate_random_data(n_samples=100, config=config)

    # Initialise and train SOM
    som = SelfOrganisingMap(config=config)
    som.fit(data)

    # Create plots directory if it doesn't exist
    plots_dir = Path(config.plots_dir)
    plots_dir.mkdir(exist_ok=True)

    # Visualise results
    SOMVisuaisr.plot_som_grid(
        som.weights, save_path=str(plots_dir / "som_visualisation.png"), config=config
    )

    logger.info("Training and visualisation completed successfully")


if __name__ == "__main__":
    main()
