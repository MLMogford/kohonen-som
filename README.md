# Kohonen Self-Organising Map (SOM)

A Python implementation of the Kohonen Self-Organising Map algorithm with a focus on conversion to production-ready code.

# Optimisation Summary

The notebook provided has had the following optimisations and improvements applied to it.

Notebook-specific improvements are:

    1. Pre-calculated coordinate matrices instead of nested loops
    2. Vectorised operations instead of element-wise calculations
    3. Reduced redundant calculations by storing intermediate results
    4. Added broadcasting for weight updates
    5. PEP 8 standard variable names
    
    These improvements significantly reduce computation time by:
    - Eliminating nested loops which are slow in Python
    - Leveraging NumPy's efficient array operations
    - Minimising repeated calculations
    - Using memory more efficiently through broadcasting

Further to this, the code has been modified to prepare it for production.


- **Modular Structure:** Separated concerns into data, models, processing, and visualisation modules
- **Type Hints:** Added type annotations for better code maintainability
- **Logging:** Incorporated basic logging
- **Configuration:** Used Poetry for dependency management and yaml for config definitions
- **Documentation:** Added docstrings and comments
- **Error Handling:** Added input validation
- **Batch Processing:** Added mini-batch training option
- **Visualisation:** Separated visualisation logic
- **Testing Structure:** Added tests directory for unit tests
- **Flexibility:** Made the SOM implementation more configurable


--------------------------------

# Using this Package

## Poetry environment management
### Ensure poetry is installed
```bash
poetry --version
```

### Install Poetry if not installed (Mac via homebrew)

```bash
brew install poetry
```


### Setting Up the Project

1. Clone the repository:
```bash
git clone https://github.com/mlmogford/kohonen-som.git
cd kohonen-som
```

### Run poetry commands in the project directory

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry env activate
```

## Usage

1. run example training script using poetry

```bash
poetry run python examples/train_som.py
```


2. Run the test suite:
```bash
poetry run pytest
```

3. Run tests with coverage:
```bash
poetry run pytest --cov=kohonen_som
```

4. Run specific test files:
```bash
poetry run pytest tests/test_som.py
```

A Makeflie has been added to help automte cleanup, but tests etc tasks run locally or in CI pipelines.

### Using makefile

```bash
make format
make lint
make test
make coverage
make clean
``` 

## Project Structure

```
kohonen-som/
├── pyproject.toml                  # Project configuration and dependencies
├── README.md                       # Project documentation
├── Makefile                       
├── kohonen_notebook/               # Improved notebook
├── config/                         # Deployment-sepcific configs
├── src/
│   └── kohonen_som/                # Main package directory
│       ├── __init__.py
│       ├── data/                   # Data loading and preprocessing
│       ├── models/                 # SOM implementation
│       ├── processing/             # Data processing utilities
│       └── visualisation/          # Plotting and visualisation
└── tests/                          # Test directory
    ├── __init__.py
    ├── conftest.py                 # Test configurations and fixtures
    ├── test_data_loader.py
    ├── test_som.py
    └── test_visualisation.py
```


## Future Work:
**Improved Experimentation Set-Up**
- Adding performance metrics
- model tracking with ML Flow (or WandB/cloud native service)
    - model check pointing 
    - metric tracking

**Code**
- Add and improve unit tests
- Adding and improve data validation tests
- Adding parallel processing for large datasets

**Deployment**
- Implement CI/CD pipelines
- Containerisation and integration with cloud container repository and service
- Implement model serialisation, versioning with cloud ML service
- Git development strategy
    - Dev, Test and Production branches with cloud environments managed by CI/CD pipelines
    - Feature branches, independent code reviews, CI pipeline based automated unit testing


--------------------------------
<!-- 


    - Git Ops
        1. Create a feature branch
        2. Make changes
        3. Run tests
        4. Formatting and linting
        5. Submit a pull request
        6. Code review and merge to Dev
        7. Functional testing in Test environment
        8. Deploy to Production envitonment


# Kohonen Self-Organising Map (SOM)

A Python implementation of the Kohonen Self-Organising Map algorithm with a focus on production-ready code.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Poetry (Python package manager)

### Installing Poetry

If you don't have Poetry installed, you can install it using:

install homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```bash
brew install poetry
``` -->

<!-- For Windows, use PowerShell:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
``` -->

<!-- ### Setting Up the Project

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kohonen-som.git
cd kohonen-som
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry env activate
```

## Usage -->

<!-- ### Basic Example

```python
from kohonen_som.data.loader import DataLoader
from kohonen_som.models.som import SelfOrganisingMap
from kohonen_som.visualisation.plotter import SOMVisuaisr

# Generate sample data
data = DataLoader.generate_random_data(
    n_samples=100,
    n_features=3,
    random_state=42
)

# Initiais and train SOM
som = SelfOrganisingMap(
    width=10,
    height=10,
    input_dim=3,
    random_state=42
)

# Train the model
som.fit(data, n_iterations=1000, batch_size=32)

# Visuais results
SOMVisuaisr.plot_som_grid(
    som.weights,
    save_path="som_visualisation.png"
)
```

### Loading Your Own Data

```python
# Load data from numpy file
data = DataLoader.load_from_numpy(
    "path/to/your/data.npy",
    normais=True
)
``` -->

<!-- ## Running Tests

1. Ensure you're in the project directory and the virtual environment is activated:
```bash
cd kohonen-som
poetry env activate
``` -->
<!-- 
2. Run the test suite:
```bash
poetry run pytest
```

3. Run tests with coverage:
```bash
poetry run pytest --cov=kohonen_som
```

4. Run specific test files:
```bash
poetry run pytest tests/test_som.py
``` -->

<!-- Now you can run the following commands:
Run tests:
pytest
Run tests with coverage:
cov
Format code:
.
Sort imports:
.
Run linting:
flake8
The updates include:
Added pytest-cov for coverage reporting
Configured pytest settings
Configured Black formatter settings
Configured isort settings
Added coverage configuration
Added script shortcuts
The configuration:
Sets Python 3.8 as minimum version
Configures test paths and patterns
Sets up coverage reporting
Aligns Black and isort configurations
Excludes appropriate files from coverage
Sets up consistent code style rules -->


<!-- using makefile

```bash
make format
make lint
make test
make coverage
make clean
```  -->


<!-- ## Project Structure

```
kohonen-som/
├── pyproject.toml          # Project configuration and dependencies
├── README.md              # Project documentation
├── src/
│   └── kohonen_som/      # Main package directory
│       ├── __init__.py
│       ├── data/         # Data loading and preprocessing
│       ├── models/       # SOM implementation
│       ├── processing/   # Data processing utilities
│       └── visualisation/# Plotting and visualisation
└── tests/                # Test directory
    ├── __init__.py
    ├── conftest.py      # Test configurations and fixtures
    ├── test_data_loader.py
    ├── test_som.py
    └── test_visualisation.py
``` -->

<!-- ## Development

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

To format code:
```bash
poetry run black .
poetry run isort .
poetry run flake8 .
``` -->

<!-- ### Adding Dependencies

To add a new package:
```bash
poetry add package-name
``` -->

<!-- For development dependencies:
```bash
poetry add --group dev package-name
```

## Troubleshooting

Common issues and solutions:

1. Poetry installation fails:
   - Ensure Python 3.8+ is installed
   - Try installing Poetry with pip: `pip install poetry`

2. Package import errors:
   - Verify virtual environment is activated: `poetry shell`
   - Check installation: `poetry install`

3. Tests failing:
   - Ensure all dependencies are installed: `poetry install`
   - Check Python version compatibility
   - Verify test data paths

## Contributing -->

<!-- 1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request -->

<!-- ## License

This project is licensed under the MIT License - see the LICENSE file for details. -->
<!-- ```

This documentation provides:
1. Clear setup instructions
2. Usage examples
3. Test running instructions
4. Project structure overview
5. Development guidelines
6. Troubleshooting tips
7. Contributing guidelines

Users can follow these instructions to get the package up and running on their local machine, whether they're using it for development or production purposes.


```bash
poetry run pytest
```



These tests cover the basic functionality of each module:


These tests cover the basic functionality of each module:
DataLoader Tests:
Test random data generation
Test data loading and normalisation
Test reproducibility with random state
SOM Tests:
Test initialisation
Test fitting
Test batch processing
Test input validation
Test transformation
Visualisation Tests:
Test plot generation
Test plot saving
The tests include:
Basic functionality testing
Edge case handling
Input validation
Fixtures for common test data
File I/O testing
Shape and value range assertions
You can expand these tests by adding:
More edge cases
Performance tests
Integration tests
More comprehensive input validation
Memory usage tests
Convergence tests
Stress tests with large datasets






config

This configuration setup provides several benefits:
Centraisd Configuration: All parameters are in one place
Flexibility: Easy to modify parameters without changing code
Version Control: Configuration can be version controlled
Multiple Environments: Can have different configs for development/production
Type Safety: Using dataclass ensures type checking
Modularity: Parameters are grouped by functionality
Serialisation: Easy to save/load configurations
Default Values: Sensible defaults provided
Documentation: Parameters are self-documenting
Validation: Can add parameter validation if needed
To use different configurations:


config = SOMConfig.from_yaml("config/production_config.yaml") -->