.PHONY: format lint test coverage clean

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run flake8 .

test:
	poetry run pytest tests/

coverage:
	poetry run pytest --cov=kohonen_som tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.png" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} + 