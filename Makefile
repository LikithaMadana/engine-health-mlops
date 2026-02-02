.PHONY: help install test lint clean train tune serve docker-build docker-run deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := engine-health-mlops
DOCKER_TAG := latest

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy

test: ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=120

format: ## Format code with black
	black src/ tests/ api/ kubeflow_pipeline/

clean: ## Clean build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf mlruns/ mlartifacts/ optuna_studies/

train: ## Train both models
	$(PYTHON) run_pipeline.py

tune: ## Run hyperparameter tuning
	$(PYTHON) run_tuning.py

select-champion: ## Select and register champion model
	$(PYTHON) select_champion.py

serve: ## Start inference API locally
	uvicorn api.inference:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui: ## Start MLflow UI
	mlflow ui --port 5000

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: ## Run Docker container
	docker run -p 8000:8000 -v $(PWD)/models:/app/models $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## Show docker-compose logs
	docker-compose logs -f

compile-pipeline: ## Compile Kubeflow pipeline
	$(PYTHON) kubeflow_pipeline/pipeline.py

deploy-k8s: ## Deploy to Kubernetes
	kubectl apply -f deployment/k8s-deployment.yaml

deploy-kserve: ## Deploy with KServe
	kubectl apply -f deployment/kserve-inference.yaml

test-api: ## Test API endpoints
	$(PYTHON) test_api.py

all: clean install lint test ## Run all checks

setup: ## Initial setup
	$(PIP) install -r requirements.txt
	mkdir -p models data logs
	@echo "Setup complete! Run 'make train' to start training."
