.PHONY: help install test test-one run hooks-install

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Editable install with dev + cpu extras
	pip install -e .[dev,cpu]

test: ## Run the offline test suite (no GPU/dataset needed)
	pytest tests/ -q

test-one: ## Run tests matching an expression: make test-one k=stages
	pytest tests/ -q -k "$(k)"

run: ## Run the full pipeline (downloads the sample dataset on first run)
	datascience run

hooks-install: ## Install the pre-commit hooks (detect-secrets)
	pre-commit install
