# PromptShield/Makefile
#
# This file gives us clean terminal shortcuts for common project commands.
#
# Instead of typing long commands repeatedly, we can run:
#
#   make doctor
#   make test
#   make lint
#   make format
#   make data-check




# Use bash instead of the default sh shell.
SHELL := /bin/bash

# Python executable inside the active environment.
PYTHON := python

# Source and test directories.
SRC_DIR := src
TEST_DIR := tests
PACKAGE := promptshield

# Help

.PHONY: help
help:
	@echo ""
	@echo "PromptShield developer commands"
	@echo "--------------------------------"
	@echo "make doctor       Check Python version, package import, and key tools"
	@echo "make data-check   Verify required raw dataset files exist"
	@echo "make test         Run the test suite"
	@echo "make lint         Run Ruff lint checks"
	@echo "make format       Auto-format code with Ruff"
	@echo "make typecheck    Run mypy static type checks"
	@echo "make quality      Run lint, typecheck, and tests"
	@echo "make clean        Remove local caches and generated files"
	@echo ""


# Environment diagnostics

.PHONY: doctor
doctor:
	@echo ""
	@echo "Checking PromptShield environment..."
	@echo "------------------------------------"
	@$(PYTHON) --version
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'"
	@$(PYTHON) -c "import $(PACKAGE); print('Package import OK:', $(PACKAGE).__name__)"
	@$(PYTHON) -m pip --version
	@$(PYTHON) -c "import pandas, sklearn, fastapi, pydantic; print('Core dependencies OK')"
	@echo "Environment looks good."
	@echo ""


# Dataset checks

.PHONY: data-check
data-check:
	@echo ""
	@echo "Checking required raw datasets..."
	@echo "---------------------------------"
	@test -f data/raw/deepset_prompt_injections/train.parquet || \
		(echo "Missing: data/raw/deepset_prompt_injections/train.parquet" && exit 1)
	@test -f data/raw/deepset_prompt_injections/test.parquet || \
		(echo "Missing: data/raw/deepset_prompt_injections/test.parquet" && exit 1)
	@echo "Found deepset prompt injection dataset files."
	@ls -lh data/raw/deepset_prompt_injections/*.parquet
	@echo ""


# Code quality

.PHONY: lint
lint:
	@echo ""
	@echo "Running Ruff lint checks..."
	@echo "---------------------------"
	@ruff check $(SRC_DIR) scripts $(TEST_DIR)
	@echo ""


.PHONY: format
format:
	@echo ""
	@echo "Formatting code with Ruff..."
	@echo "----------------------------"
	@ruff format $(SRC_DIR) scripts $(TEST_DIR)
	@ruff check --fix $(SRC_DIR) scripts $(TEST_DIR)
	@echo ""


.PHONY: typecheck
typecheck:
	@echo ""
	@echo "Running mypy type checks..."
	@echo "---------------------------"
	@mypy $(SRC_DIR)
	@echo ""


# Testing

.PHONY: test
test:
	@echo ""
	@echo "Running tests..."
	@echo "----------------"
	@pytest
	@echo ""


.PHONY: quality
quality: lint typecheck test
	@echo ""
	@echo "All quality checks completed."
	@echo ""

# Cleanup

.PHONY: clean
clean:
	@echo ""
	@echo "Cleaning local caches..."
	@echo "------------------------"
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf htmlcov .coverage coverage.xml
	@echo "Clean complete."
	@echo ""