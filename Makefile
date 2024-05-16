.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install install-dev install-envpool install-envpool-dev check bump-version release format
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
ruff: ## run ruff as a formatter
	python -m ruff --exit-zero arlbench
	python -m ruff --silent --exit-zero --no-cache --fix arlbench
isort:
	python -m isort arlbench tests

test: ## run tests quickly with the default Python
	python -m pytest tests
cov-report:
	coverage html -d coverage_html

coverage: ## check code coverage quickly with the default Python
	coverage run --source arlbench -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/arlbench.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ arlbench
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html
bump-version: ## bump the version -- add current version number and kind of upgrade (minor, major, patch) as arguments
	bump-my-version bump --current-version

release: dist ## package and upload a release
	twine upload --repository testpypi dist/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uploaded distribution into"
	@echo "* Run the following:"
	@echo
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arlbench"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import arlbench'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload dist/*"

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist
install: clean ## install the package to the active Python's site-packages
	pip install -e .

install-dev: clean ## install with dev tools
	pip install -e ".[dev,examples]"

install-envpool: clean ## include envpool
	pip install -e ".[envpool]"

install-envpool-dev: clean ## include envpool and dev
	pip install -e ".[dev,examples,envpool]"

check:
	pre-commit run --all-files

format:
	make ruff
	make isort