# Common tasks. Run `make help` for the list.
.PHONY: help install test notebooks notebooks-dry clean

PYTHON ?= python

help:
	@echo "install        install the dependencies"
	@echo "test           run the pytest suite"
	@echo "notebooks      rebuild and execute every notebook"
	@echo "notebooks-dry  assemble the notebooks without executing them"
	@echo "clean          remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

notebooks:
	$(PYTHON) tools/make_notebooks.py

notebooks-dry:
	$(PYTHON) tools/make_notebooks.py --dry-run

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
