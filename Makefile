# Convenience targets; every target is one documented command (see docs/INSTALL.md,
# docs/RUN_METRICS.md). Run `make help` for the list. Activate the venv first or set
# PYTHON=.venv/bin/python.
PYTHON ?= python
BATCH  ?= 16

.PHONY: help install install-uv assets cache tune test metrics diagnostics demo clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:      ## editable install with pip (venv made by `python3 -m venv`)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e .

install-uv:   ## same with uv (venv made by `uv venv`; such venvs have no pip)
	uv pip install --python $(PYTHON) -e .

assets:       ## download pinned Flickr8k splits + SigLIP 2 checkpoint, write manifests
	$(PYTHON) scripts/setup_assets.py
	$(PYTHON) scripts/prepare_manifest.py

cache:        ## encode global + five-crop image embeddings for validation and test (once)
	$(PYTHON) scripts/build_image_cache.py --split all --batch-size $(BATCH)

tune:         ## select five-crop alpha and its threshold on validation only
	$(PYTHON) scripts/tune_alpha.py --batch-size 64

test:         ## unit tests (no model, no data needed)
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

metrics:      ## full metric pipeline for every method -> artifacts/reports/summary.md
	PYTHON=$(PYTHON) bash scripts/run_metrics.sh

diagnostics:  ## metrics + IVF nprobe sweep + index scaling study
	PYTHON=$(PYTHON) bash scripts/run_metrics.sh --diagnostics

demo:         ## Gradio demo on http://127.0.0.1:7860
	$(PYTHON) app.py

clean:        ## remove caches of the tooling (never data, models or artifacts)
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -rf src/*.egg-info .pytest_cache
