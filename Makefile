PYTHON ?= python
CONFIG ?= config.yaml
PIPELINE = scripts/run_pipeline.py

.PHONY: help install plan core core-force metadata preprocess downstream validation supplemental external clean-reports status

help:
	@echo "UCEC methylation biomarker pipeline"
	@echo ""
	@echo "Common commands:"
	@echo "  make install       Install Python dependencies with pip"
	@echo "  make plan          Preview the core pipeline without running it"
	@echo "  make core          Run the standard 01-08 reproducible workflow"
	@echo "  make metadata      Query metadata/clinical info without downloading beta files"
	@echo "  make preprocess    Run preprocessing only"
	@echo "  make downstream    Run differential analysis through final evaluation, assuming clean data exist"
	@echo "  make validation    Run supplemental panel/nested-CV/external-validation scripts"
	@echo "  make external      Run GSE155760 external validation only"
	@echo "  make status        Show git status and key output folders"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=/path/to/python   Python interpreter to use"
	@echo "  CONFIG=config.yaml       Pipeline config file"

install:
	$(PYTHON) -m pip install -r requirements.txt

plan:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --dry-run

core:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages core

core-force:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages core --force

metadata:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages download --skip-download --force

preprocess:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages preprocess --force

downstream:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages core --from differential

validation:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages validation

supplemental:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages supplemental

external:
	$(PYTHON) $(PIPELINE) --config $(CONFIG) --stages external_validation

clean-reports:
	@find outputs/reports -name "pipeline_run_report.json" -print

status:
	@git status --short --branch
	@find data/processed data/results outputs/tables outputs/figures outputs/reports -maxdepth 2 -type f 2>/dev/null | sed -n '1,80p'

