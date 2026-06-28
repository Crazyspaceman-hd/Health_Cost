# Task runner for the Synthetic Health Cost Growth Target Analytics pipeline.
# On Windows without `make`, run the underlying `python -m ...` commands shown
# in each target directly.

PY ?= python

.PHONY: help install data validate db metrics visualize report pipeline dashboard all test clean

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  data      Generate synthetic source tables  -> data/raw/"
	@echo "  validate  Run data-quality validation        -> outputs/validation_*.csv"
	@echo "  db        Build the SQLite analytics database -> data/processed/"
	@echo "  metrics   Compute cost & utilization metrics  -> outputs/*_summary.csv"
	@echo "  visualize Render charts                        -> outputs/figures/*.png"
	@echo "  report    Generate the executive summary       -> reports/executive_summary.md"
	@echo "  pipeline  Run the entire workflow end to end in one process"
	@echo "  dashboard Launch the interactive Streamlit dashboard"
	@echo "  test      Run the pytest suite"
	@echo "  all       data + validate + metrics + visualize + report"
	@echo "  clean     Remove generated data and outputs"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.generate_data

validate: data
	$(PY) -m src.validate

db: data
	$(PY) -m src.load_db

metrics: data
	$(PY) -m src.metrics

visualize: data
	$(PY) -m src.visualize

report: data
	$(PY) -m src.report

# Run every stage in a single process (shared in-memory data).
pipeline:
	$(PY) -m src.pipeline

dashboard:
	$(PY) -m streamlit run src/dashboard.py

all: data validate metrics visualize report

test:
	$(PY) -m pytest -q

clean:
	$(PY) -c "import shutil,glob,os; [os.remove(f) for f in glob.glob('data/raw/*.csv')+glob.glob('outputs/*.csv')+glob.glob('outputs/figures/*.png') if os.path.exists(f)]; print('cleaned')"
