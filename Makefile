# Reproduce everything from a clean clone:  make setup && make all
# Override the interpreter if needed:      make train PYTHON=.venv/Scripts/python
PYTHON ?= python

T2_MODELS ?= majority logreg knn svm_rbf decision_tree random_forest lightgbm lit_logreg lit_mlp
T1_MODELS ?= mean ols ridge lasso decision_tree_reg lightgbm_reg lit_linear

.PHONY: setup features eda eda-target train train-t2 train-t1 artifacts eval figures score test test-notebooks lint all clean-outputs

setup:            ## install exact package versions (Python 3.12)
	$(PYTHON) -m pip install -r requirements-lock.txt

features:         ## data/raw -> data/processed/features.parquet (+ leakage checks)
	$(PYTHON) -m scripts.build_features

eda:              ## full EDA report -> reports/EDA_REPORT.md (+ eda_* figures and tables)
	$(PYTHON) -m scripts.make_eda_report

eda-target:       ## target / leakage / 2023+ shift tables -> reports/tables
	$(PYTHON) -m scripts.run_target_eda

train-t2:         ## Task 2 models on all three validation schemes, plus Test predictions
	$(PYTHON) -m scripts.train --task task2 --models $(T2_MODELS) --schemes fold_group fold_strat temporal --predict-test

train-t1:         ## Task 1 models on GroupKFold and temporal
	$(PYTHON) -m scripts.train --task task1 --models $(T1_MODELS) --schemes fold_group temporal

artifacts:        ## calibrated scorer bundle -> artifacts/
	$(PYTHON) -m scripts.train --save-artifacts

train: train-t2 train-t1 artifacts

eval:             ## metrics for every file in preds/ -> reports/tables
	$(PYTHON) -m scripts.evaluate

figures:          ## defence figure pack -> reports/figures + reports/FIGURES.md
	$(PYTHON) -m scripts.make_figures

score:            ## sample decisions + 1,000-call latency benchmark
	$(PYTHON) -m scripts.score

test:             ## unit + end-to-end tests on synthetic data (what CI runs)
	$(PYTHON) -m pytest

test-notebooks:   ## execute all 4 notebooks top to bottom (needs real data; slow)
	RUN_NOTEBOOKS=1 $(PYTHON) -m pytest tests/test_scripts_and_notebooks.py -k notebook

lint:
	$(PYTHON) -m ruff check src scripts tests

all: features train eval eda figures score test

clean-outputs:    ## delete generated predictions, artifacts and reports (data/ is kept)
	rm -f preds/*.parquet artifacts/*.joblib reports/figures/*.png reports/tables/*.csv reports/*.md
