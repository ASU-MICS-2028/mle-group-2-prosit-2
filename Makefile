# Reproduce everything from a clean clone:  make setup && make all
# Override the interpreter if needed:      make train PYTHON=.venv/Scripts/python
PYTHON ?= python

T2_MODELS ?= majority logreg knn svm_rbf decision_tree random_forest lightgbm lit_logreg lit_mlp
T1_MODELS ?= mean ols ridge lasso decision_tree_reg lightgbm_reg lit_linear

.PHONY: setup features eda-target train train-t2 train-t1 artifacts eval figures score test lint all clean-outputs

setup:            ## install exact package versions (Python 3.12)
	$(PYTHON) -m pip install -r requirements-lock.txt

features:         ## data/raw -> data/processed/features.parquet (+ leakage checks)
	$(PYTHON) -m scripts.build_features

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

figures:          ## report figures -> reports/figures
	$(PYTHON) -m scripts.make_figures

score:            ## sample decisions + 1,000-call latency benchmark
	$(PYTHON) -m scripts.score

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests

all: features train eval figures score test

clean-outputs:    ## delete generated predictions, artifacts and reports (data/ is kept)
	rm -f preds/*.parquet artifacts/*.joblib reports/figures/*.png reports/tables/*.csv
