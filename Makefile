PYTHON ?= python

.PHONY: features eda-target train eval score test lint all

features:
	$(PYTHON) -m src.features

eda-target:
	$(PYTHON) -m src.eda_target

train:
	$(PYTHON) -m src.train --task task2 --models majority logreg decision_tree random_forest lightgbm --schemes fold_group fold_strat temporal --predict-test --save-artifacts
	$(PYTHON) -m src.train --task task1 --models mean ols ridge lasso decision_tree_reg lightgbm_reg --schemes fold_group temporal --save-artifacts

eval:
	$(PYTHON) -m src.evaluate

score:
	$(PYTHON) -m src.score

lint:
	$(PYTHON) -m ruff check src/ tests/

test:
	$(PYTHON) -m pytest tests/ -v

all: features train eval score test
