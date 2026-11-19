# P3 · Bernard (`adjeibernardmarfo`): Regression, Scaffolding & Deployment

**Mission:** build the Task 1 loan-amount regression pipeline, maintain repo scaffolding & CI guardrails as PM, and ship the Part E scoring service with plain-language reason codes and latency benchmarks.
**You own:** `src/config.py` schemas, `src/processing/preprocess.py`, `src/serving/`, `tests/`, CI/pre-commit/Makefile tooling, `03_regression.ipynb`, regression & deployment sections of the report, linear regression derivation. **Trains:** OLS, Ridge, Lasso, Decision Tree (and LightGBM/Lightning linear cross-check). **LOs:** LO2, LO3, LO4, LO6 (see [LEARNING_OUTCOMES](LEARNING_OUTCOMES.md)).

## Steps

**0 · Scaffolding & Guardrails (PM)**
- [x] Shared schemas (`NUM_FEATURES_T1`, `NUM_FEATURES_T2`, `PRED_COLUMNS`), `src/processing/preprocess.py`, `src/models/lightning_modules.py`, `src/training/cross_validation.py`, `src/evaluation/`, `src/evaluation/fairness.py`, `src/serving/`.
- [x] Automated `pytest` guardrails (`tests/`), `ci.yml`, `pr-issue-check.yml`, `.pre-commit-config.yaml`, `Makefile`.
- [x] Restructure into packages (`src/models/` one file per family, `training/`, `evaluation/`, `serving/`, `visualization/`), `scripts/`, `configs/`, `reports/`; restore Henry's `02_classification` (D-20).
- [x] Python 3.12 everywhere: `.python-version`, cross-platform `requirements-lock.txt`, CI installs the lock.

**1 · Explore amounts, durations, and leak suspects (#7)**
- [ ] Profile `Total_Amount`, `duration`, and `Lender_portion_Funded` skew in `03_regression.ipynb` (with charts).
- [ ] Document findings in `docs/WORKLOG.md` and confirm leak rulings in Part A session (#9).

**2 · Task 1 Models & Diagnostics (#17, #21, #22)**
- [ ] Train the pruned decision tree for Task 2 (`decision_tree`, the P3 classifier in the brief's model table) and compare it in `03_regression.ipynb`.
- [ ] Run Task 1 models (`mean`, `ols`, `ridge`, `lasso`, `decision_tree_reg`, `lightgbm_reg`, `lit_linear`) via `python -m scripts.train` across `fold_group`, `fold_strat`, and `temporal`.
- [ ] Report RMSE and MAE on both log and original scales + residual-vs-fitted diagnostic plot in `03_regression.ipynb`.
- [ ] Extract feature importances / coefficients / SHAP interpretation for loan limits.

**3 · Part E Scoring Service (#30, #31)**
- [x] Implement `src/serving/` (`score_application`, `explain`, `benchmark_latency`, `REQUIRED_APPLICATION_FIELDS`).
- [x] Fix #30 bugs: limit in local currency via USD, no macro extrapolation for Ghana (D-18), Platt calibration (never 0%), reasons a borrower can act on and that never contradict their record, `needs_review` flags.
- [ ] The calibrated logistic regression under-predicts new borrowers (2.2% for a profile whose real rate is 5.7%, n=264): compare LightGBM / MLP as the deployed classifier.
- [ ] Swap in the final winning Task 2 & Task 1 models once team model comparison finishes, and record final median / p95 latency over 1,000 calls in `03_regression.ipynb` & report.

**4 · Write up & Derivation (#32, #33, #34)**
- [x] `docs/derivations/linear.md`: Gaussian MLE -> normal equations -> Ridge (MAP), checked against scikit-learn (`tests/test_linear_derivation.py`).
- [ ] `03_regression.ipynb` runs top to bottom from a fresh kernel.
- [ ] Regression + Deployment sections in Technical Report + individual reflection.
