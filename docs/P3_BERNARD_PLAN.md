# P3 · Bernard (`adjeibernardmarfo`): Regression, Scaffolding & Deployment

**Mission:** build the Task 1 loan-amount regression pipeline, maintain repo scaffolding & CI guardrails as PM, and ship the Part E scoring service with plain-language reason codes and latency benchmarks.
**You own:** `src/config.py` schemas, `src/preprocess.py`, `src/score.py`, `tests/`, CI/pre-commit/Makefile tooling, `03_regression.ipynb`, regression & deployment sections of the report, linear regression derivation. **Trains:** OLS, Ridge, Lasso, Decision Tree (and LightGBM/Lightning linear cross-check). **LOs:** LO2, LO3, LO4, LO6 (see [LEARNING_OUTCOMES](LEARNING_OUTCOMES.md)).

## Steps

**0 · Scaffolding & Guardrails (PM)**
- [x] Shared schemas (`NUM_FEATURES_T1`, `NUM_FEATURES_T2`, `PRED_COLUMNS`), `src/preprocess.py`, `src/transforms.py`, `src/datamodule.py`, `src/train.py`, `src/evaluate.py`, `src/fairness.py`, `src/score.py`.
- [x] Automated `pytest` guardrails (`tests/`), `ci.yml`, `pr-issue-check.yml`, `.pre-commit-config.yaml`, `Makefile`.

**1 · Explore amounts, durations, and leak suspects (#7)**
- [ ] Profile `Total_Amount`, `duration`, and `Lender_portion_Funded` skew in `03_regression.ipynb` (with charts).
- [ ] Document findings in `docs/WORKLOG.md` and confirm leak rulings in Part A session (#9).

**2 · Task 1 Models & Diagnostics (#17, #21, #22)**
- [ ] Run Task 1 models (`mean`, `ols`, `ridge`, `lasso`, `decision_tree_reg`, `lightgbm_reg`, `lit_linear`) via `src/train.py` across `fold_group`, `fold_strat`, and `temporal`.
- [ ] Report RMSE and MAE on both log and original scales + residual-vs-fitted diagnostic plot in `03_regression.ipynb`.
- [ ] Extract feature importances / coefficients / SHAP interpretation for loan limits.

**3 · Part E Scoring Service (#30, #31)**
- [x] Implement `src/score.py` (`score_application`, `explain_prediction`, `benchmark_latency`, `REQUIRED_APPLICATION_FIELDS`).
- [ ] Swap in the final winning Task 2 & Task 1 models once team model comparison finishes, and record final median / p95 latency over 1,000 calls in `03_regression.ipynb` & report.

**4 · Write up & Derivation (#32, #33, #34)**
- [ ] `docs/derivations/linear.md`: Gaussian MLE -> normal equations -> Ridge (MAP), checked against scikit-learn.
- [ ] `03_regression.ipynb` runs top to bottom from a fresh kernel.
- [ ] Regression + Deployment sections in Technical Report + individual reflection.
