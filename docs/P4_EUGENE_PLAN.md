# P4 · Eugene (`eugene-sew`): Transfer, Ensembles & Ethical Audit

**Mission:** validate cross-market transfer from Kenya to Ghana, train ensemble models (Random Forest & LightGBM), and lead the ethical, fairness, and regulatory audit.
**You own:** `04_transfer_and_fairness.ipynb`, `docs/audit/` (Model Card, Datasheet, Regulatory Mapping, Borrower Explainer), gradient boosting derivation. **Trains:** Random Forest, LightGBM (Task 2 & Task 1). **LOs:** LO2, LO3, LO4, LO5, LO6 (see [LEARNING_OUTCOMES](LEARNING_OUTCOMES.md)).

## Steps

**1 · Explore Ghana & Context Files (#8)**
- [ ] Compare Ghana vs Kenya loan types, lenders, USD amounts, durations, and macro indicators (`macro_inflation`, `macro_fx`).
- [ ] Analyze `micro_gha.csv` (World Bank Global Findex Ghana 2021) using survey weight `wgt` to profile credit exclusion by gender, rural/urban, and income quintile.

**2 · Ensemble Models (#18)**
- [ ] Run Random Forest and LightGBM via `python -m scripts.train` (`--task task2 --models random_forest lightgbm --schemes fold_group fold_strat temporal --predict-test`).
- [ ] Compare against LogReg, MLP, kNN, SVM, and majority baseline using `src/evaluation/`.

**3 · Cross-Market Transfer (#23, #24)**
- [ ] Run `run_domain_classifier()` from `src/evaluation/shift.py` to quantify Kenya -> Ghana covariate shift and importance weights.
- [ ] Document the concrete evidence NkwaPay needs before trusting Ghana predictions (pilot labeled repayments, local calibration check, monitoring triggers).

**4 · Ethical & Regulatory Audit (#25, #26, #27, #28, #29, #35)**
- [ ] Disaggregated fairness metrics (`disaggregated_evaluation()` + `add_loan_size_band()`) across `country_id`, `loan_type`, `lender_id`, `is_new_borrower`, and loan-size band; identify proxy risks and implement + evaluate one mitigation (e.g., group-aware threshold or reweighting) with before/after table.
- [ ] Write `docs/audit/model_card.md` (Mitchell et al.), `docs/audit/datasheet.md` (Gebru et al.), and regulatory compliance mapping (BoG Digital Credit Directive, Data Protection Act 2012 / Act 843, Credit Reporting Act 2007 / Act 726).
- [ ] Draft plain-language borrower decline explanation (#29) aligned with `src/serving/` reason templates.

**5 · Write up & Derivation (#32, #33, #34, #35)**
- [ ] `docs/derivations/boosting.md`: Gradient boosting derivation (functional gradient descent, residuals/pseudo-residuals, shrinkage) checked on a small example.
- [ ] `04_transfer_and_fairness.ipynb` runs top to bottom from a fresh kernel.
