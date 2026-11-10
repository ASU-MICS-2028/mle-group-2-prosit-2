# P2 · Henry (`henryhkb`): Classification and evaluation

**Mission:** build the shared train and evaluate tools, and turn a default probability into an approve/decline rule justified in cedis.
**You own:** Lightning template and DataModule, `train` and `evaluate` scripts, validation comparison, cost threshold, calibration, bootstrap CIs, `02_classification`, logistic regression derivation. **Trains:** logistic regression (scikit-learn and Lightning), MLP. **LOs:** LO1, LO3, LO4 (see [LEARNING_OUTCOMES](LEARNING_OUTCOMES.md)).

## Steps

**0 · Tidy**
- [x] Move `01_eda.ipynb` to `notebooks/explore_target_henry.ipynb`, read from `data/raw/` (D-12), rerun so outputs save, open a PR.

**1 · Explore the target (Day 1)**
- [ ] Default rate by month, lender, loan type, tenure (you have year).
- [ ] Is 2023's jump a different population or a different lender mix?
- [ ] Finish the leak check with a plot of `extra_pct` by outcome.

**2 · Part A: label, tasks, cost matrix**
- [ ] Define label and window; write Task 1 (amount) and Task 2 (default).
- [ ] Cost matrix in cedis/shillings from the data. **Don't reuse the brief's example numbers.**
- [ ] Threshold = C_FP / (C_FP + C_FN); show it under two other recovery assumptions.

**3 · Shared tools on the rough file**
- [x] `src/config.py` (seed 42, paths; written by Perfect)
- [ ] `src/datamodule.py` (Lightning DataModule; for now the Lightning model is defined inside `02_classification.ipynb`).
- [ ] `src/train.py`: any model over every fold of all schemes; writes ID, split, fold, model, y_true, y_pred.
- [ ] `src/evaluate.py`: PR-AUC, ROC-AUC, P/R/F1 sweep, confusion matrix, calibration, bootstrap CIs.
- [x] Logistic regression in Lightning (`BCEWithLogitsLoss(pos_weight≈53.6)`, Adam, early stopping); cross-check against scikit-learn (done: both give ROC-AUC 0.845, PR-AUC 0.170; coefficients not compared).

**4 · Validation study** (after Perfect's folds): stratified vs GroupKFold vs temporal; quantify inflation; decide D-7.

**5 · Models and noise floor**
- [ ] MLP vs logistic regression.
- [ ] Calibrate on held-out data (isotonic vs Platt); reliability curve before and after.
- [ ] Cost-based operating point; bootstrap 95% CIs on F1 and PR-AUC against the ±0.0263 noise floor.

**6 · Write up**: `02_classification` runs from a fresh kernel; logistic derivation (log-odds → likelihood → BCE → gradient Xᵀ(σ(Xw) − y), checked against autograd); results; reflection.

## You understand it when you can answer
- Why does minimising BCE equal maximum likelihood?
- Why is "approve everyone" 98% accurate, and what do PR-AUC and F1 show instead?
- What does `pos_weight` do to calibration, and why calibrate afterwards?
- Why is the temporal score lower than the random one, and what do we promise the committee?
