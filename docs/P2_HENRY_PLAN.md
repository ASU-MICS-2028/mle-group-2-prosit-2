# P2 · Henry (`henryhkb`): Classification and evaluation

**Mission:** build the shared train and evaluate tools, and turn a default probability into an approve/decline rule justified in cedis.
**You own:** Lightning template and DataModule, `train` and `evaluate` scripts, validation comparison, cost threshold, calibration, bootstrap CIs, `02_classification`, logistic regression derivation. **Trains:** logistic regression (scikit-learn and Lightning), MLP. **LOs:** LO1, LO3, LO4 (see [LEARNING_OUTCOMES](LEARNING_OUTCOMES.md)).

## Steps

**0 · Tidy**
- [x] Converted `notebooks/explore_target_henry.ipynb` into reproducible script `src/analysis/target_eda.py` (D-17) so `notebooks/` keeps only the 4 deliverable notebooks.

**1 · Explore the target (Day 1)**
- [x] Default rate by month, lender, loan type, tenure (`src/analysis/target_eda.py`).
- [x] Is 2023's jump a different population or a different lender mix? (`shift_2023_summary` + `shift_2023_by_lender` in `src/analysis/target_eda.py`: both!).
- [x] Finish the leak check (`extra_pct` median & IQR by outcome in `src/analysis/target_eda.py`).

**2 · Part A: label, tasks, cost matrix**
- [x] Define label and window; write Task 1 (amount) and Task 2 (default) (`02_classification` §0, PR #60).
- [x] Cost matrix in cedis/shillings from the data. **Don't reuse the brief's example numbers.** (KES draft by Bernard, D-19; cedis from Ghana's own loans, D-22: 3,002 GHS, 3.0% fee)
- [x] Threshold = C_FP / (C_FP + C_FN); show it under two other recovery assumptions (Ghana p\* 0.029 / 0.036 / 0.047; 0.15–0.23 if 6 future loans are lost).

**3 · Shared tools on the rough file**
- [x] `src/config.py` (seed 42, paths; written by Perfect)
- [x] `src/config.py` (seed 42, paths; written by Perfect, schema constants extended (PR #46))
- [x] `src/models/lightning_modules.py` (scaffolded (PR #46): `LoanDataModule`, `LitLogReg`, `LitMLP`, `LitLinearReg`)
- [x] `src/training/cross_validation.py` (scaffolded (PR #46): writes standardized `C.PRED_COLUMNS` parquet files to `preds/`)
- [x] `src/evaluation/` (scaffolded (PR #46): PR-AUC, ROC-AUC, P/R/F1 sweep, confusion matrix, calibration, cost threshold, bootstrap CIs)
- [x] Logistic regression in Lightning (`BCEWithLogitsLoss(pos_weight≈53.6)`, Adam, early stopping); cross-check against scikit-learn (done: both give ROC-AUC 0.845, PR-AUC 0.170; coefficients not compared).

**4 · Validation study** (after Perfect's folds): stratified vs GroupKFold vs temporal; quantify inflation; decide D-7.
- [x] Done (`02_classification` §5–6); D-7 decided: compare with GroupKFold, quote by-time (ROC-AUC 0.63–0.65) as the honest forecast.

**5 · Models and noise floor**
- [x] MLP vs logistic regression (calibrated PR-AUC 0.204 vs 0.175; customer-level CIs overlap, so the lead is not established).
- [x] Calibrate on held-out data (isotonic vs Platt); reliability curve before and after (§8, added by Bernard, verified by Henry).
- [x] Cost-based operating point; bootstrap 95% CIs on F1 and PR-AUC against the ±0.0263 noise floor (§9–10 by Bernard, verified; §9b Ghana by Henry).

**6 · Write up**: `02_classification` runs from a fresh kernel; logistic derivation (log-odds → likelihood → BCE → gradient Xᵀ(σ(Xw) − y), checked against autograd); results; reflection.
- [x] `02_classification` runs top to bottom from a fresh kernel (`nbconvert --execute`, macOS); conclusions in §12 (added in PR #60 as §11, renumbered when §11 calibration over time was added).
- [x] Calibration over time (#55): pre-2023 Platt under-predicts 2023–24 risk; recalibrate on recent loans and monitor monthly (`02` §11, D-24, PR #61).
- [x] Logistic regression derivation (`docs/derivations/logistic.md`, #32): log-odds → sigmoid → likelihood → BCE → gradient Xᵀ(σ(Xw) − y), weighted version, convexity; checked against autograd (PR #62).
- [x] Personal reflection (#37): `docs/reflections/henry.md` and `henry.pdf` (one page).

## You understand it when you can answer
- Why does minimising BCE equal maximum likelihood?
- Why is "approve everyone" 98% accurate, and what do PR-AUC and F1 show instead?
- What does `pos_weight` do to calibration, and why calibrate afterwards?
- Why is the temporal score lower than the random one, and what do we promise the committee?
