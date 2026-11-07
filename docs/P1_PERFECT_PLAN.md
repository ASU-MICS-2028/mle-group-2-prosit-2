# P1 · Perfect (`peaElorm`): Data and features

**Mission:** produce the one cleaned, time-safe feature file and fold splits everyone trains on. If it's wrong, every result is wrong.
**You own:** feature file and folds, `01_eda`, data section of the report, README, baselines, kNN, SVM, SVM derivation. **LOs:** LO1, LO2, LO3 (see [LEARNING_OUTCOMES](LEARNING_OUTCOMES.md)).

## Steps

**0 · Setup**
- [ ] Put the three zips in `data/raw/` (git-ignored); create a venv; install `requirements.txt`.
- [ ] Pull Henry's branch and read his notebook.

**1 · Explore Train/Test structure (Day 1)**
- [ ] Two-lender loans (2,134 repeated `tbl_loan_id`): decide D-3.
- [ ] Customer and loan overlap with Test: decide D-4.
- [ ] Duplicates, outliers (`Total_Amount` max 23M). Bring 3–5 findings and a chart each.

**2 · Rough feature file (Day 1)**: safe raw columns only, so Henry isn't blocked.

**3 · Part A session with Henry**
- [ ] Settle D-2 to D-7. Write the feature audit table and the loan timeline.

**4 · Shared features (Days 2–3)**
- [ ] Clean, de-duplicate, drop leaks.
- [ ] Borrower history via backward `merge_asof` (past loans, past default rate, days since last loan).
- [ ] Calendar features; previous-year macro join; USD conversion.
- [ ] Folds: 5-fold GroupKFold by `customer_id`, one temporal cutoff, one stratified set for comparison. Seed 42.
- [ ] Assertions: no customer in both sides of a GroupKFold split; history uses only earlier dates.
- [ ] Log it in WORKLOG and tell Henry.

**5 · Your models**: baselines (majority, mean), kNN, SVM RBF on a subsample, through Henry's scripts.

**6 · Write up**: `01_eda` runs from a fresh kernel; SVM derivation (margin, hinge loss, dual, kernel) checked against scikit-learn; data section; reflection.

## You understand it when you can answer
- Why does a random split inflate our score here, and by how much?
- What does a backward `merge_asof` guarantee that a plain merge doesn't?
- Why is `New_versus_Repeat` risky, given 21% vs 1.7% default?
