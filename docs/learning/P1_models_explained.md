# P1 models explained: majority baseline, kNN and SVM (issue #15)

**Owner:** Perfect (P1) · **Numbers from:** [`reports/tables/task2_model_summary.csv`](../../reports/tables/task2_model_summary.csv) and [`task2_bootstrap_ci.csv`](../../reports/tables/task2_bootstrap_ci.csv) (produced by `make train eval`; nothing here was re-run for this note) · **Code:** [`src/models/baselines.py`](../../src/models/baselines.py), [`neighbors.py`](../../src/models/neighbors.py), [`svm.py`](../../src/models/svm.py) · **Settings:** [`configs/models.yaml`](../../configs/models.yaml) · **SVM maths:** [`docs/derivations/svm.md`](../derivations/svm.md)

Task 2 predicts whether a loan defaults. 1.83% of Train loans do. Scores below use **by-customer 5-fold validation** (`fold_group`, 68,654 loans) unless marked "by time" (train on loans before 2023, validate on 2023 onwards, 4,210 loans).

## 1. Majority baseline: the floor

**What it does.** Predicts "repays" for every loan, so it never flags anyone (`DummyClassifier(strategy="most_frequent")`).

**Result.** ROC-AUC 0.500 and PR-AUC 0.018. Both numbers are the "no skill" level: ROC-AUC 0.5 is a coin flip, and PR-AUC of 0.018 equals the default rate. It is 98% accurate and finds zero defaulters, which is why accuracy is not our metric.

**Why we keep it.** Every other model must beat it. By time, its PR-AUC is **0.095**, because 9.5% of validation loans default; any by-time PR-AUC has to be read against that floor.

## 2. k-nearest neighbours (k = 25, distance-weighted)

**What it does.** To score a loan, find the 25 training loans whose features are closest (after preprocessing) and take their default rate, with nearer loans counting more. It learns nothing in advance; it just remembers the training set.

**Result.** ROC-AUC 0.769, PR-AUC 0.155 (95% interval 0.134 to 0.180). By time: ROC-AUC 0.600, PR-AUC 0.140.

**Reading it.**
- It beats the floor clearly: PR-AUC is about 8.5 times the 0.018 floor.
- A guess, not tested: "distance" weighs every column on one scale, so it cannot learn that the borrower-history columns matter more than, say, day of the month. Trees and the neural network can learn that, which may be why they score higher.
- Its probabilities come in steps of 1/25 (4%), too coarse to price a loan. It is a ranking tool.
- In the summary table it declines about **21%** of borrowers (14,715 of 68,654) and catches 67% of the defaulters (846 of 1,258), with precision 5.7% (3 times the 1.83% base rate). `selection_rate` in that table is the share **approved** (78.6%), not the share flagged. The decline cut-off is the cost threshold from `configs/cost_matrix.yaml`; that value moved from 0.036 (D-22) to about 0.10 (D-28), and the CSV does not record which one produced these rows, so treat the confusion counts as illustrative. ROC-AUC and PR-AUC do not depend on any cut-off.

## 3. SVM with an RBF kernel (C = 1, `gamma="scale"`, balanced class weights)

**What it does.** Finds a boundary that keeps the two classes as far apart as possible, with a penalty for loans on the wrong side, and bends it using the RBF kernel (full derivation in `svm.md`). Its raw output is a score, not a probability, so Platt scaling converts it.

**Result.** ROC-AUC 0.805, PR-AUC 0.073 (interval 0.065 to 0.081). By time: ROC-AUC 0.570, PR-AUC 0.116.

**Reading it.**
- It orders loans overall better than kNN (0.805 vs 0.769), but its PR-AUC is less than half of kNN's (0.073 vs 0.155). The bootstrap intervals do not overlap (0.065 to 0.081 vs 0.134 to 0.180), so this gap is larger than resampling noise at customer level. The top of its list has too many repayers.
- It was trained on a **12,000-row subsample**, because a kernel SVM needs an n × n matrix and the folds have about 55,000 loans. The subsample holds roughly 220 defaulters. That *may* explain the weak PR-AUC, but we have not tested it (a larger `max_train_rows` would).
- At the same kind of cut-off it declines about 22% of borrowers (14,964), catches 68% of defaulters (855 of 1,258) and has precision 5.7%, almost the same operating point as kNN. So the two differ in how they *rank*, not in the one decision they produce.
- By time it sits near chance (0.570), among the lowest of our models.

## 4. Side by side

| Model | By customer ROC-AUC | By customer PR-AUC | By time ROC-AUC | By time PR-AUC | Can it price a loan? |
| --- | :---: | :---: | :---: | :---: | --- |
| Majority | 0.500 | 0.018 | 0.500 | 0.095 | No (same score for all) |
| kNN | 0.769 | 0.155 | 0.600 | 0.140 | Only roughly (steps of 4%) |
| SVM (RBF) | 0.805 | 0.073 | 0.570 | 0.116 | After Platt scaling |

## 5. Three sentences for the report

> The majority-class baseline reaches PR-AUC 0.018 (the default rate) and ROC-AUC 0.500, so every learned model is judged against that floor. Distance-based kNN (k = 25) reaches PR-AUC 0.155 and a kernel SVM reaches 0.073 under by-customer validation, both clearly above the floor but below the tree ensembles and the neural network (0.19 to 0.21); the SVM was fitted on a 12,000-row subsample because its cost grows with the square of the training size. Under validation by time all three fall towards the floor (kNN 0.600, SVM 0.570 ROC-AUC), so none of them is a strong candidate for deployment (the choice is issue #51).

## 6. You understand it when you can answer

- Why is "approve everyone" 98% accurate, and why is that a useless number here?
- What does kNN do at prediction time, and why does it not need training?
- Why can two models disagree, one better on ROC-AUC and one better on PR-AUC?
- What would you change to test whether the SVM's weak PR-AUC is caused by the 12,000-row subsample?
- Why does the by-time majority baseline score 0.095 rather than 0.018?
- In the summary table, what does `selection_rate` mean, and why is it 78% for a model that declines 21% of borrowers?
