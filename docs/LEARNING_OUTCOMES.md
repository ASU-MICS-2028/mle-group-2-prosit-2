# Learning outcomes: what to show, and how to know you understand it

At the debrief you must "explain, defend and reproduce your work without AI". If you can't answer a self-check question out loud, you don't own that outcome yet. Tick boxes as evidence lands in the repo. Owners: P1 Perfect, P2 Henry, P3 Bernard, P4 Eugene.

## LO1 · Formulate problems as ML tasks

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | Loan timeline; mark the decision moment and closest column (`disbursement_date`) | Report Part A | P1 |
| ☐ | Label definition and time window | Report Part A | P2 |
| ☐ | Feature audit table for all 16 Train columns | `docs/feature_audit.md` | P1 |
| ☐ | Task 1 (amount) and Task 2 (default), business goal with threshold and approval floor | Report Part A | P2 |
| ☐ | Cost matrix in money, with sensitivity at 2 other recovery rates | Report Part A | P2 |

**Self-check:** Why can't `Total_Amount_to_Repay` be used at decision time? What does a false negative cost in cedis? Why is the threshold C_FP / (C_FP + C_FN)?

## LO2 · End-to-end pipelines

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | One shared feature file and folds, seed 42 | `src/features.py` | P1 |
| ☐ | Preprocessor fitted on training rows only | `src/preprocess.py` | P1/P2 |
| ☐ | One train script and one evaluate script for all models | `src/train.py`, `src/evaluate.py` | P2 |
| ☐ | A teammate reproduces results from a fresh clone | PR reviews | all |
| ☐ | Scorer: probability, decision, limit, top-3 reasons, latency | `src/score.py` | Bernard |

**Self-check:** Where does leakage enter if the scaler sees validation rows? Why does a backward `merge_asof` keep history safe? Why reuse the training preprocessor at inference?

## LO3 · Derive algorithms from principles

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | Logistic regression: log-odds → likelihood → BCE → gradient | `docs/derivations/logistic.md` | Henry |
| ☐ | Linear regression: Gaussian MLE → normal equations (Ridge as MAP) | `docs/derivations/linear.md` | Bernard |
| ☐ | One of SVM (Perfect) / boosting / tree | `docs/derivations/<name>.md` | Perfect |
| ☐ | Check each against scikit-learn on a small example | derivation notebook | author |

**Self-check:** Show ∂BCE/∂w = Xᵀ(σ(Xw) − y). Why is minimising BCE maximum likelihood? Why do normal equations fail with collinear features? What does `pos_weight ≈ 53.6` do?

## LO4 · Evaluate with metrics and validation

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | Stratified vs GroupKFold vs temporal; how inflated the naive score was; which we trust | `02_classification` | Henry |
| ☐ | ≥ 5 models incl. majority baseline | `preds/`, results table | all |
| ☐ | Threshold sweep, confusion matrix, cost-based operating point | `02_classification` | Henry |
| ☐ | Calibration before/after | `02_classification` | Henry |
| ☐ | Bootstrap 95% CIs on F1 and PR-AUC vs the ±0.0263 noise floor | `02_classification` | Henry |
| ☐ | RMSE/MAE on log and original scale; residual plot | `03_regression` | Bernard |

**Self-check:** Why is accuracy useless at 1.83% defaults? Why PR-AUC over ROC-AUC? Which split gives the honest number? What does "calibrated" mean?

## LO5 · Bias, fairness, ethics

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | Datasheet and Model Card | `docs/audit/` | Eugene |
| ☐ | PR-AUC, F1, selection rate by country, loan type, lender, tenure, size band | `04_transfer_and_fairness` | Eugene |
| ☐ | Proxy analysis and one mitigation with before/after | `04_transfer_and_fairness` | Eugene |
| ☐ | Regulatory map: BoG Directive, Act 843, Act 726; what we can't satisfy | `docs/audit/` | Eugene |
| ☐ | Findex Ghana: who lacks credit; does the model narrow or widen the gap | `04_transfer_and_fairness` | Eugene |

**Self-check:** Can a model be accurate and unfair? What is a proxy, with an example from this data? What must we tell a declined borrower?

## LO6 · Communicate

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | 15-min debrief; each presents their part | `slides/` | all |
| ☐ | Plain-language decline explanation | `docs/audit/` | Bernard / Eugene |
| ☐ | Q&A rehearsal: ask each other the self-check questions, no notes | WORKLOG | all |
| ☐ | One-page reflection each; AI-use declaration | `docs/reflections/` | each |

## Reading by task

| Task | Read |
| --- | --- |
| Leakage, features | README §1 "The leak, from first principles" |
| Logistic regression, class weights | ISLR ch. 4; HOMLP ch. 4 |
| Validation, calibration | ISLR ch. 5; scikit-learn "Probability calibration" |
| Linear models, Ridge/Lasso | ISLR ch. 3, 6 |
| Trees, forests, boosting | ISLR ch. 8; HOMLP ch. 6–7 |
| SVM | ISLR ch. 9 |
| Fairness, Model Card | Mitchell et al. 2019; Barocas, Hardt, Narayanan |
| Reason codes | Molnar, *Interpretable ML* (SHAP) |
