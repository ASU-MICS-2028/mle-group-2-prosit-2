# Learning outcomes: what to show, and how to know you understand it

At the debrief you must "explain, defend and reproduce your work without AI". If you can't answer a self-check question out loud, you don't own that outcome yet. Tick boxes as evidence lands in the repo. Owners: P1 Perfect, P2 Henry, P3 Bernard, P4 Eugene.

## LO1 · Formulate problems as ML tasks

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | Loan timeline; mark the decision moment and closest column (`disbursement_date`) | Report Part A | P1 |
| ✅ | Label definition and time window | `02_classification` §0 (PR #60) | P2 |
| 🟨 | Feature audit table for all 16 Train columns (draft; confirm in Part A session) | `docs/feature_audit.md` | P1 |
| ✅ | Task 1 (amount) and Task 2 (default), business goal with threshold and approval floor | `02_classification` §0, D-22 (PR #60) | P2 |
| ✅ | Cost matrix in money, with sensitivity at 2 other recovery rates: in cedis from Ghana's loans, p\* 0.029 / 0.036 / 0.047 (D-22) | `02_classification` §9b (PR #60) | P2 |

**Self-check:** Why can't `Total_Amount_to_Repay` be used at decision time? What does a false negative cost in cedis? Why is the threshold C_FP / (C_FP + C_FN)?

## LO2 · End-to-end pipelines

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ✅ | One shared feature file and folds, seed 42 (merged #39) | `src/features.py` | P1 |
| ✅ | Preprocessor fitted on training rows only, inside every fold | `src/processing/preprocess.py`, `src/training/cross_validation.py` | P1/P2/P3 |
| ✅ | One train script and one evaluate script for all models (16 models, 50 prediction files) | `scripts/train.py`, `scripts/evaluate.py`, `reports/tables/` | P2/P3 |
| ☐ | A teammate reproduces results from a fresh clone | PR reviews | all |
| 🟨 | Scorer: probability, decision, limit, top-3 reasons, review flags, latency (median 44 ms, p95 75 ms); final model choice pending | `src/serving/` | Bernard |

**Self-check:** Where does leakage enter if the scaler sees validation rows? Why does a backward `merge_asof` keep history safe? Why reuse the training preprocessor at inference?

## LO3 · Derive algorithms from principles

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ✅ | Logistic regression: log-odds → likelihood → BCE → gradient, plus `pos_weight` and convexity; gradient checked against PyTorch autograd (`tests/test_logistic_derivation.py`) | `docs/derivations/logistic.md` (PR #62) | Henry |
| ✅ | Linear regression: Gaussian MLE → normal equations (Ridge as MAP), checked against `LinearRegression`, `Ridge(alpha=10)` and PyTorch autograd (`tests/test_linear_derivation.py`, 5 tests) | `docs/derivations/linear.md` (#32) | Bernard |
| ✅ | SVM: widest street → hinge loss → Lagrangian → dual → RBF kernel; each claim checked against scikit-learn `SVC` on toy data (`tests/test_svm_derivation.py`, 8 tests) | `docs/derivations/svm.md` (PR #67) | Perfect |
| ✅ | Gradient boosting: functional gradient, pseudo-residuals, Newton leaf values, shrinkage; worked example checked numerically | `docs/derivations/boosting.md` | Eugene |
| ✅ | Check each against scikit-learn / PyTorch on a small example | `tests/test_{logistic,svm,linear}_derivation.py`, `docs/derivations/boosting.md` | all |

**Self-check:** Show ∂BCE/∂w = Xᵀ(σ(Xw) − y). Why is minimising BCE maximum likelihood? Why do normal equations fail with collinear features? What does `pos_weight ≈ 53.6` do?

## LO4 · Evaluate with metrics and validation

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ✅ | Stratified vs GroupKFold vs temporal; how inflated the naive score was; which we trust | `02_classification` §5, `reports/tables/task2_model_summary.csv` | Henry |
| ✅ | ≥ 5 models incl. majority baseline (9 Task 2, 7 Task 1; best by customer: MLP / Random Forest PR-AUC 0.21) | `reports/tables/*_model_summary.csv` | all |
| ✅ | Threshold sweep, confusion matrix, cost-based operating point (Ghana p\* 0.036: approve 88.6%, −44% expected loss) | `02_classification` §9, §9b (PR #60) | Henry |
| ✅ | Calibration before/after (Platt: ECE 0.30 → 0.002); over time: pre-2023 calibration under-predicts 2024 by half, recalibration plan (D-24) | `02_classification` §8, §11 (PR #61) | Henry |
| ✅ | Bootstrap 95% CIs on F1 and PR-AUC vs the ±0.0263 noise floor (customer-level) | `02_classification` §10 | Henry |
| ✅ | RMSE/MAE on log and original (USD) scale; residual plot | `03_regression`, `reports/figures/t1_residuals_*.png` | Bernard |

**Self-check:** Why is accuracy useless at 1.83% defaults? Why PR-AUC over ROC-AUC? Which split gives the honest number? What does "calibrated" mean?

## LO5 · Bias, fairness, ethics

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ✅ | Datasheet and Model Card | `docs/audit/datasheet.md`, `docs/audit/model_card.md` | Eugene |
| ✅ | PR-AUC, F1, selection rate by country, loan type, lender, tenure, size band (first-time borrowers 49% vs 98%) | `04_transfer_and_fairness` §3.1 | Eugene |
| ✅ | Proxy analysis and one mitigation with before/after and its cost (25% starter loan, now in the scorer: 49% → 81%, GHS 18 per first-time applicant) | `04_transfer_and_fairness` §3.3–3.4 | Eugene |
| ✅ | Regulatory map: BoG Directive, Act 843, Act 726; what we can't satisfy (Act section numbers to confirm) | `docs/audit/regulatory_map.md` | Eugene |
| ✅ | Findex Ghana: who lacks credit; does the model narrow or widen the gap (widens, as shipped) | `04_transfer_and_fairness` §4 | Eugene |

**Self-check:** Can a model be accurate and unfair? What is a proxy, with an example from this data? What must we tell a declined borrower?

## LO6 · Communicate

| ☐ | Work | Evidence | Owner |
| --- | --- | --- | --- |
| ☐ | 15-min debrief; each presents their part | `slides/` | all |
| ✅ | Plain-language decline explanation | `docs/audit/decline_explanation.md` | Bernard / Eugene |
| ☐ | Q&A rehearsal: ask each other the self-check questions, no notes | WORKLOG | all |
| ☐ | One-page reflection each; AI-use declaration | `docs/reflections/` | each |

## Reading by task

| Task | Read |
| --- | --- |
| Leakage, features | [DATA_AND_PIPELINE_GUIDE](DATA_AND_PIPELINE_GUIDE.md#1-understand-the-data) ("The leak, from first principles") |
| Logistic regression, class weights | ISLR ch. 4; HOMLP ch. 4 |
| Validation, calibration | ISLR ch. 5; scikit-learn "Probability calibration" |
| Linear models, Ridge/Lasso | ISLR ch. 3, 6 |
| Trees, forests, boosting | ISLR ch. 8; HOMLP ch. 6–7 |
| SVM | ISLR ch. 9 |
| Fairness, Model Card | Mitchell et al. 2019; Barocas, Hardt, Narayanan |
| Reason codes | Molnar, *Interpretable ML* (SHAP) |
