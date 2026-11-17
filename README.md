# Prosit 2: Responsible Credit Scoring for NkwaPay

ICS553 Machine Learning Essentials · Group 2

We train a default-risk classifier (**Task 2**) and a loan-limit regressor (**Task 1**) on Kenyan digital loans, evaluate Kenya → Ghana cross-market transfer, audit the models for fairness and regulation (BoG Digital Credit Directive, Act 843, Act 726), and ship a scoring service with plain-language reason codes. Everyone builds against **one shared feature file (`src/features.py`), one library of models and metrics (`src/`), and one set of entry points (`scripts/`)**, so every result is comparable.

> **Data dictionary, leakage proof and glossary:** [docs/DATA_AND_PIPELINE_GUIDE.md](docs/DATA_AND_PIPELINE_GUIDE.md).

---

## Status

Detailed, dated status lives in [docs/PROGRESS.md](docs/PROGRESS.md); this table only says where each deliverable stands.

| Deliverable | Purpose | Owner | Status |
| --- | --- | --- | --- |
| `01_eda.ipynb` + `src/features.py` | Train/Test structure EDA; time-safe features and 3 fold schemes | Perfect (P1) | ✅ merged (PRs #38, #39) |
| `src/analysis/target_eda.py` | Henry's target EDA as a script: default rate by year, month, lender, type, tenure; 2023+ shift | Henry (P2) | ✅ merged (PR #46) |
| `02_classification.ipynb` | LogReg (scikit-learn + Lightning), MLP, validation study | Henry (P2) | ✅ Henry's original restored on `p3/30-scorer-fixes-and-structure`; calibration, cost threshold and CIs still to add (#19, #20) |
| `03_regression.ipynb` | Task 1 limits, residuals, scorer demo and latency | Bernard (P3) | 🟨 rerun on the restructured pipeline; amounts/leak-suspect EDA (#7) and Task 2 tree (#17) still to do |
| `src/serving/` scorer | Probability, decision, limit, 3 reasons, review flags | Bernard (P3) | 🟨 Ghana currency and 0%-risk bugs fixed on `p3/30-scorer-fixes-and-structure` (#30) |
| `04_transfer_and_fairness.ipynb` | RF/LightGBM, Kenya → Ghana shift, fairness, Findex | Eugene (P4) | 🟥 AI-generated draft; Eugene to own and fix (see the note at its top) |

Status key: ⬜ Not started · 🟨 In progress · ✅ Done · 🟥 Needs owner action

---

## Team & Ownership

| Role | Name | Owns | Plan |
| --- | --- | --- | --- |
| **P1 · Data & Features** | Perfect ([@peaElorm](https://github.com/peaElorm)) | `src/features.py`, `01_eda.ipynb`, baselines/kNN/SVM, SVM derivation | [P1_PERFECT_PLAN.md](docs/P1_PERFECT_PLAN.md) |
| **P2 · Classification & Evaluation** | Henry ([@henryhkb](https://github.com/henryhkb)) | `02_classification.ipynb`, LogReg & MLP, validation study, calibration, cost matrix, CIs, LogReg derivation | [P2_HENRY_PLAN.md](docs/P2_HENRY_PLAN.md) |
| **P3 · Regression & Deployment** | Bernard ([@adjeibernardmarfo](https://github.com/adjeibernardmarfo)) | `03_regression.ipynb`, OLS/Ridge/Lasso/tree, `src/serving/`, CI & `tests/`, linear regression derivation | [P3_BERNARD_PLAN.md](docs/P3_BERNARD_PLAN.md) |
| **P4 · Transfer & Fairness** | Eugene ([@eugene-sew](https://github.com/eugene-sew)) | `04_transfer_and_fairness.ipynb`, RF & LightGBM, Kenya→Ghana shift, Model Card/Datasheet/BoG audit, boosting derivation | [P4_EUGENE_PLAN.md](docs/P4_EUGENE_PLAN.md) |

**Review rotation:** Perfect → Henry → Bernard → Eugene → Perfect. Each person owns their own notebook; nobody rewrites another member's notebook without their approval.

---

## Getting started

### 1. Environment (Python 3.12)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1     macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins every package to an exact version for every platform; `requirements.txt` lists only the direct dependencies. CI installs the lock file on Python 3.12, so it tests exactly what you run.

### 2. Data (never committed)

Unzip into `data/raw/` with these exact folder names:

```text
data/raw/african-credit-scoring-challenge/   Train.csv, Test.csv, economic_indicators.csv, ...   (Zindi)
data/raw/GHA_2021_FINDEX_v02_M_csv/          micro_gha.csv                                       (World Bank)
data/raw/financial-inclusion-in-africa/      Train.csv, ...                                      (Zindi, optional)
```

### 3. Reproduce everything

```bash
make features     # data/raw -> data/processed/features.parquet, fails loudly on leakage (~5 s)
make train        # every model on every validation scheme -> preds/, plus the scorer -> artifacts/
make eval         # metrics for every prediction file -> reports/tables/
make figures      # report figures -> reports/figures/
make score        # sample decisions + 1,000-call latency benchmark
make test         # pytest; `make lint` runs ruff
```

On Windows without `make`, run the same modules directly, e.g. `python -m scripts.train --task task2 --models logreg lightgbm --schemes fold_group temporal`. Every script has `--help`.

---

## Repository structure

```text
├── configs/
│   ├── models.yaml                 # hyperparameters for every model (edit here, not in code)
│   └── cost_matrix.yaml            # cost-matrix inputs: measured from Train.csv vs assumptions (#11, draft)
├── data/                           # git-ignored: raw/ (the three zips) and processed/features.parquet
├── scripts/                        # entry points, run as `python -m scripts.<name>`
│   ├── build_features.py           # -> data/processed/features.parquet
│   ├── run_target_eda.py           # target / leakage / 2023+ shift tables
│   ├── train.py                    # any model x any validation scheme -> preds/; --save-artifacts -> artifacts/
│   ├── evaluate.py                 # all preds/ -> reports/tables/*_model_summary.csv, cost sensitivity
│   ├── make_figures.py             # -> reports/figures/*.png
│   └── score.py                    # sample decisions and latency
├── src/
│   ├── config.py                   # seed, paths, feature lists (single source of truth)
│   ├── features.py                 # time-safe borrower history, calendar, macro, fold schemes (P1)
│   ├── analysis/target_eda.py      # Henry's target EDA as a reusable function
│   ├── processing/preprocess.py    # fold-safe ColumnTransformer (D-5) and TargetLogScaler (clipped)
│   ├── models/                     # one file per model family + registry
│   │   ├── __init__.py             # build_estimator(name, task): the registry, settings from configs/models.yaml
│   │   ├── baselines.py            # majority class, mean
│   │   ├── linear.py               # logistic regression, OLS, Ridge, Lasso
│   │   ├── neighbors.py            # kNN
│   │   ├── svm.py                  # RBF SVM on a stratified subsample
│   │   ├── trees.py                # pruned decision trees
│   │   ├── ensembles.py            # Random Forest, LightGBM
│   │   ├── lightning_modules.py    # LitLogReg, LitMLP, LitLinearReg (Henry's architectures)
│   │   └── data_module.py          # LoanDataModule (customer-grouped early-stopping split)
│   ├── training/
│   │   ├── cross_validation.py     # run_cv, predict_test_split: one prediction format for every model
│   │   ├── lightning_runner.py     # one quiet Trainer loop for every Lightning model
│   │   └── artifacts.py            # calibrated scorer bundle for deployment
│   ├── evaluation/
│   │   ├── metrics.py              # PR-AUC, ROC-AUC, F1 at threshold, reliability/ECE, RMSE/MAE
│   │   ├── calibration.py          # Platt (deployment) and isotonic, out-of-fold
│   │   ├── cost_matrix.py          # threshold from configs/cost_matrix.yaml + sensitivity table
│   │   ├── bootstrap.py            # 95% CIs, resampling customers (not rows)
│   │   ├── fairness.py             # metrics and selection rate per group
│   │   └── shift.py                # Kenya-vs-Ghana domain classifier, importance weights
│   ├── serving/                    # Part E scoring service
│   │   ├── schema.py               # required fields, exchange rate, application -> feature row
│   │   ├── scorer.py               # score_application(): probability, decision, limit, reasons, review flags
│   │   ├── reasons.py              # plain-language reason codes a borrower can act on
│   │   └── latency.py              # median / p95 over 1,000 calls
│   └── visualization/              # reusable plots; save_figure() -> reports/figures/
├── notebooks/                      # the brief's 4 deliverable notebooks; each owner writes their own
├── reports/                        # committed outputs: tables/*.csv and figures/*.png for the report
├── preds/ · artifacts/             # git-ignored predictions and model bundles (regenerate with make train)
├── tests/                          # one file per area; run in CI on synthetic data (no real data in CI)
└── docs/                           # tracking, decisions, derivations, audit, reflections
```

### Model and prediction contract

Every model, whatever its library, goes through `src/training/cross_validation.py` and writes the same parquet columns to `preds/`:
`ID, customer_id, country_id, lender_id, loan_type, is_new_borrower, split, scheme, fold, model, y_true, y_pred`.
Task 2 `y_pred` is a raw (uncalibrated) probability; Task 1 `y_pred` is an amount in **USD**. `scripts/evaluate.py` calibrates Task 2 out of fold before any threshold metric, because the cost threshold only means something on calibrated probabilities.

### Where did the old modules go? (PR #46 → this structure)

| Before | Now |
| --- | --- |
| `src/preprocess.py`, `src/transforms.py` | `src/processing/preprocess.py` |
| `src/datamodule.py` | `src/models/lightning_modules.py`, `src/models/data_module.py`, `src/training/lightning_runner.py` |
| `src/train.py` | `src/models/*` (estimators), `src/training/cross_validation.py`, `src/training/artifacts.py`, `scripts/train.py` |
| `src/evaluate.py` | `src/evaluation/{metrics,calibration,cost_matrix,bootstrap}.py`, `scripts/evaluate.py` |
| `src/fairness.py` | `src/evaluation/fairness.py`, `src/evaluation/shift.py` |
| `src/score.py` | `src/serving/*`, `scripts/score.py` |
| `src/eda_target.py` | `src/analysis/target_eda.py`, `scripts/run_target_eda.py` |
| `python -m src.<module>` | `python -m scripts.<name>` or `make <target>` (`python -m src.features` still works) |

---

## Live roadmap and GitHub issues

**Master tracker: [Roadmap issue #1](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/1).**
Filter open issues by owner: [P1](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP1) · [P2](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP2) · [P3](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP3) · [P4](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP4) · [Whole team](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3Aall).

> GitHub ticks tasklist items inside an issue automatically when the referenced issue closes, so issue #1 is the live checklist. The copy below reflects GitHub as of 2026-11-17.

<details>
<summary>Copy-paste tasklist for GitHub issue #1 (auto-ticks when issues close)</summary>

```markdown
### P1 · Data & Features
- [x] #4 Repo skeleton, `.gitignore`, shared config
- [x] #5 Explore Train/Test structure (`01_eda.ipynb`)
- [x] #12 Shared feature file & fold splits (`src/features.py`)
- [ ] #15 Models: baselines, kNN, SVM

### P2 · Classification & Evaluation
- [x] #6 Explore target by month, lender, loan type, tenure
- [ ] #10 Define Task 1, Task 2, and business goal
- [ ] #11 Cost matrix in cedis/shillings + sensitivity checks
- [x] #13 Shared `train` / `evaluate` / `datamodule` scripts
- [x] #14 Validation study (stratified vs GroupKFold vs temporal; D-7)
- [x] #16 Models: logistic regression and MLP
- [ ] #19 Calibration, threshold sweep, cost operating point
- [ ] #20 Bootstrap 95% CIs on F1 and PR-AUC vs noise floor

### P3 · Regression & Deployment
- [ ] #7 Explore amounts, durations, and leak suspects
- [ ] #17 Models: OLS, Ridge, Lasso, decision tree
- [ ] #21 Regression metrics (log + original scale) & residuals
- [ ] #22 Interpretation (coefficients / importances / SHAP)
- [ ] #30 Scoring service (`src/serving/`)
- [x] #31 Latency figures (1,000 calls) & input specification

### P4 · Transfer & Fairness
- [ ] #8 Explore Ghana vs Kenya, macro, Findex, inclusion files
- [ ] #18 Models: Random Forest, LightGBM
- [ ] #23 Cross-market transfer & domain classifier / importance weights
- [ ] #24 Evidence required for NkwaPay to trust Ghana predictions
- [ ] #25 Model Card
- [ ] #26 Fairness audit across country, loan type, lender, tenure, loan size + proxy + mitigation
- [ ] #27 Regulatory mapping (BoG Directive, Act 843, Act 726)
- [ ] #28 Findex Ghana exclusion analysis
- [ ] #29 Plain-language decline explanation
- [ ] #35 Ethical & regulatory audit document

### Whole Team Deliverables
- [x] #2 Profile all three datasets
- [x] #3 Assign roles (P1–P4)
- [ ] #9 Part A session: confirm D-2 to D-7, timeline, label, feature audit
- [ ] #32 Derivations (`docs/derivations/`: SVM, LogReg, Linear, Boosting)
- [ ] #33 All 4 notebooks run top-to-bottom on clean clone
- [ ] #34 Technical report (6–10 pages)
- [ ] #36 Debrief slides & mock Q&A
- [ ] #37 Individual one-page reflections
```

</details>

---

## Project docs

| Page | Purpose |
| --- | --- |
| [docs/DATA_AND_PIPELINE_GUIDE.md](docs/DATA_AND_PIPELINE_GUIDE.md) | Dataset profiles, column tables, leakage proof, glossary, Lightning primer |
| [docs/PROGRESS.md](docs/PROGRESS.md) | What's done, what's left, blockers, session log |
| [docs/WORKLOG.md](docs/WORKLOG.md) | Session handoff notes (newest first) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Data and modelling decisions (never edit silently; mark superseded) |
| [docs/feature_audit.md](docs/feature_audit.md) | 16-column leakage and role audit |
| [docs/LEARNING_OUTCOMES.md](docs/LEARNING_OUTCOMES.md) | Learning outcomes and self-check questions |
| [docs/DELIVERABLES.md](docs/DELIVERABLES.md) | 100-point rubric checklist |
| [docs/AI_USE_LOG.md](docs/AI_USE_LOG.md) | AI-use declaration log |

---

## Team rules and GitHub workflow

**Shared logic lives in `src/`; entry points in `scripts/`; settings in `configs/`.** Notebooks are where each owner shows, explains and interprets their own work; they may import from `src/` or build a model step by step in plain sight, whichever the owner needs to explain it at the debrief.

1. **One issue, one branch, one PR:** branch from the latest `main` as `<role>/<issue>-<short-name>` (e.g. `p3/17-task1-models`). Never push to `main`.
2. **Link real issues:** `Closes #17` or `Refs #17` in the PR description (checked by `PR Issue Link Check`).
3. **Docs ship with code:** update `docs/WORKLOG.md` and `docs/PROGRESS.md` (checked by `Docs freshness`) and any doc listed in `CLAUDE.md`.
4. **Tests must pass:** CI runs `ruff check src/ scripts/ tests/` and `pytest` on Python 3.12.
5. **Review before merge:** one approval from the rotation reviewer (P1 → P2 → P3 → P4 → P1), who reruns the branch. Nobody merges their own PR unreviewed.
6. **Owners own their notebooks:** propose changes to someone else's notebook as a PR they review, never as a rewrite.
