# Prosit 2: Responsible Credit Scoring for NkwaPay

ICS553 Machine Learning Essentials · Group 2

We train a default-risk classifier (**Task 2**) and a loan-limit regressor (**Task 1**) on Kenyan digital loans, evaluate Kenya → Ghana cross-market transfer, conduct an ethical and regulatory audit (BoG Digital Credit Directive, Act 843, Act 726), and ship a low-latency scoring service with plain-language reason codes. Everyone builds against **one shared feature table (`src/features.py`), one preprocessing/training/evaluation pipeline (`src/`), and automated CI guardrails (`tests/`)**.

> **Detailed Data Dictionary, Leakage Proof & Glossary:** moved to [docs/DATA_AND_PIPELINE_GUIDE.md](docs/DATA_AND_PIPELINE_GUIDE.md) so this README stays crisp and actionable.

---

## Status

| Notebook / code | Purpose | Owner | Status |
| --- | --- | --- | --- |
| `01_eda.ipynb` + `src/features.py` + `src/eda_target.py` | Explore & clean data; target/shift EDA; build time-safe features & 3 fold schemes | Perfect (P1) + Henry (P2) | ✅ EDA §1–5, `features.parquet`, and `src/eda_target.py` ready |
| `src/` shared ML pipeline + `tests/` | Shared `preprocess`, `datamodule`, `train`, `evaluate`, `fairness`, `score` + CI guardrails | Bernard (P3) | 🟨 Built on `p3/13-30-senior-scaffolding` (PR ready for review) |
| `02_classification.ipynb` | Thin script-backed driver: LogReg/MLP, 3-scheme CV, calibration, cost threshold, bootstrap CIs | Henry (P2) | ✅ Refactored to import 100% from `src/` on `p3/13-30-senior-scaffolding` |
| `03_regression.ipynb` | Thin script-backed driver: Task 1 limits, residual diagnostics, Part E scorer + latency | Bernard (P3) | ✅ Thin script-backed deliverable notebook ready on `p3/13-30-senior-scaffolding` |
| `04_transfer_and_fairness.ipynb` | Thin script-backed driver: RF/LightGBM, Kenya → Ghana shift, fairness + mitigation, Findex | Eugene (P4) | 🟨 Script-backed notebook scaffolded; `docs/audit/` & boosting derivation next |

Status key: ⬜ Not started · 🟨 In progress · ✅ Done

---

## Team & Ownership

| Role | Name | Focus & Owned Deliverables | Plan |
| --- | --- | --- | --- |
| **P1 · Data & Features** | Perfect ([@peaElorm](https://github.com/peaElorm)) | `src/features.py`, `01_eda.ipynb`, baselines/kNN/SVM, SVM derivation | [P1_PERFECT_PLAN.md](docs/P1_PERFECT_PLAN.md) |
| **P2 · Classification & Evaluation** | Henry ([@henryhkb](https://github.com/henryhkb)) | `02_classification.ipynb`, LogReg & MLP, validation study, calibration, cost matrix, CIs, LogReg derivation | [P2_HENRY_PLAN.md](docs/P2_HENRY_PLAN.md) |
| **P3 · Regression, Infra & Deployment** | Bernard ([@adjeibernardmarfo](https://github.com/adjeibernardmarfo)) | `03_regression.ipynb`, OLS/Ridge/Lasso/Tree, `src/score.py`, CI & `tests/`, Linear Reg derivation | [P3_BERNARD_PLAN.md](docs/P3_BERNARD_PLAN.md) |
| **P4 · Transfer & Fairness** | Eugene ([@eugene-sew](https://github.com/eugene-sew)) | `04_transfer_and_fairness.ipynb`, RF & LightGBM, Kenya→Ghana shift, Model Card/Datasheet/BoG audit, Boosting derivation | [P4_EUGENE_PLAN.md](docs/P4_EUGENE_PLAN.md) |

**Review rotation:** Perfect → Henry → Bernard → Eugene → Perfect.

---

## Getting Started & End-to-End Reproduction

### 1. Environment Setup
```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download Raw Data (Never Commit Data)
Unzip into `data/raw/`:
- `african-credit-scoring-challenge/` from [Zindi](https://zindi.africa/competitions/african-credit-scoring-challenge)
- `GHA_2021_FINDEX_v02_M_csv/` from [World Bank Microdata Library](https://microdata.worldbank.org/index.php/catalog/4646)
- `financial-inclusion-in-africa/` from Zindi (optional)

### 3. Run the Reproducible Pipeline
Using `Makefile` (or run `python -m ...` directly):

```bash
make features   # python -m src.features -> builds data/processed/features.parquet (~4s) with leakage assertions
make eda-target # python -m src.eda_target -> reproducible target, leakage & 2023+ shift tables
make train      # runs Task 2 & Task 1 models across fold_group, fold_strat, temporal -> writes preds/ & artifacts/
make eval       # python -m src.evaluate -> aggregates all predictions in preds/
make score      # python -m src.score -> scores sample Ghana application + runs 1,000-call latency benchmark
make test       # pytest tests/ -v && ruff check src/ tests/
```

Or run individual models directly:
```bash
python -m src.train --task task2 --models majority logreg lightgbm --schemes fold_group temporal --predict-test
python -m src.train --task task1 --models mean ols ridge lasso --schemes fold_group temporal --save-artifacts
```

---

## Repository & Shared Pipeline Architecture

```text
├── .github/
│   ├── pull_request_template.md
│   └── workflows/
│       ├── ci.yml                  # Runs ruff lint + pytest guardrails on every PR
│       ├── docs-freshness.yml      # Enforces WORKLOG.md & PROGRESS.md updates when code changes
│       └── pr-issue-check.yml      # Enforces real issue linking ("Closes #13" / "Refs #13")
├── .pre-commit-config.yaml         # Blocks accidental large-file commits + runs ruff locally
├── Makefile                        # One-command reproduction (features, train, eval, score, test)
├── pyproject.toml                  # Ruff & Pytest configuration
├── data/                           # Gitignored raw CSVs & processed features.parquet
├── artifacts/                      # Gitignored fitted joblib deployment bundles (task2_clf.joblib, task1_reg.joblib)
├── preds/                          # Gitignored standardized prediction parquet files
├── src/
│   ├── config.py                   # Seed (42), paths, feature/drop schemas, cost constants
│   ├── features.py                 # Time-safe merge_asof borrower history, macro join, 3 fold schemes (P1)
│   ├── eda_target.py               # Reproducible target, leakage & 2023+ shift EDA script (converted from explore_target_henry)
│   ├── preprocess.py               # ColumnTransformer (D-5 unseen category handling) & TargetLogScaler
│   ├── transforms.py               # Alias re-exporting src.preprocess
│   ├── datamodule.py               # Shared PyTorch Lightning DataModule, LitLogReg, LitMLP, LitLinearReg
│   ├── train.py                    # Unified CV runner writing standardized preds/ & joblib artifacts/
│   ├── evaluate.py                 # PR-AUC, ROC-AUC, cost threshold, calibration, bootstrap CIs, RMSE/MAE
│   ├── fairness.py                 # Disaggregated group metrics, loan-size bands, Kenya->Ghana domain classifier
│   └── score.py                    # Part E scoring service (probability, decision, limit, 3 plain reasons, latency)
├── tests/
│   ├── test_leakage_and_splits.py  # Automated assertions against leaky cols, split integrity, and 4-notebook rule
│   └── test_pipeline_and_scorer.py # Preprocessor unseen-category, TargetLogScaler, metric & scorer contract tests
├── notebooks/                      # ONLY the 4 required Prosit deliverable notebooks (thin script-backed drivers)
│   ├── 01_eda.ipynb                # P1
│   ├── 02_classification.ipynb     # P2 (imports from src/)
│   ├── 03_regression.ipynb         # P3 (imports from src/)
│   └── 04_transfer_and_fairness.ipynb # P4 (imports from src/)
└── docs/                           # Tracking, decisions, derivations, audit artifacts, reflections
```

### Model & Prediction Contract (`preds/`)
Every model—whether scikit-learn, LightGBM, or PyTorch Lightning—runs through `src/train.py` and writes identical parquet columns:
`ID, customer_id, country_id, lender_id, loan_type, is_new_borrower, split, scheme, fold, model, y_true, y_pred`.
Because all models share this contract, `src/evaluate.py` and `04_transfer_and_fairness.ipynb` evaluate every model identically without custom glue code.

---

## Live Roadmap & GitHub Issue Tracking

**Master tracker: [Roadmap Issue #1](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/1).**
Filter open issues by owner: [P1](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP1) · [P2](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP2) · [P3](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP3) · [P4](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3AP4) · [Whole team](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aissue+label%3Aall).

> **Why GitHub Issue #1 is the live checklist:** Markdown checkboxes inside `README.md` do not auto-tick when PRs merge, whereas GitHub **automatically ticks** tasklist items inside an Issue when the referenced issue closes via `Closes #<num>`. Use the collapsible list below inside Issue #1 to keep progress 100% automatic.

<details>
<summary>Copy-paste tasklist for GitHub Issue #1 (auto-ticks when issues close)</summary>

```markdown
### P1 · Data & Features
- [x] #4 Repo skeleton, `.gitignore`, shared config
- [x] #5 Explore Train/Test structure (`01_eda.ipynb`)
- [x] #12 Shared feature file & fold splits (`src/features.py`)
- [ ] #15 Models: baselines, kNN, SVM

### P2 · Classification & Evaluation
- [ ] #6 Explore target by month, lender, loan type, tenure
- [ ] #10 Define Task 1, Task 2, and business goal
- [ ] #11 Cost matrix in cedis/shillings + sensitivity checks
- [ ] #13 Shared `train` / `evaluate` / `datamodule` scripts
- [ ] #14 Validation study (stratified vs GroupKFold vs temporal; D-7)
- [x] #16 Models: logistic regression and MLP
- [ ] #19 Calibration, threshold sweep, cost operating point
- [ ] #20 Bootstrap 95% CIs on F1 and PR-AUC vs noise floor

### P3 · Regression & Deployment
- [ ] #7 Explore amounts, durations, and leak suspects
- [ ] #17 Models: OLS, Ridge, Lasso, decision tree
- [ ] #21 Regression metrics (log + original scale) & residuals
- [ ] #22 Interpretation (coefficients / importances / SHAP)
- [ ] #30 Scoring service (`src/score.py`)
- [ ] #31 Latency figures (1,000 calls) & input specification

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

## Project Docs

| Page | Purpose |
| --- | --- |
| [docs/DATA_AND_PIPELINE_GUIDE.md](docs/DATA_AND_PIPELINE_GUIDE.md) | Full dataset profiles, column tables, leakage proof, glossary, Lightning primer |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Dashboard of completed work, current findings, blockers, session log |
| [docs/WORKLOG.md](docs/WORKLOG.md) | Session handoff notes (newest first) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Data & modeling decision log (never edit silently; mark superseded) |
| [docs/feature_audit.md](docs/feature_audit.md) | Full 16-column leakage & role audit |
| [docs/LEARNING_OUTCOMES.md](docs/LEARNING_OUTCOMES.md) | Rubric-mapped LOs and oral-defense self-check questions |
| [docs/DELIVERABLES.md](docs/DELIVERABLES.md) | 100-point rubric checklist |
| [docs/AI_USE_LOG.md](docs/AI_USE_LOG.md) | Mandatory AI-use declaration log |

---

## Team Rules & GitHub Workflow

**Code against the shared `src/` files, never around them.** Notebooks (`01`–`04`) should stay thin—import from `src/`, run experiments, plot diagnostics, and document findings.

1. **One issue, one branch, one PR:** branch from latest `main` as `<role>/<issue>-<short-name>` (e.g., `p3/17-task1-models`). Never push directly to `main`.
2. **Link real issues in PR descriptions:** replace `Closes #<issue>` with e.g. `Closes #17` (or `Refs #17`). Enforced by `PR Issue Link Check`.
3. **Docs ship with code:** update `docs/WORKLOG.md`, `docs/PROGRESS.md` (enforced by `Docs freshness`), and any affected doc per `CLAUDE.md`.
4. **Tests must pass:** `CI (Lint & Guardrail Tests)` runs `ruff check src/ tests/` and `pytest tests/ -v` on every PR.
5. **Peer review & rerun:** each PR requires 1 approval from the rotation (`P1 → P2 → P3 → P4 → P1`) before **Squash and merge**.
