<h1 align="center">Responsible Credit Scoring for NkwaPay</h1>

<p align="center">
  <b>ICS553 Machine Learning Essentials · Prosit 2 · Group 2 · Ashesi University</b><br>
  Default-risk and loan-limit models trained on Kenyan digital loans, tested for transfer to Ghana,<br>
  audited for fairness and regulation, and shipped as a scoring service with plain-language reasons.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-1f4e79?style=flat-square">
  <img alt="Tests" src="https://img.shields.io/badge/tests-127_passing-2e7d32?style=flat-square">
  <img alt="Models" src="https://img.shields.io/badge/models-16_trained-1f4e79?style=flat-square">
  <img alt="Figures" src="https://img.shields.io/badge/figures-47-1f4e79?style=flat-square">
  <img alt="Issues" src="https://img.shields.io/badge/issues-10_of_45_closed-d4a72c?style=flat-square">
  <img alt="Deadline" src="https://img.shields.io/badge/deadline-19_Nov_2026-b03a2e?style=flat-square">
</p>

<p align="center">
  <a href="#team">Team</a> ·
  <a href="#progress">Progress</a> ·
  <a href="#whats-left-by-person">What's left</a> ·
  <a href="#key-results-so-far">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#repository-structure">Structure</a> ·
  <a href="#how-we-work">How we work</a>
</p>

---

## Team

<table>
  <tr>
    <td align="center" width="25%">
      <a href="https://github.com/peaElorm"><img src="https://github.com/peaElorm.png?size=120" width="80" alt="Perfect"><br><b>Perfect Elorm Avugla</b></a><br>
      <sub><code>peaElorm</code></sub><br>
      <sub>Data &amp; features</sub>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/henryhkb"><img src="https://github.com/henryhkb.png?size=120" width="80" alt="Henry"><br><b>Henry Kofi Brandoh</b></a><br>
      <sub><code>henryhkb</code></sub><br>
      <sub>Classification &amp; evaluation</sub>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/adjeibernardmarfo"><img src="https://github.com/adjeibernardmarfo.png?size=120" width="80" alt="Bernard"><br><b>Bernard Marfo Adjei</b></a><br>
      <sub><code>adjeibernardmarfo</code></sub><br>
      <sub>Regression, deployment &amp; project lead</sub>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/eugene-sew"><img src="https://github.com/eugene-sew.png?size=120" width="80" alt="Eugene"><br><b>Eugene</b></a><br>
      <sub><code>eugene-sew</code></sub><br>
      <sub>Transfer &amp; fairness</sub>
    </td>
  </tr>
  <tr>
    <td valign="top"><sub><code>src/features.py</code>, <code>01_eda</code>, baselines, kNN, SVM, SVM derivation</sub></td>
    <td valign="top"><sub><code>02_classification</code>, logistic regression, MLP, validation study, calibration, cost matrix, CIs, logistic derivation</sub></td>
    <td valign="top"><sub><code>03_regression</code>, OLS/Ridge/Lasso/tree, scoring service, CI and tests, linear derivation</sub></td>
    <td valign="top"><sub><code>04_transfer_and_fairness</code>, Random Forest, LightGBM, Kenya→Ghana shift, Model Card, Datasheet, regulation, boosting derivation</sub></td>
  </tr>
</table>

Reviews rotate **Perfect → Henry → Bernard → Eugene → Perfect**. Every pull request needs one approval from the next person in the rotation, and nobody rewrites another member's notebook.

---

## Progress

Status as of **17 Nov 2026**. GitHub's [roadmap issue #1](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/1) is the live source; this table adds work that is finished but still in review.

<sub><img src="https://img.shields.io/badge/-done-2e7d32?style=flat-square"> closed on GitHub &nbsp; <img src="https://img.shields.io/badge/-in_review-1f4e79?style=flat-square"> finished, waiting in a pull request &nbsp; <img src="https://img.shields.io/badge/-in_progress-d4a72c?style=flat-square"> started &nbsp; <img src="https://img.shields.io/badge/-to_do-9aa0a6?style=flat-square"> not started</sub>

| Part | Rubric | Issues closed | Status |
| --- | :---: | :---: | --- |
| **Setup** | — | ![3/4](https://img.shields.io/badge/3%2F4-done-2e7d32?style=flat-square) | Data profiled, roles, repo, shared config. Left: protect `main` ([#56](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/56)) |
| **A · Frame the problem** | 10 pts | ![2/7](https://img.shields.io/badge/2%2F7-in_progress-d4a72c?style=flat-square) | Structure and target EDA done; full EDA report in review. Left: Part A session, tasks and goal, cost matrix sign-off |
| **B · Build & evaluate** | 30 + 20 pts | ![4/14](https://img.shields.io/badge/4%2F14-in_review-1f4e79?style=flat-square) | All 16 models trained; validation study, calibration, CIs and Task 1 metrics done (in review). Left: model choice, interpretation write-up, ablation, temporal calibration |
| **C · Cross-market transfer** | (in B/D) | ![0/3](https://img.shields.io/badge/0%2F3-in_progress-d4a72c?style=flat-square) | Shift figures and domain classifier done; Eugene to own the analysis and notebook 04 |
| **D · Ethics & regulation** | 20 pts | ![0/7](https://img.shields.io/badge/0%2F7-to_do-9aa0a6?style=flat-square) | Fairness and Findex figures ready; Model Card, Datasheet, regulation, mitigation not started |
| **E · Deployment** | 15 pts | ![1/4](https://img.shields.io/badge/1%2F4-in_review-1f4e79?style=flat-square) | Scorer with reasons, review flags and latency (in review). Left: final model, limit rule |
| **Deliverables** | 5 pts + all | ![0/6](https://img.shields.io/badge/0%2F6-to_do-9aa0a6?style=flat-square) | Derivations, report, audit, slides, reflections |

### Notebooks

| Notebook | Owner | Status |
| --- | --- | --- |
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Perfect | ![done](https://img.shields.io/badge/-done-2e7d32?style=flat-square) Train/Test structure (PRs #38, #39); link the EDA report ([#48](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/48)) |
| [`02_classification.ipynb`](notebooks/02_classification.ipynb) | Henry | ![in review](https://img.shields.io/badge/-in_review-1f4e79?style=flat-square) Henry's original restored + calibration, cost threshold, CIs (PR #47) |
| [`03_regression.ipynb`](notebooks/03_regression.ipynb) | Bernard | ![in review](https://img.shields.io/badge/-in_review-1f4e79?style=flat-square) Task 1 models, residuals, scorer demo (PR #47); Task 2 tree to add |
| [`04_transfer_and_fairness.ipynb`](notebooks/04_transfer_and_fairness.ipynb) | Eugene | ![needs owner](https://img.shields.io/badge/-needs_owner-b03a2e?style=flat-square) AI draft with known problems ([#54](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/54)) |

### Done

| Area | What | Evidence |
| --- | --- | --- |
| Data | Three datasets profiled; time-safe features (borrower history, calendar, macro) and 3 fold schemes with leakage checks | [`src/features.py`](src/features.py), [`docs/feature_audit.md`](docs/feature_audit.md) |
| EDA | 10-question data-science report: structure, target, money, leakage, history, categories, Kenya vs Ghana, macro, Findex | [`reports/EDA_REPORT.md`](reports/EDA_REPORT.md) |
| Models | 9 default models and 7 loan-amount models on random, by-customer and by-time validation | [`reports/tables/`](reports/tables) |
| Evaluation | Calibration (Platt), cost threshold with sensitivity, customer-level bootstrap CIs | [`02_classification`](notebooks/02_classification.ipynb) §7–10 |
| Product | Scorer: probability, decision, local-currency limit, 3 reasons, review flags; median latency about 25 ms | [`src/serving/`](src/serving) |
| Evidence | 35 defence figures, each tied to the claim it supports | [`reports/FIGURES.md`](reports/FIGURES.md) |
| Quality | 127 tests incl. end-to-end on fake data; architecture, normalization and time-safety tests; CI on Python 3.12 | [`tests/`](tests) |

---

## What's left, by person

<table>
<tr><td valign="top" width="50%">

**<img src="https://github.com/peaElorm.png?size=40" width="20" align="center"> Perfect**
- [#15](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/15) Baselines, kNN, SVM: review results (trained)
- [#48](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/48) Link the EDA report from `01_eda`
- [#32](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/32) SVM derivation
- Review Eugene's pull requests

**<img src="https://github.com/henryhkb.png?size=40" width="20" align="center"> Henry**
- Review PR #47 and the stacked EDA/figures PR
- [#10](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/10) Tasks and business goal · [#11](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/11) settle the cost assumptions
- [#19](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/19) · [#20](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/20) Check §7–10 of 02, then close
- [#55](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/55) Calibration on 2023+ loans
- [#32](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/32) Logistic regression derivation

</td><td valign="top" width="50%">

**<img src="https://github.com/adjeibernardmarfo.png?size=40" width="20" align="center"> Bernard**
- [#7](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/7) Leak-suspect ruling (evidence in EDA §5)
- [#17](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/17) Task 2 decision tree in 03 · [#21](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/21) close after review
- [#22](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/22) Interpretation write-up · [#52](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/52) ablation
- [#51](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/51) Final scorer model · [#53](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/53) limit rule
- [#32](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/32) Linear regression derivation

**<img src="https://github.com/eugene-sew.png?size=40" width="20" align="center"> Eugene**
- [#54](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/54) Own and fix notebook 04
- [#8](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/8) · [#23](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/23) · [#24](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/24) Kenya→Ghana shift and trust evidence
- [#25](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/25) Model Card · [#49](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/49) Datasheet · [#35](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/35) audit document
- [#26](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/26) Fairness + proxies · [#50](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/50) mitigation
- [#27](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/27) Regulation · [#28](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/28) Findex · [#29](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/29) decline explanation
- [#18](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/18) Random Forest and LightGBM: review results (trained)

</td></tr>
</table>

**Everyone:** [#9](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/9) Part A session · [#33](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/33) notebooks rerun from a clean clone · [#34](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/34) report · [#36](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/36) slides and Q&A · [#37](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/37) reflections · [#56](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/56) protect `main`

---

## Key results so far

| Question | Answer | Evidence |
| --- | --- | --- |
| How rare is default? | 1.83% of 68,654 Kenyan loans; 1.3% in 2022, about 10% from 2023 | [EDA §3](reports/EDA_REPORT.md#3-how-rare-is-default-and-does-it-move) |
| Which columns leak? | Repayment markup alone separates defaulters at AUC 0.98; both repayment columns dropped | [EDA §5](reports/EDA_REPORT.md#5-which-columns-leak-the-answer) |
| How much do random folds flatter? | LightGBM PR-AUC 0.30 random → 0.19 by customer → 0.15 on 2023+ loans | [`val_scheme_pr_auc`](reports/figures/val_scheme_pr_auc.png) |
| Best default models? | MLP 0.207 and Random Forest 0.204 PR-AUC by customer; intervals overlap logistic regression (0.175) | [`eval_model_intervals`](reports/figures/eval_model_intervals.png) |
| Calibrated? | Platt scaling: calibration error 0.30 → 0.002 | [`eval_reliability_before_after`](reports/figures/eval_reliability_before_after.png) |
| Does it transfer to Ghana? | No direct evidence: Ghana loans are 10x larger, 97% from unseen lenders, no labelled history; every Ghana score is flagged for review | [EDA §8](reports/EDA_REPORT.md#8-is-ghana-like-kenya), [`app_decision_mix`](reports/figures/app_decision_mix.png) |
| Open concerns | Month and lender share act as period markers in the tree models ([#22](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/22)); the limit cuts half of requests ([#53](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/53)) | [`interp_shap_lightgbm`](reports/figures/interp_shap_lightgbm.png) |

---

## Quick start

**1. Environment (Python 3.12)**

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt
```

**2. Data** (never committed): unzip the three downloads into `data/raw/` with these folder names:

```text
data/raw/african-credit-scoring-challenge/   Train.csv, Test.csv, economic_indicators.csv   (Zindi)
data/raw/GHA_2021_FINDEX_v02_M_csv/          micro_gha.csv                                  (World Bank)
data/raw/financial-inclusion-in-africa/      Train.csv, ...                                 (Zindi, optional)
```

**3. Run** (each command prints one line per step and a short summary)

| Command | What it does | Output |
| --- | --- | --- |
| `make features` | Build features and folds; fails on any leakage | `data/processed/features.parquet` |
| `make train` | All 16 models on every validation scheme, plus the scorer | `preds/`, `artifacts/` |
| `make eval` | Metrics for every model, cost sensitivity | `reports/tables/` |
| `make eda` | Data-science EDA report | `reports/EDA_REPORT.md` |
| `make figures` | 35 defence figures and their catalogue | `reports/figures/`, `reports/FIGURES.md` |
| `make score` | Sample decisions and 1,000-call latency | terminal, `reports/tables/scorer_latency.csv` |
| `make test` | 127 tests, including end-to-end on fake data | — |
| `make test-notebooks` | Run all 4 notebooks top to bottom (real data, slow) | — |

Without `make` (e.g. on Windows): `python -m scripts.<name>`, for example `python -m scripts.train --task task2 --models logreg lightgbm --schemes fold_group temporal`. Every script has `--help`.

---

## Repository structure

```text
configs/            models.yaml (hyperparameters) · cost_matrix.yaml (measured vs assumed costs)
scripts/            entry points: build_features, train, evaluate, make_eda_report, make_figures, score
src/
  config.py         seed, paths, feature lists (single source of truth)
  features.py       time-safe borrower history, calendar, macro, fold schemes
  analysis/         data_profile (EDA questions), target_eda
  processing/       fold-safe ColumnTransformer, TargetLogScaler (log, standardize, clipped inverse)
  models/           one file per family + registry: baselines, linear, neighbors, svm, trees,
                    ensembles, lightning_modules, data_module
  training/         cross_validation (one prediction format), lightning_runner, artifacts
  evaluation/       metrics, calibration, cost_matrix, bootstrap, fairness, shift
  serving/          schema, scorer, batch, reasons, latency
  visualization/    one house style at 200 dpi: eda, classification, regression,
                    interpretation, fairness, transfer, product
  console.py        the one-line-per-step terminal output every script uses
notebooks/          the four deliverable notebooks, one per owner
reports/            EDA_REPORT.md, FIGURES.md, figures/*.png, tables/*.csv (committed)
tests/              end-to-end, architecture, normalization, feature time-safety, serving, plots
docs/               progress, work log, decisions, feature audit, plans, AI-use log
```

Every model writes the same prediction columns to `preds/` (`ID, customer_id, country_id, lender_id, loan_type, is_new_borrower, split, scheme, fold, model, y_true, y_pred`), so one evaluation script judges them all. Task 2 predictions are raw probabilities, calibrated out of fold before any threshold; Task 1 predictions are amounts in USD.

---

## How we work

1. **One issue, one branch, one pull request.** Branch from the latest `main` as `<role>/<issue>-<short-name>`; never push to `main`.
2. **Link the issue** in the PR description (`Closes #17` or `Refs #17`); a check enforces it.
3. **Docs ship with code:** update `docs/WORKLOG.md` and `docs/PROGRESS.md` (checked) and any doc listed in [`CLAUDE.md`](CLAUDE.md).
4. **Tests and lint pass** (`make test`, `make lint`); CI runs both on Python 3.12.
5. **One approval from the rotation reviewer**, who reruns the branch. Nobody merges their own PR unreviewed.
6. **Owners own their notebooks.** Suggest changes to someone else's notebook in a PR they review.
7. **Log AI use** in [`docs/AI_USE_LOG.md`](docs/AI_USE_LOG.md); each of us must explain our own work without AI at the debrief.

## Documentation

| Page | Purpose |
| --- | --- |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Dated status, findings, blockers, session log |
| [`docs/WORKLOG.md`](docs/WORKLOG.md) | Handoff notes, newest first |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Every data and modelling decision, with its reason |
| [`docs/DATA_AND_PIPELINE_GUIDE.md`](docs/DATA_AND_PIPELINE_GUIDE.md) | Data dictionary, leakage proof, glossary |
| [`docs/feature_audit.md`](docs/feature_audit.md) | All 16 Train columns: available at decision time? used how? |
| [`docs/DELIVERABLES.md`](docs/DELIVERABLES.md) · [`docs/LEARNING_OUTCOMES.md`](docs/LEARNING_OUTCOMES.md) | Rubric checklist and learning outcomes |
| [`reports/EDA_REPORT.md`](reports/EDA_REPORT.md) · [`reports/FIGURES.md`](reports/FIGURES.md) | Exploration report and figure catalogue |
