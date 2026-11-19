<h1 align="center">Responsible Credit Scoring for NkwaPay</h1>

<p align="center">
  <b>ICS553 Machine Learning Essentials · Prosit 2 · Group 2 · Ashesi University</b><br>
  Default-risk and loan-limit models trained on Kenyan digital loans, tested for transfer to Ghana,<br>
  audited for fairness and regulation, and shipped as a scoring service with plain-language reasons.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-1f4e79?style=flat-square">
  <img alt="Tests" src="https://img.shields.io/badge/tests-163_passing-2e7d32?style=flat-square">
  <img alt="Models" src="https://img.shields.io/badge/models-16_trained-1f4e79?style=flat-square">
  <img alt="Figures" src="https://img.shields.io/badge/figures-47-1f4e79?style=flat-square">
  <!-- status:badge --><img alt="Issues" src="https://img.shields.io/badge/issues-39_of_46_closed-d4a72c?style=flat-square"><!-- /status:badge -->
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

<!-- status:progress -->
Generated from GitHub on **19 Nov 2026** by `scripts/update_readme.py` (the *README status* Action keeps it current).

| Part | Rubric | Closed | Still open |
| --- | :---: | :---: | --- |
| **Setup** | — | ![3/4](https://img.shields.io/badge/3%2F4-closed-d4a72c?style=flat-square) | [#56](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/56) |
| **A · Frame the problem** | 10 pts | ![5/7](https://img.shields.io/badge/5%2F7-closed-d4a72c?style=flat-square) | [#7](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/7) [#9](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/9) |
| **B · Build & evaluate** | 30 + 20 pts | ![13/14](https://img.shields.io/badge/13%2F14-closed-d4a72c?style=flat-square) | [#22](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/22) ⏳ |
| **C · Cross-market transfer** | in B, D | ![3/3](https://img.shields.io/badge/3%2F3-closed-2e7d32?style=flat-square) | — |
| **D · Ethics & regulation** | 20 pts | ![6/7](https://img.shields.io/badge/6%2F7-closed-d4a72c?style=flat-square) | [#27](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/27) ⏳ |
| **E · Deployment** | 15 pts | ![4/4](https://img.shields.io/badge/4%2F4-closed-2e7d32?style=flat-square) | — |
| **Deliverables** | 5 pts + all | ![4/6](https://img.shields.io/badge/4%2F6-closed-d4a72c?style=flat-square) | [#36](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/36) [#37](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/37) ⏳ |

<sub>⏳ = a pull request for it is open.</sub>

<!-- /status:progress -->

### Notebooks

| Notebook | Owner | Status |
| --- | --- | --- |
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Perfect | ![done](https://img.shields.io/badge/-done-2e7d32?style=flat-square) Train/Test structure (PRs #38, #39); §6 links the EDA report and records the checks against it ([#48](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/48), branch `p1/32-svm-derivation`, awaiting review) |
| [`02_classification.ipynb`](notebooks/02_classification.ipynb) | Henry | ![merged](https://img.shields.io/badge/-merged-2e7d32?style=flat-square) Henry's original (PR #47) + problem framing (§0), cost matrix in cedis from Ghana's loans (§9b, p\* 0.036) and conclusions (§12); calibration and CIs verified (PR #60); calibration over time and recalibration plan (§11, #55, PR #61) |
| [`03_regression.ipynb`](notebooks/03_regression.ipynb) | Bernard | ![done](https://img.shields.io/badge/-done-2e7d32?style=flat-square) Task 1 models (`ols`, `ridge`, `lasso`, `decision_tree_reg`, `lightgbm_reg`, `lit_linear`), residuals, Task 2 pruned `decision_tree` & pruning settings (#17), and D-29/D-30 scorer demo (executed end-to-end) |
| [`04_transfer_and_fairness.ipynb`](notebooks/04_transfer_and_fairness.ipynb) | Eugene | ![in review](https://img.shields.io/badge/-in_review-1f4e79?style=flat-square) Rebuilt on the shipped scorer: ensembles, Kenya→Ghana shift, calibrated fairness audit, proxies, costed mitigation, Findex ([#54](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/54)); audit docs in [`docs/audit/`](docs/audit/audit.md) |

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

<!-- status:people -->
Find your name, open each issue card and follow the steps. Every card says what the task is, why it matters for the grade, what already exists, where to work, how to run it, and when it is done. Close an issue with a pull request whose description says `Closes #<number>`. Every open issue is **assigned on GitHub** to its owner; signed in, [see the issues assigned to you](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aopen+assignee%3A%40me).

<sub><img alt="a pull request for it is open" src="https://img.shields.io/badge/-in_review-1f4e79?style=flat-square"> a pull request for it is open &nbsp; <img alt="work is on main; owner verifies and closes" src="https://img.shields.io/badge/-merged,_verify_%26_close-2e7d32?style=flat-square"> work is on main; owner verifies and closes &nbsp; <img alt="evidence exists, owner must finish" src="https://img.shields.io/badge/-partly_done-d4a72c?style=flat-square"> evidence exists, owner must finish &nbsp; <img alt="models already ran, owner must review" src="https://img.shields.io/badge/-trained,_review-d4a72c?style=flat-square"> models already ran, owner must review &nbsp; <img alt="not started" src="https://img.shields.io/badge/-to_do-9aa0a6?style=flat-square"> not started &nbsp; <img alt="start today" src="https://img.shields.io/badge/-start_now-b03a2e?style=flat-square"> start today</sub>

| | Person | Open | In review or merged | Not started | Jump to |
| :---: | --- | :---: | :---: | :---: | --- |
| <img src="https://github.com/peaElorm.png?size=60" width="28"> | **Perfect Elorm Avugla** · Data & features | 0 | 0 | 0 | [Perfect's issues](#peaelorm) |
| <img src="https://github.com/henryhkb.png?size=60" width="28"> | **Henry Kofi Brandoh** · Classification & evaluation | 0 | 0 | 0 | [Henry's issues](#henryhkb) |
| <img src="https://github.com/adjeibernardmarfo.png?size=60" width="28"> | **Bernard Marfo Adjei** · Regression, deployment & project lead | 2 | 1 | 0 | [Bernard's issues](#adjeibernardmarfo) |
| <img src="https://github.com/eugene-sew.png?size=60" width="28"> | **Eugene** · Transfer & fairness | 1 | 1 | 0 | [Eugene's issues](#eugene-sew) |
| 👥 | **Everyone** | 4 | 1 | 3 | [Shared issues](#everyone) |

### How to pick up an issue

```bash
git switch main && git pull                      # start from the latest main
git switch -c p4/54-fix-notebook-04              # <role>/<issue>-<short-name>
# ... do the steps on the card, run `make test` ...
git add -A && git commit -m "Fix notebook 04 calibration (#54)"
git push -u origin p4/54-fix-notebook-04
gh pr create --fill                               # description must say: Closes #54
```

Then add a line to `docs/WORKLOG.md` and `docs/PROGRESS.md` (a check enforces it) and ask the next person in the review rotation to approve. The cards below update themselves when issues and pull requests change.

<h3 id="peaelorm"><img src="https://github.com/peaElorm.png?size=60" width="32" align="center"> Perfect Elorm Avugla</h3>

Data & features · [peaElorm](https://github.com/peaElorm) · **0 open** · [Perfect's assigned issues on GitHub](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aopen+assignee%3ApeaElorm)

All of this person's issues are closed. ✅


<h3 id="henryhkb"><img src="https://github.com/henryhkb.png?size=60" width="32" align="center"> Henry Kofi Brandoh</h3>

Classification & evaluation · [henryhkb](https://github.com/henryhkb) · **0 open** · [Henry's assigned issues on GitHub](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aopen+assignee%3Ahenryhkb)

All of this person's issues are closed. ✅


<h3 id="adjeibernardmarfo"><img src="https://github.com/adjeibernardmarfo.png?size=60" width="32" align="center"> Bernard Marfo Adjei</h3>

Regression, deployment & project lead · [adjeibernardmarfo](https://github.com/adjeibernardmarfo) · **2 open** · [Bernard's assigned issues on GitHub](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aopen+assignee%3Aadjeibernardmarfo)

<details>
<summary><img alt="evidence exists, owner must finish" src="https://img.shields.io/badge/-partly_done-d4a72c?style=flat-square"> <b>#7 · Explore amounts, durations and the two leak suspects</b></summary>

| | |
| --- | --- |
| **What it is** | Explore amounts and terms, and rule on the two repayment columns. |
| **Why it matters** | Rubric A (10 pts): leakage correctly identified |
| **Already done** | EDA §4–5: amounts are heavy-tailed (log needed); the markup separates defaulters at AUC 0.98, so both columns are dropped (D-2). |
| **Where to work** | `docs/DECISIONS.md` |
| **How to run** | `make eda` |
| **Done when** | D-2 is decided at the session; close. |
| **Automated check** | The leak-suspect ruling (D-2) is decided, backed by the leakage section of the EDA report. Run `make acceptance ISSUE=7`; it also runs on every pull request that says `Closes #7` and when the issue is closed (a failing check reopens it). |

**Steps**

1. Present the ruling at the Part A session (#9) using `eda_leakage_univariate_auc.png` and `eda_markup_by_outcome.png`.
2. Mark D-2 decided in `docs/DECISIONS.md`.

[Open issue #7 on GitHub →](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/7)

</details>

<details>
<summary><img alt="a pull request for it is open" src="https://img.shields.io/badge/-in_review-1f4e79?style=flat-square"> <b>#22 · Interpretation: coefficients, importances or SHAP</b></summary>

| | |
| --- | --- |
| **What it is** | Explain, in plain language, what drives the default model. |
| **Why it matters** | Rubric B: interpretation; LO6: explanations |
| **Already done** | Figures: `interp_coefficients.png` (logistic regression), `interp_shap_lightgbm.png`. Finding: the tree model's top drivers (lender share, month) mostly mark the period, not the borrower. |
| **Where to work** | `docs/interpretation.md` (then the report's Part B) |
| **How to run** | `make figures` |
| **Done when** | The interpretation is written and matches the figures. |
| **Automated check** | docs/interpretation.md explains the main drivers in plain language, using the coefficient and SHAP figures. Run `make acceptance ISSUE=22`; it also runs on every pull request that says `Closes #22` and when the issue is closed (a failing check reopens it). |
| **Pull request** | [#76](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/pull/76) |

**Steps**

1. Write one page: top drivers, direction, and what they mean for a borrower.
2. State the period-marker finding and what the ablation (#52) shows.

[Open issue #22 on GitHub →](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/22)

</details>


<h3 id="eugene-sew"><img src="https://github.com/eugene-sew.png?size=60" width="32" align="center"> Eugene</h3>

Transfer & fairness · [eugene-sew](https://github.com/eugene-sew) · **1 open** · [Eugene's assigned issues on GitHub](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues?q=is%3Aopen+assignee%3Aeugene-sew)

<details>
<summary><img alt="a pull request for it is open" src="https://img.shields.io/badge/-in_review-1f4e79?style=flat-square"> <b>#27 · Regulatory mapping: BoG Digital Credit Directive, Act 843, Act 726</b></summary>

| | |
| --- | --- |
| **What it is** | Map our design to the Bank of Ghana Digital Credit Directive, the Data Protection Act 2012 (Act 843) and the Credit Reporting Act 2007 (Act 726). |
| **Why it matters** | Rubric D: regulatory mapping |
| **Already done** | Nothing yet. |
| **Where to work** | `docs/audit/regulatory_map.md` |
| **How to run** | — |
| **Done when** | A table of requirements, our response, and gaps. |
| **Automated check** | docs/audit/regulatory_map.md maps the design to the BoG Directive, Act 843 and Act 726, with gaps. Run `make acceptance ISSUE=27`; it also runs on every pull request that says `Closes #27` and when the issue is closed (a failing check reopens it). |
| **Pull request** | [#75](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/pull/75) |

**Steps**

1. List each relevant requirement and how we meet it (reason codes, review flags, data minimisation).
2. List what we cannot satisfy yet.

[Open issue #27 on GitHub →](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/27)

</details>


<h3 id="everyone">👥 Everyone</h3>

<details>
<summary><img alt="start today" src="https://img.shields.io/badge/-start_now-b03a2e?style=flat-square"> <b>#9 · Part A session: loan timeline, label definition, feature audit, leakage rulings</b></summary>

| | |
| --- | --- |
| **What it is** | Part A session: agree the loan timeline, the label, the feature audit and the leakage rulings. |
| **Why it matters** | Rubric A (10 pts) |
| **Already done** | Drafts: `docs/feature_audit.md`, `docs/DECISIONS.md` (D-2 to D-7, D-18 to D-20 are drafts). |
| **Where to work** | `docs/part_a.md` (timeline, label, decision moment), `docs/DECISIONS.md` |
| **How to run** | — |
| **Done when** | All Part A decisions are marked decided. |
| **Automated check** | Part A notes exist (timeline, label, decision moment) and decisions D-2 to D-7 are decided. Run `make acceptance ISSUE=9`; it also runs on every pull request that says `Closes #9` and when the issue is closed (a failing check reopens it). |

**Steps**

1. Meet (30–45 min).
2. Draw the timeline: application → approval → disbursement → due date → repaid/defaulted; mark the decision moment (`disbursement_date`).
3. Define the label (default = not repaid by the due date).
4. Mark each draft decision decided or changed.

[Open issue #9 on GitHub →](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/9)

</details>

<details>
<summary><img alt="not started" src="https://img.shields.io/badge/-to_do-9aa0a6?style=flat-square"> <b>#56 · Protect main: require one approving review and passing CI</b></summary>

| | |
| --- | --- |
| **What it is** | Turn on branch protection so nothing reaches `main` without review and passing CI. |
| **Why it matters** | Process |
| **Already done** | PR #46 was merged without review. |
| **Where to work** | GitHub settings |
| **How to run** | — |
| **Done when** | Direct pushes to `main` are blocked. |
| **Automated check** | main requires a pull request with one approving review (checked through the GitHub API). Run `make acceptance ISSUE=56`; it also runs on every pull request that says `Closes #56` and when the issue is closed (a failing check reopens it). |

**Steps**

1. An admin: Settings → Branches → add rule for `main` → require a pull request, 1 approval, and status checks `lint-and-test`, `docs-updated`, `check-issue-link`.
2. Allow GitHub Actions to push to `main` (add it to the bypass list), or the *README status* Action cannot commit its updates.

[Open issue #56 on GitHub →](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/56)

</details>

<details>
<summary><img alt="not started" src="https://img.shields.io/badge/-to_do-9aa0a6?style=flat-square"> <b>#36 · Debrief slides and mock Q&A</b></summary>

| | |
| --- | --- |
| **What it is** | Debrief slides and mock Q&A. |
| **Why it matters** | Rubric E (15 pts): presentation and defence |
| **Already done** | `reports/FIGURES.md` lists what each figure defends. |
| **Where to work** | `slides/` |
| **How to run** | — |
| **Done when** | Slides ready and rehearsed. |
| **Automated check** | Debrief slides are in slides/. Run `make acceptance ISSUE=36`; it also runs on every pull request that says `Closes #36` and when the issue is closed (a failing check reopens it). |

**Steps**

1. Each presents their part (about 3–4 minutes).
2. Rehearse questions without notes; the AI policy requires explaining your work without AI.

[Open issue #36 on GitHub →](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/36)

</details>

<details>
<summary><img alt="a pull request for it is open" src="https://img.shields.io/badge/-in_review-1f4e79?style=flat-square"> <b>#37 · Individual reflections</b></summary>

| | |
| --- | --- |
| **What it is** | One-page individual reflection each. |
| **Why it matters** | Rubric F (5 pts) |
| **Already done** | — |
| **Where to work** | `docs/reflections/<name>.md` |
| **How to run** | — |
| **Done when** | Four reflections submitted. |
| **Automated check** | Four individual reflections are in docs/reflections/. Run `make acceptance ISSUE=37`; it also runs on every pull request that says `Closes #37` and when the issue is closed (a failing check reopens it). |
| **Pull request** | [#75](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/pull/75) |

**Steps**

1. Your contribution, what you learned, what you would do differently.

[Open issue #37 on GitHub →](https://github.com/ASU-MICS-2028/mle-group-2-prosit-2/issues/37)

</details>


<!-- /status:people -->

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
| `streamlit run app/streamlit_app.py` | Borrower app over the live scorer (needs `make artifacts`) | browser, http://localhost:8501 |
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
7. **Every issue has an acceptance test** in [`tests/acceptance/test_issues.py`](tests/acceptance/test_issues.py) that passes only when it is done. `make acceptance` shows the scoreboard (`ISSUE=53` for one). A pull request that says `Closes #N` must pass issue N's test, and closing an issue whose test fails reopens it automatically.
8. **The README status updates itself.** Close issues and open PRs as usual; the *README status* Action rebuilds the progress table and issue cards. To change a card's text, edit [`docs/issue_cards.yaml`](docs/issue_cards.yaml).
9. **Log AI use** in [`docs/AI_USE_LOG.md`](docs/AI_USE_LOG.md); each of us must explain our own work without AI at the debrief.

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
