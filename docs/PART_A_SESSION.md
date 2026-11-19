# Part A session: agenda and decision sheet (issue #9)

**Goal:** leave the session with every Part A decision marked `decided` in [DECISIONS.md](DECISIONS.md). Part A is 10 points, and the whole team's results already rely on these drafts. **Time:** about 45 minutes. **Who:** all four; Perfect chairs (the data decisions are his), Henry speaks to the cost matrix and label, Bernard to the pipeline, Eugene to Ghana.

## 1. What the brief asks for in Part A, and where each piece is

| Brief item | Where it is | State |
| --- | --- | --- |
| 1a. Timeline of one loan, decision moment, closest column | [`reports/figures/part_a_loan_timeline.png`](../reports/figures/part_a_loan_timeline.png) (made by `python -m scripts.make_timeline_figure`) | Drawn; confirm at the session |
| 1b. Label and the period it is measured over | `02_classification` §0: `target` = lender's recorded outcome for loans disbursed Oct 2021 to Nov 2024; known only after the due date (D-13) | Written; confirm |
| 1c. Feature audit table for every column of `Train.csv` | [`feature_audit.md`](feature_audit.md) (16 columns plus engineered features) | Draft; confirm row by row |
| 2. Two tasks and the business goal | `02_classification` §0 (Task 1 amount, Task 2 default; goal: decline if p > p\*, with a proposed 80% approval floor, D-22) | Written; confirm |
| 3. Cost matrix in cedis | `configs/cost_matrix.yaml` (D-22, D-28): C_FP 270 GHS, C_FN 2,416 GHS, threshold about 0.10 | Decided |

## 2. The decisions to settle (draft → decided)

Each row: the question for the room, the evidence, what I recommend, and what changes if the team disagrees. Evidence is in the repo; nothing needs re-running.

| # | Question | Evidence | Recommendation | If we disagree |
| --- | --- | --- | --- | --- |
| **D-2** | Are `Total_Amount_to_Repay` and `Lender_portion_to_be_repaid` leaks? | Markup alone separates default at AUC 0.98 (`eda_leakage_univariate_auc.png`); repaid loans owe a median 0.76% extra, defaulted 22.1% (`eda_markup_by_outcome.png`) | **Drop both.** Say in the report that the data owner's definitions call the column "expected", so this is a judgement from the data, not a proof | Every score becomes near-perfect and meaningless; all results must be re-run |
| **D-3** | Two-lender loans: keep rows, group by loan | 2,134 loans appear twice, same borrower, amount and date; labels agree 99.9% | **Confirm.** The "default far more (8.3% vs 1.4%)" figure is mostly timing (two-lender loans are 1% of 2022 loans but 46% of 2023 and 64% of 2024); do not use it as a reason | Fold leakage between sibling rows; scores inflate |
| **D-4** | The 4,306 shared customers and 920 shared loans in Train and Test | The 920 are the other lender's share of two-lender loans, not duplicates | **Confirm:** keep in training, fold by customer, never validate on a row whose sibling is in training | Same as D-3 |
| **D-5** | Unseen lenders and loan types in Test | 4 unseen lenders (18% of Test rows), Type_3 and Type_8 (17%); 97% of Ghana rows have an unseen lender | **Confirm:** one-hot with `handle_unknown="ignore"`; the scorer flags unseen values for review | Ghana scoring crashes or silently mislabels |
| **D-6** | Currency: amounts in USD via the previous year's exchange rate | `log_amount_usd` makes Kenya and Ghana comparable | **Confirm.** Note D-18 already stops models from using `log_amount_local` | Limits and costs mix shillings and cedis |
| **D-13** | When is a past default "known"? | Only when the earlier loan's **due date** is before this loan's disbursement; brute-force checked on 400 rows in `validate()` | **Confirm.** This is the most important leakage rule in the project | History features leak the answer |
| **D-14** | Defaulters are cut off, so labels cover only approved borrowers (status: open) | Only 3.5% of customers who default ever get another loan | **Mark decided as a stated limitation** (selective labels), repeated in the audit | Not a choice; it is a fact. We only decide whether to state it |
| **D-18** | Models do not use the 4 macro columns or `log_amount_local`; Task 1 predicts the amount in USD | Kenya-only training gives each macro column 4 yearly values; with `log_amount_usd` the local amount encodes the exchange rate, i.e. the country | **Confirm** (Bernard proposed it 2026-11-17, team to confirm) | Models can read the country off the features and extrapolate wildly on Ghana |

## 3. Questions to leave the session able to answer

- Which column is the closest to the decision moment, and why is it not the application date? (`disbursement_date`; the data has no application or approval date.)
- Why is the label only known after the due date, and what does that do to borrower history? (D-13)
- Why do we call the repayment columns "invalid", and what is the honest caveat?
- Who is missing from the labels, and what does that do to Ghana? (D-14)

## 4. After the session (15 minutes, Perfect)

1. In [DECISIONS.md](DECISIONS.md), change the status of each confirmed row to `decided`, with the date and names in the last column. Do not edit a decision silently: if the team changes one, mark it `superseded` and add a row.
2. Tick the Part A rows in [PROGRESS.md](PROGRESS.md), [LEARNING_OUTCOMES.md](LEARNING_OUTCOMES.md) and the feature audit's status line.
3. Close #9 (and #7 once D-2 is decided). Issue #9's card still lists D-7, D-19 and D-20 as drafts; D-7 and D-20 are decided and D-19 is superseded by D-28.
