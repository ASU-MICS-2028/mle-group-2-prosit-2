# Model Card: NkwaPay default-risk scorer (Task 2)

Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*. Owner: Eugene (P4). Figures come from `notebooks/04_transfer_and_fairness.ipynb` (run 2026-11-17 on the pipeline after PR #60) unless noted.

## Model details

- **What it is:** a logistic regression with balanced class weights, followed by **Platt calibration** fitted on out-of-fold (GroupKFold) scores. It predicts the probability that a digital micro-loan defaults. It is built with scikit-learn: training in `src/training/artifacts.py` (`fit_and_save_deployment_artifacts`, `clf_model="logreg"`) and serving in `src/serving/scorer.py`. It is saved as `artifacts/task2_clf.joblib` by `make artifacts`.
- **Inputs:** 14 numeric features and 2 categorical ones (`config.NUM_FEATURES_T2`, `CAT_FEATURES`):
  - loan amount in USD and term;
  - borrower history built from earlier loans only;
  - calendar fields;
  - lender and loan type.

  Since D-18 there are **no macro columns and no local-currency amount**.
- **Decision rule** (`decide` in `src/serving/scorer.py`):
  - **Decline** when the calibrated probability is above $C_{FP}/(C_{FP}+C_{FN})$. With the shipped cedi cost matrix (D-28) that is **0.10**.
  - **Starter loan (D-26):** a first-time borrower between 0.10 and 0.30 is offered a loan of 25% of the request instead of a decline.
  - Otherwise, **approve** the request, or **limit** it to the Task 1 recommended amount (LightGBM, in USD, converted to local currency) when the request is larger than that amount.
  - Every decision comes with 3 reasons for the credit team (`top_3_reasons`) and a borrower-facing version (`borrower_reasons`). The borrower version rewords the funding-lender reason and drops filler.
  - A `needs_review` flag marks applications outside the training scope: an unseen country, lender or product, or an amount outside the range.
- **Why this model:** Random Forest and LightGBM rank defaulters slightly better by customer (PR-AUC 0.203 and 0.195, against 0.176). That edge disappears by time, where every model falls to a ROC-AUC of about 0.6. The credit committee asked for a model it can justify, and logistic regression gives one weight per input.
- **Version:** course submission, November 2026, ICS553 Group 2.

## Intended use

- **Primary use:** decision support for small, short-term (mostly 7-day) digital loans to **Kenyan** applicants who resemble the training population. Declines have a human review route.
- **Users:** NkwaPay's credit team. Borrowers see the decision and the reasons in the app.
- **Out of scope:**
  - **automated lending in Ghana** (see *Caveats*);
  - pricing loans;
  - inferring gender, ethnicity, religion or region, directly or through proxies.

## Factors

- **Groups audited:** country, borrower tenure (first-time or repeat), funding lender, product (`loan_type`) and loan-size quartile. These are the slices the brief names.
- **Not available:** gender, age, region and income. The only view of these is Global Findex Ghana 2021, which describes the adult population, not our applicants.

## Metrics

- **Ranking:** PR-AUC (headline, because defaults are rare) and ROC-AUC.
- **At the threshold:** approval rate, precision, recall, F1, and the false-positive rate (good borrowers declined).
- **Fairness:** approval rate per group and the disparate-impact ratio, which is group approval ÷ highest group approval. Below 0.8 is the usual warning sign.
- **Money:** expected cost of decisions in GHS, from the cost matrix.
- **Uncertainty:** customer-level bootstrap 95% intervals, in `reports/tables/task2_bootstrap_ci.csv` (P2).

## Training data

- **Training:** `Train.csv`, 68,654 Kenyan loan rows from 6,540 customers, Oct 2021 – Nov 2024, with a 1.83% default rate. Leaky repayment columns are removed. History is built only from earlier loans whose due date has passed. See `docs/audit/datasheet.md`.

## Evaluation data

- **Evaluation:** 5-fold GroupKFold by customer, and a temporal split (train before 2023-01-01, validate after). Ghana appears only in `Test.csv` (3,525 rows) and has no labels, so it can be described but not evaluated.

## Quantitative analyses

**Overall (by customer, calibrated unless stated)**

| | ROC-AUC | PR-AUC | Approval rate | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| Raw scores (class-weighted) at 0.10 | 0.843 | 0.176 | 14.5% | n/a | n/a |
| **Calibrated, shipped threshold 0.10** | 0.843 | 0.175 | **97.5%** | 0.262 | 0.192 |
| Calibrated, old KES threshold 0.019 (D-19) | 0.843 | 0.175 | 82.8% | 0.672 | 0.072 |
| By time (raw) | 0.633 | 0.134 | n/a | n/a | n/a |

Calibration error (ECE) falls from 0.304 to 0.002 after calibration. Raw class-weighted scores (mean 0.32, against a true rate of 0.018) must never be compared with the threshold.

**By group (calibrated, by customer, threshold 0.10, before the starter loan)**

| Group | n | Approval | Actual default rate | Disparate-impact ratio |
| --- | --- | --- | --- | --- |
| Repeat borrowers | 68,087 | 97.9% | 1.7% | 1.00 |
| **First-time borrowers** | 567 | **48.7%** | 21.0% | **0.50** |
| **First-time borrowers, with the starter loan** | 567 | **81.3%** | 21.0% | **0.83** |
| Loan size, smallest quartile | 17,166 | 99.6% | 1.3% | 1.00 |
| Loan size, largest quartile | 17,164 | 93.9% | 2.9% | 0.94 |
| Lender 267278 (94% of rows) | 64,653 | 99.0% | 1.2% | 1.00 |
| Lender 251804 | 3,542 | 78.4% | 8.9% | 0.79 |
| Lenders 267277 / 245684 | 271 / 188 | 25.8% / 34.6% | 29.5% / 41.5% | 0.26 / 0.35 |
| Product Type_1 (90% of rows) | 61,723 | 100% | 0.8% | 1.00 |
| Products Type_4, 6, 9 | 1,797 | 32–57% | 11–18% | 0.32–0.57 |
| Product Type_14 | 99 | 0% | 71.7% | 0.00 |

**Ghana (Test, no labels), through the full scorer:**
- Only 7% are approved in full: 79% are limited, to a median 46% of the request (66% in Kenya, under the D-30 quantile cap), and 14% are declined.
- Mean probability is 0.081, against 0.017 in Kenya.
- 100% are flagged `needs_review`.

## Ethical considerations

- **First-time borrowers carry the largest gap.**
  - They are approved 48.7% of the time against 97.9% (ratio 0.50). The gap tracks real risk (21% default), but it falls on people with no history, who Findex suggests are disproportionately women, rural and poorer adults.
  - The size of the gap is a cost-matrix choice: with the same model, first-time approval is 0% under the old KES matrix, 12% in cedis with 1 lost loan, 49% with 3 (shipped) and 70% with 6.
  - **Mitigation adopted (D-26, in the scorer):** a 25% starter loan up to p = 0.30. It lifts first-time approval to 81.3% (ratio 0.83) for GHS 10.0k of extra expected cost on 567 applicants, about GHS 18 each. Approving the same band at full size would cost GHS 38.7k.
- **Proxies.**
  - Lender and product identify Ghana perfectly (proxy strength 1.0), and USD amount nearly so (0.90).
  - `n_past_loans` (0.91) and `n_past_labelled` (0.83) identify first-time borrowers.
  - So dropping a group flag (country or tenure) would not remove its effect.
- **Who the data leaves out.** Labels exist only for people previously approved, and defaulters are rarely lent to again (3.5%), so this is the reject-inference problem (D-14). The data also misses the roughly 40% of Ghanaian adults with no mobile-money account.
- **A looser threshold means more defaults approved.** At 0.10 the model catches only 26% of defaulters (67% at 0.019). That is what the cedi costs imply, so the recovery and collection assumptions need checking in the pilot.
- **Regulation.** See `docs/audit/regulatory_map.md`. Several BoG Directive requirements are not yet met: a regulatory loan cap, pre-agreement cost disclosure, decision logging and bureau reporting.

## Caveats and recommendations

1. **Do not automate Ghana with this model.** The domain classifier separates Ghana from Kenya with an AUC of 0.985 even after removing the label-history artefact, and importance weighting would rest on 4% of Kenyan rows. Ghanaian probabilities depend on how the unknown label history is encoded: the median moves from 0.011 to 0.028. Run a **capped pilot of about 3,500 loans over about 3 months**, then recalibrate on Ghanaian outcomes (notebook §2.4).
2. **Encode missing history as unknown.** Label-derived history is structurally 0 for every Ghanaian row because no Ghanaian loan has a label.
3. **Use calibrated probabilities everywhere.** Rates read off raw class-weighted scores are wrong by an order of magnitude.
4. **Measure the cost assumptions.** Recovery, collection cost and the 3 future loans lost (D-28) are assumptions, and first-time approval moves from 12% to 70% across them. Replace them with pilot figures.
5. **Expect drift.** Default rates moved from 1.3% (2022) to 11.5% (2023), and ROC-AUC by time is about 0.6. Monitor monthly and recalibrate on recent outcomes (#55).
6. **Keep a human in the loop for declines**, and log every decision for the Bank of Ghana (Directive 3.7).
