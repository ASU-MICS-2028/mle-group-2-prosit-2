# Model Interpretation: What Drives Default Risk and Loan Limits (#22)

**Owner:** Bernard (P3) · **Figures:** [`interp_coefficients.png`](../reports/figures/interp_coefficients.png), [`interp_shap_lightgbm.png`](../reports/figures/interp_shap_lightgbm.png), [`interp_ablation.png`](../reports/figures/interp_ablation.png), [`interp_ablation_roc.png`](../reports/figures/interp_ablation_roc.png) · **Data:** [`reports/tables/ablation.csv`](../reports/tables/ablation.csv) · **Reason-code engine:** [`src/serving/reasons.py`](../src/serving/reasons.py)

---

## 1. Plain-language summary

What actually drives NkwaPay's default-risk predictions? Looking at the deployed logistic regression's **coefficients** alongside LightGBM's **SHAP** values and our feature-group **ablation study (#52)** reveals two distinct stories:

1. **Genuine borrower and product risk:** whether a borrower has built a track record of past loans (`n_past_loans`, `days_since_last_loan`, `is_new_borrower`), which loan product they apply for (`loan_type`), and their loan terms (`duration`) carry real credit signal that holds across customers and into 2023+ loans.
2. **Period markers inside tree models:** LightGBM's top SHAP drivers — the lender's funding share (`Lender_portion_Funded`, mean |SHAP| 1.21) and disbursement **month** (`disb_month`, mean |SHAP| 0.61) — largely mark *when* a loan was issued (the 2022 low-default regime at 1.3% vs the 2023+ regime at ~10%), not individual borrower creditworthiness.

This contrast is a central reason we deploy Platt-calibrated logistic regression (D-29) and hide calendar and internal funding-share fields from borrower-facing reasons (`NEVER_SHOWN` in `src/serving/reasons.py`).

---

## 2. Deployed model: logistic regression coefficients (`interp_coefficients.png`)

In logistic regression on standardized inputs, a positive weight **raises** the log-odds of default and a negative weight **lowers** it. The largest weights in the deployed model are:

| Feature | Standardized weight (log-odds) | Direction | Plain-language meaning for a borrower |
| --- | ---: | --- | --- |
| `loan_type_Type_23` | +4.59 | Raises risk | High-default specialty product (small count, high observed loss rate). |
| `loan_type_Type_14` | +1.67 | Raises risk | Longer-term / higher-default product category. |
| `loan_type_Type_20`, `Type_2`, `Type_5` | +1.40, +1.29, +1.04 | Raises risk | Products with historically higher missed-repayment rates. |
| `n_past_loans` | −3.62 | Lowers risk | Returning borrowers with more prior loans default far less often than first-time borrowers (1.7% vs 21.0% on Train). |
| `n_past_labelled` | +2.35 | Partial offset (collinear) | Strongly correlated with `n_past_loans`; because `n_past_loans` carries −3.62, the *net* effect of each completed repaid loan remains strongly protective (−1.27 log-odds per SD). |
| `loan_type_Type_17`, `Type_16`, `Type_1` | −2.95, −2.01, −1.65 | Lowers risk | Standard short-term products (`Type_1` is Kenya's core 7-day loan, ~1.2% default rate). |

**Why collinearity shows up in coefficients:** As derived in [`docs/derivations/logistic.md`](derivations/logistic.md) §9, `n_past_loans` and `n_past_labelled` move closely together, so L2-regularised logistic regression splits weight across them with opposite signs. In production, `src/serving/reasons.py` checks `_is_true_of_borrower()` so a clean returning borrower is never shown a contradictory reason (such as "missed repayments" when `n_past_defaults == 0`).

---

## 3. Tree model: LightGBM SHAP values (`interp_shap_lightgbm.png`)

Across a 6,000-loan sample of Train, LightGBM's top drivers by **mean |SHAP value|** (average magnitude of push on the log-odds) are:

| Rank | Feature | Mean \|SHAP\| | What it is actually capturing |
| ---: | --- | ---: | --- |
| 1 | `Lender_portion_Funded` | 1.209 | **Period / lender-mix marker:** funding share changed sharply when lender `267278`'s share dropped from 97% (2022) to 49% (2023+). |
| 2 | `days_since_last_loan` | 0.613 | **Borrower activity:** recent repeat borrowers repay reliably; blank (missing indicator) flags first-time borrowers. |
| 3 | `disb_month` | 0.606 | **Calendar period marker:** default rates spiked in specific months of 2023–2024; trees split on calendar month to separate regimes. |
| 4 | `n_past_loans` | 0.588 | **Borrower tenure:** core behavioural signal separating repeat vs first-time borrowers. |
| 5 | `n_past_labelled` | 0.385 | **Borrower track record:** how many earlier loans have already passed their due date (always 0 for Ghana Test rows). |
| 6 | `loan_type_Type_1` | 0.382 | **Product type:** distinguishes Kenya's main 7-day product from higher-risk products. |
| 7 | `log_amount_usd` | 0.331 | **Loan size:** larger requests carry higher exposure and slightly different risk profiles. |
| 8–9 | `disb_dom` (0.214), `disb_dow` (0.092) | 0.306 combined | **Calendar noise:** day-of-month and day-of-week splits that memorise training dates rather than borrower risk. |

The pruned Task 2 decision tree (`03_regression.ipynb` §4, depth 6, 47 leaves) echoes the same pattern: its top five splits are `loan_type_Type_1` (0.571), `days_since_last_loan` (0.137), `n_past_loans` (0.078), `disb_month` (0.077), and `Lender_portion_Funded` (0.068).

---

## 4. What the ablation study proves (#52)

To separate genuine credit signal from random-fold / period-marker artefacts, [`scripts/run_ablation.py`](../scripts/run_ablation.py) retrained both deployed `logreg` and `lightgbm` after removing one feature group at a time — and after adding the 4 macro columns back — on random folds (`fold_strat`), by-customer folds (`fold_group`), and 2023+ loans (`temporal`). Full table in [`reports/tables/ablation.csv`](../reports/tables/ablation.csv); charts in `interp_ablation.png` and `interp_ablation_roc.png`:

| Variant | `logreg` ΔPR (Group) | `logreg` ΔROC (Group) | `logreg` ΔROC (Time) | `lightgbm` ΔPR (Random) | `lightgbm` ΔROC (Time) | `lightgbm` ΔPR (Time) | Takeaway |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `- borrower_history` | +0.002 | **−0.053** | **−0.036** | **−0.056** | **−0.028** | −0.001 | Largest ROC-AUC loss on every split: borrower track record carries core ranking power. |
| `- lender_product` | **−0.031** | −0.006 | −0.009 | −0.015 | **−0.024** | **−0.020** | Biggest PR-AUC driver for `logreg` (Group) and `lightgbm` (Temporal). |
| `- loan_terms` | −0.001 | −0.003 | **−0.018** | −0.010 | **−0.024** | −0.002 | Genuine temporal signal: longer terms default far more often (1.2% at 7d vs 31% >60d). |
| `- calendar` | **+0.008** | −0.002 | −0.001 | **−0.044** | **+0.020** | **+0.005** | **Smoking gun:** calendar inflates LightGBM random-fold PR-AUC (+0.044) but *hurts* 2023+ temporal accuracy! |
| `- amount` | +0.001 | −0.000 | +0.000 | −0.008 | −0.009 | −0.004 | Minimal marginal gain once product and duration are known. |
| `+ macro` | **−0.007** | +0.000 | **−0.005** | +0.004 | 0.000 | 0.000 | Confirms D-18: macro indicators hurt `logreg` on every split and add 0.000 temporal value for LightGBM (and removing them raised `lit_mlp` temporal ROC-AUC from **0.48 → 0.65**). |

**Key lesson for NkwaPay:**
- **Calendar features (`disb_month`, `disb_dow`, `disb_dom`) are period markers, not creditworthiness.** On random folds, LightGBM leans on exact dates to inflate PR-AUC by +0.044, but when tested honestly on 2023+ loans, removing calendar *raises* LightGBM ROC-AUC from 0.582 → **0.602** and PR-AUC from 0.148 → **0.153**.
- **Borrower history, product type, and loan duration carry real signal** — yet Ghana has 0% labelled borrower history and 87% unseen loan types, which is why every Ghana score is flagged for human review and must be validated in a capped pilot (D-27, `docs/audit/ghana_trust.md`).

---

## 5. How interpretations become borrower-facing reasons

In [`src/serving/reasons.py`](../src/serving/reasons.py):
- **Exact per-borrower contributions:** Because the deployed classifier is `logreg` (D-29), each feature's contribution to an applicant's log-odds is computed directly as `w_j × x_j` in ~21 ms (no slow SHAP approximation needed at serving time).
- **Actionable filter (`NEVER_SHOWN`):** `disb_month`, `disb_dow`, `disb_dom`, `Lender_portion_Funded`, and `n_lenders_on_loan` are never shown to borrowers — a borrower cannot change the month of the year or internal lender funding split.
- **Record consistency & plain language:** `_is_true_of_borrower()` blocks contradictory reasons caused by correlated linear weights, and `borrower_reasons()` rewords internal lender risk as a product suggestion while dropping filler text, matching `docs/audit/decline_explanation.md`.
- **Task 1 loan-limit drivers:** In `03_regression.ipynb` §3, `lightgbm_reg` splits primarily on loan duration, borrower history (`n_past_loans`, `days_since_last_loan`), lender share, and product type; using its 85th-quantile version (`lightgbm_reg_q85`, D-30) turns that prediction into an upper cap so 84.3% of repaid Kenyan loans stay below their limit.
