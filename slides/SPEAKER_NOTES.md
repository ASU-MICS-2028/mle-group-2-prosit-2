# Debrief Presentation Guide & Mock Q&A (#36)

**Deck:** [`NkwaPay_Prosit2_Debrief.pptx`](NkwaPay_Prosit2_Debrief.pptx) · **Length:** 10 slides · **Target time:** 15 minutes (~3.5 min per member)

---

## 1. Run of Show (10 Slides)

| Slide | Title | Presenter | Time | Key figure |
| :---: | --- | --- | :---: | --- |
| 1 | Title & Executive Recommendation | Bernard (opens) | 1.0 min | — |
| 2 | Problem Framing, Data Reality & Leakage Guardrails | Perfect (P1) | 2.0 min | `part_a_loan_timeline.png` |
| 3 | Honest Validation: Why Random Splits Mislead | Perfect / Henry | 1.5 min | `val_scheme_pr_auc.png` |
| 4 | Model Comparison, Calibration & Cedi Cost Threshold | Henry (P2) | 2.0 min | `eval_reliability_before_after.png` |
| 5 | What Drives Risk? Coefficients, SHAP & Ablation | Bernard (P3) | 2.0 min | `interp_ablation.png` |
| 6 | Loan Limits: 85th-Quantile Cap (#53) | Bernard (P3) | 1.5 min | `app_decision_mix.png` |
| 7 | Kenya → Ghana Transfer Analysis (Part C) | Eugene (P4) | 1.5 min | `shift_domain_scores.png` |
| 8 | Fairness, Starter Loan & Regulation (Part D) | Eugene (P4) | 2.0 min | `fair_selection_tenure.png` |
| 9 | Live Scoring Service, Reasons & Streamlit App (Part E) | Bernard & Eugene | 1.0 min | `app_latency.png` |
| 10 | Four-Step Rollout & Monitoring Plan | Bernard (closes) | 0.5 min | — |

---

## 2. Mock Q&A Rehearsal (Answer Without Notes)

Every member should be able to answer questions across all sections (AI policy: explain and defend without AI).

1. **Why did you drop `Total_Amount_to_Repay` and `Lender_portion_to_be_repaid`? Can you prove they leak?**
   *Answer (D-2):* On their own the markup separates defaults at **AUC 0.98** (median markup 0.8% on repaid loans vs 22.1% on defaulted loans), which looks like late penalties added after default. The honest caveat: the data dictionary calls the column "expected", so it is an empirical judgement from the data, not legal proof — we drop both so no score is artificially perfect.
2. **Why is accuracy useless here, and why PR-AUC over ROC-AUC?**
   *Answer:* Defaults are **1.83%** of loans; predicting "everyone repays" gives 98.2% accuracy with zero defaults caught. ROC-AUC rewards true-negative volume (98% repayers), whereas PR-AUC focuses directly on precision and recall of the rare positive class (baseline = 0.018).
3. **Why did LightGBM jump to 0.303 PR-AUC on random folds but drop to 0.195 by customer and 0.148 by time?**
   *Answer (D-7):* Random folds split the same customer's multiple loans (and 920 two-lender sibling rows) across train and validation. Flexible trees memorise customer/period patterns; `GroupKFold` removes customer leakage, and the 2023+ temporal split measures real forward-in-time drift (when defaults rose from 1.3% to ~9.5%).
4. **Why deploy Logistic Regression when Random Forest and MLP had slightly higher GroupKFold PR-AUC?**
   *Answer (D-29):* Their customer-level 95% bootstrap intervals **overlap** (`logreg` 0.175 [0.154–0.199] vs `RF` 0.203 [0.180–0.229]), `logreg` actually **beats** RF and LightGBM on 2023+ temporal ROC-AUC (0.633 vs 0.619 / 0.582), matches RF on new-borrower calibration (−2.4 pp), gives exact linear reason codes in **21 ms** (vs 289 ms for RF TreeSHAP), and fits the credit committee's preference for simple auditable models.
5. **Why did class weighting break raw probabilities, and how does Platt scaling fix it?**
   *Answer (`docs/derivations/logistic.md`):* Weighting defaulters by `pos_weight ≈ 54` shifts every predicted log-odds up by `log(54) ≈ 4.0`, making a 1.8% risk look like ~50%. Out-of-fold Platt scaling fits a logistic curve on the raw scores, bringing ECE from **0.304 → 0.002** while preserving ranking.
6. **What happens if calibration goes stale over time?**
   *Answer (D-24, `02` §11):* Pre-2023 Platt scaling predicts 4.5% on 2024 loans that default at 8.7%, causing heavy losses if trusted blindly. Monthly monitoring triggers recalibration whenever actual vs predicted default drifts by > 2 pp for two consecutive months.
7. **What did the ablation study (#52) reveal about calendar features and macro columns?**
   *Answer (`reports/tables/ablation.csv`):* Removing disbursement month/day drops LightGBM random-fold PR-AUC by **−0.044**, yet **improves** 2023+ temporal ROC-AUC by **+0.020** (0.582 → 0.602) — proving trees used calendar dates as period markers. Adding macro columns back hurts `logreg` on every split and adds 0.000 temporal gain for LightGBM (removing macro lifted MLP temporal ROC-AUC 0.48 → 0.65, D-18).
8. **Why do linear regression models lose to the simple mean in USD on Task 1?**
   *Answer (`docs/derivations/linear.md`):* Linear models optimise log-scale squared error (where they beat `mean`, R² 0.14–0.16), but exponentiating small log-space extrapolation errors creates huge dollar errors (`ols` RMSE $2,089 vs `mean` $1,275). Tree leaves average real training loans and cannot extrapolate (`lightgbm_reg` $1,114).
9. **Why did you change the limit rule to an 85th-quantile cap?**
   *Answer (D-30, #53):* Predicting the *typical* loan amount cut **50.1%** of repaid Kenyan loans (48.9% of Kenyan Test). Switching to `lightgbm_reg_q85` (`objective="quantile", alpha=0.85`) cuts only **15.7%** of repaid Kenyan loans (14.6% of Kenyan Test; 83% approved in full) while still capping 19.6% of defaults.
10. **Why can't we use importance weighting to estimate Ghana performance?**
    *Answer (D-27):* Kenya vs Ghana domain classifier AUC is **0.996** (**0.985** even without `n_past_labelled = 0`), leaving an effective sample share of just **4%**. Any reweighted estimate would depend on a tiny handful of Kenyan loans.
11. **Why not just drop `is_new_borrower` to fix the first-time borrower fairness gap?**
    *Answer (D-26):* `n_past_loans` (proxy strength 0.91) and `n_past_labelled` (0.83) reconstruct first-time status anyway. Instead, the **25% starter loan** for borderline first-time borrowers (`0.101 < p ≤ 0.30`) lifts approval from **49% → 81%** (ratio 0.83) at a modest cost of **GHS 18 per first-time applicant**.
12. **What must NkwaPay build before launching unsupervised scoring in Ghana?**
    *Answer (`docs/audit/ghana_trust.md`, `regulatory_map.md`):* Run a 3-month capped pilot of ~3,500 small loans with human review on declines, fix unknown-history encoding, recalibrate on Ghanaian outcomes, connect to a licensed credit bureau (Act 726), and keep plain-language decline reasons + appeal channels (BoG Directive, Act 843).
