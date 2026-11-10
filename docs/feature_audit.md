# Feature audit: every column in Train.csv

**Decision moment:** NkwaPay decides when a borrower applies; the closest column is `disbursement_date` (money is sent right after approval). Anything recorded or derived **after** disbursement is invalid. Status: **draft**, to be confirmed in the Part A session (D-2).

| # | Column | Available at decision time? | Role | Reason |
| --- | --- | --- | --- | --- |
| 1 | `ID` | Yes | Key | Customer + loan + lender ids joined; submission key (the brief calls it `TransactionId`) |
| 2 | `customer_id` | Yes | Key (groups, history) | Not a feature. Groups every fold; keys the borrower history |
| 3 | `country_id` | Yes | Fairness slice (kept in file, not fitted) | Train is all Kenya, so it carries no signal to learn; used to slice Kenya vs Ghana |
| 4 | `tbl_loan_id` | Yes | Key (groups) | Not a feature. A loan can have two lender rows; keeps them in one fold |
| 5 | `lender_id` | Yes | Feature (categorical) | Known at funding. 4 values in Train (one has 94%); 4 new in Test, so unseen values must be handled. Proxy risk: audit in Part D |
| 6 | `loan_type` | Yes | Feature (categorical) | Product chosen at application. Type_3 and Type_8 only in Test (Ghana). Default rate 0.8% to 72% across types |
| 7 | `Total_Amount` | Yes | Feature (log, in USD) **and Task 1 target** | Principal is requested and set at approval. Used as a feature for Task 2; excluded when it is the Task 1 target |
| 8 | `Total_Amount_to_Repay` | **Invalid** | Dropped | Markup over principal separates the label at AUC 0.98. Repaid loans follow a fee schedule by term (median 0.7% at 7 days); defaulted 7-day loans owe a median 32% extra, which looks like penalties added after default |
| 9 | `disbursement_date` | Yes | Time anchor; calendar features | Month, day of week and day of month are derived; the raw date orders history and drives the temporal split |
| 10 | `due_date` | Yes, but redundant | Dropped | Exactly `disbursement_date + duration` (checked on 100% of rows) |
| 11 | `duration` | Yes | Feature | Term is agreed at approval. 7 days for 95% of loans |
| 12 | `New_versus_Repeat` | Yes | Feature (`is_new_borrower`) | Known at application. Only 0.8% are new, but they default at 21% vs 1.7% |
| 13 | `Amount_Funded_By_Lender` | Yes, but redundant | Dropped | Exactly `Total_Amount × Lender_portion_Funded` (100% of rows); 8,520 rows are zero from rounding |
| 14 | `Lender_portion_Funded` | Yes | Feature | Lender's share of the loan, set at funding |
| 15 | `Lender_portion_to_be_repaid` | **Invalid** | Dropped | Owed ÷ funded separates the label at AUC 0.98, same pattern as #8; the definition says "outstanding", which implies after-the-fact |
| 16 | `target` | **No (it is the outcome)** | **Task 2 label** | 1 = defaulted, 0 = repaid |

**Tasks.** Task 1 (regression) target: `Total_Amount`. Task 2 (classification) target: `target`.

**Caveat on #8 and #15.** The data owner's `VariableDefinitions.txt` calls `Total_Amount_to_Repay` the amount "expected" to be repaid, which would make it known at issuance. The data say otherwise (above). We cannot prove which is true, so we drop both; keeping them would inflate every score to near-perfect and teach nothing usable at decision time. Worth stating in the report as a limitation.

## Engineered features (built in `src/features.py`, all time-safe)

| Feature | What it is | Why it is safe |
| --- | --- | --- |
| `n_past_loans`, `days_since_last_loan` | Borrower's earlier loans, counted once per loan | Backward `merge_asof`, no exact matches: only loans disbursed strictly before this one |
| `n_past_labelled`, `n_past_defaults`, `past_default_rate`, `days_since_last_default` | Outcomes of earlier loans | Counted only if the earlier loan's **due date** is before this loan's disbursement, because the outcome is unknown until then. Only Train loans have labels |
| `disb_month`, `disb_dow`, `disb_dom` | Calendar | From `disbursement_date` |
| `macro_inflation`, `macro_fx`, `macro_deposit_rate`, `macro_unemployment` | FRED indicators by country | Previous year's value (annual figures arrive after year-end). Lending rate, real rate, spread, precipitation and fossil use are empty for Ghana, so dropped |
| `log_amount_usd`, `log_amount_local` | log(1 + amount) | USD uses the previous year's exchange rate (D-6) |
| `n_lenders_on_loan` | Rows sharing the `tbl_loan_id` | Known at funding |
