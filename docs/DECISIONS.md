# Decision log

Models are only comparable if everyone made the same data decisions. Record each decision with its reason. Never silently edit one: mark it `superseded` and add a row. Status: `open` · `decided` · `superseded`.

| ID | Question | Status | Decision / proposal | Why | Who / when |
| --- | --- | --- | --- | --- | --- |
| D-1 | Roles | decided | P1 Perfect (`peaElorm`), P2 Henry (`henryhkb`); P3 Bernard (`adjeibernardmarfo`), P4 Eugene (`eugene-sew`) | Roles assigned as members became available (P3 and P4 added 2026-11-10) | Perfect, 2026-11-07 / 2026-11-10 |
| D-2 | Are `Total_Amount_to_Repay` and `Lender_portion_to_be_repaid` leaks? | draft, implemented | Drop both (`config.LEAKY_COLUMNS`). Evidence and caveat in `docs/feature_audit.md` | AUC ≈ 0.98 each; markup 0.8% repaid vs 22% defaulted, looks like penalties added after default | Perfect + Henry |
| D-3 | Two-lender loans (2,134 repeated `tbl_loan_id`) | draft, implemented | Keep rows, but group by loan/customer in every split | Same borrower, amount, date; labels agree in 2,132 of 2,134; they default 8.3% vs 1.4%. Evidence: `notebooks/01_eda.ipynb` §1 | P1 |
| D-4 | 4,306 customers and 920 loans in both Train and Test | draft, implemented | The 920 are sibling lender rows of two-lender loans, not duplicates; keep in training, fold by `customer_id`, never validate on a row whose sibling is in training | Sibling labels agree 99.9%, so Test labels for those rows are effectively known (8.8% default). Evidence: EDA §2 | P1 |
| D-5 | Unseen lenders and loan types in Test | draft | "unseen" category; one-hot ignoring unknowns (applied in the preprocessor, not yet written) | All unseen rows are Ghana: 97% of Ghana rows have an unseen lender (4 new), 87% an unseen loan type (Type_3, Type_8). Evidence: EDA §3 | P1 + P2 |
| D-6 | Currency | draft, implemented | `log_amount_usd` via previous-year exchange rate; `log_amount_local` also kept; local currency for the cost matrix | Shillings vs cedis | P1 |
| D-7 | Validation scheme to trust | draft | Temporal as the trusted scheme, GroupKFold as a check. Cutoff 2023-01-01 gives 64,444 train rows (857 defaults) and 4,210 valid rows (401 defaults, 9.5%) | Rate drifts 1.3% → 11.5% → 8.7%; final choice after the validation study | P2 |
| D-8 | Framework | decided | Lightning for LogReg/linear/MLP; scikit-learn and LightGBM for the rest | README §2 | Bernard |
| D-9 | Who covers P3/P4 work? | decided | Bernard is P3, Eugene is P4 | All four roles filled | Perfect, 2026-11-10 |
| D-10 | Interim PR review | superseded | Was: Perfect and Henry review each other. Now the full rotation applies (P1→P2→P3→P4→P1) | Four members available | Perfect, 2026-11-10 |
| D-11 | Seed and paths | decided | Seed 42; paths via `src/config.py` | Reproducibility | Bernard |
| D-12 | Henry's `01_eda.ipynb` | decided | Merged to `main`, moved to `notebooks/explore_target_henry.ipynb`, reads `data/raw/` | `01_eda` is P1's; avoids a clash | Henry |
| D-13 | When is a past default "known"? | draft, implemented | Only for earlier loans whose **due date** is before this loan's disbursement | An outcome does not exist until the due date; counting earlier would leak. Brute-force checked on 400 random rows in `validate()` | P1 |
| D-14 | Defaulters are cut off, so labels are biased | open | Report as a limitation; consider in Part D | Only 3.5% of customers who default ever get another loan; only 152 rows have a known past default. We only see outcomes for people NkwaPay already approved (reject inference problem) | P1 |
