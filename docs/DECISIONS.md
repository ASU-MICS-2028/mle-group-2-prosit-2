# Decision log

Models are only comparable if everyone made the same data decisions. Record each decision with its reason. Never silently edit one: mark it `superseded` and add a row. Status: `open` · `decided` · `superseded`.

| ID | Question | Status | Decision / proposal | Why | Who / when |
| --- | --- | --- | --- | --- | --- |
| D-1 | Roles | decided | P1 Perfect (`peaElorm`), P2 Henry (`henryhkb`); P3, P4 unassigned | Available now | Perfect, 2026-11-07 |
| D-2 | Are `Total_Amount_to_Repay` and `Lender_portion_to_be_repaid` leaks? | open | Proposed: drop both | AUC ≈ 0.98 each; markup 0.8% repaid vs 22% defaulted, looks like penalties added after default | Perfect + Henry |
| D-3 | Two-lender loans (2,134 repeated `tbl_loan_id`) | open | Proposed: keep rows, but group by loan/customer in every split | Same borrower, amount, date; labels agree in 2,132 of 2,134; they default 8.3% vs 1.4%. Evidence: `notebooks/01_eda.ipynb` §1 | P1 |
| D-4 | 4,306 customers and 920 loans in both Train and Test | open | Proposed: the 920 are sibling lender rows of two-lender loans, not duplicates; keep in training, fold by `customer_id`, never validate on a row whose sibling is in training | Sibling labels agree 99.9%, so Test labels for those rows are effectively known (8.8% default). Evidence: EDA §2 | P1 |
| D-5 | Unseen lenders and loan types in Test | open | Proposed: "unseen" category; one-hot ignoring unknowns | All unseen rows are Ghana: 97% of Ghana rows have an unseen lender (4 new), 87% an unseen loan type (Type_3, Type_8). Evidence: EDA §3 | P1 + P2 |
| D-6 | Currency | open | Proposed: USD via previous-year exchange rate for comparisons; local currency for cost matrix | Shillings vs cedis | P1 |
| D-7 | Validation scheme to trust | open | Proposed: temporal, GroupKFold as check | Rate drifts 1.3% → 11.5% → 8.7% | P2 |
| D-8 | Framework | decided | Lightning for LogReg/linear/MLP; scikit-learn and LightGBM for the rest | README §2 | Bernard |
| D-9 | Who covers P3/P4 work? | open | (a) recruit others, (b) Perfect takes P3 and Henry P4, (c) split by model | About half the grade | Perfect + Henry |
| D-10 | Interim PR review | decided | Perfect and Henry review each other | Rotation needs four | Perfect, 2026-11-07 |
| D-11 | Seed and paths | decided | Seed 42; paths via `src/config.py` | Reproducibility | Bernard |
| D-12 | Henry's `01_eda.ipynb` | decided | Merged to `main`, moved to `notebooks/explore_target_henry.ipynb`, reads `data/raw/` | `01_eda` is P1's; avoids a clash | Henry |
