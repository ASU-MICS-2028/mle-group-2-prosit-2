# Work log (handoff notes)

Add an entry at the **top** at the end of each session, 5–10 lines. Link to notebooks or PRs instead of pasting results.

```
## YYYY-MM-DD · Name (role) · branch or PR
**Did:** …   **Found:** … (numbers + where)   **Next:** …   **Watch out:** …
```

---

## 2026-11-09 · Perfect (P1) · `p1/explore-train-test-structure`
**Did:** `notebooks/01_eda.ipynb` section on Train/Test structure; filled evidence into D-3, D-4, D-5.
**Found:** (1) Two-lender loans = same borrower, amount, date, labels agree 2,132/2,134, default 8.3% vs 1.4%. (2) The 920 "shared loans" are the other lender's row of a two-lender loan: 100% same customer/amount/date, 0% same lender, so Train labels leak Test labels. (3) Ghana: 0% customers seen, 97% rows with unseen lender, 87% with unseen loan type. (4) Kenyan Test spans 2021-2024 like Train, so it is not temporal. (5) 57 loans over 1M include 27 defaults.
**Next:** Part A session with Henry (D-2 to D-7); then rough feature file and the feature audit table.
**Watch out:** Data is symlinked into `data/raw/` from the folder above the repo (git-ignored). The Findex and inclusion files are not yet used.

## 2026-11-07 · Perfect (P1) · setup, not yet pushed
**Did:** Assigned P1/P2; created `docs/`, README tracking section. Reviewed Henry's branch.
**Found:** The brief's debrief window was Nov 2–6. README plan misses Datasheet, proxy analysis, a mitigation and the AI declaration.
**Next:** PR for setup; Part A session with Henry (D-2 to D-7).
**Watch out:** P3 and P4 have nobody. Confirm the deadline.

## 2026-11-01 · Henry · `explore/henry-eda`
**Did:** `01_eda.ipynb` at repo root: shapes, default rate, missing values, repayment markup by outcome, new vs repeat, default rate by year.
**Found:** Rate by year: 2021 53.8% (39 loans), 2022 1.3% (64,405), 2023 11.5% (1,240), 2024 8.7% (2,970). Only this output is saved.
**Next:** Rate by month, lender, loan type; compare random vs customer vs time splits.
**Watch out:** Merged 2026-11-07 by Perfect; moved to `notebooks/explore_target_henry.ipynb` and path changed to `data/raw/`. Not yet rerun after the path change, and only the by-year output is saved.

## 2026-10-21 · Bernard Adjei · `main`
**Did:** README: data guide for all three zips, framework, pipeline, roles, rules, roadmap issues #1–#37.
**Found:** Leak suspects, drift by year, unseen lenders and loan types, Findex access gaps.
**Next:** Assign roles; explore by slice.
**Watch out:** No code or requirements yet.
