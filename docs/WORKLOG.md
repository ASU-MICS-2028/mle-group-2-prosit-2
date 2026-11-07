# Work log (handoff notes)

Add an entry at the **top** at the end of each session, 5–10 lines. Link to notebooks or PRs instead of pasting results.

```
## YYYY-MM-DD · Name (role) · branch or PR
**Did:** …   **Found:** … (numbers + where)   **Next:** …   **Watch out:** …
```

---

## 2026-11-07 · Perfect (P1) · setup, not yet pushed
**Did:** Assigned P1/P2; created `docs/`, README tracking section. Reviewed Henry's branch.
**Found:** The brief's debrief window was Nov 2–6. README plan misses Datasheet, proxy analysis, a mitigation and the AI declaration.
**Next:** PR for setup; Part A session with Henry (D-2 to D-7).
**Watch out:** P3 and P4 have nobody. Confirm the deadline.

## 2026-11-01 · Henry · `explore/henry-eda`
**Did:** `01_eda.ipynb` at repo root: shapes, default rate, missing values, repayment markup by outcome, new vs repeat, default rate by year.
**Found:** Rate by year: 2021 53.8% (39 loans), 2022 1.3% (64,405), 2023 11.5% (1,240), 2024 8.7% (2,970). Only this output is saved.
**Next:** Rate by month, lender, loan type; compare random vs customer vs time splits.
**Watch out:** Reads `datasets/`; plan uses `data/raw/`. Not merged (D-12).

## 2026-10-21 · Bernard Adjei · `main`
**Did:** README: data guide for all three zips, framework, pipeline, roles, rules, roadmap issues #1–#37.
**Found:** Leak suspects, drift by year, unseen lenders and loan types, Findex access gaps.
**Next:** Assign roles; explore by slice.
**Watch out:** No code or requirements yet.
