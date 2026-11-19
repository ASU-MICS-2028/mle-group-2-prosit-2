# Deliverables, mapped to the rubric (100 points)

| Criterion | Pts | Excellent looks like |
| --- | --- | --- |
| A. Problem framing, cost matrix | 10 | Label defined; complete feature audit with leakage found; cost matrix in money |
| B. Modelling, pipeline | 30 | Reproducible; ≥ 5 models; ≥ 2 correct derivations; time-safe features |
| C. Evaluation, validation | 20 | PR-AUC + calibration; stratified vs grouped vs temporal; noise floor respected |
| D. Ethical and regulatory audit | 20 | Model Card; disaggregated metrics; proxy analysis; regulatory map; a mitigation |
| E. Deployment, presentation | 15 | Working scorer with reason codes; strong defence |
| F. Process, contribution | 5 | Even contribution; thoughtful reflection |

Modelling and evaluation are half the grade.

## Package

Group: GitHub repo (README to reproduce end to end, data-loading code, notebooks 01 to 04) · technical report 6–10 pages with derivations and AI-use declaration · ethical and regulatory audit · scoring service and explainer · debrief slides.
Individual: one-page reflection each.

## Checklist

**A:** ☐ timeline ☑ label ☐ audit of every column ☑ two tasks and goal ☑ cost matrix in money with ≥ 2 sensitivity cases (cedis, D-22)
**B:** ☑ EDA (imbalance, missing, skew, cardinality; `reports/EDA_REPORT.md`) ☐ `merge_asof` history, calendar, macro ☑ three validation schemes, inflation quantified ☑ ≥ 5 models ☑ classification and regression metrics ☑ bootstrap CIs on F1 and PR-AUC 🟨 interpretation (coefficient and SHAP figures; plain-language write-up #22)
**C:** ☑ Kenya vs Ghana distributions (EDA §8, `shift_*` figures) ☑ domain classifier and importance weighting (`04` §2.2: AUC 0.985 without the label-history artefact; ESS share 0.04, so weighting is unusable) ☑ evidence needed to trust Ghana predictions (`04` §2.4: capped pilot of ~3,500 loans)
**D:** ☑ Model Card ☑ Datasheet ☑ disaggregated metrics ☑ proxy analysis ☑ mitigation with cost ☑ regulatory map (Act 843 sections confirmed; Act 726 to confirm) ☑ Findex analysis ☑ borrower explanation · all in `docs/audit/` (summary: `audit.md`) and `04` §3–4
**E:** ☑ `joblib` bundle and `score_application()` in `src/serving/` (deployed models decided D-29: calibrated `logreg` + `lightgbm_reg`; refitted on D-28 cedi config + D-26 starter loan) ☑ latency (median 24 ms, p95 35 ms over 1,000 calls; `reports/tables/scorer_latency.csv`) ☑ required inputs (`REQUIRED_APPLICATION_FIELDS`)

## Gaps in the README plan

| Gap | Brief says | Fix |
| --- | --- | --- |
| No Datasheet | LO5 | `docs/audit/datasheet.md` (Eugene) |
| No proxy analysis | LO5, rubric D | In `04_transfer_and_fairness` (Eugene) |
| No explicit mitigation | Rubric D | e.g. group-aware threshold, with before/after (Eugene) |
| No AI-use declaration | Required in the report | [AI_USE_LOG](AI_USE_LOG.md) |
| PR-AUC CI not named | CIs on F1 and PR-AUC | Added to LO4 |
| Cost sensitivity not named | Appendix A: ≥ 2 alternatives | Added to LO1 |
| Currency | Cost matrix in cedis/shillings | Decide D-6 |
