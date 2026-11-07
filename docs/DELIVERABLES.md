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

**A:** ☐ timeline ☐ label ☐ audit of every column ☐ two tasks and goal ☐ cost matrix in money with ≥ 2 sensitivity cases
**B:** ☐ EDA (imbalance, missing, skew, cardinality) ☐ `merge_asof` history, calendar, macro ☐ three validation schemes, inflation quantified ☐ ≥ 5 models ☐ classification and regression metrics ☐ bootstrap CIs on F1 and PR-AUC ☐ interpretation
**C:** ☐ Kenya vs Ghana distributions ☐ domain classifier or importance weighting ☐ evidence needed to trust Ghana predictions
**D:** ☐ Model Card ☐ Datasheet ☐ disaggregated metrics ☐ proxy analysis ☐ mitigation ☐ regulatory map ☐ Findex analysis ☐ borrower explanation
**E:** ☐ joblib model and `score()` ☐ latency (median, p95 over 1,000 calls) ☐ required inputs

## Gaps in the README plan

| Gap | Brief says | Fix |
| --- | --- | --- |
| No Datasheet | LO5 | `docs/audit/datasheet.md` (P4) |
| No proxy analysis | LO5, rubric D | In `04_transfer_and_fairness` (P4) |
| No explicit mitigation | Rubric D | e.g. group-aware threshold, with before/after (P4) |
| No AI-use declaration | Required in the report | [AI_USE_LOG](AI_USE_LOG.md) |
| PR-AUC CI not named | CIs on F1 and PR-AUC | Added to LO4 |
| Cost sensitivity not named | Appendix A: ≥ 2 alternatives | Added to LO1 |
| Currency | Cost matrix in cedis/shillings | Decide D-6 |
