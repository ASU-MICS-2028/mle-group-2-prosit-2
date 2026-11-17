# Ethical and regulatory audit: NkwaPay credit-scoring tool (#35, rubric D)

Owner: Eugene (P4). This is the summary. The evidence is in the linked parts and in `notebooks/04_transfer_and_fairness.ipynb`.

| Part | Document |
| --- | --- |
| Model Card (Mitchell et al.) | [model_card.md](model_card.md) |
| Datasheet (Gebru et al.) | [datasheet.md](datasheet.md) |
| Fairness results, proxies, mitigation | `04` notebook §3; summary below |
| Regulatory map (BoG Directive 2025, Act 843, Act 726) | [regulatory_map.md](regulatory_map.md) |
| Borrower decline message | [decline_explanation.md](decline_explanation.md) |

## Executive summary

**What we audited.** The shipped scorer is a Platt-calibrated logistic regression. It declines above p\* = 0.019 (KES cost matrix) and otherwise approves the request or limits it to the Task 1 amount. All figures below are out-of-fold on Kenyan loans, unless they say Ghana.

1. **First-time borrowers are shut out.**
   - At p\* = 0.019, 0% of first-time borrowers are approved, against 83.5% of repeat borrowers. The disparate-impact ratio is 0.
   - Their risk is genuinely higher (21% default, against 1.7%). Even so, a rule that never admits a new customer works against the BoG Directive's access objective (1.3), and it falls on the groups Findex shows are already excluded: women, rural and poorer adults.
2. **The size of the exclusion is a cost assumption, not a model property.** First-time approval depends on the threshold:

   | Threshold | First-time approval |
   | --- | --- |
   | 0.019 (KES, 1 lost loan) | 0% |
   | 0.036 (cedis, D-22) | 12% |
   | 0.106 (6 lost loans) | 49% |

   The team must settle how many future loans a wrongly declined customer is worth.
3. **Mitigation, with its price.**
   - Offer first-time borrowers at p ≤ 0.05 a starter loan of 25% of the request. Their approval rises from 0% to 36%, and 4.9% of those approved default.
   - The extra expected cost is KES 8.7k across 567 applicants, about KES 15 each. Approving the same people at full size would cost KES 27.4k.
   - The ratio stays at 0.43. We recommend the starter loan and report that it narrows the gap but does not close it.
4. **Proxies.**
   - Lender and product identify Ghana perfectly, and the USD amount nearly so (proxy strength 1.0, 1.0, 0.90).
   - The history fields identify first-time borrowers (0.91).
   - Dropping the group flag would therefore not remove the effect, which is why the mitigation acts on the decision.
5. **Other groups.**
   - Loan size: approval falls from 88% (smallest quartile) to 75.5% (largest), ratio 0.86, in line with default rates.
   - Lender and product: the gaps are large but track default rates of 8–72%. The decline message must not blame a borrower for the lender that funds their loan.
6. **Ghana.**
   - The model has no Ghanaian labels. A domain classifier separates Ghana from Kenya with an AUC of 0.985 even after removing the label-history artefact.
   - Through the scorer, no Ghanaian is approved in full: 70% are limited to about 16% of the request and 30% are declined.
   - Treating the unknown history as clean moves declines to 79%.
   - **We advise against automated lending in Ghana before a capped pilot** of about 3,500 loans over about 3 months, followed by recalibration on Ghanaian outcomes.
7. **Regulation.** These items are not met yet:
   - a hard cap at the Bank's loan limit;
   - pre-agreement and total-cost disclosure;
   - decision logging for supervisors;
   - a DPIA, consent capture and DPC registration;
   - credit-bureau reporting and consent;
   - a real 20-day review process.

   The first three, plus review logging, are changes to our own code. The rest are NkwaPay processes. See [regulatory_map.md](regulatory_map.md) §4.

## Open items before submission

- Confirm the section numbers for Act 843 and Act 726 on GhaLII. They are marked "to be confirmed" because automated access was blocked.
- Team decision on the cost matrix: KES (D-19) or cedis (D-22), and the future-loans assumption. Then re-run `04`.
- P3: the borrower-facing reason list and the starter-loan offer in the scorer (see [decline_explanation.md](decline_explanation.md)).
