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

**What we audited.** The shipped scorer is a Platt-calibrated logistic regression. It declines above p\* = 0.10 (cedi cost matrix, D-26), offers first-time borrowers just above that a 25% starter loan (D-24), and otherwise approves the request or limits it to the Task 1 amount. All figures below are out-of-fold on Kenyan loans, unless they say Ghana.

1. **First-time borrowers carry the largest gap.**
   - At p\* = 0.10, 48.7% of first-time borrowers are approved, against 97.9% of repeat borrowers. The disparate-impact ratio is 0.50.
   - Their risk is genuinely higher (21% default against 1.7%). Even so, the gap falls on the groups Findex shows are already excluded (women, rural and poorer adults), and it works against the BoG Directive's access objective (1.3).
2. **The size of the gap is a cost assumption, not a model property.** With the same model, first-time approval depends on the threshold:

   | Cost matrix | Threshold | First-time approval |
   | --- | --- | --- |
   | Old KES, 1 lost loan (D-19) | 0.019 | 0% |
   | Cedis, 1 lost loan (D-22) | 0.036 | 12% |
   | **Cedis, 3 lost loans (shipped, D-26)** | **0.10** | **49%** |
   | Cedis, 6 lost loans | 0.18 | 70% |

   Three lost loans assumes that about 1 in 4 declined good borrowers leave, taking the 8 later loans a typical customer takes with them. The pilot should measure it.
3. **Mitigation, now in the scorer (D-24).**
   - First-time borrowers between 0.10 and 0.30 get a starter loan of 25% of the request instead of a decline. Their approval rises from 49% to 81% (ratio 0.83, above 0.8).
   - 185 people get a starter loan, and 18% of them default. The extra expected cost is GHS 10.0k across 567 applicants, about GHS 18 each. Approving the same band at full size would cost GHS 38.7k.
4. **Proxies.**
   - Lender and product identify Ghana perfectly, and the USD amount nearly so (proxy strength 1.0, 1.0, 0.90).
   - The history fields identify first-time borrowers (0.91).
   - Dropping the group flag would therefore not remove the effect, which is why the mitigation acts on the decision.
5. **Other groups.**
   - Loan size: 99.6% approved for the smallest quartile against 93.9% for the largest (ratio 0.94), in line with default rates.
   - Lender 251804: 78% approved (ratio 0.79, 8.9% default).
   - The two small lenders and products Type_4, 6, 9 and 14 have large gaps that track default rates of 11–72%.
   - The borrower message rewords the lender reason, because borrowers do not choose their funding lender.
6. **Ghana.**
   - The model has no Ghanaian labels. A domain classifier separates Ghana from Kenya with an AUC of 0.985 even after removing the label-history artefact.
   - Through the scorer, no Ghanaian is approved in full: 86% are limited to about 15% of the request and 14% are declined.
   - Treating the unknown history as clean more than doubles the median probability (0.011 to 0.028).
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
- Replace the cost assumptions (recovery, collection cost, 3 lost loans) with pilot figures when they exist, then re-run `04`.
- P3 review of the scorer changes (`decide`, `borrower_reasons`, starter loan) in this pull request.
