# Regulatory map: NkwaPay credit-scoring tool vs Ghanaian law

Owner: Eugene (P4). Part D.3 of the brief. Status as of 2026-11-17.

**What is being assessed:** the scoring tool, meaning `src/serving/` (probability of default, approve/limit/decline, recommended limit, top-3 reasons, `needs_review` flags), the models behind it, and the borrower app in `app/` (separate pull request). NkwaPay's wider business (licensing, capital, AML) is out of scope except where the tool touches it.

**Sources.**
- **BoG Directive:** Bank of Ghana, *Directive for Digital Credit Services Providers in Ghana, 2025* (September 2025, in force **1 November 2025**), read in full from the PDF linked on bog.gov.gh. Paragraph numbers below are the Directive's own.
- **Act 843 and Act 726:** the official texts could not be retrieved for this audit (GhaLII returned 403 to automated access, and the Data Protection Commission site blocked it). Their rows are written at the level of the Acts' principles, and **section numbers are left for a team member to confirm** against the text on GhaLII before submission.

Status key: ✅ met by our design · 🟨 partly met, or met only in principle · ❌ not met; NkwaPay must act before launch.

## 1. BoG Directive for Digital Credit Services Providers (2025)

| Para | Requirement (paraphrased) | How the tool relates | Status |
| --- | --- | --- | --- |
| 1.3(1), 1.3(3) | Objectives: promote **access** to digital credit; **responsible** and cost-effective credit | The fairness audit measures exactly this trade-off: at the shipped threshold 49% of first-time borrowers are approved (98% of repeat), and no Ghanaian gets the full amount. The scorer's starter loan (D-24) lifts first-time approval to 81% at a known cost (about GHS 18 per first-time applicant) | 🟨 |
| 2.1(5)(d)(ii) | Licence application needs a risk-management policy and a **comprehensive risk assessment** | Model Card, the Kenya→Ghana shift study and the monitoring triggers are the model-risk part of that assessment | 🟨 |
| 2.1(5)(d)(x), (xi) | A **Data Protection Impact Assessment approved by the Data Protection Commission** and a data protection certificate | Not done. The Datasheet lists the personal data used, which is the starting point for a DPIA | ❌ |
| 2.1(5)(d)(xiv) | A written **Digital Credit Policy** | The cost-matrix threshold, limit model and decline rules are the technical core of that policy; NkwaPay must adopt them formally | 🟨 |
| 2.3(1) | The physical office must **not** be used for manual onboarding, approval, disbursement or collection | Approval is automated by the scorer. Our "ask for a review" route for declined borrowers must be handled digitally, not as an in-office manual approval | 🟨 |
| 3.5(5), 7.1(2) | No customer may be lent more than the **limit set by the Bank**; limits must follow BoG requirements | The limit model predicts an amount but has **no hard cap**. A configurable regulatory cap must clip `recommended_limit` | ❌ |
| 3.7 | Submit any information the Bank requires; give access to core lending platforms | Every decision should be logged (inputs, probability, threshold, reasons, model version). Not implemented yet | ❌ |
| §4(1)–(2) | Identify risks on an ongoing basis and put controls in place | Part C defines monitoring triggers (drift in the domain classifier, calibration on Ghana pilot labels) | 🟨 |
| 6(1)(a) | Clear, accurate information on the credit arrangement and the customer's **rights and responsibilities** | The scorer returns the decision, limit and 3 plain-language reasons; the decline message (`docs/audit/decline_explanation.md`) adds next steps and rights | 🟨 |
| 6(1)(b) | A **digital pre-agreement** before the loan: principal, disbursement schedule, cost if the borrower defaults, fees, insurance | The app shows principal and term only. Fees, default costs and insurance are pricing decisions outside the model | ❌ |
| 6(1)(c), 7.1(3)(b) | Interest disclosed for the tenure and **not compounded** | Out of scope: the tool does not price loans | ❌ (out of scope) |
| 6(1)(d) | **No misleading information** | Probabilities are calibrated (Platt) before the threshold, and every fairness table uses them (`04` §1.1). Reasons come from the model's own contributions, never contradict the borrower's record, and omit calendar and loan-structure factors. Lender and product can still appear as reasons, although the borrower does not choose them freely | 🟨 |
| 6(1)(e) | Complaints received and **resolved within 20 days** | The app offers "ask for a review"; the 20-day resolution process is NkwaPay's to build | 🟨 |
| 6(1)(g) | No compulsory **bundled** products | The tool offers none | ✅ |
| 7.1(1) | Credit policy in line with the **Borrowers and Lenders Act, 2020 (Act 1052)** | Not reviewed in this audit | ⬜ not assessed |
| 7.1(3)(c)–(d) | Disclose **total cost of credit** and the **due date** | The app shows the term but not the total cost or due date | ❌ |
| 7.2(4)(c) | No unauthorised calls or messages to a customer's **contacts** during collection | The model uses no phone contacts, call logs or location; features are loan history only | ✅ |
| 7.3(5)–(7) | Submit customer credit information to licensed **credit bureaus, daily**, complete and accurate | Not part of the tool. Using bureau data as a model input would also help first-time borrowers who have no NkwaPay history | ❌ |
| 7.3(8)–(9) | **Consent** before sharing credit information with a bureau (oral, print or electronic) | Consent capture is not in the app | ❌ |

**Not in the Directive:** it does not mention automated or algorithmic decisions, explanations of declines, or non-discrimination. Those obligations come from Act 843 (below) and from the brief's own fairness requirement.

## 2. Data Protection Act, 2012 (Act 843)

Section numbers to be confirmed against the official text.

| Requirement (by principle) | How the tool relates | Status |
| --- | --- | --- |
| **Lawful processing and purpose specification:** collect personal data for a specific, lawful purpose | Purpose is creditworthiness assessment. Inputs are limited to the loan application and NkwaPay's own loan history | 🟨 (purpose must be stated in the privacy notice) |
| **Minimality:** data adequate, relevant, not excessive | No contacts, SMS, location or device data. `customer_id` is used only to look up history and to group folds, never as a model feature | ✅ |
| **Consent** | The app has no consent step yet | ❌ |
| **Data subject access and correction** | A borrower can see their decision and reasons, but cannot yet view or correct the history the model used | 🟨 |
| **Automated decision-taking:** a data subject's right concerning decisions based solely on automated processing | Every decision is automated. The app's "ask for a review, a person will look at it" route is the human-review safeguard; it needs a real, logged process behind it | 🟨 |
| **Security safeguards** | Raw data is git-ignored and never committed. No production access controls exist yet (student project) | 🟨 |
| **Registration** with the Data Protection Commission | NkwaPay's responsibility; also required by Directive 2.1(5)(d)(xi) | ❌ |

## 3. Credit Reporting Act, 2007 (Act 726) and Credit Reporting Regulations, 2020 (L.I. 2394)

Section numbers to be confirmed against the official text. L.I. 2394 is named in Directive 1.5(14).

| Requirement (by principle) | How the tool relates | Status |
| --- | --- | --- |
| Lenders supply **accurate** credit data to licensed bureaus | Not implemented (see Directive 7.3) | ❌ |
| **Consent** before a borrower's credit report is obtained or shared | No bureau data is used, so no report is pulled today. If bureau features are added, consent must be captured first | ✅ today / ❌ if extended |
| Borrowers can **access** their report and **dispute** errors | Relevant once NkwaPay reports to bureaus: a wrong "default" record would then follow the borrower to other lenders | ❌ |

## 4. What we cannot currently satisfy (summary for the report)

1. **Regulatory loan cap:** the limit model has no hard cap at the Bank's limit (3.5(5), 7.1(2)).
2. **Pre-agreement and cost disclosure:** total cost of credit, fees, non-compounded interest, due date and default costs are not shown (6(1)(b)–(c), 7.1(3)).
3. **DPIA, consent and DPC registration** (2.1(5)(d)(x)–(xi); Act 843).
4. **Bureau reporting and bureau consent** (7.3; Act 726).
5. **Decision logging for supervisors** (3.7).
6. **A real human-review process** for automated declines, resolving complaints within 20 days (6(1)(e); Act 843 automated decisions).
7. **Act 1052** (Borrowers and Lenders Act) was not assessed.

Items 1, 5 and 6 are changes to our own code and app. The rest are business processes NkwaPay must put in place around the tool.
