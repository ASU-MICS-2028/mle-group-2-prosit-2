# Regulatory map: NkwaPay credit-scoring tool vs Ghanaian law

Owner: Eugene (P4). Part D.3 of the brief. Status as of 2026-11-17.

**What is being assessed:** the scoring tool, meaning `src/serving/` (probability of default, approve/limit/decline, recommended limit, top-3 reasons, `needs_review` flags), the models behind it, and the borrower app in `app/` (separate pull request). NkwaPay's wider business (licensing, capital, AML) is out of scope except where the tool touches it.

**Sources.**
- **BoG Directive:** Bank of Ghana, *Directive for Digital Credit Services Providers in Ghana, 2025* (September 2025, in force **1 November 2025**), read in full from the PDF linked on bog.gov.gh. Paragraph numbers below are the Directive's own.
- **Act 843:** *Data Protection Act, 2012*, read from the official PDF published by the Data Protection Commission (dpc.gov.gh, `data-protection-act-2012-act-843.pdf`). Section numbers below are the Act's own.
- **Act 726:** the official text could not be retrieved (GhaLII and ILO NATLEX refused automated access, and the Bank of Ghana site has no copy). Its rows are written at the level of principle, and **section numbers remain to be confirmed** against a printed copy. Act 843 s.36 confirms how the two Acts meet: a credit bureau answering a data subject's request must satisfy both.

Status key: ✅ met by our design · 🟨 partly met, or met only in principle · ❌ not met; NkwaPay must act before launch.

## 1. BoG Directive for Digital Credit Services Providers (2025)

| Para | Requirement (paraphrased) | How the tool relates | Status |
| --- | --- | --- | --- |
| 1.3(1), 1.3(3) | Objectives: promote **access** to digital credit; **responsible** and cost-effective credit | The fairness audit measures exactly this trade-off: at the shipped threshold 49% of first-time borrowers are approved (98% of repeat), and no Ghanaian gets the full amount. The scorer's starter loan (D-26) lifts first-time approval to 81% at a known cost (about GHS 18 per first-time applicant) | 🟨 |
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

Section numbers from the official text (see *Sources*).

| Section | Requirement | How the tool relates | Status |
| --- | --- | --- | --- |
| s.17 | Process data by eight principles: accountability, lawfulness, specification of purpose, compatibility of further processing, quality of information, openness, security safeguards, data subject participation | The rows below take them one by one | 🟨 |
| s.18 | Personal data processed lawfully and reasonably, without infringing privacy | Inputs are the loan application and NkwaPay's own loan history; no third-party scraping | 🟨 (needs the privacy notice) |
| s.18(2) | Data of **foreign** data subjects processed in line with their own country's data-protection law | All training data is Kenyan, so Kenya's Data Protection Act, 2019 also governs its use for training. Not assessed in this audit | ⬜ not assessed |
| s.19 | **Minimality:** data adequate, relevant and not excessive for the purpose | No contacts, SMS, location or device data. `customer_id` is used only to look up history and to group folds, never as a model feature | ✅ |
| s.20 | **Consent, justification and objection** | The app has no consent step yet | ❌ |
| s.22, s.23 | Data collected for a **specific purpose**, and the data subject made **aware** of it | Purpose is creditworthiness assessment. The decline message says what data was used and what was not (`decline_explanation.md`); the app does not show it before applying | 🟨 |
| s.25 | Further processing compatible with the purpose of collection | Training on past loan records to score new loans is arguably compatible; the data's original consent wording is undocumented (Datasheet) | 🟨 |
| s.26 | **Quality of information:** complete, accurate, up to date | Label history is structurally 0 for Ghana, which is inaccurate as stated; the audit recommends encoding it as unknown | ❌ for Ghana |
| s.27 | Data controller **registered** with the Commission | NkwaPay's responsibility; also required by Directive 2.1(5)(d)(xi) | ❌ |
| s.28 | **Security measures** | Raw data is git-ignored and never committed. No production access controls exist yet (student project) | 🟨 |
| s.32, s.33, s.35 | **Access** to personal information, **correction** of personal data, right of access | A borrower sees their decision and reasons but cannot yet view or correct the history the model used | 🟨 |
| s.41 | **Automated decision-taking:** where a decision that significantly affects an individual is based solely on automatic processing, the controller must notify them, and they may require it to be **reconsidered** within 21 days; the controller replies within 21 days | Every decision is automated. The decline message and the app offer "ask for a review, a person will look at it". Note s.41(4)(a)–(b): the section does not apply to decisions taken in considering whether to enter a contract, which arguably covers a loan application. We keep the review route anyway: the BoG Directive (6(1)(e), 20 days) requires a complaints route, and it is the fair thing to do | 🟨 (needs a real, logged process) |

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
3. **DPIA, consent and DPC registration** (2.1(5)(d)(x)–(xi); Act 843 s.20, s.27).
4. **Bureau reporting and bureau consent** (7.3; Act 726).
5. **Decision logging for supervisors** (3.7).
6. **A real human-review process** for automated declines, resolving complaints within 20 days (6(1)(e); Act 843 s.41, 21-day reconsideration).
7. **Act 1052** (Borrowers and Lenders Act) was not assessed.
8. **Kenya's Data Protection Act, 2019** (Act 843 s.18(2): the training data is Kenyan) was not assessed.

Items 1, 5 and 6 are changes to our own code and app. The rest are business processes NkwaPay must put in place around the tool.
