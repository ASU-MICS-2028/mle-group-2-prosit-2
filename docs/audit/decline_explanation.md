# Explaining a declined application to the borrower (#29, Part D.5)

Owner: Eugene (P4). This is the message a declined borrower receives in the app and by SMS. It is built from the 3 reasons `src/serving/reasons.py` returns, and it follows BoG Directive 6(1)(a): clear information on the decision and on the borrower's rights.

## The message (example: first-time borrower, GHS 800 over 30 days)

The three numbered reasons are the scorer's own wording for this applicant (`make score` prints real examples).

> **We can't offer you this loan today.**
>
> We looked at your request for GHS 800 over 30 days. Here is what mattered most:
> 1. **This is your first loan with us, so there is no repayment record yet.**
> 2. **The repayment period is longer than most loans we see repaid.**
> 3. **The amount requested is large compared with similar loans.**
>
> **What you can do**
> - **Start smaller.** We can offer you a starter loan of GHS 200 today. Repaying it on time builds your record with us, and your next limit grows with it.
> - **Ask for a review.** Reply REVIEW or tap *Ask for a review*. A person on our team will look at your application and reply within 20 days.
> - **Your data.** We used only your loan request and your history of loans with NkwaPay. We did not use your contacts, messages or location. You can ask to see or correct the information we hold about you.
>
> This decision is not a judgement of you, and it does not stop you from applying again.

## Writing rules

1. **Plain words, no scores.** Never show a probability, threshold or model name.
2. **Only reasons the borrower can understand and act on.** The scorer already leaves out the calendar fields (`disb_month`, `disb_dow`, `disb_dom`) and the loan-structure fields (`Lender_portion_Funded`, `n_lenders_on_loan`) through `NEVER_SHOWN`. It also never states a history reason that contradicts the borrower's own record. Two reasons it can still return need care in the borrower message:

   | Scorer reason | Show to borrower? | Why |
   | --- | --- | --- |
   | First loan / few earlier loans / earlier loan not repaid / recent missed repayment | Yes | True of the person, and they can act on it |
   | Repayment period longer than most | Yes | They chose it and can change it |
   | Amount large compared with similar loans | Yes | They chose it and can change it |
   | "Loans through this lender have a higher rate of missed repayments" | **No, rephrase**: "This loan product carries more risk for us; a different product may suit you" | The borrower does not pick the funding lender. It is NkwaPay's arrangement |
   | "This loan product has a higher rate of missed repayments" | Yes, with the alternative product named | Only if another product is actually offered |
   | "Your overall application profile" (filler when fewer than 3 reasons apply) | **No** | It says nothing. Show fewer reasons instead |

3. **Always offer a next step**, and prefer a starter loan to a flat decline. The scorer does this itself (D-24): a first-time borrower between the threshold (0.10) and 0.30 gets decision `starter` and an offer of 25% of the request (`recommended_limit`). A decline message is only sent above 0.30, or to repeat borrowers above the threshold; it should still suggest a smaller loan.
4. **Always state the review route and the 20-day response time** (BoG Directive 6(1)(e)).
5. **Say what data was used and what was not**, and that the borrower can ask to see or correct it (Act 843 principles; section numbers to be confirmed).

## In code

`score_application` returns `top_3_reasons` (credit team) and `borrower_reasons` (rule 2 applied: lender reason reworded, filler dropped, so it may hold fewer than 3). The borrower app shows only `borrower_reasons`. Tested in `tests/test_serving.py`.
