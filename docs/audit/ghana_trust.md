# What NkwaPay needs before trusting Ghana scores (#24, Part C)

Owner: Eugene (P4). The evidence behind each point is in `notebooks/04_transfer_and_fairness.ipynb` §2.

## Why Ghana scores cannot be trusted today

- **No Ghanaian outcomes.** Every label is Kenyan, so no Ghana metric (PR-AUC, calibration, approval quality) can be measured.
- **Ghana is a different population.** A domain classifier separates Ghana from Kenya with an AUC of 0.997 on the model's features. It is still 0.985 after removing the label-history artefact, and 0.958 after also removing the loan amount.
  - Ghanaian loans are about 10 times larger in USD and about 4 times longer.
  - 97% of Ghanaian rows come from lenders the model never saw, and 87% are for products it never saw.
- **Reweighting cannot bridge it.** The importance weights have an effective sample share of 0.04, so a reweighted Kenyan estimate would rest on about 4% of Kenyan rows (D-27).
- **One input is unknowable.** Label history (`n_past_labelled`, past defaults) is 0 for every Ghanaian row, because no Ghanaian loan has a label. Treating it as clean instead more than doubles the median predicted risk (0.011 → 0.028).
- **The product already behaves oddly.** Only 7% of Ghanaian applicants are approved in full; 79% are limited to a median 46% of the request, and every application is flagged `needs_review`.

## The evidence plan

1. **Capped pilot.**
   - **Size:** about **3,500 Ghanaian loans**. Measuring a default rate near 8% to within ±2 points (95%) takes about 700 loans, times 5 calibration bins.
   - **Terms:** the smallest product size, with a hard GHS cap per loan.
   - **Oversight:** a person reviews every decline.
   - **Timing:** Ghanaian loans run 36 days on average, so outcomes mature in about **3 months**.
2. **Fix the inputs first.**
   - Encode label history as *unknown*, not 0, for borrowers whose earlier loans are not in our labelled data.
   - Map Ghanaian lenders and products to Kenyan ones where the terms match.
   - Re-run the domain classifier: the AUC should fall as the artefacts are removed.
3. **Calibrate on Ghanaian outcomes.**
   - Fit a Ghana-specific Platt calibrator on the pilot labels.
   - **Accept the model for Ghana only if** its PR-AUC on Ghanaian outcomes is clearly above Ghana's base rate, and the reliability curve is within ±2 points in every bin.
4. **Re-cost the threshold.** The cost matrix is in cedis (D-28, threshold 0.10), but recovery, collection cost and the 3 future loans lost are assumptions. Replace them with pilot figures, and check the starter-loan repayment rate (D-26).
5. **Monitor, with triggers that stop automation.** Go back to manual review if any of these happens:
   - the domain-classifier AUC on new applications moves by more than 0.05 from the pilot;
   - the monthly default rate leaves the calibrated band (recalibrate monthly; see D-24 for the Kenyan drift evidence);
   - first-time-borrower approval falls below the floor the starter loan is designed to hold (ratio 0.8).
