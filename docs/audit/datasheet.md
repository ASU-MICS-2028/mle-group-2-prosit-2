# Datasheet: African Credit Scoring Challenge data, as used by NkwaPay's scoring tool

Owner: Eugene (P4). Format follows Gebru et al. (2021), *Datasheets for Datasets*. Figures are measured on the files in `data/raw/` (see `reports/EDA_REPORT.md`, `docs/feature_audit.md`, `notebooks/01_eda.ipynb`).

## Motivation

- **Why was it created?** Zindi published it for the *African Credit Scoring Challenge*: predict whether a digital loan will default, so lenders can extend credit to people without formal credit histories. We use it as a stand-in for NkwaPay's own portfolio.
- **Who created it and who funded it?** Released on Zindi. The underlying lenders are not named; lender IDs are numeric codes. We do not know who collected the records or under what agreement.
- **Our purpose:** train a default classifier (Task 2) and a loan-amount regressor (Task 1) on Kenyan loans, and assess how they transfer to Ghana.

## Composition

| File | One row is | Rows | Notes |
| --- | --- | --- | --- |
| `Train.csv` | one lender's share of one loan | 68,654 | All Kenya, Oct 2021 – Nov 2024. 6,540 customers (median 6 loans each, max 208). 1,258 defaults (**1.83%**) |
| `Test.csv` | same, without the label | 18,594 | 15,069 Kenya + 3,525 Ghana. Ghana runs Nov 2021 – Mar 2023 |
| `economic_indicators.csv` | one indicator × country, one column per year 2001–2023 | 27 | Ghana, Côte d'Ivoire, Kenya × 9 indicators |
| `micro_gha.csv` (Global Findex Ghana 2021) | one surveyed adult | 1,000 | Survey weights in `wgt`. Used only for the inclusion analysis, never for training |

- **Label.** `target`: 1 = defaulted, 0 = repaid. The data owner does not define "default" (days past due, write-off or other), nor the window it is measured over. We treat it as given.
- **Missing values.** None in Train or Test. Gaps exist in the macro file: Ghana has no lending rate, real interest rate or spread; precipitation stops in 2021; fossil-fuel use is empty for both countries.
- **Personal data.** `customer_id` is a pseudonymous ID, but it links all of a person's loans, so the data is **personal data** under Act 843. There are no names, phone numbers, gender, age or location. Gender and region appear only in Findex.
- **Sensitive attributes.** None directly. **Proxies exist:** `lender_id` and `loan_type` almost perfectly identify the country (97% of Ghana rows have a lender unseen in Kenya; 87% a loan type unseen in Kenya); loan amount tracks income.
- **Duplicates and overlap.** 2,134 loans are funded by two lenders and appear as two rows; their labels agree in 2,132 cases. 4,306 customers appear in both Train and Test, and 920 Test loans are the *other lender's row* of a Train loan, so their labels are effectively known. Every split is therefore grouped by customer.
- **Leaky columns.** `Total_Amount_to_Repay` and `Lender_portion_to_be_repaid` separate the label at AUC ≈ 0.98: defaulted 7-day loans owe a median 32% over principal against 0.7% for repaid loans, which looks like penalties added after default. Both are dropped (decision D-2).
- **Imbalance and drift.** Default rate by year: 1.3% in 2022 (64,405 rows), 11.5% in 2023, 8.7% in 2024. One lender holds 94% of Train rows.

## Collection process

- **How was it collected?** Unknown. The rows look like loan-servicing records exported by lenders, aggregated by Zindi. No collection protocol, consent wording or sampling design is published.
- **Who is represented?** Only people who **applied, were approved and were funded**. Rejected applicants are absent, and defaulters are mostly cut off afterwards: only 3.5% of customers who default ever get another loan. So the labels describe approved borrowers, not the population NkwaPay will score (**reject inference problem**, decision D-14).
- **Consent.** Not documented. We cannot confirm that borrowers agreed to secondary use for model training.

## Preprocessing (ours, in `src/features.py`)

- Leaky and redundant columns dropped (D-2; `due_date` and `Amount_Funded_By_Lender` are exact functions of other columns).
- Borrower history built with a backward `merge_asof` (strictly earlier loans only). A past default counts only once that loan's due date has passed (D-13).
- Calendar features from `disbursement_date`; macro indicators joined from the **previous** year; amounts converted to USD with the previous year's exchange rate (D-6).
- Since D-18 the models use only the USD amount and **no macro columns**: in Kenya-only training each macro column takes 4 yearly values, and Ghana's values lie far outside them. Both stay in `features.parquet` for the shift study.
- Folds: 5-fold GroupKFold by customer (trusted), stratified 5-fold (naive comparison), and a temporal cutoff at 2023-01-01.
- Raw files are kept unchanged in `data/raw/` (git-ignored). `make features` (`python -m scripts.build_features`) rebuilds everything and fails if any leakage check fails.

## Uses

- **Appropriate:** research and coursework on credit scoring, imbalance, grouped and temporal validation, covariate shift, and fairness auditing.
- **Use with care:** scoring Ghanaian applicants. There are no Ghanaian labels, Ghana's loans are about 10× larger in USD and about 4× longer, and most Ghanaian lenders and products never appear in training. Any Ghana score is an extrapolation.
- **Inappropriate:** fully automated live lending decisions without local validation; any inference about gender, ethnicity, religion or region (not present, and proxies must not be used to recover them); identifying individuals.

## Distribution

- Distributed by Zindi under the competition's terms (free account required). Findex Ghana 2021 comes from the World Bank Microdata Library (free registration). Neither may be redistributed by us, so `data/` is git-ignored and each team member downloads the files.

## Maintenance

- Zindi does not promise updates. For deployment, NkwaPay would replace this with its own Ghanaian portfolio and refresh the datasheet whenever the data source, label definition or population changes.
- Contact for this datasheet: the P4 owner of this repository.
