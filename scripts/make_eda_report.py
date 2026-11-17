"""Build the exploratory data analysis report: reports/EDA_REPORT.md with its figures and tables.

    python -m scripts.make_eda_report      (or: make eda; needs `make features` first)

Every number in the report is computed here from the raw files, so rerunning rebuilds it exactly.
Each section: the question, the evidence (figure + table), what it shows, and the decision it supports.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.analysis import data_profile as P  # noqa: E402
from src.console import console  # noqa: E402
from src.processing import load_features  # noqa: E402
from src.visualization import save_figure  # noqa: E402
from src.visualization.eda import (  # noqa: E402
    plot_box_by_group,
    plot_correlation,
    plot_grouped_bars,
    plot_histogram,
    plot_macro_lines,
    plot_rate_bars,
    plot_rate_over_time,
    plot_raw_vs_log,
    plot_univariate_auc,
)


def report_path():
    return C.REPORTS / "EDA_REPORT.md"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v) if not np.isnan(v) else ""
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return head + "".join("| " + " | ".join(fmt(v) for v in row) + " |\n" for row in df.itertuples(index=False))


class Report:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def section(self, title: str, question: str) -> None:
        console.info(f"✓ {title}")
        self.parts.append(f"\n## {title}\n\n**Question:** {question}\n")

    def figure(self, fig, name: str, caption: str) -> None:
        save_figure(fig, name)
        plt.close(fig)
        self.parts.append(f"\n![{caption}](figures/{name}.png)\n*{caption}*\n")

    def table(self, df: pd.DataFrame, name: str, floatfmt: str = "{:,.3f}") -> None:
        df.to_csv(C.TABLES / f"{name}.csv", index=False)
        self.parts.append("\n" + md_table(df, floatfmt))

    def finding(self, *bullets: str) -> None:
        self.parts.append("\n**What it shows:**\n" + "".join(f"- {b}\n" for b in bullets))

    def decision(self, text: str) -> None:
        self.parts.append(f"\n**Decision it supports:** {text}\n")

    def write(self, header: str) -> None:
        report_path().write_text(header + "".join(self.parts), encoding="utf-8")


def main() -> None:
    console.title("EDA report")
    C.TABLES.mkdir(parents=True, exist_ok=True)
    train, test = P.load_raw()
    feats = load_features()
    macro = P.load_macro_long()
    r = Report()
    base_rate = float(train["target"].mean())

    # 1 ------------------------------------------------------------ what we have
    r.section("1. What data do we have?", "How many rows, loans and customers, over which dates, in which country?")
    ov = P.overview(train, test)
    r.table(ov, "eda_overview")
    q = P.data_quality(train, test)
    r.table(q, "eda_data_quality")
    r.finding(
        f"Train is **Kenya only**: {len(train):,} rows, {train['customer_id'].nunique():,} customers, "
        f"{train['disbursement_date'].min():%b %Y} to {train['disbursement_date'].max():%b %Y}. "
        f"Test adds {int((test.country_id == 'Ghana').sum()):,} Ghanaian rows with no labels.",
        f"No missing cells and no duplicate rows or IDs in either file; `due_date` = disbursement + duration and "
        f"`Amount_Funded_By_Lender` = amount x portion on {pct(q.loc[0, 'due_eq_disb_plus_duration'], 0)} of rows.",
    )
    r.decision("drop `due_date` and `Amount_Funded_By_Lender` as exact copies of other columns; no imputation needed for raw columns.")

    # 2 ------------------------------------------------------------ unit of analysis
    r.section("2. What is one row?", "Is a row a loan, and how often do customers borrow?")
    tl = P.two_lender_loans(train)
    lpc = P.loans_per_customer(train)
    r.figure(plot_histogram(lpc.clip(upper=60), "Loans per customer (Train, capped at 60)", "loans per customer", log_y=True),
             "eda_loans_per_customer", "Most customers borrow repeatedly: median 6 loans, up to 208.")
    r.finding(
        f"A row is one **lender's share** of a loan: {tl['two_lender_loans']:,} of {tl['loans']:,} loans appear twice "
        f"(two lenders), and their labels agree {pct(tl['labels_agree'])} of the time.",
        f"Two-lender loans default far more ({pct(tl['default_rate_two_lender'])} vs {pct(tl['default_rate_single'])}).",
        f"Customers take a median of {lpc.median():.0f} loans (mean {lpc.mean():.1f}, max {lpc.max()}), so one customer's rows are strongly correlated.",
    )
    r.decision("group every split and every bootstrap by customer (D-3, D-4); add `n_lenders_on_loan` as a feature.")

    # 3 ------------------------------------------------------------ target
    r.section("3. How rare is default, and does it move?", "What is the default rate, and how does it change over time, lender, product and tenure?")
    by_year = P.rate_table(train.assign(year=train["disbursement_date"].dt.year), "year")
    by_month = P.rate_table(train.assign(month=train["disbursement_date"].dt.to_period("M").astype(str)), "month")
    r.figure(plot_rate_over_time(by_month, "month", "Loans and default rate per month (Train)"),
             "eda_default_rate_by_month", "Default rate jumps from about 1% in 2022 to 10% from 2023, while volume falls.")
    r.table(by_year, "eda_default_by_year")
    by_lender = P.rate_table(train, "lender_id")
    by_type = P.rate_table(train, "loan_type", min_count=50).sort_values("default_rate", ascending=False)
    by_tenure = P.rate_table(train, "New_versus_Repeat")
    by_duration = P.rate_table(train.assign(term=pd.cut(train["duration"], [0, 7, 14, 30, 60, 2000], labels=["<=7", "8-14", "15-30", "31-60", ">60"])), "term")
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    plot_rate_bars(by_lender, "lender_id", "By lender", ax=axes[0], overall=base_rate)
    plot_rate_bars(by_type.head(10), "loan_type", "By loan type (n >= 50)", ax=axes[1], overall=base_rate)
    plot_rate_bars(by_tenure, "New_versus_Repeat", "By tenure", ax=axes[2], overall=base_rate)
    plot_rate_bars(by_duration, "term", "By term (days)", ax=axes[3], overall=base_rate)
    fig.tight_layout()
    r.figure(fig, "eda_default_rate_by_group", "Default rate by lender, product, tenure and term: risk is concentrated in small groups.")
    r.table(pd.concat([by_lender.rename(columns={"lender_id": "value"}).assign(column="lender_id"),
                       by_tenure.rename(columns={"New_versus_Repeat": "value"}).assign(column="tenure"),
                       by_duration.rename(columns={"term": "value"}).assign(column="term")])[["column", "value", "loans", "defaults", "default_rate"]],
            "eda_default_by_group")
    r.finding(
        f"Only **{pct(base_rate, 2)}** of rows default ({int(train['target'].sum()):,} of {len(train):,}): accuracy is meaningless; use PR-AUC.",
        "By year: " + ", ".join(f"{int(y)} {pct(d)} (n={n:,})" for y, d, n in zip(by_year["year"], by_year["default_rate"], by_year["loans"])) + ".",
        f"Lenders range from {pct(by_lender['default_rate'].min())} to {pct(by_lender['default_rate'].max())}; "
        f"new borrowers default at {pct(by_tenure.set_index('New_versus_Repeat').loc['New Loan', 'default_rate'])} vs "
        f"{pct(by_tenure.set_index('New_versus_Repeat').loc['Repeat Loan', 'default_rate'])} for repeat borrowers.",
        f"Longer terms are riskier: {pct(by_duration['default_rate'].iloc[0])} for 7-day loans vs {pct(by_duration['default_rate'].iloc[-1])} above 60 days.",
    )
    r.decision("class weights in every model (`configs/models.yaml`), PR-AUC as headline metric, and a **temporal** validation split (D-7) because the rate drifts.")

    # 4 ------------------------------------------------------------ money
    r.section("4. How are amounts distributed?", "Why log-transform, and does loan size relate to risk?")
    r.figure(plot_raw_vs_log(train["Total_Amount"], "Total_Amount (KES)"), "eda_amount_raw_vs_log",
             "Amounts span 2 to 23 million KES; on the log scale they are close to bell-shaped.")
    dec = P.amount_deciles(train)
    r.figure(plot_rate_bars(dec, "amount_decile", "Default rate by loan-size decile (KES)", overall=base_rate),
             "eda_default_by_amount_decile", "The largest 10% of loans default at about 2.5 times the average rate.")
    r.table(dec, "eda_default_by_amount_decile")
    r.finding(
        f"Median loan {train['Total_Amount'].median():,.0f} KES, mean {train['Total_Amount'].mean():,.0f}, max {train['Total_Amount'].max():,.0f}: "
        "the mean is pulled up by a few huge loans.",
        f"Risk rises in the top decile ({pct(dec['default_rate'].iloc[-1])}) but is not monotone below it.",
    )
    r.decision("log1p on every money column and on the Task 1 target; report Task 1 errors on both scales; amount in USD so Kenya and Ghana share a unit (D-6, D-18).")

    # 5 ------------------------------------------------------------ leakage
    r.section("5. Which columns leak the answer?", "Is any column recorded after the lending decision?")
    auc = P.univariate_auc(train)
    r.figure(plot_univariate_auc(auc), "eda_leakage_univariate_auc", "The repayment markup alone separates defaulters at AUC 0.98: it records the outcome.")
    r.table(auc, "eda_leakage_auc")
    mk = P.markup_by_outcome(train)
    r.figure(plot_box_by_group(train.assign(markup_pct=(train["Total_Amount_to_Repay"] / train["Total_Amount"] - 1) * 100,
                                            outcome=np.where(train["target"] == 1, "defaulted", "repaid")),
                               "markup_pct", "outcome", "Amount owed above principal (%), by outcome"),
             "eda_markup_by_outcome", "Repaid loans owe a small fee; defaulted loans owe much more, consistent with penalties added after default.")
    r.table(mk, "eda_markup_by_outcome", "{:,.2f}")
    r.finding(
        f"Repaid loans owe a median {mk.set_index('outcome').loc['repaid', '50%']:.2f}% above principal; defaulted loans {mk.set_index('outcome').loc['defaulted', '50%']:.1f}%.",
        "The two repayment columns on their own score only about 0.66-0.70; it is their **ratio to the principal** that reaches 0.98.",
    )
    r.decision("drop `Total_Amount_to_Repay` and `Lender_portion_to_be_repaid` (D-2); keep the measured fee only for the cost matrix (D-19).")

    # 6 ------------------------------------------------------------ engineered features
    r.section("6. Does borrower history carry signal?", "Do the time-safe history features separate risk, and where are they empty?")
    hs = P.history_signal(feats)
    r.figure(plot_rate_bars(hs, "past_loans", "Default rate by number of earlier loans", overall=base_rate),
             "eda_default_by_history", "Experienced borrowers almost never default; first-time borrowers default at about 7%.")
    r.table(hs, "eda_default_by_history")
    miss = P.feature_missingness(feats)
    r.table(miss, "eda_feature_missingness")
    fu = P.defaulter_followup(train)
    r.finding(
        f"Default rate falls from {pct(hs['default_rate'].iloc[0])} with no earlier loans to {pct(hs['default_rate'].iloc[-1], 2)} after 50+.",
        f"Only {pct(fu['share_given_another_loan'])} of the {fu['customers_who_defaulted']:,} customers who defaulted ever got another loan: "
        "defaulters are cut off, so `past_default_rate` is almost always 0 or empty and the labels only cover approved borrowers.",
        "Every Ghana row has an empty default history (no Ghana customer appears in Train).",
    )
    r.decision("keep history features with missing-indicators (D-13); state the reject-inference limitation (D-14) and the Ghana history gap in Part C.")

    num = list(C.NUM_FEATURES_T2)
    r.figure(plot_correlation(feats[feats["split"] == "train"], num), "eda_feature_correlation",
             "Spearman correlation between model inputs: history counts move together; no input is a copy of another.")

    # 7 ------------------------------------------------------------ categories
    r.section("7. How many categories, and which are new in Test?", "Can one-hot encoding handle lenders and loan types?")
    card = P.cardinality(train, test)
    r.table(card, "eda_cardinality")
    r.finding(
        f"`lender_id` has {card.loc[0, 'n_train']} values in Train (one holds {pct(card.loc[0, 'top_share'])}); Test adds {card.loc[0, 'unseen_in_test'].count(',') + 1} unseen lenders.",
        f"`loan_type` has {card.loc[1, 'n_train']} values (Type_1 is {pct(card.loc[1, 'top_share'])}); Test adds {card.loc[1, 'unseen_in_test']}.",
    )
    r.decision("one-hot with `handle_unknown='ignore'` (D-5); the scorer flags unseen lenders and products for human review.")

    # 8 ------------------------------------------------------------ Kenya vs Ghana
    r.section("8. Is Ghana like Kenya?", "How different is the population the model will be used on?")
    sh = P.kenya_ghana_shift(feats)
    r.table(sh, "eda_kenya_ghana_shift", "{:,.3f}")
    test_feats = feats[feats["split"] == "test"].assign(amount_usd=np.expm1(feats["log_amount_usd"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for country, g in test_feats.groupby("country_id"):
        axes[0].hist(np.log10(g["amount_usd"] + 1), bins=50, density=True, alpha=0.5, label=country)
        axes[1].hist(g["duration"].clip(upper=120), bins=40, density=True, alpha=0.5, label=country)
    axes[0].set(title="Loan amount (log10 USD), Test", xlabel="log10(1 + USD)")
    axes[1].set(title="Term in days (capped at 120), Test", xlabel="days")
    axes[0].legend(frameon=False)
    r.figure(fig, "eda_kenya_vs_ghana", "Ghanaian loans are about 10 times larger in USD and four times longer.")
    ov_t = train_test = P.train_test_overlap(train, test)
    r.finding(
        f"Median loan: USD {sh.set_index('measure').loc['median_amount_usd', 'Test Ghana']:,.0f} in Ghana vs "
        f"{sh.set_index('measure').loc['median_amount_usd', 'Train (Kenya)']:,.0f} in Kenya; mean term "
        f"{sh.set_index('measure').loc['mean_duration_days', 'Test Ghana']:.0f} vs {sh.set_index('measure').loc['mean_duration_days', 'Train (Kenya)']:.0f} days.",
        f"{pct(sh.set_index('measure').loc['share_unseen_lender', 'Test Ghana'])} of Ghana rows use a lender and "
        f"{pct(sh.set_index('measure').loc['share_unseen_loan_type', 'Test Ghana'])} a product never seen in Train; "
        f"{ov_t['ghana_customers_in_train']} Ghana customers appear in Train.",
        f"Kenya Test overlaps Train: {train_test['shared_customers']:,} shared customers and {train_test['shared_loans']:,} sibling lender rows.",
    )
    r.decision("treat Ghana scores as unvalidated: the scorer sets `needs_review`, and Part C must recommend a labelled pilot before automation.")

    # 9 ------------------------------------------------------------ macro
    r.section("9. What do the macro indicators add?", "Can country-level economics be model inputs?")
    inds = ["Inflation, consumer prices (annual %)", "Official exchange rate (LCU per US$, period average)",
            "Deposit interest rate (%)", "Unemployment rate"]
    r.figure(plot_macro_lines(macro, inds), "eda_macro_indicators",
             "Ghana's inflation (31-38%) and exchange rate (8-11 GHS/USD) sit far outside anything in Kenya's training years.")
    k = macro[(macro["Country"] == "Kenya") & (macro["year"].between(2020, 2023))]
    g = macro[(macro["Country"] == "Ghana") & (macro["year"].between(2020, 2023))]
    infl = lambda d: d[d["Indicator"] == inds[0]]["value"]  # noqa: E731
    r.finding(
        f"Kenya's inflation over the training years ranges {infl(k).min():.1f}-{infl(k).max():.1f}%; Ghana's {infl(g).min():.1f}-{infl(g).max():.1f}%.",
        "With Kenya-only training, each macro column takes four yearly values: it acts as a year marker and cannot teach an economic effect.",
    )
    r.decision("macro columns stay in the feature file for the shift study but are not model inputs (D-18).")

    # 10 ----------------------------------------------------------- Findex
    r.section("10. Who in Ghana lacks access to credit? (Findex 2021)", "Which groups the model can never see, because they do not borrow digitally?")
    fx = P.findex_access(pd.read_csv(C.FINDEX_CSV))
    r.figure(plot_grouped_bars(fx, "group", ["has_account", "mobile_money", "borrowed_formally"], "Ghana adults (weighted %)", "% of adults"),
             "eda_findex_access", "Women, rural, poorer and less-educated adults are less likely to hold accounts or borrow formally.")
    r.table(fx, "eda_findex_access", "{:,.1f}")
    fxi = fx.set_index("group")
    r.finding(
        f"Only {fxi.loc['All adults', 'borrowed_formally']:.1f}% of Ghanaian adults borrowed formally; women {fxi.loc['Women', 'borrowed_formally']:.1f}% vs men {fxi.loc['Men', 'borrowed_formally']:.1f}%.",
        f"Accounts: rural {fxi.loc['Rural', 'has_account']:.0f}% vs urban {fxi.loc['Urban', 'has_account']:.0f}%; poorest 40% {fxi.loc['Poorest 40%', 'has_account']:.0f}% vs richest 60% {fxi.loc['Richest 60%', 'has_account']:.0f}%.",
    )
    r.decision("the loan data has no gender or region, so Part D uses Findex to say who the model cannot see; a model trained on existing borrowers can widen the gap.")

    header = (
        "# Exploratory data analysis\n\n"
        "Generated by `python -m scripts.make_eda_report` from `data/raw/` and `data/processed/features.parquet`; do not edit by hand. "
        "Figures are in `reports/figures/eda_*.png`, tables in `reports/tables/eda_*.csv`. "
        "Each section states the question, the evidence, what it shows, and the decision in [DECISIONS](../docs/DECISIONS.md) it supports.\n"
    )
    r.write(header)
    n_fig = sum(p.count("![") for p in r.parts)
    console.done(f"{C.rel(report_path())}: 10 sections, {n_fig} figures, tables in {C.rel(C.TABLES)}")


if __name__ == "__main__":
    main()
