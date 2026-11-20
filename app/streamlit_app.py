"""NkwaPay loan app: a borrower-facing UI over the shared scorer (`src.serving.score_application`, P3).

Run from the repo root, after `make features artifacts`:  streamlit run app/streamlit_app.py

The app never re-implements features, thresholds or the decision rule. It builds the application dict the
scorer documents (`src.serving.REQUIRED_APPLICATION_FIELDS`) and displays what comes back:
decision (approve / limit / starter / decline), recommended_limit, currency, borrower_reasons (shown to the
borrower), top_3_reasons and review_flags (credit team only).
Ghana only (NkwaPay's market), amounts in GHS. The borrower types only their mobile-money number, the amount and
the term. Everything else the model needs comes from NkwaPay's records, looked up by that number (`lookup`):
borrower history, funding lender and product. In this demo the records are real Ghanaian customers from
features.parquet behind made-up phone numbers; an unknown number is scored as a new customer on NkwaPay's standard
Ghana product. The date is today.
"""
import base64
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.serving import score_application  # noqa: E402
from src.training.artifacts import CLF_ARTIFACT, REG_ARTIFACT  # noqa: E402

LIVE = (C.ARTIFACTS / CLF_ARTIFACT).exists() and (C.ARTIFACTS / REG_ARTIFACT).exists()


COUNTRY, CUR = "Ghana", "GHS"
AMOUNT = (50, 10_000, 50, 3_000)  # slider min, max, step, default (median Ghanaian loan is about GHS 3,000)
NEW_CUSTOMER = {"lender_id": "296542", "loan_type": "Type_3"}  # Ghana's most common lender and product
RECORD_FIELDS = ["n_past_loans", "days_since_last_loan", "n_past_labelled", "n_past_defaults", "days_since_last_default",
                 "Lender_portion_Funded", "n_lenders_on_loan"]


def digits(phone: str) -> str:
    return "".join(ch for ch in phone if ch.isdigit())


@st.cache_data
def records() -> dict[str, dict]:
    """Demo stand-in for NkwaPay's customer database: 8 real Ghanaian customers (their latest application in
    features.parquet) behind made-up numbers 024 000 0001-0008. Empty if the data is not built."""
    import pandas as pd

    if not C.FEATURES_PARQUET.exists():
        return {}
    df = pd.read_parquet(C.FEATURES_PARQUET)
    gh = df[(df["split"] == "test") & (df["country_id"] == COUNTRY)].sort_values("disbursement_date")
    last = gh.groupby("customer_id").tail(1)
    new = last[last["is_new_borrower"] == 1].sort_values("customer_id").head(2)
    # repeat customers with history on file, spread from few to many earlier loans
    rep = last[(last["is_new_borrower"] == 0) & (last["n_past_loans"] > 0)]
    rep = rep.sort_values(["n_past_loans", "customer_id"]).reset_index(drop=True)
    picks = rep.iloc[[int(q * (len(rep) - 1)) for q in (0.0, 0.3, 0.6, 0.8, 0.95, 1.0)]]
    out = {}
    for i, (_, row) in enumerate(pd.concat([new, picks]).iterrows(), 1):
        rec = {k: (None if pd.isna(row[k]) else float(row[k])) for k in RECORD_FIELDS}
        rec.update(customer_id=str(row["customer_id"]), lender_id=str(row["lender_id"]), loan_type=str(row["loan_type"]),
                   New_versus_Repeat="New Loan" if row["is_new_borrower"] == 1 else "Repeat Loan")
        out[f"024000000{i}"] = rec
    return out


def lookup(phone: str) -> tuple[dict, bool]:
    """NkwaPay's records for this number, or a new-customer record if there is none."""
    rec = records().get(digits(phone)[-10:].rjust(10, "0"))
    if rec:
        return rec, True
    return {"customer_id": "new", "New_versus_Repeat": "New Loan", **NEW_CUSTOMER}, False


def label(phone: str, rec: dict) -> str:
    who = "first-time" if rec["New_versus_Repeat"] == "New Loan" else f"{int(rec['n_past_loans'] or 0)} earlier loans"
    return f"0{phone[1:3]} {phone[3:6]} {phone[6:]} · {who} · lender {rec['lender_id']} · {rec['loan_type']}"


# Stylised Adinkra symbols for the page background (simplified line drawings, not exact reproductions):
#   Sankofa (heart form): learn from the past -> scoring from loan history
#   Bese Saka (sack of cola nuts): abundance and trade
#   Dwennimmen (ram's horns): humility with strength -> responsible lending
SANKOFA = ('<path d="M0 28 C-26 8 -32 -8 -24 -20 C-16 -32 -2 -28 0 -16 C2 -28 16 -32 24 -20 C32 -8 26 8 0 28 Z"/>'
           '<path d="M0 -16 C-2 -4 -12 2 -16 -6 C-19 -12 -12 -16 -9 -11"/>'
           '<path d="M0 -16 C2 -4 12 2 16 -6 C19 -12 12 -16 9 -11"/>')
BESE_SAKA = ('<circle r="6"/>' + "".join(
    f'<ellipse cx="0" cy="-18" rx="6" ry="10" transform="rotate({a})"/>' for a in (0, 90, 180, 270)))
DWENNIMMEN = "".join(f'<path d="M0 0 C0 -16 16 -16 16 -5 C16 3 7 3 7 -3" transform="rotate({a})"/>'
                     for a in (0, 90, 180, 270))
TILE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="260" height="260">'
    '<g fill="none" stroke="#065F46" stroke-width="2.5" stroke-linecap="round" opacity=".08">'
    f'<g transform="translate(60 60)">{SANKOFA}</g>'
    f'<g transform="translate(195 80)">{BESE_SAKA}</g>'
    f'<g transform="translate(120 190)">{DWENNIMMEN}</g>'
    f'<g transform="translate(235 225) scale(.6)">{SANKOFA}</g>'
    '</g></svg>'
)
PATTERN = f'url("data:image/svg+xml,{quote(TILE)}")'

HERO_IMG = base64.b64encode((Path(__file__).parent / "assets" / "hero.png").read_bytes()).decode()


st.set_page_config(page_title="NkwaPay · Quick loan", page_icon="💚", layout="wide")
st.markdown(f"<style>.stApp {{background-color:#F4F7F5; background-image:{PATTERN};}}</style>",
            unsafe_allow_html=True)
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
      html, body, [class*="css"], .stMarkdown, button, input, label {font-family:'Plus Jakarta Sans', sans-serif !important;}
      .block-container {padding: 1rem 2.5rem 1.5rem; max-width: 100%;}
      #MainMenu, footer, [data-testid="stHeader"], [data-testid="stSidebar"] {display: none;}

      .sb-brand {display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .3rem;}
      .sb-logo {width:36px; height:36px; border-radius:11px; background:#10B981; display:flex; align-items:center;
                justify-content:center; font-weight:800; color:#053B2D !important; font-size:1.1rem;}
      .sb-title {font-weight:800; font-size:1.15rem;}
      .hero-brand {margin-bottom:.9rem;} .hero-brand .sb-title {color:#fff; font-size:1.3rem;}
      .dot {width:9px; height:9px; border-radius:50%; flex:0 0 9px;}

      /* ---------- main ---------- */
      .hero {background: linear-gradient(120deg, #064E3B 0%, #047857 55%, #10B981 100%); color:#fff;
             border-radius: 24px; padding: 2.2rem 2.4rem 2.5rem; margin: 2.2rem 0 1.6rem; min-height: 210px;
             position:relative; display:flex; align-items:center;}
      .hero-text {max-width: 60%;}
      .hero-img {position:absolute; right:4rem; bottom:7px; height:265px; pointer-events:none;
             filter: drop-shadow(0 12px 18px rgba(0,0,0,.25));}
      .hero h1 {color:#fff; font-size:2.4rem; font-weight:800; margin:0; line-height:1.15; padding:0;}
      .hero p {color:#D1FAE5; margin:.5rem 0 0; font-size:1.08rem;}
      .kente {position:absolute; left:0; right:0; bottom:0; height:7px; border-radius:0 0 24px 24px;
              background: repeating-linear-gradient(90deg, #F5B700 0 28px, #C8102E 28px 40px,
                          #F5B700 40px 52px, #0B0B0B 52px 60px, #047857 60px 88px, #F5B700 88px 96px);}
      [data-testid="stForm"] {background:#fff; border:1px solid #E5E7EB !important; border-radius:22px;
             padding:1.8rem 1.9rem; box-shadow:0 8px 24px rgba(6,78,59,.06);}
      [data-testid="stForm"] [data-baseweb="input"] {min-height:52px; border-radius:14px !important;
             background:#F8FAF9; border:1px solid #E5E7EB !important;}
      [data-testid="stForm"] input {font-size:1.05rem !important;}
      [data-testid="stForm"] [data-testid="stButtonGroup"] button {min-height:46px; padding:.6rem .75rem; font-size:.95rem;}
      [data-testid="stForm"] .label {margin: .4rem 0 .35rem;}
      .card {background:#fff; border:1px solid #E5E7EB; border-radius:20px; padding:1.2rem 1.4rem;
             box-shadow:0 8px 24px rgba(6,78,59,.06); margin-bottom:.9rem;}
      .result-approve {border-left: 6px solid #10B981;}
      .result-decline {border-left: 6px solid #F59E0B;}
      .result-limit {border-left: 6px solid #10B981;}
      .result-starter {border-left: 6px solid #0EA5E9;}
      .empty {text-align:center; padding:3.2rem 1.5rem; color:#6B7280; border-style:dashed;}
      .empty .icon {font-size:2.2rem; margin-bottom:.4rem;}
      .big {font-size:2rem; font-weight:800; color:#064E3B; line-height:1.1; margin:.15rem 0 .2rem;}
      .label {color:#6B7280; font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em;}
      .stats {display:flex; gap:.7rem; margin-top:.9rem;}
      .stat {flex:1; background:#F0FDF4; border-radius:14px; padding:.65rem .9rem;}
      .stat b {display:block; font-size:1.15rem; color:#064E3B;}
      .bar {height:8px; background:#E5E7EB; border-radius:999px; position:relative; margin:.5rem 0 .25rem;}
      .bar > span {position:absolute; left:0; top:0; bottom:0; border-radius:999px;}
      .bar > i {position:absolute; top:-4px; bottom:-4px; width:2px; background:#111827;}
      .reason {display:flex; gap:.7rem; align-items:flex-start; padding:.5rem 0; border-bottom:1px solid #F3F4F6;
               font-size:.92rem;}
      .reason:last-child {border-bottom:none; padding-bottom:0;}
      .num {flex:0 0 26px; height:26px; border-radius:50%; background:#ECFDF5; color:#047857;
            font-weight:800; display:flex; align-items:center; justify-content:center; font-size:.8rem;}
      .muted {color:#6B7280; font-size:.86rem;}
      .rec td {padding:.15rem .8rem .15rem 0; font-size:.84rem; color:#374151;}
      div[data-testid="stFormSubmitButton"] > button {border-radius:999px; padding:.85rem 1rem; font-weight:700; font-size:1.05rem;
            background:#047857; border:none;}
      div[role="dialog"] {max-width: 600px !important; border-radius: 20px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""<div class="hero"><div class="hero-text"><div class="sb-brand hero-brand"><div class="sb-logo">N</div><div class="sb-title">NkwaPay</div></div><h1>Quick cash, clear answers.</h1>
    <p>Tell us how much you need and for how long. You'll get a decision, and the reasons, in seconds.</p></div>
    <img class="hero-img" src="data:image/png;base64,{HERO_IMG}" alt=""><div class="kente"></div></div>""",
    unsafe_allow_html=True,
)
left, right = st.columns([1.1, 1.35], gap="large")

# ---------- main ----------

with left:
    demo = records()
    if demo:  # demo only: sign in as one of the sample customers (fills the number field)
        st.selectbox("Demo · sign in as a sample customer", [""] + list(demo), key="demo_pick",
                     format_func=lambda k: "Type any number, or pick a sample…" if not k else label(k, demo[k]),
                     on_change=lambda: st.session_state.update(phone=st.session_state.demo_pick))
    with st.form("apply"):
        st.markdown('<div class="label">Your mobile money number</div>', unsafe_allow_html=True)
        phone = st.text_input("Mobile money number", key="phone", placeholder="e.g. 024 000 0003",
                              label_visibility="collapsed")
        st.markdown('<div class="label">How much do you need?</div>', unsafe_allow_html=True)
        lo, hi, step, default = AMOUNT
        amount = st.slider("Amount", lo, hi, default, step=step, format=f"{CUR} %d", label_visibility="collapsed")
        st.markdown('<div class="label">Pay back in</div>', unsafe_allow_html=True)
        duration = st.segmented_control("Term", [7, 14, 30, 60, 90], default=7, format_func=lambda d: f"{d} days",
                                        label_visibility="collapsed")
        submitted = st.form_submit_button("Check my offer", type="primary", use_container_width=True)
    st.markdown('<div class="muted" style="text-align:center;margin-top:.6rem">'
                '🔒 We only use your loan history with us. No hidden checks.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted" style="text-align:center;margin-top:.3rem">'
                f'<span class="dot" style="display:inline-block;background:{"#34D399" if LIVE else "#FBBF24"}"></span> '
                f'{"Live model connected" if LIVE else "Model not built · run make artifacts"}</div>', unsafe_allow_html=True)

def _fmt_record_value(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return v


@st.dialog("For the credit team", width="small")
def show_credit_team_dialog(state: dict) -> None:
    r = state["r"]
    record = state["record"]
    found = state["found"]
    decision, cur = r["decision"], r["currency"]
    p, thr = r["probability_of_default"], r["threshold"]
    fill = min(p / (2 * thr), 1.0) * 100 if thr else 0
    color = {"approve": "#10B981", "limit": "#10B981", "starter": "#0EA5E9"}.get(decision, "#F59E0B")
    flags = "".join(f"<li>{f}</li>" for f in r["review_flags"]) or "<li>None</li>"

    st.markdown(
        f"""<div class="muted">Decision <b>{decision}</b> · chance of default <b>{p:.1%}</b> ·
        decline above <b>{thr:.1%}</b></div>
        <div class="bar"><span style="width:{fill:.0f}%; background:{color}"></span><i style="left:50%"></i></div>
        <div class="muted">Black line = decline threshold · scored in {r['latency_ms']:.1f} ms ·
        recommended amount {cur} {r['recommended_limit']:,.0f}</div>
        <div class="muted" style="margin-top:.6rem"><b>Model reasons:</b> {' · '.join(r['top_3_reasons'])}</div>
        <div class="muted" style="margin-top:.4rem"><b>Review flags:</b><ul>{flags}</ul></div>""",
        unsafe_allow_html=True,
    )
    shown_fields = ["customer_id", "New_versus_Repeat", "lender_id", "loan_type", *RECORD_FIELDS]
    mid = (len(shown_fields) + 1) // 2
    left_rows = "".join(
        f"<tr><td>{k}</td><td><b>{_fmt_record_value(record.get(k))}</b></td></tr>"
        for k in shown_fields[:mid]
    )
    right_rows = "".join(
        f"<tr><td>{k}</td><td><b>{_fmt_record_value(record.get(k))}</b></td></tr>"
        for k in shown_fields[mid:]
    )
    note = (
        "Found in NkwaPay's records."
        if found
        else "No record for this number: scored as a new customer on NkwaPay's standard Ghana product."
    )
    st.markdown(
        f"""<div class="muted" style="margin-top:.6rem"><b>Records used</b> · {note}
        Labelled history is 0 for every Ghanaian customer (no repayment outcomes yet).</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:.5rem 1.2rem; margin-top:.4rem;">
          <table class="rec" style="width:100%;">{left_rows}</table>
          <table class="rec" style="width:100%;">{right_rows}</table>
        </div>""",
        unsafe_allow_html=True,
    )


with right:
    validation_error = None
    if submitted:
        if len(digits(phone)) < 9:
            st.session_state.pop("last_offer", None)
            validation_error = "Please enter your mobile money number."
        else:
            record, found = lookup(phone)  # NkwaPay's side: history, lender, product
            application = {
                **{k: v for k, v in record.items() if v is not None},
                "country_id": COUNTRY,
                "Total_Amount": float(amount),
                "duration": int(duration or 7),
                "disbursement_date": date.today().isoformat(),  # scored at the moment of applying
            }
            try:
                r = score_application(application)
            except Exception as e:  # show scorer errors plainly instead of a stack trace
                st.error(f"We couldn't check this application right now: {e}")
                st.stop()
            st.session_state["last_offer"] = {
                "r": r,
                "amount": amount,
                "days": application["duration"],
                "record": record,
                "found": found,
            }

    state = st.session_state.get("last_offer")
    if validation_error:
        st.error(validation_error)
    elif not state:
        st.markdown(
            """<div class="card empty"><div class="icon">💚</div><b>Your offer will appear here.</b>
            <div class="muted">Fill in the form and press <b>Check my offer</b>.</div></div>""",
            unsafe_allow_html=True,
        )
    else:
        r = state["r"]
        req_amount = state["amount"]
        days = state["days"]
        decision, cur = r["decision"], r["currency"]
        offer = req_amount if decision == "approve" else r["recommended_limit"]
        headline, sub = {
            "approve": ("You're approved.", f"<b>{cur} {offer:,.0f}</b> for {days} days, sent to your mobile money wallet."),
            "limit": ("Approved, for a smaller amount.",
                      f"We can lend you <b>{cur} {offer:,.0f}</b> for {days} days today. Repay on time and your limit grows."),
            "starter": ("Let's start smaller.",
                        f"As a new customer you can start with <b>{cur} {offer:,.0f}</b> for {days} days. "
                        "Repaying it on time builds your record with us."),
            "decline": ("Not this time.", "We can't offer this loan today. Here's why, and what you can do next."),
        }[decision]
        if r["needs_review"] and decision != "decline":
            sub += " A member of our team will confirm the offer before the money is sent."

        st.markdown(
            f"""<div class="card result-{decision}">
            <div class="label">Your decision</div><div class="big">{headline}</div><div class="muted">{sub}</div>
            <div class="stats">
              <div class="stat"><span class="label">Requested</span><b>{cur} {req_amount:,.0f}</b></div>
              <div class="stat"><span class="label">Offered</span><b>{cur} {offer:,.0f}</b></div>
              <div class="stat"><span class="label">Term</span><b>{days} days</b></div>
            </div></div>""",
            unsafe_allow_html=True,
        )

        shown = r["borrower_reasons"] or ["Your overall application"]
        reasons = "".join(f'<div class="reason"><div class="num">{i}</div><div>{t}</div></div>'
                          for i, t in enumerate(shown, 1))
        why = f'<div class="card"><div class="label">Why we decided this</div>{reasons}</div>'
        if decision == "approve":
            st.markdown(why, unsafe_allow_html=True)
        else:
            steps = ["Ask for a review: reply REVIEW. A person will reply within 20 days."]
            steps += (["Repay on time to unlock a bigger loan next time."] if decision in ("limit", "starter")
                      else ["Try a smaller amount or a shorter term.", "Repay on time to improve future offers."])
            next_steps = "".join(f'<div class="reason"><div class="num">→</div><div>{t}</div></div>' for t in steps)
            c1, c2 = st.columns(2, gap="small")
            c1.markdown(why, unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><div class="label">What you can do</div>{next_steps}</div>',
                        unsafe_allow_html=True)

        if st.button("For the credit team", use_container_width=True):
            show_credit_team_dialog(state)
