"""NkwaPay loan app: a borrower-facing UI over the shared scorer (`src.serving.score_application`, P3).

Run from the repo root, after `make features artifacts`:  streamlit run app/streamlit_app.py

The app never re-implements features, thresholds or the decision rule. It builds the application dict the
scorer documents (`src.serving.REQUIRED_APPLICATION_FIELDS`) and displays what comes back:
decision (approve / limit / starter / decline), recommended_limit, currency, borrower_reasons (shown to the
borrower), top_3_reasons and review_flags (credit team only).
Borrowers enter only what they know (phone, amount, term); NkwaPay-side fields sit in the demo-controls panel.
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


FALLBACK = {"loan_type": ({"Type_1"}, {"Type_1", "Type_3", "Type_8"}), "lender_id": ({"267278"}, {"267278"})}


@st.cache_data
def catalogue(col: str) -> tuple[list[str], set[str]]:
    """Every value of `col` in Train + Test, and those the model never saw in training (Test-only, i.e. Ghana)."""
    import pandas as pd

    try:
        train = set(pd.read_csv(C.TRAIN_CSV, usecols=[col])[col].astype(str))
        test = set(pd.read_csv(C.TEST_CSV, usecols=[col])[col].astype(str))
    except FileNotFoundError:  # data not downloaded: fall back to what the EDA documented
        train, test = FALLBACK[col]
    every = sorted(train | test, key=lambda v: (len(v), v))
    return every, test - train


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

MARKETS = {"🇬🇭 Ghana": ("Ghana", "GHS"), "🇰🇪 Kenya": ("Kenya", "KES")}

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

      /* ---------- floating control panel ---------- */
      .st-key-controls {background: linear-gradient(160deg, #053B2D 0%, #064E3B 55%, #065F46 100%);
            border-radius: 24px; padding: 1.8rem 1.8rem 1.5rem; position: sticky; top: 1rem;
            box-shadow: 0 24px 48px rgba(5,59,45,.28), 0 2px 6px rgba(5,59,45,.2);}
      .st-key-controls * {color: #ECFDF5;}
      .st-key-controls [data-testid="stWidgetLabel"] p {font-size: .78rem !important; font-weight: 600; color: #A7F3D0 !important;
                                         text-transform: uppercase; letter-spacing: .05em;}
      .st-key-controls [data-baseweb="input"], .st-key-controls [data-baseweb="base-input"],
      .st-key-controls [data-baseweb="select"] > div {
            background: rgba(255,255,255,.08) !important; border: 1px solid rgba(167,243,208,.25) !important;
            border-radius: 14px !important; min-height: 50px;}
      .st-key-controls [data-baseweb="input"], .st-key-controls [data-baseweb="base-input"],
      .st-key-controls [data-baseweb="select"] > div {background: #fff !important; border-color: transparent !important;}
      .st-key-controls input, .st-key-controls [data-baseweb="select"] div, .st-key-controls [data-baseweb="select"] span,
      .st-key-controls [data-testid="stNumberInput"] button {
            color: #053B2D !important; -webkit-text-fill-color: #053B2D !important; font-weight: 600;}
      .st-key-controls [data-testid="stNumberInput"] button {background: #ECFDF5 !important;}
      .st-key-controls [data-testid="stDateInput"] [role="group"] *, .st-key-controls [data-testid="stDateInput"] input {color: #053B2D !important; -webkit-text-fill-color: #053B2D !important;}
      .st-key-controls svg {fill: #047857 !important; color: #047857 !important;}
      .st-key-controls [data-testid="stButtonGroup"] button {
            background: rgba(255,255,255,.08); border: 1px solid rgba(167,243,208,.25); border-radius: 14px;
            padding: .6rem 1.2rem; min-height: 46px; font-size: 1rem;}
      .st-key-controls [data-testid="stButtonGroup"] {gap: .5rem;}
      .st-key-controls [data-testid="stButtonGroup"] button[kind$="Active"] {
            background: #10B981; border-color: #10B981;}
      .st-key-controls hr {border-color: rgba(167,243,208,.2);}
      .sb-brand {display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .3rem;}
      .sb-logo {width:36px; height:36px; border-radius:11px; background:#10B981; display:flex; align-items:center;
                justify-content:center; font-weight:800; color:#053B2D !important; font-size:1.1rem;}
      .sb-title {font-weight:800; font-size:1.15rem;}
      .hero-brand {margin-bottom:.9rem;} .hero-brand .sb-title {color:#fff; font-size:1.3rem;}
      .sb-sub {color:#A7F3D0 !important; font-size:.9rem; margin-bottom:.4rem; line-height:1.45;}
      .sb-section {color:#6EE7B7 !important; font-size:.7rem; font-weight:700; letter-spacing:.12em;
                   text-transform:uppercase; margin: 1.5rem 0 .4rem;}
      .sb-status {display:flex; align-items:center; gap:.5rem; background:rgba(255,255,255,.07);
                  border-radius:14px; padding:.8rem 1rem; font-size:.88rem;}
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
      .chip {background:rgba(255,255,255,.16); color:#fff; border-radius:999px; padding:.3rem .9rem;
             font-size:.8rem; font-weight:700; white-space:nowrap;
             display:inline-block; margin-bottom:.8rem;}
      [data-testid="stForm"] {background:#fff; border:1px solid #E5E7EB !important; border-radius:22px;
             padding:1.8rem 1.9rem; box-shadow:0 8px 24px rgba(6,78,59,.06);}
      [data-testid="stForm"] [data-baseweb="input"] {min-height:52px; border-radius:14px !important;
             background:#F8FAF9; border:1px solid #E5E7EB !important;}
      [data-testid="stForm"] input {font-size:1.05rem !important;}
      [data-testid="stForm"] [data-testid="stButtonGroup"] button {min-height:46px; padding:.6rem 1.1rem; font-size:.98rem;}
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
      .team {margin-top:.9rem; padding-top:.8rem; border-top:1px dashed #E5E7EB;}
      div[data-testid="stFormSubmitButton"] > button {border-radius:999px; padding:.85rem 1rem; font-weight:700; font-size:1.05rem;
            background:#047857; border:none;}
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
ctrl, left, right = st.columns([1, 1.25, 1.35], gap="large")

# ---------- floating panel: NkwaPay-side inputs, not typed by the borrower ----------
with ctrl, st.container(key="controls"):
    st.markdown(
        """<div class="sb-title">Demo controls</div>
        <div class="sb-sub">In production these come from NkwaPay's records, not the borrower.</div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-section">Market</div>', unsafe_allow_html=True)
    market = st.segmented_control("Market", list(MARKETS), default="🇬🇭 Ghana", label_visibility="collapsed") or "🇬🇭 Ghana"
    country, cur = MARKETS[market]

    st.markdown('<div class="sb-section">Borrower record</div>', unsafe_allow_html=True)
    first_time = st.toggle("First-time borrower", value=False)
    tenure = "New Loan" if first_time else "Repeat Loan"
    history = {}
    if not first_time:  # in production these come from NkwaPay's loan records
        h1, h2, h3 = st.columns(3)
        n_past = h1.number_input("Earlier loans", 1, 300, 6)
        n_missed = h2.number_input("Missed", 0, int(n_past), 0)
        days_since = h3.number_input("Days since last", 0, 2000, 14)
        history = {"n_past_loans": int(n_past), "n_past_labelled": int(n_past), "n_past_defaults": int(n_missed),
                   "days_since_last_loan": float(days_since)}

    st.markdown('<div class="sb-section">Loan setup</div>', unsafe_allow_html=True)
    lenders, unseen_lenders = catalogue("lender_id")
    lender_id = st.selectbox("Funding lender", lenders, accept_new_options=True,  # default: the main lender (94% of Train)
                             index=lenders.index("267278") if "267278" in lenders else 0,
                             format_func=lambda v: f"{v} · unseen in training" if v in unseen_lenders else v)
    types, unseen = catalogue("loan_type")
    loan_type = st.selectbox("Product", types, accept_new_options=True,
                             format_func=lambda t: f"{t} · unseen in training" if t in unseen else t)
    app_date = st.date_input("Application date", date.today())

    st.divider()
    st.markdown(
        f"""<div class="sb-status"><span class="dot" style="background:{'#34D399' if LIVE else '#FBBF24'}"></span>
        {'Live model connected' if LIVE else 'Model not built · run make artifacts'}</div>""",
        unsafe_allow_html=True,
    )

# ---------- main ----------

with left:
    with st.form("apply"):
        st.markdown('<div class="label">Your mobile money number</div>', unsafe_allow_html=True)
        customer_id = st.text_input("Mobile money number", placeholder="e.g. 024 123 4567", label_visibility="collapsed")
        st.markdown('<div class="label">How much do you need?</div>', unsafe_allow_html=True)
        amount = st.slider("Amount", 50, 5000, 500, step=50, format=f"{cur} %d", label_visibility="collapsed")
        st.markdown('<div class="label">Pay back in</div>', unsafe_allow_html=True)
        duration = st.segmented_control("Term", [7, 14, 30, 60, 90], default=7, format_func=lambda d: f"{d} days",
                                        label_visibility="collapsed")
        submitted = st.form_submit_button("Check my offer", type="primary", use_container_width=True)
    st.markdown('<div class="muted" style="text-align:center;margin-top:.6rem">'
                '🔒 We only use your loan history with us. No hidden checks.</div>', unsafe_allow_html=True)

with right:
    if not submitted:
        st.markdown(
            """<div class="card empty"><div class="icon">💚</div><b>Your offer will appear here.</b>
            <div class="muted">Fill in the form and press <b>Check my offer</b>.</div></div>""",
            unsafe_allow_html=True,
        )
    elif not customer_id.strip():
        st.error("Please enter your mobile money number.")
    else:
        application = {
            "customer_id": customer_id.strip(),
            "country_id": country,
            "lender_id": str(lender_id).strip(),
            "loan_type": str(loan_type).strip(),
            "Total_Amount": float(amount),
            "duration": int(duration or 7),
            "disbursement_date": app_date.isoformat(),
            "New_versus_Repeat": tenure,
            **history,
        }
        try:
            r = score_application(application)
        except Exception as e:  # show scorer errors plainly instead of a stack trace
            st.error(f"We couldn't check this application right now: {e}")
            st.stop()

        decision, cur = r["decision"], r["currency"]
        offer = amount if decision == "approve" else r["recommended_limit"]
        days = application["duration"]
        p, thr = r["probability_of_default"], r["threshold"]
        fill = min(p / (2 * thr), 1.0) * 100 if thr else 0
        color = {"approve": "#10B981", "limit": "#10B981", "starter": "#0EA5E9"}.get(decision, "#F59E0B")
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
              <div class="stat"><span class="label">Requested</span><b>{cur} {amount:,.0f}</b></div>
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

        with st.expander("For the credit team"):
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
