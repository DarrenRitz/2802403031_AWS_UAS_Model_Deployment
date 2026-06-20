import streamlit as st
import pandas as pd
import boto3
import json
import numpy as np

ENDPOINT_NAME = "uas-endpoint"
REGION        = "us-east-1"

SCORE_COLOR = {"Good": "#2ecc71", "Standard": "#f39c12", "Poor": "#e74c3c"}
SCORE_EMOJI = {"Good": "✅",      "Standard": "⚠️",       "Poor": "❌"}

OCCUPATIONS = [
    "Teacher", "Doctor", "Accountant", "Writer", "Musician",
    "Engineer", "Architect", "Journalist", "Entrepreneur",
    "Developer", "Media_Manager", "Scientist", "Manager",
    "Lawyer", "Mechanic",
]

PAYMENT_BEHAVIOURS = [
    "Low_spent_Small_value_payments",  "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",  "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments","High_spent_Large_value_payments",
]

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .section-title {
        font-size: 13px; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #7c8db5;
        margin-bottom: 10px; margin-top: 24px;
        border-bottom: 1px solid #1e2235; padding-bottom: 6px;
    }
    .result-card { border-radius: 12px; padding: 28px 32px; margin-top: 24px; text-align: center; }
    .result-label { font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: #7c8db5; margin-bottom: 6px; }
    .result-score { font-size: 48px; font-weight: 800; line-height: 1.1; }
    .result-model { font-size: 12px; color: #555e7a; margin-top: 8px; }
    .prob-label   { font-size: 12px; color: #7c8db5; margin-bottom: 2px; }
    label { color: #b0b8d0 !important; font-size: 13px !important; }
    div.stButton > button {
        background: linear-gradient(135deg, #3b5bdb, #5c7cfa); color: white;
        border: none; border-radius: 8px; padding: 12px 0;
        font-size: 15px; font-weight: 600; width: 100%;
        letter-spacing: 0.03em; cursor: pointer; transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.88; }
    section[data-testid="stSidebar"] { background-color: #090b10; border-right: 1px solid #1a1d2e; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💳 Credit Score")
    st.markdown("---")
    st.caption("Model: Random Forest")
    st.caption("Target: Good / Standard / Poor")
    st.caption(f"Endpoint: `{ENDPOINT_NAME}`")
    st.markdown("---")
    st.caption("2802403031 · Darren Ritz Junaidi")

st.markdown("# 💳 Credit Score Predictor")
st.markdown("Masukkan data nasabah di bawah, lalu klik **Predict** untuk melihat prediksi skor kredit.")
st.markdown("---")

with st.form("predict_form"):

    st.markdown('<div class="section-title">Identitas Nasabah</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: age           = st.number_input("Age", min_value=17, max_value=100, value=30)
    with c2: occupation    = st.selectbox("Occupation", OCCUPATIONS)
    with c3: annual_income = st.number_input("Annual Income (USD)", min_value=0.0, value=50000.0, step=1000.0)

    st.markdown('<div class="section-title">Informasi Bank & Kredit</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, value=3)
    with c2: num_credit_card   = st.number_input("Num Credit Cards",  min_value=0, value=2)
    with c3: interest_rate     = st.number_input("Interest Rate (%)", min_value=0.0, value=15.0, step=0.5)
    with c4: num_of_loan       = st.number_input("Num of Loans",      min_value=0, value=2)

    c1, c2, c3 = st.columns(3)
    with c1: credit_mix               = st.selectbox("Credit Mix", ["Standard", "Good", "Bad"])
    with c2: outstanding_debt         = st.number_input("Outstanding Debt (USD)", min_value=0.0, value=500.0, step=10.0)
    with c3: credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)

    c1, c2 = st.columns(2)
    with c1: credit_history_age   = st.number_input("Credit History Age (months)", min_value=0, value=60)
    with c2: num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, value=2)

    st.markdown('<div class="section-title">Perilaku Pembayaran</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: delay_from_due_date    = st.number_input("Delay from Due Date (days)", min_value=0, value=5)
    with c2: num_of_delayed_payment = st.number_input("Num of Delayed Payments",   min_value=0, value=3)
    with c3: payment_of_min_amount  = st.selectbox("Payment of Min Amount", ["No", "Yes"])

    c1, c2 = st.columns(2)
    with c1: payment_behaviour   = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOURS)
    with c2: changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, value=10.0, step=0.5)

    st.markdown('<div class="section-title">Keuangan Bulanan</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: monthly_inhand_salary   = st.number_input("Monthly Inhand Salary (USD)",   min_value=0.0, value=4000.0, step=100.0)
    with c2: total_emi_per_month     = st.number_input("Total EMI per Month (USD)",     min_value=0.0, value=200.0,  step=10.0)
    with c3: amount_invested_monthly = st.number_input("Amount Invested Monthly (USD)", min_value=0.0, value=150.0,  step=10.0)
    monthly_balance = st.number_input("Monthly Balance (USD)", min_value=0.0, value=500.0, step=10.0)

    st.markdown('<div class="section-title">Jenis Pinjaman yang Dimiliki</div>', unsafe_allow_html=True)
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        student_loan  = st.checkbox("Student Loan")
        personal_loan = st.checkbox("Personal Loan")
    with lc2:
        credit_builder_loan = st.checkbox("Credit-Builder Loan")
        mortgage_loan       = st.checkbox("Mortgage Loan")
    with lc3:
        debt_consolidation_loan = st.checkbox("Debt Consolidation Loan")
        payday_loan             = st.checkbox("Payday Loan")
    with lc4:
        auto_loan        = st.checkbox("Auto Loan")
        home_equity_loan = st.checkbox("Home Equity Loan")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Credit Score")

if submitted:
    payload = {
        "instances": [[
            age, occupation, annual_income, monthly_inhand_salary,
            num_bank_accounts, num_credit_card, interest_rate, num_of_loan,
            delay_from_due_date, num_of_delayed_payment, changed_credit_limit,
            num_credit_inquiries, credit_mix, outstanding_debt,
            credit_utilization_ratio, credit_history_age, payment_of_min_amount,
            total_emi_per_month, amount_invested_monthly, payment_behaviour,
            monthly_balance,
            int(student_loan), int(personal_loan), int(credit_builder_loan),
            int(mortgage_loan), int(debt_consolidation_loan), int(payday_loan),
            int(auto_loan), int(home_equity_loan),
        ]]
    }

    try:
        runtime  = get_runtime_client()
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        result       = json.loads(response["Body"].read().decode("utf-8"))
        credit_score = result["labels"][0]
        proba_list   = result["probabilities"][0]
        class_names  = ["Good", "Poor", "Standard"]  # urutan alfabetis LabelEncoder
        probabilities = dict(zip(class_names, proba_list))

        color = SCORE_COLOR.get(credit_score, "#888")
        emoji = SCORE_EMOJI.get(credit_score, "")

        st.markdown(f"""
        <div class="result-card" style="background:{color}18; border:1.5px solid {color}55;">
            <div class="result-label">Prediksi Credit Score</div>
            <div class="result-score" style="color:{color};">{emoji} {credit_score}</div>
            <div class="result-model">Model: Random Forest · Endpoint: {ENDPOINT_NAME}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Probabilitas per Kelas")
        for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
            st.markdown(f'<div class="prob-label">{label}</div>', unsafe_allow_html=True)
            st.progress(float(prob))
            st.caption(f"{prob * 100:.1f}%")

    except Exception as e:
        st.error(f"Error: {e}")