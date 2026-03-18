import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# CUSTOM CSS - LIGHT PROFESSIONAL UI
# ============================================================
st.markdown("""
<style>
/* ===== Global App Background ===== */
html, body, [data-testid="stAppViewContainer"], .main {
    background: linear-gradient(180deg, #f4f8ff 0%, #eef4ff 100%) !important;
    color: #111827 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== Main Block Container ===== */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ===== Remove Dark Layer if Any ===== */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0) !important;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

/* ===== Titles ===== */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #163a70;
    text-align: center;
    margin-bottom: 8px;
}

.sub-title {
    font-size: 18px;
    color: #4b5563;
    text-align: center;
    margin-bottom: 28px;
}

/* ===== Cards ===== */
.custom-card {
    background: #ffffff !important;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
    border: 1px solid #e5ecf6;
    margin-bottom: 22px;
}

.result-card {
    background: linear-gradient(135deg, #ffffff, #f8fbff) !important;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
    border: 1px solid #dbeafe;
    margin-bottom: 22px;
}

/* ===== Section Heading ===== */
.section-heading {
    font-size: 24px;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 14px;
}

/* ===== Labels ===== */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #111827 !important;
    font-weight: 600 !important;
}

/* ===== Input Boxes ===== */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1.5px solid #d1d9e6 !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
    font-size: 15px !important;
}

/* ===== Selectbox ===== */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1.5px solid #d1d9e6 !important;
    border-radius: 12px !important;
    min-height: 46px !important;
}

/* selected text inside selectbox */
div[data-baseweb="select"] span {
    color: #111827 !important;
}

/* ===== Slider ===== */
.stSlider {
    padding-top: 8px;
}
.stSlider [data-baseweb="slider"] {
    padding-left: 6px;
    padding-right: 6px;
}

/* ===== Buttons ===== */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
    transition: 0.2s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.30);
}

/* ===== Metrics ===== */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e5ecf6 !important;
    padding: 12px !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 10px rgba(17, 24, 39, 0.05);
}

div[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-weight: 600 !important;
}

div[data-testid="metric-container"] div {
    color: #111827 !important;
}

/* ===== Dataframe Wrapper ===== */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #e5ecf6 !important;
}

/* ===== Risk Badges ===== */
.risk-badge {
    padding: 10px 18px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
    margin-top: 8px;
    margin-bottom: 14px;
}

.low-risk {
    background: #dcfce7;
    color: #166534;
}

.medium-risk {
    background: #fef3c7;
    color: #92400e;
}

.high-risk {
    background: #fee2e2;
    color: #991b1b;
}

/* ===== Info Note ===== */
.small-note {
    font-size: 14px;
    color: #6b7280;
    margin-top: 8px;
}

/* ===== Expanders ===== */
.streamlit-expanderHeader {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* ===== Markdown text ===== */
p, li, span, div {
    color: #111827;
}

/* ===== Sidebar if appears ===== */
[data-testid="stSidebar"] {
    background: #f8fbff !important;
}

/* ===== Fix dropdown popup text (important) ===== */
ul[role="listbox"] li {
    color: #111827 !important;
    background-color: #ffffff !important;
}

/* ===== Success/Warning/Error text better ===== */
[data-testid="stAlert"] {
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🚗 Insurance Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Professional Claim Analysis using Machine Learning + Rule-Based Risk Intelligence</div>', unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
try:
    saved_system = joblib.load("insurance_fraud_pipeline.pkl")
    pipeline = saved_system["pipeline"]
    selected_features = saved_system["selected_features"]
    best_threshold = saved_system["best_threshold"]
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

# ============================================================
# MAIN 2-COLUMN LAYOUT
# ============================================================
left_col, right_col = st.columns([1.05, 1], gap="large")

# ============================================================
# LEFT COLUMN - FORM
# ============================================================
with left_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📝 Claim Details Input</div>', unsafe_allow_html=True)

    age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
    auto_year = st.number_input("Vehicle Manufacturing Year", min_value=1990, max_value=2025, value=2016)

    incident_severity = st.selectbox(
        "Incident Severity",
        ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"]
    )

    incident_type = st.selectbox(
        "Incident Type",
        [
            "Single Vehicle Collision",
            "Multi-vehicle Collision",
            "Vehicle Theft",
            "Parked Car"
        ]
    )

    incident_hour_of_the_day = st.slider("Incident Hour of the Day", 0, 23, 14)

    total_claim_amount = st.number_input("Total Claim Amount (₹)", min_value=0, value=25000)
    witnesses = st.number_input("Number of Witnesses", min_value=0, max_value=10, value=1)
    bodily_injuries = st.number_input("Number of Bodily Injuries", min_value=0, max_value=10, value=0)

    property_damage = st.selectbox("Property Damage", ["YES", "NO"])
    police_report_available = st.selectbox("Police Report Available", ["YES", "NO"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze Fraud Risk", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# RIGHT COLUMN - INFO / RESULTS PLACEHOLDER
# ============================================================
with right_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Smart Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    This system evaluates insurance claims using:

    - **Machine Learning Probability**
    - **Suspicious Risk Signals**
    - **Business Rule Override Logic**
    - **Low / Medium / High Risk Classification**
    - **Detailed Claim & Feature Summary**
    """)
    st.info("Fill the form on the left and click **Analyze Fraud Risk**.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PREDICTION LOGIC
# ============================================================
if predict_btn:
    vehicle_age = 2026 - auto_year

    high_claim_flag = 1 if total_claim_amount > 50000 else 0
    late_night_flag = 1 if 0 <= incident_hour_of_the_day <= 5 else 0
    no_witness_flag = 1 if witnesses == 0 else 0
    no_police_report_flag = 1 if police_report_available == "NO" else 0
    total_loss_flag = 1 if incident_severity == "Total Loss" else 0
    vehicle_theft_flag = 1 if incident_type == "Vehicle Theft" else 0

    suspicious_score = (
        high_claim_flag
        + late_night_flag
        + no_witness_flag
        + no_police_report_flag
        + total_loss_flag
        + vehicle_theft_flag
    )

    input_data = {
        "age": [age],
        "auto_year": [auto_year],
        "incident_hour_of_the_day": [incident_hour_of_the_day],
        "total_claim_amount": [total_claim_amount],
        "witnesses": [witnesses],
        "bodily_injuries": [bodily_injuries],
        "incident_severity": [incident_severity],
        "incident_type": [incident_type],
        "property_damage": [property_damage],
        "police_report_available": [police_report_available],
        "vehicle_age": [vehicle_age],
        "high_claim_flag": [high_claim_flag],
        "late_night_flag": [late_night_flag],
        "no_witness_flag": [no_witness_flag],
        "no_police_report_flag": [no_police_report_flag],
        "total_loss_flag": [total_loss_flag],
        "vehicle_theft_flag": [vehicle_theft_flag],
        "suspicious_score": [suspicious_score]
    }

    input_df = pd.DataFrame(input_data)

    try:
        input_df = input_df[selected_features]
    except Exception as e:
        st.error(f"❌ Feature mismatch error: {e}")
        st.write("Model Expected Features:", selected_features)
        st.write("Available Input Features:", list(input_df.columns))
        st.stop()

    try:
        fraud_probability = pipeline.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    # Rule-based override
    if suspicious_score >= 5:
        fraud_probability = max(fraud_probability, 0.85)
    elif suspicious_score == 4:
        fraud_probability = max(fraud_probability, 0.65)
    elif suspicious_score == 3:
        fraud_probability = max(fraud_probability, 0.45)

    prediction = 1 if fraud_probability >= best_threshold else 0

    # Risk levels
    if fraud_probability < 0.30:
        risk_level = "Low Risk"
        risk_class = "low-risk"
    elif fraud_probability < 0.70:
        risk_level = "Medium Risk"
        risk_class = "medium-risk"
    else:
        risk_level = "High Risk"
        risk_class = "high-risk"

    # Reasons
    reasons = []
    if high_claim_flag:
        reasons.append("High claim amount detected")
    if late_night_flag:
        reasons.append("Late-night incident timing")
    if no_witness_flag:
        reasons.append("No witness available")
    if no_police_report_flag:
        reasons.append("Police report unavailable")
    if total_loss_flag:
        reasons.append("Claim marked as total loss")
    if vehicle_theft_flag:
        reasons.append("Incident involves vehicle theft")

    # ============================================================
    # RESULTS SECTION
    # ============================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📌 Final Fraud Analysis Result")

    res1, res2 = st.columns([1, 1], gap="large")

    # LEFT RESULT PANEL
    with res1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if prediction == 1:
            st.error("⚠️ Potential Fraudulent Claim Detected")
        else:
            st.success("✅ Claim Appears Genuine")

        st.markdown(f'<div class="risk-badge {risk_class}">{risk_level}</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Fraud Probability", f"{fraud_probability:.2%}")
        with m2:
            st.metric("Decision Threshold", f"{best_threshold:.2f}")
        with m3:
            st.metric("Suspicious Score", f"{suspicious_score}/6")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🚨 Suspicious Indicators")

        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.success("No major suspicious indicators found.")

        st.markdown("### 🧾 Decision Logic")
        if suspicious_score >= 5:
            st.warning("Strong business rule override applied due to multiple suspicious indicators.")
        elif suspicious_score == 4:
            st.info("Moderate-to-high risk adjustment applied using business rules.")
        elif suspicious_score == 3:
            st.info("Moderate suspicious activity triggered partial fraud probability adjustment.")
        else:
            st.success("Prediction mainly based on machine learning model output.")

        st.markdown('<div class="small-note">This project combines ML prediction with real-world fraud risk rules for more reliable claim screening.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT RESULT PANEL
    with res2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Claim Summary")

        summary_df = pd.DataFrame({
            "Feature": [
                "Customer Age",
                "Vehicle Year",
                "Vehicle Age",
                "Incident Severity",
                "Incident Type",
                "Incident Hour",
                "Claim Amount",
                "Witnesses",
                "Bodily Injuries",
                "Property Damage",
                "Police Report"
            ],
            "Value": [
                age,
                auto_year,
                vehicle_age,
                incident_severity,
                incident_type,
                incident_hour_of_the_day,
                total_claim_amount,
                witnesses,
                bodily_injuries,
                property_damage,
                police_report_available
            ]
        })

        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Engineered Risk Features")

        engineered_df = pd.DataFrame({
            "Risk Feature": [
                "High Claim Flag",
                "Late Night Flag",
                "No Witness Flag",
                "No Police Report Flag",
                "Total Loss Flag",
                "Vehicle Theft Flag",
                "Suspicious Score"
            ],
            "Value": [
                high_claim_flag,
                late_night_flag,
                no_witness_flag,
                no_police_report_flag,
                total_loss_flag,
                vehicle_theft_flag,
                suspicious_score
            ]
        })

        st.dataframe(engineered_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)