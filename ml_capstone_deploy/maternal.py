# ─────────────────────────────────────────────────────────────────────────────
# app.py — Maternal Health Risk Predictor
# Run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    # Full pipeline: scaler + SMOTE + RandomForest — NO separate scaler needed
    model   = joblib.load("trained_models/maternal_risk_model.pkl")
    encoder = joblib.load("trained_models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

st.title("🏥 Maternal Health Risk Predictor")
st.markdown("Enter the patient's vital signs and click **Predict Risk Level**.")
st.markdown("---")

st.subheader("Patient Vitals")
col1, col2, col3 = st.columns(3)

with col1:
    age    = st.number_input("Age (years)",                     min_value=10,  max_value=70,  value=30)
    sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50,  max_value=200, value=120)

with col2:
    dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30,  max_value=150, value=80)
    bs     = st.number_input("Blood Sugar (mmol/L)",            min_value=1.0, max_value=30.0, value=7.0, step=0.1)

with col3:
    temp       = st.number_input("Body Temperature (F)",  min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    heart_rate = st.number_input("Heart Rate (bpm)",      min_value=40,   max_value=150,   value=76)

st.markdown("---")
predict_clicked = st.button("Predict Risk Level", use_container_width=True)

if predict_clicked:
    # Build input with exact same column names used during training
    data = pd.DataFrame({
        "Age"        : [age],
        "SystolicBP" : [sys_bp],
        "DiastolicBP": [dia_bp],
        "BS"         : [bs],
        "BodyTemp"   : [temp],
        "HeartRate"  : [heart_rate]
    })

    # Pipeline handles scaling — do NOT scale manually
    prediction_num = model.predict(data)[0]
    probabilities  = model.predict_proba(data)[0]

    # CORRECT mapping — LabelEncoder is alphabetical: high risk=0, low risk=1, mid risk=2
    risk_label = encoder.classes_[prediction_num]

    st.subheader("Prediction Result")

    m1, m2, m3 = st.columns(3)
    for col_widget, cls, prob in zip([m1, m2, m3], encoder.classes_, probabilities):
        col_widget.metric(f"{cls.title()} Probability", f"{prob*100:.1f}%")

    st.markdown("---")

    if risk_label == "high risk":
        st.error("HIGH RISK — Immediate medical attention recommended.")
    elif risk_label == "mid risk":
        st.warning("MID RISK — Increased monitoring and care advised.")
    else:
        st.success("LOW RISK — Patient vitals appear within safe range.")

    st.markdown("### Clinical Recommendation")
    if risk_label == "high risk":
        st.error("Urgent: Refer to specialist, monitor BP/BS continuously, check for preeclampsia.")
    elif risk_label == "mid risk":
        st.warning("Schedule follow-up in 1-2 weeks, daily home BP monitoring, dietary guidance.")
    else:
        st.success("Routine care: Continue regular prenatal check-ups and healthy lifestyle.")

    st.markdown("### Patient Summary")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
| Vital | Value | Normal Range |
|---|---|---|
| Age | {age} years | 18-40 |
| Systolic BP | {sys_bp} mmHg | 90-120 |
| Diastolic BP | {dia_bp} mmHg | 60-80 |
        """)
    with s2:
        st.markdown(f"""
| Vital | Value | Normal Range |
|---|---|---|
| Blood Sugar | {bs} mmol/L | 4.0-5.5 |
| Body Temp | {temp} F | 97-99 |
| Heart Rate | {heart_rate} bpm | 60-100 |
        """)

st.markdown("---")
st.markdown(
    "<center><small>Maternal Health Risk Predictor - Random Forest + SMOTE - Built with Streamlit</small></center>",
    unsafe_allow_html=True
)
