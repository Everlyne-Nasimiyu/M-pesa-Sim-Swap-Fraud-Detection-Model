
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model  = joblib.load("xgboost_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="M-Pesa SIM Swap Fraud Detector",
    page_icon="🔐",
    layout="centered"
)

# Header
st.title("🔐 M-Pesa SIM Swap Fraud Detection")
st.markdown("Enter transaction details below to check if it is fraudulent.")
st.divider()

# ── Numerical Inputs ───────────────────────────────────────────
st.subheader("Transaction Details")

amount              = st.text_input("Transaction Amount (KES)", "500")
sender_bal_before   = st.text_input("Sender Balance Before (KES)", "5000")
sender_bal_after    = st.text_input("Sender Balance After (KES)", "4500")
receiver_bal_before = st.text_input("Receiver Balance Before (KES)", "1000")
receiver_bal_after  = st.text_input("Receiver Balance After (KES)", "1500")
hour                = st.text_input("Hour of Transaction (0-23)", "14")

st.divider()

# ── Categorical Inputs ─────────────────────────────────────────
st.subheader("Transaction Context")
st.markdown("**Transaction Type** — type: `peer`, `till`, or `paybill`")
transaction_type = st.text_input("Transaction Type", "peer")

st.markdown("**Device Type** — type: `smartphone` or `feature`")
device_type = st.text_input("Device Type", "smartphone")

st.markdown("**Region** — type: `Nairobi`, `Mombasa`, `Kisumu`, `Nakuru`, or `Eldoret`")
region = st.text_input("Region", "Nairobi")

st.markdown("**Day of Week** — type: `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, or `Sun`")
day_of_week = st.text_input("Day of Week", "Tue")

st.divider()

# ── Predict ────────────────────────────────────────────────────
if st.button("🔍 Check Transaction"):
    try:
        # Convert inputs to correct types
        amount_val              = float(amount)
        sender_bal_before_val   = float(sender_bal_before)
        sender_bal_after_val    = float(sender_bal_after)
        receiver_bal_before_val = float(receiver_bal_before)
        receiver_bal_after_val  = float(receiver_bal_after)
        hour_val                = int(hour)

        # Engineer features
        balance_depletion_rate = amount_val / (sender_bal_before_val + 1)
        is_high_value          = 1 if amount_val > 3800 else 0
        is_balance_wipeout     = 1 if sender_bal_after_val < 100 else 0
        sender_balance_ratio   = sender_bal_after_val / (sender_bal_before_val + 1)

        # Build feature dict
        input_data = {
            "amount":                   amount_val,
            "sender_balance_before":    sender_bal_before_val,
            "sender_balance_after":     sender_bal_after_val,
            "receiver_balance_before":  receiver_bal_before_val,
            "receiver_balance_after":   receiver_bal_after_val,
            "hour":                     hour_val,
            "transaction_type_peer":    1 if transaction_type == "peer"       else 0,
            "transaction_type_till":    1 if transaction_type == "till"       else 0,
            "device_type_smartphone":   1 if device_type      == "smartphone" else 0,
            "region_Kisumu":            1 if region           == "Kisumu"     else 0,
            "region_Mombasa":           1 if region           == "Mombasa"    else 0,
            "region_Nairobi":           1 if region           == "Nairobi"    else 0,
            "region_Nakuru":            1 if region           == "Nakuru"     else 0,
            "day_of_week_Mon":          1 if day_of_week      == "Mon"        else 0,
            "day_of_week_Sat":          1 if day_of_week      == "Sat"        else 0,
            "day_of_week_Sun":          1 if day_of_week      == "Sun"        else 0,
            "day_of_week_Thu":          1 if day_of_week      == "Thu"        else 0,
            "day_of_week_Tue":          1 if day_of_week      == "Tue"        else 0,
            "day_of_week_Wed":          1 if day_of_week      == "Wed"        else 0,
            "balance_depletion_rate":   balance_depletion_rate,
            "is_high_value":            is_high_value,
            "is_balance_wipeout":       is_balance_wipeout,
            "sender_balance_ratio":     sender_balance_ratio,
        }

        input_df = pd.DataFrame([input_data])

        # Scale
        scale_cols = ["amount", "sender_balance_before", "sender_balance_after",
                      "receiver_balance_before", "receiver_balance_after",
                      "balance_depletion_rate", "sender_balance_ratio"]
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        # Predict
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Display result
        st.subheader("Detection Result")

        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
            st.metric("Fraud Probability", f"{probability*100:.1f}%")
            st.warning("""
            **Recommended Actions:**
            - Freeze the account immediately
            - Notify subscriber via alternative contact
            - Escalate to fraud investigation team
            - Block pending Fuliza / M-Shwari drawdowns
            """)
        else:
            st.success("✅ TRANSACTION APPEARS LEGITIMATE")
            st.metric("Fraud Probability", f"{probability*100:.1f}%")
            st.info("Transaction cleared. No action required.")

        # Show engineered features
        with st.expander("View Engineered SIM Swap Signals"):
            st.write(f"**Balance Depletion Rate:** {balance_depletion_rate:.4f}")
            st.write(f"**Sender Balance Ratio:**   {sender_balance_ratio:.4f}")
            st.write(f"**Balance Wipeout Flag:**   {is_balance_wipeout}")
            st.write(f"**High Value Flag:**        {is_high_value}")

    except ValueError:
        st.error("Please ensure all numerical fields contain valid numbers.")

# Footer
st.divider()
st.caption("SIM Swap Fraud Detection System — "
           "M-Pesa Ecosystem | XGBoost Model | "
           "GoMyCode Data Science Capstone 2026")
