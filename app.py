import streamlit as st
import pandas as pd
import joblib

# ========================
# Load Trained Model
# ========================
model_dict = joblib.load('xgb_smoten_jcoba.pkl')
model = model_dict['model']
threshold = model_dict['threshold']  # ambil threshold custom

# ========================
# Show SVG Logo (optional)
# ========================
try:
    with open("olist.svg", "r") as f:
        svg_logo = f.read()

    svg_logo = svg_logo.replace(
        '<svg',
        '<svg style="width: 350px; display: block; margin: auto; margin-bottom: 10px;"'
    )
    st.markdown(svg_logo, unsafe_allow_html=True)

except FileNotFoundError:
    st.warning("⚠️ File 'olist.svg' not found. Please make sure it's in the same folder.")

# ========================
# Title and Description
# ========================
st.markdown(
    "<h1 style='text-align: center;'>🔍 E-commerce Customer Satisfaction Prediction (Olist)</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Enter transaction details to predict whether the customer is satisfied or not based on a machine learning model.</p>",
    unsafe_allow_html=True
)

# ========================
# Input Form
# ========================
st.markdown("## 💡 Transaction Data")
col1, col2 = st.columns(2)

with col1:
    review_time_days = st.number_input("📝 Time Gap to Review Days", value=0, step=1, help="Days between delivery and review from the customer.")
    processing_time_days = st.number_input("🛠️ Processing Time Days", value=0, step=1, help="Time taken by the seller to process the order.")
    quantity = st.number_input("💰 Quantity", value=1, step=1, help="Number of items ordered.")

with col2:
    payment_installments = st.number_input("💳 Number of Installments", value=0, step=1, help="Total number of payments made in installments.")
    review_response_time_days = st.number_input("💬 Seller Response Time Gap Days", value=0, step=1, help="Time between review and seller response.")
    delivery_time_days = st.number_input("🚚 Delivery Time Days", value=0, step=1, help="Days from shipping to delivery.")

# ========================
# Prediction
# ========================
st.markdown("---")
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([{
        'processing_time_days': processing_time_days,
        'review_time_days': review_time_days,
        'quantity': quantity,
        'review_response_time_days': review_response_time_days,
        'payment_installments': payment_installments,
        'delivery_time_days': delivery_time_days
    }])

    # Prediksi probabilitas tiap kelas
    probs = model.predict_proba(input_df)[0]  # [prob_kelas_0, prob_kelas_1]
    prediction = int(probs[1] >= threshold)

    # ========================
    # Output
    # ========================
    st.markdown("### 🎯 Class Probabilities")
    st.markdown(f"- ❌ Not Satisfied (Class 0): `{probs[0]:.2f}`")
    st.markdown(f"- ✅ Satisfied (Class 1): `{probs[1]:.2f}`")

    st.bar_chart(pd.DataFrame({
        'Probability': probs
    }, index=['Not Satisfied', 'Satisfied']))

    st.markdown(f"### 🔮 Final Prediction: `{probs[1]:.2f}` (based on threshold `{threshold}`)")

    if prediction == 1:
        st.success("✅ Prediction: **Satisfied**")
        st.markdown("> This customer is likely to leave a **positive review** based on the transaction details.")
    else:
        st.error("❌ Prediction: **Not Satisfied**")
        st.markdown("> This customer may be **dissatisfied**. Please review the delivery, response, or processing time.")
