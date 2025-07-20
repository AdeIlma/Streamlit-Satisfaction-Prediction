import streamlit as st
import pandas as pd
import joblib
from collections import Counter
import os

# ========================
# Display SVG Logo
# ========================
if os.path.exists("olist.svg"):
    with open("olist.svg", "r") as f:
        svg_logo = f.read()

    svg_logo = svg_logo.replace(
        '<svg',
        '<svg style="width: 350px; display: block; margin: auto; margin-bottom: 20px;"'
    )
    st.markdown(svg_logo, unsafe_allow_html=True)
else:
    st.warning("⚠️ 'olist.svg' logo file not found.")

# =======================
# Load Model and Categories
# =======================
def load_categories(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

model_files = [f'LightGBM_ADASYN_RSCV_fold{i}.pkl' for i in range(1, 6)]
models = [joblib.load(open(filename, 'rb')) for filename in model_files]
category_options = load_categories('kategori.txt')

# =======================
# UI Header
# =======================
st.markdown("""
    <h1 style='text-align: center; color: #4F8BF9;'>🔍 E-commerce Customer Satisfaction Prediction</h1>
    <p style='text-align: center; font-size: 18px;'>Enter transaction data and product category to predict whether a customer will be satisfied or not.</p>
""", unsafe_allow_html=True)

# =======================
# Input Section
# =======================
with st.expander("📥 Input Transaction Data & Product Category", expanded=True):
    st.markdown("### 💡 Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        total_freight = st.number_input("💰 Freight Cost", min_value=0.0, step=0.01)
        processing_time_days = st.number_input("🛠️ Processing Time (days)", value=0, step=1, help="Number of days from order placed to shipped.")
        review_time_days = st.number_input("📝 Review Time Lag (days)", value=0, step=1, help="Number of days between delivery and review.")
        review_response_time_days = st.number_input("📝 Review Response Time (days)", value=0, step=1, help="Number of days before the customer gave a review.")

    with col2:
        delivery_time_days = st.number_input("🚚 Delivery Time (days)", value=0, step=1, help="Actual shipping duration in days.")
        delivery_delay_days = st.number_input("⏰ Delivery Delay (days)", value=0, step=1, help="Difference between estimated and actual delivery time.")
        estimated_delivery_time_days = st.number_input("📦 Estimated Delivery Time (days)", value=0, step=1, help="Estimated delivery time based on system prediction.")
        max_processing_time_days = st.number_input("🔧 Max Processing Time (days)", value=0, step=1, help="Longest recorded processing time.")

    st.markdown("---")
    st.markdown("### 🗂️ Customer & Product Info")

    col3, col4 = st.columns(2)
    with col3:
        customer_state = st.selectbox("🌎 Customer Region", [
            "Southeast (Sudeste)", "South (Sul)", "Northeast (Nordeste)",
            "Central-West (Centro-Oeste)", "North (Norte)"
        ])
    with col4:
        product_category = st.selectbox("🏷️ Product Category", category_options)

# =======================
# Prediction Button
# =======================
st.markdown("---")
st.markdown("### 🚀 Click the button below to make a prediction:")

if st.button("🔍 Predict Satisfaction"):
    input_df = pd.DataFrame([{
        'processing_time_days': processing_time_days,
        'delivery_time_days': delivery_time_days,
        'delivery_delay_days': delivery_delay_days,
        'review_time_days': review_time_days,
        'product_category_name_english': product_category,
        'new_customer_state': customer_state,
        'estimated_delivery_time_days': estimated_delivery_time_days,
        'total_freight': total_freight,
        'max_processing_time_days': max_processing_time_days,
        'review_response_time_days': review_response_time_days
    }])

    votes = [model.predict(input_df)[0] for model in models]
    final_prediction = Counter(votes).most_common(1)[0][0]

    st.markdown("---")
    st.markdown("### 📊 Prediction Result:")

    if final_prediction == 1:
        st.success("✅ The customer is predicted to be **SATISFIED** with the service.")
        st.markdown("<div style='text-align:center; font-size:40px;'>😄</div>", unsafe_allow_html=True)
    else:
        st.error("❌ The customer is predicted to be **NOT SATISFIED** with the service.")
        st.markdown("<div style='text-align:center; font-size:40px;'>😟</div>", unsafe_allow_html=True)
