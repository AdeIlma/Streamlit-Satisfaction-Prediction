import streamlit as st
import pandas as pd
import joblib
from collections import Counter
import os

# ========================
# Page Configuration
# ========================
st.set_page_config(page_title="Customer Satisfaction Prediction", page_icon="✅", layout="centered")

# ========================
# Display SVG Logo (Optional)
# ========================
if os.path.exists("olist.svg"):
    with open("olist.svg", "r") as f:
        svg_logo = f.read()
    svg_logo = svg_logo.replace('<svg', '<svg style="width: 350px; display: block; margin: auto; margin-bottom: 20px;"')
    st.markdown(svg_logo, unsafe_allow_html=True)
else:
    st.warning("⚠️ 'olist.svg' not found.")

# ========================
# Load Product Categories
# ========================
def load_categories(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        st.error(f"❌ File '{file_path}' not found.")
        return []

categories = load_categories('kategori.txt')

# ========================
# App Header
# ========================
st.markdown("<h2 style='text-align: center;'>🔍 E-commerce Customer Satisfaction Prediction (Olist)</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill in the transaction details below to predict customer satisfaction using a machine learning model.</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================
# Load Pre-trained Models
# ========================
model_files = [f'XGBoost_ADASYN_fold{i}.pkl' for i in range(1, 6)]
models = []

for file in model_files:
    try:
        models.append(joblib.load(file))
    except Exception as e:
        st.error(f"❌ Failed to load model '{file}': {e}")

# ========================
# Input Form
# ========================
with st.form("prediction_form"):
    st.markdown("### 📝 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        review_time = st.number_input("📆 Review Time (days)", min_value=0, step=1, help="Days between delivery and customer review.")
        processing_time = st.number_input("🏷️ Processing Time (days)", min_value=0, step=1, help="Days between order and shipment.")
        delivery_time = st.number_input("🚚 Delivery Time (days)", min_value=0, step=1, help="Days between shipment and delivery.")
        delivery_delay = st.number_input("⏱️ Delivery Delay (days)", min_value=0, step=1, help="Number of days late compared to the estimated delivery.")

    with col2:
        payment_value = st.number_input("💰 Payment Value", min_value=0.0, step=0.01, help="Total transaction amount.")
        payment_type = st.selectbox("💳 Payment Method", [
            'credit_card', 'boleto', 'voucher', 'debit_card',
            'credit_card,voucher', 'voucher,credit_card'
        ], help="Method used for payment.")
        order_status = st.selectbox("📦 Order Status", ['delivered', 'canceled'], help="Final status of the order.")
        customer_state = st.selectbox("🗺️ Customer Region", [
            "Tenggara (Sudeste)", "Selatan (Sul)", "Timur Laut (Nordeste)",
            "Tengah-Barat (Centro-Oeste)", "Utara (Norte)"
        ], help="Region of the customer.")

    product_category = st.selectbox("🛍️ Product Category", categories, help="Product category purchased by the customer.")

    submitted = st.form_submit_button("🔍 Predict")


if submitted:
    if not models:
        st.error("❌ No models loaded. Cannot proceed with prediction.")
    else:
        df_input = pd.DataFrame([{
            'processing_time_days': processing_time,
            'delivery_time_days': delivery_time,
            'delivery_delay_days': delivery_delay,
            'review_time_days': review_time,
            'payment_value': payment_value,
            'new_customer_state': customer_state,
            'product_category_name_english': product_category,
            'order_status': order_status,
            'payment_type': payment_type
        }])

        # Voting from all models
        votes = [model.predict(df_input)[0] for model in models]
        final_prediction = Counter(votes).most_common(1)[0][0]

        st.markdown("---")
        st.markdown("### 📊 Prediction Result:")

        if final_prediction == 1:
            st.success("✅ The customer is predicted to be **SATISFIED** with the service.")
            st.markdown("<div style='text-align:center; font-size:40px;'>😄</div>", unsafe_allow_html=True)
        else:
            st.error("❌ The customer is predicted to be **NOT SATISFIED** with the service.")
            st.markdown("<div style='text-align:center; font-size:40px;'>😟</div>", unsafe_allow_html=True)

