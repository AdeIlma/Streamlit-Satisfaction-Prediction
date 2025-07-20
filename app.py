import streamlit as st
import pandas as pd
import joblib
from collections import Counter
import os

# ========================
# Page Configuration
# ========================
st.set_page_config(page_title="Customer Satisfaction Prediction", page_icon="✅")

# ========================
# Display SVG Logo
# ========================
if os.path.exists("olist.svg"):
    with open("olist.svg", "r") as f:
        svg_logo = f.read()

    svg_logo = svg_logo.replace(
        '<svg',
        '<svg style="width: 350px; display: block; margin: auto; margin-bottom: 10px;"'
    )
    st.markdown(svg_logo, unsafe_allow_html=True)
else:
    st.warning("⚠️ File 'olist.svg' not found.")

# ========================
# Load Category Function
# ========================
def load_categories(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        st.error(f"File '{file_path}' not found.")
        return []

# ========================
# App Title & Description
# ========================
st.title("🔍 E-commerce Customer Satisfaction Prediction (Olist)")
st.markdown("Enter transaction details to predict whether the customer is satisfied or not based on a machine learning model.")

# ========================
# Load Models
# ========================
model_files = [
    'XGBoost_ADASYN_fold1.pkl',
    'XGBoost_ADASYN_fold2.pkl',
    'XGBoost_ADASYN_fold3.pkl',
    'XGBoost_ADASYN_fold4.pkl',
    'XGBoost_ADASYN_fold5.pkl'
]

models = []
for file in model_files:
    try:
        models.append(joblib.load(file))
    except Exception as e:
        st.error(f"Failed to load model '{file}': {e}")

# ========================
# Load Product Categories
# ========================
categories = load_categories('kategori.txt')

# ========================
# Input Form
# ========================
with st.form("prediction_form"):
    review_time = st.number_input("Review Time (days)", value=0, step=1, value=0, step=1, help="Days between delivery and customer review.")
    processing_time = st.number_input("Processing Time (days)", value=0, step=1, help="Days between order and shipment by the seller.")
    delivery_time = st.number_input("Delivery Time (days)", value=0, step=1, help="Days between delivery and review from the customer.")
    payment_type = st.selectbox("Payment Method", [
        'credit_card', 'boleto', 'voucher', 'debit_card',
        'credit_card,voucher', 'voucher,credit_card'], 
        help="The payment method used by the customer. May include combinations of methods."
    )
    customer_state = st.selectbox("Customer Region", [
        "Tenggara (Sudeste)", "Selatan (Sul)", "Timur Laut (Nordeste)", 
        "Tengah-Barat (Centro-Oeste)", "Utara (Norte)"],
        help="The geographic region where the customer is located."
    )
    delivery_delay = st.number_input("Delivery Delay (days)", value=0, step=1, help="Days of delay beyond estimated delivery date.")
    product_category = st.selectbox("Product Category", categories, help="The category of the product purchased by the customer.")
    payment_value = st.number_input("Payment Value", value=00.00, step=00.01, value=0, step=1, help="Total payment value of the order.")
    order_status = st.selectbox("Order Status", ['delivered', 'canceled'], help="The final status of the order: whether it was delivered or canceled.")


    submitted = st.form_submit_button("🔍 Predict")

# ========================
# Prediction Processing
# ========================
if submitted:
    if not models:
        st.error("No models loaded. Cannot proceed with prediction.")
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

        try:
            predictions = [model.predict(df_input)[0] for model in models]
            majority_vote = Counter(predictions).most_common(1)[0][0]

            if majority_vote == 1:
                st.success("✅ Prediction: **Satisfied**")
            else:
                st.error("❌ Prediction: **Not Satisfied**")

            st.markdown(f"📊 **Model Voting Results:** `{dict(Counter(predictions))}`")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
