import streamlit as st
import pandas as pd
import joblib
import altair as alt

# ========================
# Load Trained Model
# ========================
try:
    model_dict = joblib.load('best_model.pkl')

    # Coba ambil threshold jika ada, jika tidak, gunakan default
    model = model_dict['model'] if isinstance(model_dict, dict) and 'model' in model_dict else model_dict
    default_threshold = model_dict.get('threshold', 0.5) if isinstance(model_dict, dict) else 0.5

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

threshold = default_threshold  # Bisa diubah jika ingin threshold dinamis

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
    st.warning("⚠️ File 'olist.svg' tidak ditemukan. Pastikan file berada di direktori yang sama.")

# ========================
# Title and Description
# ========================
st.markdown(
    "<h1 style='text-align: center;'>🔍 E-commerce Customer Satisfaction Prediction (Olist)</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Masukkan data transaksi atau upload file CSV untuk memprediksi kepuasan pelanggan berdasarkan model machine learning.</p>",
    unsafe_allow_html=True
)

# ========================
# Section: Single Prediction
# ========================
st.markdown("## 🧾 Manual Transaction Input")

col1, col2 = st.columns(2)
with col1:
    review_time_days = st.number_input("📝 Time Gap to Review Days", value=0, step=1)
    processing_time_days = st.number_input("🛠️ Processing Time Days", min_value=1, step=1)
    quantity = st.number_input("💰 Quantity", min_value=1, step=1)
with col2:
    payment_installments = st.number_input("💳 Number of Installments", min_value=1, step=1)
    review_response_time_days = st.number_input("💬 Seller Response Time Gap Days", value=0, step=1)
    delivery_time_days = st.number_input("🚚 Delivery Time Days", min_value=1, step=1)

# ========================
# Predict Single Input
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

    try:
        probs = model.predict_proba(input_df)[0]
        prediction = 1 if probs[1] >= threshold else 0

        st.markdown("### 🎯 Class Probabilities")
        st.markdown(f"- ❌ Not Satisfied (Class 0): `{probs[0]:.2f}`")
        st.markdown(f"- ✅ Satisfied (Class 1): `{probs[1]:.2f}`")

        chart_df = pd.DataFrame({
            'Satisfaction': ['Not Satisfied', 'Satisfied'],
            'Probability': probs
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x='Satisfaction',
            y='Probability',
            color='Satisfaction'
        ).properties(width=400, height=300)

        st.altair_chart(chart, use_container_width=True)

        if prediction == 1:
            st.success("✅ Prediction: **Satisfied**")
            st.markdown("> This customer is likely to leave a **positive review**.")
        else:
            st.error("❌ Prediction: **Not Satisfied**")
            st.markdown("> This customer may be **dissatisfied**. Review time or delivery time might need improvement.")

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")

# ========================
# Section: Bulk Prediction via CSV
# ========================
st.markdown("---")
st.markdown("## 📁 Upload CSV File for Bulk Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df_bulk = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df_bulk)} rows from uploaded file.")
        st.dataframe(df_bulk.head())

        expected_columns = [
            'processing_time_days',
            'review_time_days',
            'quantity',
            'review_response_time_days',
            'payment_installments',
            'delivery_time_days'
        ]

        if not all(col in df_bulk.columns for col in expected_columns):
            st.error(f"❌ The CSV must contain these columns:\n{expected_columns}")
        else:
            if st.button("📊 Predict CSV Data"):
                probs_bulk = model.predict_proba(df_bulk)
                df_bulk['prob_not_satisfied'] = probs_bulk[:, 0]
                df_bulk['prob_satisfied'] = probs_bulk[:, 1]
                df_bulk['prediction'] = (df_bulk['prob_satisfied'] >= threshold).astype(int)
                df_bulk['prediction_label'] = df_bulk['prediction'].map({0: 'Not Satisfied', 1: 'Satisfied'})

                st.success("✅ Bulk prediction completed!")
                st.dataframe(df_bulk.head())

                csv = df_bulk.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv,
                    file_name="bulk_prediction_results.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"❌ Error processing CSV: {e}")
