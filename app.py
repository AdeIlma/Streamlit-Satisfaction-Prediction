import streamlit as st
import pandas as pd
import joblib
from collections import Counter
import os

# ===================
# Konfigurasi Halaman
# ===================
st.set_page_config(page_title="Prediksi Kepuasan Pelanggan", page_icon="✅")

# ===================
# Tampilkan Logo SVG
# ===================
if os.path.exists("olist.svg"):
    with open("olist.svg", "r") as f:
        svg_logo = f.read()

    svg_logo = svg_logo.replace(
        '<svg',
        '<svg style="width: 350px; display: block; margin: auto; margin-bottom: 10px;"'
    )
    st.markdown(svg_logo, unsafe_allow_html=True)
else:
    st.warning("⚠️ File 'olist.svg' tidak ditemukan.")

# =====================
# Fungsi Load Kategori
# =====================
def load_kategori(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        st.error(f"File '{file_path}' tidak ditemukan.")
        return []

# =========================
# Judul dan Deskripsi App
# =========================
st.title("📦 Prediksi Kepuasan Pelanggan")
st.markdown("Masukkan informasi berikut untuk memprediksi apakah pelanggan akan *Satisfied* atau *Not Satisfied* berdasarkan model voting dari 5 model XGBoost.")

# ================
# Load Model Model
# ================
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
        st.error(f"Gagal memuat model '{file}': {e}")

# =====================
# Load Kategori Produk
# =====================
opsi = load_kategori('kategori.txt')

# ================
# Form Input Data
# ================
with st.form("prediction_form"):
    processing_time = st.number_input("Processing Time (days)", value=0, step=1)
    delivery_time = st.number_input("Delivery Time (days)", value=0, step=1)
    delivery_delay = st.number_input("Delivery Delay (days)", value=0, step=1)
    review_time = st.number_input("Review Time (days)", value=0, step=1)
    payment_value = st.number_input("Payment Value", value=0, step=1)

    customer_state = st.selectbox("Customer State", [
        "Tenggara (Sudeste)", "Selatan (Sul)", "Timur Laut (Nordeste)",
        "Tengah-Barat (Centro-Oeste)", "Utara (Norte)"
    ])
    product_category = st.selectbox("Product Category", opsi)
    order_status = st.selectbox("Order Status", ['delivered', 'canceled'])
    payment_type = st.selectbox("Payment Type", [
        'credit_card', 'boleto', 'voucher', 'debit_card',
        'credit_card,voucher', 'voucher,credit_card'
    ])

    submitted = st.form_submit_button("🔍 Prediksi")

# =================
# Proses Prediksi
# =================
if submitted:
    if not models:
        st.error("Model belum dimuat. Tidak dapat melakukan prediksi.")
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
            hasil_terbanyak = Counter(predictions).most_common(1)[0][0]

            if hasil_terbanyak == 1:
                st.success("✅ Prediksi: **Satisfied**")
            else:
                st.error("❌ Prediksi: **Not Satisfied**")

            st.markdown(f"📊 **Voting dari semua model:** `{dict(Counter(predictions))}`")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
