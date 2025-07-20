import streamlit as st
import pandas as pd
import joblib
from collections import Counter

# =======================
# Load Model dan Kategori
# =======================

# Fungsi membaca kategori produk
def load_kategori(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# Load semua model fold
model_files = ['LightGBM_ADASYN_RSCV_fold1.pkl', 'LightGBM_ADASYN_RSCV_fold2.pkl', 'LightGBM_ADASYN_RSCV_fold3.pkl', 'LightGBM_ADASYN_RSCV_fold4.pkl', 'LightGBM_ADASYN_RSCV_fold5.pkl']
models = [joblib.load(open(filename, 'rb')) for filename in model_files]

# Load kategori dari file .txt
opsi = load_kategori('kategori.txt')

# =======================
# Desain UI Streamlit
# =======================

# Judul
st.markdown("<h1 style='text-align: center;'>🔍 Prediksi Kepuasan Pelanggan E-commerce</h1>", unsafe_allow_html=True)

# Deskripsi singkat
st.markdown("Masukkan informasi transaksi dan kategori untuk memprediksi apakah pelanggan puas atau tidak berdasarkan model ML.")

# Input numerik
st.markdown("## 💡 Data Transaksi")
col1, col2 = st.columns(2)
with col1:
    total_freight = st.number_input("💰 Biaya Ongkir", min_value=0.0, step=0.01)
    processing_time_days = st.number_input("🛠️ Waktu Proses (processing_time_days)")
    review_time_days = st.number_input("📝 Jarak Waktu Review (review_time_days)")
    review_response_time_days = st.number_input("📝 Jarak Waktu Review (review_response_time_days)")

with col2:
    delivery_time_days = st.number_input("🚚 Waktu Pengiriman (delivery_time_days)")
    delivery_delay_days = st.number_input("⏰ Keterlambatan Pengiriman (delivery_delay_days)")
    estimated_delivery_time_days = st.number_input("🚚 Estimasi Waktu Pengiriman (estimated_delivery_time_days)")
    max_processing_time_days = st.number_input("🛠️ Waktu Proses Terlama (max_processing_time_days)")

# Input kategori
st.markdown("## 🗂️ Informasi Kategori")
col3, col4 = st.columns(2)
with col3:
    new_customer_state = st.selectbox("🌎 Lokasi Pelanggan (customer_state)", [
        "Tenggara (Sudeste)", "Selatan (Sul)", "Timur Laut (Nordeste)", 
        "Tengah-Barat (Centro-Oeste)", "Utara (Norte)"
    ])

    # Kategori produk full width
    product_category_name_english = st.selectbox("🏷️ Kategori Produk (product_category_name_english)", opsi)

# Tombol prediksi
st.markdown("---")
if st.button("🔍 Prediksi"):
    # Buat DataFrame dari input
    input_df = pd.DataFrame([{
        'processing_time_days': processing_time_days,
        'delivery_time_days': delivery_time_days,
        'delivery_delay_days': delivery_delay_days,
        'review_time_days': review_time_days,
        'product_category_name_english': product_category_name_english,
        'new_customer_state': new_customer_state,
        'estimated_delivery_time_days': estimated_delivery_time_days,
        'total_freight': total_freight,
        'max_processing_time_days': max_processing_time_days,
        'review_response_time_days': review_response_time_days
    }])


    # Voting dari semua model
    hasil = [model.predict(input_df)[0] for model in models]
    hasil_terbanyak = Counter(hasil).most_common(1)[0][0]

    # Output hasil
    if hasil_terbanyak == 1:
        st.success("✅ Prediksi: **Satisfied**")
    else:
        st.error("❌ Prediksi: **Not Satisfied**")
