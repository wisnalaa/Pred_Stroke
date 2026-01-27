import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Stroke - Skripsi", layout="centered")

# --- 1. Load Model & Assets ---
@st.cache_resource
def load_assets():
    try:
        # Load model LightGBM
        model = lgb.Booster(model_file='model_stroke_lgbm.txt')
        # Load encoders
        encoders = joblib.load('label_encoders.pkl')
        return model, encoders
    except Exception as e:
        st.error(f"Gagal memuat model/encoder. Pastikan file ada di folder yang sama. Error: {e}")
        return None, None

model, encoders = load_assets()

# --- 2. Judul & Deskripsi ---
st.title("🏥 Aplikasi Prediksi Risiko Stroke")
st.write("Masukkan data pasien di bawah ini untuk melihat prediksi risiko stroke.")

# --- 3. Form Input Data ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin (Gender)", ['Male', 'Female'])
        age = st.number_input("Umur (Age)", min_value=0.0, max_value=100.0, value=45.0)
        hypertension = st.selectbox("Hipertensi (Hypertension)", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        heart_disease = st.selectbox("Penyakit Jantung (Heart Disease)", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        ever_married = st.selectbox("Pernah Menikah? (Ever Married)", ['Yes', 'No'])

    with col2:
        work_type = st.selectbox("Tipe Pekerjaan (Work Type)", 
                                 ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
        residence_type = st.selectbox("Tipe Tempat Tinggal (Residence)", ['Urban', 'Rural'])
        avg_glucose = st.number_input("Rata-rata Glukosa", min_value=0.0, value=80.0)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, value=25.0)
        smoking_status = st.selectbox("Status Merokok", 
                                      ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    submit_btn = st.form_submit_button("🔍 Prediksi Sekarang")

# --- 4. Logika Prediksi ---
if submit_btn and model is not None:
    # Membuat DataFrame dari input
    data_input = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Preprocessing: Encoding Kategori
    # Kita menggunakan try-except untuk menangani jika ada kategori input yang tidak dikenali
    try:
        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in cat_cols:
            le = encoders[col]
            # Menangani input yang mungkin tidak ada saat training (safety measure)
            # Menggunakan mapping manual jika transform error, atau default ke mode
            if data_input[col][0] in le.classes_:
                data_input[col] = le.transform(data_input[col])
            else:
                st.warning(f"Nilai '{data_input[col][0]}' pada {col} tidak dikenal model. Menggunakan default.")
                data_input[col] = 0 # Default fallback

        # Pastikan urutan kolom sesuai dengan saat training
        # Berdasarkan notebook Anda:
        feature_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                         'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        data_input = data_input[feature_order]

        # Prediksi
        pred_prob = model.predict(data_input)[0]
        
        # Threshold dari notebook Anda adalah 0.4
        FINAL_THRESHOLD = 0.4
        prediction = 1 if pred_prob > FINAL_THRESHOLD else 0

        # Tampilkan Hasil
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **Hasil Prediksi: BERISIKO STROKE**")
            st.write(f"Probabilitas: {pred_prob:.2%}")
            st.info("Saran: Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.")
        else:
            st.success(f"✅ **Hasil Prediksi: TIDAK BERISIKO / RENDAH**")
            st.write(f"Probabilitas: {pred_prob:.2%}")
            st.info("Saran: Tetap jaga pola hidup sehat.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")