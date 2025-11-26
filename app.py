import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ================================
# TITLE
# ================================
st.title("Prediksi Tingkat Pengangguran Terbuka per Kabupaten")
st.write("Model: Linear Regression")

# ================================
# LOAD DATASET
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("bps-od_17044_tingkat_pengangguran_terbuka__kabupatenkota_data.csv")
    df = df.rename(columns={
        "tahun": "tahun",
        "tingkat_pengangguran_terbuka": "tpt",
        "nama_kabupaten_kota": "kabupaten"
    })
    return df

df = load_data()

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    return joblib.load("model_tpt.sav")

model = load_model()

# ================================
# INPUT USER
# ================================
kabupaten_list = sorted(df["kabupaten"].unique())
kab = st.selectbox("Pilih Kabupaten:", kabupaten_list)

tahun_pred = st.number_input("Masukkan Tahun Prediksi:", min_value=2025, max_value=2100, step=1)

# Filter dataset berdasarkan kabupaten
df_kab = df[df["kabupaten"] == kab]

if df_kab.empty:
    st.warning("Data kabupaten tidak ditemukan.")
else:

    # ================================
    # TAMPILKAN DATASET RINGKAS
    # ================================
    st.subheader(f"Data TPT Kabupaten {kab}")
    st.dataframe(df_kab)

    # ================================
    # PREDIKSI
    # ================================
    if st.button("Prediksi"):
        hasil = model.predict(np.array([[tahun_pred]]))[0]
        st.success(f"Prediksi TPT {kab} Tahun {tahun_pred}: **{hasil:.2f}%**")

    # ================================
    # PLOT REGRESI LINEAR
    # ================================
    st.subheader("Grafik Regresi Linear")

    plt.figure(figsize=(8,5))
    plt.scatter(df_kab["tahun"], df_kab["tpt"], label="Data Asli")
    plt.plot(df_kab["tahun"], model.predict(df_kab[["tahun"]]), color="red", label="Regresi Linear")
    plt.xlabel("Tahun")
    plt.ylabel("TPT")
    plt.title(f"Regresi TPT Kabupaten {kab}")
    plt.grid(True)
    plt.legend()

    st.pyplot(plt)
