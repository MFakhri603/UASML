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
    # Tampilkan Data
    # ================================
    st.subheader(f"Data TPT Kabupaten {kab}")
    st.dataframe(df_kab)

    # ================================
    # Prediksi
    # ================================
    if st.button("Prediksi"):
        hasil = model.predict(np.array([[tahun_pred]]))[0]

        # Cari tahun sebelumnya
        tahun_sebelumnya = tahun_pred - 1

        # Jika datanya ada di dataset, gunakan
        if tahun_sebelumnya in df_kab["tahun"].values:
            tpt_lalu = df_kab[df_kab["tahun"] == tahun_sebelumnya]["tpt"].values[0]
        else:
            # Jika tidak ada datanya, prediksi juga
            tpt_lalu = model.predict(np.array([[tahun_sebelumnya]]))[0]

        # Tentukan status naik/turun
        if hasil > tpt_lalu:
            status = "📈 **Naik**"
            warna = "red"
        elif hasil < tpt_lalu:
            status = "📉 **Turun**"
            warna = "green"
        else:
            status = "➡️ **Tetap**"
            warna = "gray"

        # Tampilkan hasil
        st.success(f"Prediksi TPT {kab} Tahun {tahun_pred}: **{hasil:.2f}%**")
        st.write(f"TPT Tahun {tahun_sebelumnya}: **{tpt_lalu:.2f}%**")

        st.markdown(f"### Status Perubahan: <span style='color:{warna}; font-size:24px;'>{status}</span>", unsafe_allow_html=True)

    # ================================
    # Grafik Regresi
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
