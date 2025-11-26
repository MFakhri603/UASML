import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------
# LOAD MODEL .SAV
# ---------------------------------------
model = joblib.load("model_tpt.sav")

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
df = pd.read_csv("bps-od_17044_tingkat_pengangguran_terbuka__kabupatenkota_data.csv")

# Pastikan kolom sesuai dataset kamu
# kolom: tahun, nama_kabupaten_kota, tingkat_pengangguran_terbuka

st.title("📊 Prediksi Tingkat Pengangguran Terbuka per Kabupaten/Kota")

# ---------------------------------------
# DROPDOWN PILIH KABUPATEN
# ---------------------------------------
kabupaten_list = sorted(df["nama_kabupaten_kota"].unique())
kabupaten = st.selectbox("Pilih Kabupaten/Kota:", kabupaten_list)

df_kab = df[df["nama_kabupaten_kota"] == kabupaten]

st.write("### Data Historis:")
st.dataframe(df_kab)

# ---------------------------------------
# INPUT TAHUN PREDIKSI
# ---------------------------------------
tahun_max_dataset = int(df_kab["tahun"].max())

tahun_prediksi = st.number_input(
    "Prediksi sampai tahun berapa?",
    min_value=tahun_max_dataset + 1,
    max_value=2100,
    value=tahun_max_dataset + 5
)

# ---------------------------------------
# MELAKUKAN PREDIKSI
# ---------------------------------------
tahun_range = np.arange(tahun_max_dataset + 1, tahun_prediksi + 1).reshape(-1, 1)
hasil_prediksi = model.predict(tahun_range)

df_prediksi = pd.DataFrame({
    "tahun": tahun_range.flatten(),
    "prediksi_tpt": hasil_prediksi
})

st.write("### Hasil Prediksi:")
st.dataframe(df_prediksi)

# ---------------------------------------
# PLOTTING
# ---------------------------------------
st.write("### Grafik Prediksi TPT")

plt.figure(figsize=(10, 6))

# Data historis
plt.plot(df_kab["tahun"], df_kab["tingkat_pengangguran_terbuka"],
         marker='o', label="Data Historis")

# Data prediksi
plt.plot(df_prediksi["tahun"], df_prediksi["prediksi_tpt"],
         marker='x', linestyle='--', label="Prediksi")

plt.xlabel("Tahun")
plt.ylabel("Tingkat Pengangguran Terbuka (%)")
plt.title(f"Prediksi TPT Kabupaten/Kota: {kabupaten}")
plt.grid(True)
plt.legend()

st.pyplot(plt)
