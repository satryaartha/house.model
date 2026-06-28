# 🏠 House Price Prediction — Tebet, Jakarta Selatan

Aplikasi prediksi harga rumah di area **Tebet, Jakarta Selatan** menggunakan 4 model Machine Learning klasik yang digabungkan dengan metode **Ensemble Simple Average**.

🔗 **Live App:** [Klik here](https://houseprediction-jjx4dpjzbnuq9wqntthcev.streamlit.app/)

---

## 📸 Tampilan Aplikasi

Aplikasi berjalan sepenuhnya di halaman utama tanpa sidebar — semua input spesifikasi rumah tersedia langsung di tengah halaman, diikuti tombol **⚡ Run All Models** dan hasil prediksi di bawahnya.

| Tab | Deskripsi |
|-----|-----------|
| 🏠 Prediction | Input spesifikasi rumah & lihat hasil ensemble + perbandingan 4 model |
| 📊 Data Overview | Statistik dan distribusi dataset |
| 🤖 Model Details | Perbandingan performa 4 model |
| ⚙️ ML Pipeline | Arsitektur end-to-end pipeline |

---

## 🤖 Model yang Digunakan

Semua model adalah **Classical ML** (bukan Deep Learning), dilatih menggunakan scikit-learn.

| Ranking | Model | R² Score | MAE |
|---------|-------|----------|-----|
| 🥇 | Random Forest (n=300, max_depth=20) | ~0.78 | ~2.1 M |
| 🥈 | Ridge Regression (α=10) | ~0.77 | ~3.2 M |
| 🥉 | Lasso Regression (α=1000) | ~0.77 | ~3.2 M |
| 4️⃣ | Linear Regression | ~0.77 | ~3.5 M |

Hasil akhir menggunakan **Ensemble Simple Average** — rata-rata prediksi dari semua model yang berhasil dimuat.

---

## 📁 Struktur Repo

```
house.model/
├── streamlit_app.py        # Main application (no sidebar, centered layout)
├── requirements.txt        # Python dependencies
├── random_forest.pkl       # Trained Random Forest pipeline
├── ridge.pkl               # Trained Ridge pipeline
├── lasso.pkl               # Trained Lasso pipeline
├── linear_regression.pkl   # Trained Linear Regression pipeline
└── README.md
```

---

## 📊 Dataset

- **Jumlah data:** 1.010 listing rumah
- **Area:** Tebet, Jakarta Selatan

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| LB | Numerik | Luas Bangunan (m²) |
| LT | Numerik | Luas Tanah (m²) |
| KT | Numerik | Jumlah Kamar Tidur |
| KM | Numerik | Jumlah Kamar Mandi |
| GRS | Numerik | Jumlah Garasi |
| LOKASI | Kategorikal | Area dalam Tebet |
| RASIO_BANGUNAN | Derived | LB / LT |
| TOTAL_RUANGAN | Derived | KT + KM |

---

## ⚙️ ML Pipeline

```
DATA_RUMAH.csv (1.010 rows)
        ↓
Feature Engineering
  RASIO_BANGUNAN = LB / LT
  TOTAL_RUANGAN  = KT + KM
        ↓
Preprocessing (ColumnTransformer)
  Numerik  : SimpleImputer(median) + StandardScaler
  Kategorikal: SimpleImputer(most_frequent) + OneHotEncoder
        ↓
Train/Test Split — 80:20, random_state=42
        ↓
Training 4 Classical ML Models
        ↓
Evaluasi: R² Score + MAE
        ↓
Save .pkl via joblib → Deploy Streamlit Cloud
```

---

## 🚀 Cara Menjalankan Lokal

```bash
# Clone repo
git clone https://github.com/satryaartha/house.model.git
cd house.model

# Install dependencies
pip install -r requirements.txt

# Jalankan app
streamlit run streamlit_app.py
```

App terbuka di `http://localhost:8501`

---

## 💡 Cara Menggunakan Aplikasi

1. Buka link app di atas
2. Isi input di halaman utama:
   - **Luas Bangunan** & **Luas Tanah** — geser slider
   - **Kamar Tidur**, **Kamar Mandi**, **Garasi** — geser slider
   - **Lokasi** — pilih dari dropdown
3. Klik **⚡ Run All Models**
4. Lihat hasil:
   - **Kotak besar** = harga ensemble (rata-rata 4 model)
   - **4 kartu kecil** = kontribusi tiap model vs rata-rata
   - **Tabel** = perbandingan detail R², MAE, dan range estimasi

---

## 🔧 Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Language | Python 3.x |
| ML Library | scikit-learn |
| Data | pandas, numpy |
| App Framework | Streamlit |
| Serialization | joblib |
| Deployment | Streamlit Cloud |
| Version Control | Git + GitHub |
