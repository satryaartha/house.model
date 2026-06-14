# 🏠 House Price Prediction — Tebet, Jakarta Selatan

Aplikasi prediksi harga rumah di area **Tebet, Jakarta Selatan** menggunakan 4 model Machine Learning klasik yang digabungkan dengan metode **Ensemble Simple Average**.

🔗 **Live App:** [housemodel-jjx4dpjzbnuq9wqntthcev.streamlit.app](https://housemodel-jjx4dpjzbnuq9wqntthcev.streamlit.app)

---

## 📸 Preview

| Tab | Deskripsi |
|-----|-----------|
| 🏠 Prediction | Input spesifikasi rumah, jalankan 4 model, lihat ensemble result |
| 📊 Data Overview | Statistik dan distribusi dataset |
| 🤖 Model Details | Perbandingan performa 4 model |
| ⚙️ ML Pipeline | Arsitektur end-to-end pipeline |

---

## 🤖 Model yang Digunakan

| Ranking | Model | R² Score | MAE |
|---------|-------|----------|-----|
| 🥇 | Random Forest (n=300, max_depth=20) | ~0.78 | ~2.1 M |
| 🥈 | Ridge Regression (α=10) | ~0.77 | ~3.2 M |
| 🥉 | Lasso Regression (α=1000) | ~0.77 | ~3.2 M |
| 4️⃣ | Linear Regression | ~0.77 | ~3.2 M |

Hasil prediksi akhir menggunakan **Ensemble Simple Average** — rata-rata dari semua model yang berhasil dimuat.

---

## 📁 Struktur Repo

```
house.model/
├── streamlit_app.py        # Main Streamlit application
├── requirements.txt        # Python dependencies
├── random_forest.pkl       # Trained Random Forest model
├── ridge.pkl               # Trained Ridge model
├── lasso.pkl               # Trained Lasso model
├── linear_regression.pkl   # Trained Linear Regression model
└── README.md               # This file
```

---

## 📊 Dataset

- **Sumber:** DATA_RUMAH.csv
- **Jumlah data:** 1.010 listing rumah
- **Area:** Tebet, Jakarta Selatan
- **Fitur yang digunakan:**

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| LB | Numerik | Luas Bangunan (m²) |
| LT | Numerik | Luas Tanah (m²) |
| KT | Numerik | Jumlah Kamar Tidur |
| KM | Numerik | Jumlah Kamar Mandi |
| GRS | Numerik | Jumlah Garasi |
| LOKASI | Kategorikal | Area dalam Tebet |
| RASIO_BANGUNAN | Numerik (derived) | LB / LT |
| TOTAL_RUANGAN | Numerik (derived) | KT + KM |

---

## ⚙️ ML Pipeline

```
Raw Data (DATA_RUMAH.csv)
        ↓
Feature Engineering
  - Ekstrak LOKASI dari nama listing
  - Buat RASIO_BANGUNAN = LB / LT
  - Buat TOTAL_RUANGAN  = KT + KM
        ↓
Preprocessing (ColumnTransformer)
  - Numerik  : SimpleImputer(median) + StandardScaler
  - Kategorikal: SimpleImputer(most_frequent) + OneHotEncoder
        ↓
Train/Test Split (80:20, random_state=42)
        ↓
Training 4 Model (scikit-learn Pipeline)
        ↓
Evaluasi (R² Score + MAE)
        ↓
Simpan ke .pkl (joblib)
        ↓
Deploy ke Streamlit Cloud
```

---

## 🚀 Cara Menjalankan Lokal

### 1. Clone repo

```bash
git clone https://github.com/satryaartha/house.model.git
cd house.model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan app

```bash
streamlit run streamlit_app.py
```

App akan terbuka di `http://localhost:8501`

---

## 🔧 Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Language | Python 3.x |
| ML Library | scikit-learn 1.9.0 |
| Data Processing | pandas, numpy |
| App Framework | Streamlit |
| Model Serialization | joblib |
| Deployment | Streamlit Cloud |
| Version Control | Git + GitHub |

---

## 📈 Cara Menggunakan Aplikasi

1. **Buka app** di link di atas
2. **Atur input** di sidebar kiri:
   - Geser slider Luas Bangunan, Luas Tanah, Kamar Tidur, Kamar Mandi, Garasi
   - Pilih Lokasi dari dropdown
3. **Klik ⚡ Run All Models**
4. **Lihat hasil:**
   - Kotak besar kuning = harga ensemble (rata-rata 4 model)
   - 4 kartu di bawahnya = kontribusi tiap model
   - Tabel = perbandingan detail semua model

---

## 👨‍💻 Author

**Satryaartha**
