# RumahPrediktor API 🏠

Backend Flask untuk prediksi harga rumah Tebet, Jakarta Selatan.  
Menggunakan model **Random Forest** yang dilatih dari 1.010 data listing nyata.

---

## Struktur File

```
rumah-predictor-api/
├── app.py                  # Flask application utama
├── house_price_model.pkl   # Model Random Forest (joblib)
├── requirements.txt        # Python dependencies
├── Procfile                # Untuk Railway / Render
├── railway.toml            # Konfigurasi Railway
└── README.md
```

---

## Endpoint API

### `GET /`
Health check — cek apakah server aktif.

**Response:**
```json
{
  "status": "ok",
  "message": "RumahPrediktor API aktif",
  "endpoint": "POST /predict"
}
```

---

### `POST /predict`
Prediksi harga rumah berdasarkan spesifikasi.

**Request Body (JSON):**
```json
{
  "LB": 150,
  "LT": 200,
  "KT": 4,
  "KM": 3,
  "GRS": 2,
  "LOKASI": "Jakarta Selatan"
}
```

| Field  | Tipe   | Keterangan              | Range        |
|--------|--------|-------------------------|--------------|
| LB     | float  | Luas Bangunan (m²)      | 40 – 1200    |
| LT     | float  | Luas Tanah (m²)         | 25 – 1400    |
| KT     | int    | Jumlah Kamar Tidur      | 2 – 10       |
| KM     | int    | Jumlah Kamar Mandi      | 1 – 10       |
| GRS    | int    | Jumlah Garasi           | 0 – 10       |
| LOKASI | string | Area lokasi             | Lihat bawah  |

**Nilai LOKASI yang valid:**
- `Jakarta Selatan`
- `Tebet`
- `Tebet Timur`
- `Tebet Barat`
- `Tebet Utara`
- `Kebon Baru`
- `Menteng Dalam`

**Response berhasil:**
```json
{
  "success": true,
  "harga": 5415632567,
  "harga_min": 4765756659,
  "harga_max": 6065508315,
  "harga_formatted": "Rp 5,415,632,567",
  "input": {
    "LB": 150,
    "LT": 200,
    "KT": 4,
    "KM": 3,
    "GRS": 2,
    "LOKASI": "Jakarta Selatan",
    "RASIO_BANGUNAN": 0.75,
    "TOTAL_RUANGAN": 7
  }
}
```

---

## Deploy ke Railway (Gratis)

### Langkah 1 — Upload ke GitHub
1. Buat repo baru di github.com
2. Upload semua file ini ke repo tersebut
3. **Pastikan `house_price_model.pkl` ikut di-upload**

### Langkah 2 — Deploy ke Railway
1. Buka [railway.app](https://railway.app)
2. Login dengan akun GitHub
3. Klik **"New Project"** → **"Deploy from GitHub repo"**
4. Pilih repo yang baru dibuat
5. Railway otomatis detect Python dan install dependencies
6. Tunggu deploy selesai (~2–3 menit)
7. Klik **"Settings"** → **"Domains"** → **"Generate Domain"**
8. Dapat URL seperti: `https://rumah-predictor-api.up.railway.app`

### Langkah 3 — Hubungkan ke Frontend Netlify
Ganti URL API di file HTML frontend:
```javascript
const API_URL = "https://rumah-predictor-api.up.railway.app";
```

---

## Jalankan Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan server
python app.py

# Server aktif di http://localhost:5000
```

**Test dengan curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LB":150,"LT":200,"KT":4,"KM":3,"GRS":2,"LOKASI":"Jakarta Selatan"}'
```

---

## Info Model

| Properti          | Detail                          |
|-------------------|---------------------------------|
| Algoritma         | Random Forest Regressor         |
| n_estimators      | 300                             |
| max_depth         | 20                              |
| Akurasi R²        | ~0.87                           |
| MAE               | ~Rp 1.8 Miliar                  |
| Cross Validation  | 5-Fold                          |
| Data Training     | 808 listing (80%)               |
| Data Testing      | 202 listing (20%)               |
| Dataset           | 1.010 listing Tebet, Jaksel     |
