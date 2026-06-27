"""
train_model.py
==============
Training script untuk House Price Prediction — Tebet, Jakarta Selatan.

Melatih 4 model Classical ML (Linear Regression, Ridge, Lasso, Random Forest)
menggunakan scikit-learn, lalu menyimpan tiap pipeline ke file .pkl.

Cara menjalankan:
    python train_model.py

Output:
    random_forest.pkl
    ridge.pkl
    lasso.pkl
    linear_regression.pkl
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone


# ════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════
# Dataset di-encode latin-1 (mengandung karakter non-UTF8)
df = pd.read_csv("DATA_RUMAH.csv", encoding="latin-1")
df.columns = df.columns.str.strip()
print(f"Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"Kolom: {df.columns.tolist()}")


# ════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ════════════════════════════════════════
def extract_lokasi(nama):
    """Ekstrak area Tebet dari kolom nama rumah."""
    nama = str(nama).upper()
    areas = ["TEBET TIMUR", "TEBET BARAT", "TEBET UTARA",
             "KEBON BARU", "MENTENG DALAM", "TEBET"]
    for area in areas:
        if area in nama:
            return area.title()
    return "Jakarta Selatan"

# Kolom pertama berisi nama rumah → ekstrak LOKASI
df["LOKASI"] = df.iloc[:, 0].apply(extract_lokasi)

# Fitur turunan (derived features)
df["RASIO_BANGUNAN"] = df["LB"] / df["LT"].replace(0, np.nan)  # LB / LT
df["TOTAL_RUANGAN"]  = df["KT"] + df["KM"]                      # KT + KM

print("\nFeature engineering selesai:")
print("  - LOKASI         (ekstrak dari nama)")
print("  - RASIO_BANGUNAN = LB / LT")
print("  - TOTAL_RUANGAN  = KT + KM")


# ════════════════════════════════════════
# 3. SIAPKAN X DAN Y
# ════════════════════════════════════════
FEATURES = ["LB", "LT", "KT", "KM", "GRS",
            "LOKASI", "RASIO_BANGUNAN", "TOTAL_RUANGAN"]
TARGET = "HARGA"

X = df[FEATURES]
y = df[TARGET]


# ════════════════════════════════════════
# 4. TRAIN / TEST SPLIT (80:20)
# ════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")


# ════════════════════════════════════════
# 5. PREPROCESSING PIPELINE
# ════════════════════════════════════════
num_cols = ["LB", "LT", "KT", "KM", "GRS", "RASIO_BANGUNAN", "TOTAL_RUANGAN"]
cat_cols = ["LOKASI"]

preprocessor = ColumnTransformer([
    # Numerik  : imputasi median + standardisasi
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), num_cols),
    # Kategorikal : imputasi modus + one-hot encoding
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])


# ════════════════════════════════════════
# 6. DEFINISI 4 MODEL (CLASSICAL ML)
# ════════════════════════════════════════
MODELS = {
    "random_forest":     RandomForestRegressor(
                            n_estimators=300, max_depth=20,
                            random_state=42, n_jobs=-1),
    "ridge":             Ridge(alpha=10),
    "lasso":             Lasso(alpha=1000, max_iter=10000),
    "linear_regression": LinearRegression(),
}


# ════════════════════════════════════════
# 7. TRAINING + EVALUASI + SAVE
# ════════════════════════════════════════
print("\n" + "=" * 55)
print(f"{'Model':<22}{'R² Score':<12}{'MAE (Rp)':<18}")
print("=" * 55)

for name, estimator in MODELS.items():
    # Bungkus preprocessor + model dalam satu Pipeline
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", estimator),
    ])

    # Latih
    pipeline.fit(X_train, y_train)

    # Evaluasi pada test set
    y_pred = pipeline.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Simpan pipeline ke .pkl
    joblib.dump(pipeline, f"{name}.pkl")

    print(f"{name:<22}{r2:<12.4f}Rp {mae:>15,.0f}")

print("=" * 55)
print("\n✅ Semua model berhasil dilatih dan disimpan ke .pkl")


# ════════════════════════════════════════
# 8. VERIFIKASI — CONTOH PREDIKSI
# ════════════════════════════════════════
print("\nContoh prediksi (LB=150, LT=200, KT=3, KM=2, GRS=1, Tebet):")
sample = pd.DataFrame([{
    "LB": 150, "LT": 200, "KT": 3, "KM": 2, "GRS": 1,
    "LOKASI": "Tebet", "RASIO_BANGUNAN": 150/200, "TOTAL_RUANGAN": 5,
}])

for name in MODELS:
    model = joblib.load(f"{name}.pkl")
    pred = model.predict(sample)[0]
    print(f"  {name:<22}: Rp {pred:,.0f}")
