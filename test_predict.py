"""
test_predict.py
===============
Verifikasi: bandingkan prediksi model lokal vs hasil di aplikasi live.

Cara pakai:
    python test_predict.py

Masukkan input yang SAMA di aplikasi live, lalu bandingkan hasilnya.
"""

import pandas as pd
import joblib

# ── INPUT UJI (ganti sesuai keinginan) ──
LB     = 150
LT     = 200
KT     = 3
KM     = 2
GRS    = 1
LOKASI = "Tebet"

# ── Bangun input (sama persis seperti di app) ──
sample = pd.DataFrame([{
    "LB": LB, "LT": LT, "KT": KT, "KM": KM, "GRS": GRS,
    "LOKASI": LOKASI,
    "RASIO_BANGUNAN": LB / LT,
    "TOTAL_RUANGAN": KT + KM,
}])

print("=" * 55)
print(f"INPUT: LB={LB} LT={LT} KT={KT} KM={KM} GRS={GRS} Lokasi={LOKASI}")
print("=" * 55)

# ── Prediksi tiap model ──
models = ["random_forest", "ridge", "lasso", "linear_regression"]
preds = []

for name in models:
    model = joblib.load(f"{name}.pkl")
    pred = model.predict(sample)[0]
    preds.append(pred)
    print(f"  {name:<22}: Rp {pred:,.0f}")

# ── Ensemble (rata-rata) — sama seperti di app ──
ensemble = sum(preds) / len(preds)
print("-" * 55)
print(f"  {'ENSEMBLE (rata-rata)':<22}: Rp {ensemble:,.0f}")
print("=" * 55)
print("\nBandingkan angka 'ENSEMBLE' di atas dengan kotak besar")
print("kuning di aplikasi live (dengan input yang sama).")
print("Jika sama → model lokal = model deployed. ✅")
