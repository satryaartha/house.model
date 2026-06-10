from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ── Load semua 4 model dari folder models/ ──
BASE_DIR = os.path.dirname(__file__)

MODEL_FILES = {
    "random_forest":     os.path.join(BASE_DIR, "models", "random_forest.pkl"),
    "ridge":             os.path.join(BASE_DIR, "models", "ridge.pkl"),
    "lasso":             os.path.join(BASE_DIR, "models", "lasso.pkl"),
    "linear_regression": os.path.join(BASE_DIR, "models", "linear_regression.pkl"),
}

# Metadata model (dari hasil training di notebook)
MODEL_META = {
    "random_forest":     {"name": "Random Forest (n=300, d=20)", "r2": 0.87, "mae": 1_800_000, "rank": 1},
    "ridge":             {"name": "Ridge Regression (α=10)",     "r2": 0.72, "mae": 3_200_000, "rank": 2},
    "linear_regression": {"name": "Linear Regression",           "r2": 0.70, "mae": 3_500_000, "rank": 3},
    "lasso":             {"name": "Lasso Regression (α=1000)",   "r2": 0.68, "mae": 3_800_000, "rank": 4},
}

# Load semua model ke memori saat server start
MODELS = {}
for key, path in MODEL_FILES.items():
    if os.path.exists(path):
        MODELS[key] = joblib.load(path)
        print(f"✅ Loaded: {key}")
    else:
        print(f"⚠️  Not found: {path} — model ini akan dilewati")

if not MODELS:
    raise RuntimeError("❌ Tidak ada model yang berhasil dimuat! Pastikan folder models/ ada.")

print(f"\n🚀 {len(MODELS)} model siap digunakan: {list(MODELS.keys())}\n")


def build_input_df(lb, lt, kt, km, grs, lokasi):
    """Buat DataFrame sesuai format training di notebook."""
    return pd.DataFrame([{
        "LB":             lb,
        "LT":             lt,
        "KT":             kt,
        "KM":             km,
        "GRS":            grs,
        "LOKASI":         lokasi,
        "RASIO_BANGUNAN": lb / lt,
        "TOTAL_RUANGAN":  kt + km,
    }])


def validate_input(data):
    """Validasi dan parse input JSON. Return (values, error_message)."""
    required = ["LB", "LT", "KT", "KM", "GRS", "LOKASI"]
    for field in required:
        if field not in data:
            return None, f"Field '{field}' wajib diisi"

    lb     = float(data["LB"])
    lt     = float(data["LT"])
    kt     = float(data["KT"])
    km     = float(data["KM"])
    grs    = float(data["GRS"])
    lokasi = str(data["LOKASI"])

    if not (40 <= lb <= 1200):
        return None, "LB harus antara 40–1200 m²"
    if not (25 <= lt <= 1400):
        return None, "LT harus antara 25–1400 m²"
    if lt == 0:
        return None, "LT tidak boleh 0"

    return (lb, lt, kt, km, grs, lokasi), None


# ────────────────────────────────────────────
# GET /  → health check
# ────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status":         "ok",
        "message":        "RumahPrediktor API aktif",
        "models_loaded":  list(MODELS.keys()),
        "total_models":   len(MODELS),
        "endpoints": {
            "predict_all": "POST /predict/all  → semua model sekaligus",
            "predict_rf":  "POST /predict      → Random Forest saja",
        }
    })


# ────────────────────────────────────────────
# POST /predict/all → 4 model sekaligus
# ────────────────────────────────────────────
@app.route("/predict/all", methods=["POST"])
def predict_all():
    try:
        data = request.get_json()
        values, error = validate_input(data)
        if error:
            return jsonify({"error": error}), 400

        lb, lt, kt, km, grs, lokasi = values
        input_df = build_input_df(lb, lt, kt, km, grs, lokasi)

        results = []
        for key, pipeline in MODELS.items():
            meta  = MODEL_META[key]
            price = float(pipeline.predict(input_df)[0])
            price = max(price, 400_000_000)
            margin = price * 0.12

            results.append({
                "model":           key,
                "model_name":      meta["name"],
                "harga":           round(price),
                "harga_min":       round(price - margin),
                "harga_max":       round(price + margin),
                "harga_formatted": f"Rp {price:,.0f}",
                "r2_score":        meta["r2"],
                "mae":             meta["mae"],
                "rank":            meta["rank"],
            })

        # Urutkan berdasarkan rank
        results.sort(key=lambda x: x["rank"])

        return jsonify({
            "success":    True,
            "models":     results,
            "best_model": "random_forest",
            "input": {
                "LB":             lb,
                "LT":             lt,
                "KT":             kt,
                "KM":             km,
                "GRS":            grs,
                "LOKASI":         lokasi,
                "RASIO_BANGUNAN": round(lb / lt, 3),
                "TOTAL_RUANGAN":  int(kt + km),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────
# POST /predict → Random Forest saja
# ────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        values, error = validate_input(data)
        if error:
            return jsonify({"error": error}), 400

        lb, lt, kt, km, grs, lokasi = values
        input_df = build_input_df(lb, lt, kt, km, grs, lokasi)

        # Gunakan RF jika ada, fallback ke model pertama yang tersedia
        pipeline = MODELS.get("random_forest") or list(MODELS.values())[0]
        harga    = float(pipeline.predict(input_df)[0])
        harga    = max(harga, 400_000_000)
        margin   = harga * 0.12

        return jsonify({
            "success":         True,
            "harga":           round(harga),
            "harga_min":       round(harga - margin),
            "harga_max":       round(harga + margin),
            "harga_formatted": f"Rp {harga:,.0f}",
            "input": {
                "LB":             lb,
                "LT":             lt,
                "KT":             kt,
                "KM":             km,
                "GRS":            grs,
                "LOKASI":         lokasi,
                "RASIO_BANGUNAN": round(lb / lt, 3),
                "TOTAL_RUANGAN":  int(kt + km),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
