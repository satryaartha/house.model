from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # izinkan request dari Netlify frontend

# Load model saat server start
MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
model = joblib.load(MODEL_PATH)
print("✅ Model berhasil dimuat!")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "RumahPrediktor API aktif",
        "endpoint": "POST /predict"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validasi input
        required = ["LB", "LT", "KT", "KM", "GRS", "LOKASI"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Field '{field}' wajib diisi"}), 400

        lb  = float(data["LB"])
        lt  = float(data["LT"])
        kt  = float(data["KT"])
        km  = float(data["KM"])
        grs = float(data["GRS"])
        lokasi = str(data["LOKASI"])

        # Validasi range (sesuai dataset)
        if not (40 <= lb <= 1200):
            return jsonify({"error": "LB harus antara 40–1200 m²"}), 400
        if not (25 <= lt <= 1400):
            return jsonify({"error": "LT harus antara 25–1400 m²"}), 400
        if lt == 0:
            return jsonify({"error": "LT tidak boleh 0"}), 400

        # Feature engineering (sama seperti di notebook)
        rasio_bangunan = lb / lt
        total_ruangan  = kt + km

        # Buat DataFrame sesuai format training
        input_df = pd.DataFrame([{
            "LB":              lb,
            "LT":              lt,
            "KT":              kt,
            "KM":              km,
            "GRS":             grs,
            "LOKASI":          lokasi,
            "RASIO_BANGUNAN":  rasio_bangunan,
            "TOTAL_RUANGAN":   total_ruangan
        }])

        # Prediksi
        harga = model.predict(input_df)[0]
        harga = max(harga, 0)

        # Hitung kisaran estimasi ±12%
        margin = harga * 0.12
        harga_min = harga - margin
        harga_max = harga + margin

        return jsonify({
            "success":        True,
            "harga":          round(harga),
            "harga_min":      round(harga_min),
            "harga_max":      round(harga_max),
            "harga_formatted": f"Rp {harga:,.0f}",
            "input": {
                "LB":             lb,
                "LT":             lt,
                "KT":             kt,
                "KM":             km,
                "GRS":            grs,
                "LOKASI":         lokasi,
                "RASIO_BANGUNAN": round(rasio_bangunan, 3),
                "TOTAL_RUANGAN":  int(total_ruangan)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
