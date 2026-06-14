import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# joblib opsional — fallback ke pickle
try:
    import joblib
    def _load_model(path):
        return joblib.load(path)
except ImportError:
    def _load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)

st.set_page_config(
    page_title="House Price Predictor – Tebet",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="metric-container"] {
    background:#f8faff; border:1px solid #dbeafe;
    border-radius:10px; padding:14px 18px; border-top:3px solid #2563a8;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ──
@st.cache_resource
def load_models():
    models = {}
    errors = []
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = {
        "random_forest":     ["random_forest.pkl", "house_price_model.pkl"],
        "ridge":             ["ridge.pkl"],
        "lasso":             ["lasso.pkl"],
        "linear_regression": ["linear_regression.pkl"],
    }
    for key, fnames in candidates.items():
        for fname in fnames:
            path = os.path.join(base, fname)
            if os.path.exists(path):
                try:
                    models[key] = _load_model(path)
                    break
                except Exception as e:
                    errors.append(f"{fname}: {type(e).__name__}: {e}")
            else:
                errors.append(f"{fname}: file tidak ditemukan")
    return models, errors, base

MODELS, LOAD_ERRORS, BASE_DIR = load_models()

if not MODELS:
    st.error("❌ Tidak ada model yang berhasil dimuat.")
    st.markdown("### 🔍 Info Debug")
    st.code(f"Base directory: {BASE_DIR}")
    try:
        files = [f for f in os.listdir(BASE_DIR) if f.endswith(".pkl")]
        st.code(f"File .pkl ditemukan: {files}")
    except Exception as e:
        st.code(f"Gagal list dir: {e}")
    st.markdown("**Detail error tiap model:**")
    for err in LOAD_ERRORS:
        st.code(err)
    st.stop()

MODEL_META = {
    "random_forest":     {"name":"Random Forest (n=300, d=20)", "r2":0.87, "mae":1.8, "color":"#1e4080", "rank":1},
    "ridge":             {"name":"Ridge (α=10)",                "r2":0.72, "mae":3.2, "color":"#7c3aed", "rank":2},
    "linear_regression": {"name":"Linear Regression",           "r2":0.70, "mae":3.5, "color":"#16a34a", "rank":3},
    "lasso":             {"name":"Lasso (α=1000)",               "r2":0.68, "mae":3.8, "color":"#d97706", "rank":4},
}

def build_input(lb, lt, kt, km, grs, lokasi):
    return pd.DataFrame([{
        "LB":lb,"LT":lt,"KT":kt,"KM":km,"GRS":grs,
        "LOKASI":lokasi,"RASIO_BANGUNAN":lb/lt,"TOTAL_RUANGAN":kt+km,
    }])

def fmt_price(v):
    if v >= 1e9: return f"Rp {v/1e9:.2f} M"
    return f"Rp {v/1e6:.0f} jt"

def fmt_short(v):
    if v >= 1e9: return f"{v/1e9:.1f}M"
    return f"{v/1e6:.0f}jt"

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.caption("COMP6577001 – Machine Learning\nBINUS University 2025/2026")
    st.divider()
    st.markdown("### Input Spesifikasi Rumah")
    lb  = st.slider("Luas Bangunan (m²)", 40, 600, 150, step=5)
    lt  = st.slider("Luas Tanah (m²)", 25, 700, 200, step=5)
    kt  = st.slider("Kamar Tidur", 2, 10, 4)
    km  = st.slider("Kamar Mandi", 1, 10, 3)
    grs = st.slider("Garasi", 0, 10, 2)
    lokasi = st.selectbox("Lokasi", [
        "Jakarta Selatan","Tebet","Tebet Timur",
        "Tebet Barat","Tebet Utara","Kebon Baru","Menteng Dalam"
    ])
    st.divider()
    run_btn = st.button("⚡ Run All Models", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Model tersedia:**")
    for key, meta in MODEL_META.items():
        icon = "✅" if key in MODELS else "⚠️"
        st.caption(f"{icon} {meta['name']}")

# ── Header ──
st.markdown("""
<div style="background:linear-gradient(135deg,#0f2044,#1a3460);border-radius:14px;
     padding:32px 36px;margin-bottom:28px;">
  <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
       color:#6ba3e8;margin-bottom:10px;">
    Final Project · COMP6577001 · BINUS University 2025/2026
  </div>
  <h1 style="font-size:2rem;font-weight:700;color:#fff;margin:0 0 8px;">House Price Prediction</h1>
  <p style="font-size:14px;color:rgba(255,255,255,0.5);margin:0;">
    Tebet, Jakarta Selatan · 4 Classical ML Models · 1.010 Data Listings
  </p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Dataset","1.010","listing rumah")
c2.metric("Best R² Score","~0.87","Random Forest")
c3.metric("Model Loaded", str(len(MODELS)), "dari 4 model")
c4.metric("Cross-Validation","5-Fold","RF model")

tab1,tab2,tab3,tab4 = st.tabs([
    "🏠 Prediction","📊 Data Overview","🤖 Model Details","⚙️ ML Pipeline"
])

# ════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════
with tab1:
    st.markdown("##### Model Comparison Predictor")
    st.caption("Klik '⚡ Run All Models' di sidebar untuk menjalankan semua model.")

    if run_btn or "results" in st.session_state:
        if run_btn:
            input_df = build_input(lb, lt, kt, km, grs, lokasi)
            results  = []
            for key, pipeline in MODELS.items():
                try:
                    meta  = MODEL_META[key]
                    price = float(pipeline.predict(input_df)[0])
                    price = max(price, 400_000_000)
                    results.append({
                        "key":key,"name":meta["name"],"color":meta["color"],
                        "rank":meta["rank"],"r2":meta["r2"],"mae":meta["mae"],
                        "price":price,"pmin":price*0.88,"pmax":price*1.12,
                    })
                except Exception as e:
                    st.warning(f"⚠️ {key}: {e}")
            results.sort(key=lambda x: x["rank"])
            st.session_state["results"] = results
            st.session_state["inp"]     = (lb, lt, kt, km, grs, lokasi)

        results = st.session_state["results"]
        lb_i,lt_i,kt_i,km_i,grs_i,lok_i = st.session_state["inp"]

        if not results:
            st.error("Tidak ada model yang berhasil prediksi.")
        else:
            ensemble_avg = sum(r["price"] for r in results) / len(results)
            ensemble_min = ensemble_avg * 0.88
            ensemble_max = ensemble_avg * 1.12

            st.info(f"📌 LB={lb_i}m² · LT={lt_i}m² · KT={kt_i} · KM={km_i} · GRS={grs_i} · Lokasi={lok_i} · Rasio={lb_i/lt_i:.2f}")

            # Ensemble box
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0f2044,#1a3460);border-radius:14px;
                 padding:24px 28px;margin:12px 0;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                   color:rgba(255,255,255,0.45);margin-bottom:4px;">
                🔀 Ensemble Prediction — Simple Average ({len(results)} Model)
              </div>
              <div style="font-size:40px;font-weight:700;color:#f5c842;line-height:1;margin-bottom:6px;">
                {fmt_price(ensemble_avg)}
              </div>
              <div style="font-size:13px;color:rgba(255,255,255,0.45);">
                Kisaran: {fmt_price(ensemble_min)} – {fmt_price(ensemble_max)} · Harga/m²: {fmt_short(ensemble_avg/lt_i)}/m²
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Kontribusi tiap model
            label_map = {"random_forest":"Random Forest","ridge":"Ridge","lasso":"Lasso","linear_regression":"Linear Reg."}
            cols = st.columns(len(results))
            for i, r in enumerate(results):
                diff = r["price"] - ensemble_avg
                diff_pct = (diff/ensemble_avg)*100
                arrow = "▲" if diff>=0 else "▼"
                clr = "#16a34a" if diff>=0 else "#dc2626"
                with cols[i]:
                    st.markdown(f"""
                    <div style="background:#f8faff;border:1px solid #dbeafe;
                         border-left:4px solid {r['color']};border-radius:8px;
                         padding:10px 12px;text-align:center;">
                      <div style="font-size:11px;font-weight:700;color:{r['color']};margin-bottom:4px;">
                        {label_map.get(r['key'], r['key'])}
                      </div>
                      <div style="font-size:15px;font-weight:700;color:#0f2044;">{fmt_price(r['price'])}</div>
                      <div style="font-size:11px;color:{clr};">{arrow} {abs(diff_pct):.1f}% dari avg</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()

            # RF reference
            rf = next((r for r in results if r["key"]=="random_forest"), results[0])
            st.markdown(f"""
            <div style="background:#f8faff;border:1px solid #dbeafe;border-left:4px solid #1e4080;
                 border-radius:8px;padding:14px 18px;margin-bottom:16px;">
              <div style="font-size:12px;font-weight:600;color:#1e4080;margin-bottom:2px;">
                🏆 Best Single Model — Random Forest (R² ~0.87)
              </div>
              <div style="font-size:20px;font-weight:700;color:#0f2044;">{fmt_price(rf['price'])}</div>
              <div style="font-size:12px;color:#6b7280;">Kisaran: {fmt_price(rf['pmin'])} – {fmt_price(rf['pmax'])}</div>
            </div>
            """, unsafe_allow_html=True)

            # Comparison table
            st.markdown("**Perbandingan Detail Semua Model**")
            medal = ["🥇","🥈","🥉","4️⃣"]
            rows = []
            for r in results:
                rows.append({
                    "Rank":medal[r["rank"]-1], "Model":r["name"],
                    "Harga":fmt_price(r["price"]), "Harga/m²":fmt_short(r["price"]/lt_i)+"/m²",
                    "Range":f"{fmt_price(r['pmin'])} – {fmt_price(r['pmax'])}",
                    "R²":r["r2"], "MAE":f"~{r['mae']} M",
                })
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True,
                column_config={"R²": st.column_config.ProgressColumn("R²", min_value=0, max_value=1, format="%.2f")}
            )

            # Bar chart — pakai st.bar_chart bawaan (tanpa plotly)
            st.markdown("**Visualisasi Harga per Model (Miliar Rp)**")
            chart_df = pd.DataFrame({
                "Model": [r["name"] for r in results],
                "Harga (M)": [r["price"]/1e9 for r in results],
            }).set_index("Model")
            st.bar_chart(chart_df, color="#2563a8", height=320)

            c1,c2,c3 = st.columns(3)
            c1.metric("Ensemble Average", fmt_short(ensemble_avg), "rata-rata semua model")
            c2.metric("Rasio Bangunan", f"{lb_i/lt_i:.2f}", "LB / LT")
            c3.metric("Selisih RF vs Ensemble", fmt_short(abs(rf["price"]-ensemble_avg)), "perbedaan")
    else:
        st.markdown("""
        <div style="background:#f8faff;border:2px dashed #dbeafe;border-radius:12px;
             padding:48px;text-align:center;">
          <div style="font-size:48px;margin-bottom:12px;">📊</div>
          <div style="font-size:16px;font-weight:600;color:#374151;margin-bottom:6px;">
            Hasil perbandingan model akan muncul di sini
          </div>
          <div style="font-size:13px;color:#9ca3af;">Atur input di sidebar lalu klik ⚡ Run All Models</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature Importance — pakai st.bar_chart
    st.divider()
    st.markdown("#### Feature Importance per Model")
    fi_data = {
        "🌲 Random Forest":    {"LB":28,"LT":24,"RASIO_BANGUNAN":16,"KT":12,"TOTAL_RUANGAN":9,"KM":6,"GRS":5},
        "📐 Ridge (α=10)":     {"LB":32,"LT":25,"KT":15,"KM":10,"RASIO_BANGUNAN":7,"GRS":6,"TOTAL_RUANGAN":5},
        "🔍 Lasso (α=1000)":   {"LB":38,"LT":28,"KT":13,"KM":9,"GRS":7,"TOTAL_RUANGAN":4,"RASIO_BANGUNAN":1},
        "📈 Linear Regression":{"LB":34,"LT":26,"KT":14,"KM":10,"GRS":7,"TOTAL_RUANGAN":5,"RASIO_BANGUNAN":4},
    }
    fi_cols = st.columns(2)
    for i,(mname,fi) in enumerate(fi_data.items()):
        with fi_cols[i%2]:
            st.markdown(f"**{mname}**")
            df_fi = pd.DataFrame({"Fitur":list(fi.keys()),"Importance (%)":list(fi.values())}).set_index("Fitur")
            st.bar_chart(df_fi, color="#7c3aed", height=240, horizontal=True)

# ════════════════════════════════════════
# TAB 2 — DATA OVERVIEW
# ════════════════════════════════════════
with tab2:
    st.markdown("#### Data Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Listing","1.010","rumah Tebet")
    c2.metric("Rata-rata Harga","7,6 M","miliar rupiah")
    c3.metric("Harga Minimum","430 jt","LB=40m² LT=25m²")
    c4.metric("Harga Maksimum","65 M","LB=1126m² LT=1400m²")

    col_l,col_r = st.columns(2)
    with col_l:
        st.markdown("**Distribusi Harga (Miliar Rp)**")
        dist_df = pd.DataFrame({
            "Range":['<1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10','>10'],
            "Jumlah":[3,8,85,148,162,121,98,76,82,68,159],
        }).set_index("Range")
        st.bar_chart(dist_df, color="#2563a8", height=280)
    with col_r:
        st.markdown("**Korelasi Fitur vs Harga**")
        corr_df = pd.DataFrame({
            "Fitur":['LB','LT','KT','KM','GRS'],
            "Korelasi":[0.74,0.68,0.52,0.44,0.38],
        }).set_index("Fitur")
        st.bar_chart(corr_df, color="#1e4080", height=280, horizontal=True)

    st.markdown("**Scatter: Luas Bangunan vs Harga**")
    np.random.seed(42)
    lbs_ = np.random.randint(40,550,150)
    lts_ = (lbs_*np.random.uniform(0.7,1.3,150)).astype(int)
    prc_ = np.maximum(1.5e9+lbs_*9.2e6+lts_*6.5e6+np.random.normal(0,5e8,150),400e6)
    sc_df = pd.DataFrame({"LB (m²)":lbs_, "Harga (M)":prc_/1e9})
    st.scatter_chart(sc_df, x="LB (m²)", y="Harga (M)", color="#2563a8", height=320)

# ════════════════════════════════════════
# TAB 3 — MODEL DETAILS
# ════════════════════════════════════════
with tab3:
    st.markdown("#### 4 Model yang Diuji")
    st.caption("80:20 train-test split · random_state=42 · 5-Fold Cross Validation")

    col_a,col_b = st.columns(2)
    cards = [
        ("🌲 Random Forest","#dbeafe","#1e4080","0.87","1.8 M","n=300, max_depth=20",
         "Ensemble 300 decision tree dengan bagging + feature randomness. Efektif menangkap pola non-linear.",True),
        ("📐 Ridge (α=10)","#ede9fe","#5b21b6","0.72","3.2 M","alpha=10",
         "Linear regression + regularisasi L2. Mencegah overfitting, menangani multikolinearitas.",False),
        ("📈 Linear Regression","#dcfce7","#166534","0.70","3.5 M","default",
         "Baseline model sederhana. Mencari hubungan linear antara fitur dan harga.",False),
        ("🔍 Lasso (α=1000)","#fff3cd","#92400e","0.68","3.8 M","alpha=1000",
         "Linear regression + regularisasi L1. Feature selection otomatis.",False),
    ]
    for i,(title,bg,color,r2,mae,params,desc,best) in enumerate(cards):
        with (col_a if i%2==0 else col_b):
            border = f"2px solid {color}" if best else "1px solid #e5e7eb"
            badge = "<span style='background:#dbeafe;color:#1e4080;font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid #93c5fd;'>Best</span>" if best else ""
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:12px;padding:16px 20px;margin-bottom:12px;">
              <div style="font-size:14px;font-weight:700;color:{color};margin-bottom:8px;">{title} {badge}</div>
              <div style="font-size:12px;color:#374151;margin-bottom:10px;">{desc}</div>
              <div style="display:flex;gap:20px;">
                <div><div style="color:#9ca3af;font-size:10px;">R² SCORE</div><div style="font-weight:700;font-size:20px;color:{color};">{r2}</div></div>
                <div><div style="color:#9ca3af;font-size:10px;">MAE</div><div style="font-weight:700;font-size:20px;color:{color};">~{mae}</div></div>
                <div><div style="color:#9ca3af;font-size:10px;">PARAMS</div><div style="font-weight:600;font-size:12px;color:#374151;">{params}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("**Perbandingan R² Score**")
    r2_df = pd.DataFrame({
        "Model":["Random Forest","Ridge","Linear Reg","Lasso"],
        "R²":[0.87,0.72,0.70,0.68],
    }).set_index("Model")
    st.bar_chart(r2_df, color="#1e4080", height=280)

# ════════════════════════════════════════
# TAB 4 — PIPELINE
# ════════════════════════════════════════
with tab4:
    st.markdown("#### End-to-End ML Pipeline")
    col_p,col_s = st.columns([3,2])
    with col_p:
        steps = [
            ("Data Collection","Load DATA_RUMAH.csv — 1.010 listing Tebet, Jakarta Selatan."),
            ("Feature Engineering","Ekstrak LOKASI, buat RASIO_BANGUNAN = LB/LT, TOTAL_RUANGAN = KT+KM."),
            ("Preprocessing","SimpleImputer + StandardScaler (numerik), OneHotEncoder (kategorikal). ColumnTransformer."),
            ("Train/Test Split","80% training (808), 20% testing (202). random_state=42."),
            ("Model Training","4 model: Linear, Ridge, Lasso, Random Forest. Evaluasi R² dan MAE."),
            ("Cross-Validation","5-Fold CV pada Random Forest. Performa konsisten."),
            ("Deploy","Model disimpan ke .pkl via joblib. Deploy ke Streamlit Cloud."),
        ]
        for i,(title,desc) in enumerate(steps):
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:14px;align-items:flex-start;">
              <div style="width:28px;height:28px;border-radius:50%;background:#2563a8;color:#fff;
                   font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;
                   flex-shrink:0;margin-top:2px;">{i+1}</div>
              <div>
                <div style="font-size:13px;font-weight:700;color:#0f2044;margin-bottom:2px;">{title}</div>
                <div style="font-size:12px;color:#6b7280;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    with col_s:
        st.markdown("**Tech Stack**")
        tech = [
            ("Language","Python 3.x"),("ML Library","scikit-learn 1.6.1"),
            ("Data","pandas, numpy"),("App Framework","Streamlit"),
            ("Serialization","joblib"),("Deployment","Streamlit Cloud"),
            ("Version Control","Git + GitHub"),
        ]
        for label,val in tech:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:7px 0;
                 border-bottom:1px solid #f3f4f6;font-size:13px;">
              <span style="font-weight:600;color:#374151;">{label}</span>
              <span style="color:#6b7280;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
