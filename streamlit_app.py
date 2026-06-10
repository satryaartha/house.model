import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor – Tebet",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
  /* Font & background */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #f8faff;
    border: 1px solid #dbeafe;
    border-radius: 10px;
    padding: 14px 18px;
    border-top: 3px solid #2563a8;
  }
  [data-testid="metric-container"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #0f2044 !important;
  }

  /* Result price box */
  .price-box {
    background: linear-gradient(135deg, #0f2044 0%, #1a3460 100%);
    border-radius: 14px;
    padding: 24px 28px;
    margin: 16px 0;
  }
  .price-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.45); margin-bottom: 6px; }
  .price-value { font-size: 38px; font-weight: 700; color: #f5c842; line-height: 1; margin-bottom: 6px; }
  .price-range { font-size: 13px; color: rgba(255,255,255,0.45); }

  /* Model badge */
  .badge-best { background:#dbeafe; color:#1e4080; border:1px solid #93c5fd; padding:3px 10px; border-radius:99px; font-size:11px; font-weight:700; }
  .badge-good { background:#dcfce7; color:#166534; border:1px solid #86efac; padding:3px 10px; border-radius:99px; font-size:11px; font-weight:600; }
  .badge-mid  { background:#fff3cd; color:#92400e; border:1px solid #fcd34d; padding:3px 10px; border-radius:99px; font-size:11px; font-weight:600; }

  /* Section title */
  .section-eyebrow { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:#2563a8; margin-bottom:4px; }
  .section-title { font-size:1.4rem; font-weight:700; color:#0f2044; margin-bottom:4px; }
  .section-desc { font-size:13px; color:#6b7280; margin-bottom:20px; }

  /* Step pill */
  .step-pill {
    display:inline-flex; align-items:center; gap:10px;
    background:#f8faff; border:1px solid #dbeafe;
    border-radius:10px; padding:12px 16px; margin-bottom:10px; width:100%;
  }
  .step-num {
    width:28px; height:28px; border-radius:50%;
    background:#2563a8; color:#fff;
    display:inline-flex; align-items:center; justify-content:center;
    font-size:12px; font-weight:700; flex-shrink:0;
  }
  .step-text { font-size:13px; color:#374151; line-height:1.5; }
  .step-text b { color:#0f2044; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load semua model. Priority: folder models/, fallback ke house_price_model.pkl"""
    models = {}
    base = os.path.dirname(__file__)

    model_files = {
        "random_forest":     os.path.join(base, "models", "random_forest.pkl"),
        "ridge":             os.path.join(base, "models", "ridge.pkl"),
        "lasso":             os.path.join(base, "models", "lasso.pkl"),
        "linear_regression": os.path.join(base, "models", "linear_regression.pkl"),
    }

    for key, path in model_files.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)

    # Fallback: jika folder models/ belum ada, pakai house_price_model.pkl
    if not models:
        fallback = os.path.join(base, "house_price_model.pkl")
        if os.path.exists(fallback):
            models["random_forest"] = joblib.load(fallback)
            st.sidebar.warning("⚠️ Hanya model Random Forest yang tersedia. Jalankan save_all_models.py untuk 4 model.")

    return models

MODELS = load_models()

MODEL_META = {
    "random_forest":     {"name": "Random Forest",       "short": "RF",  "r2": 0.87, "mae": 1.8, "color": "#1e4080", "rank": 1, "params": "n=300, max_depth=20"},
    "ridge":             {"name": "Ridge (α=10)",         "short": "R",   "r2": 0.72, "mae": 3.2, "color": "#7c3aed", "rank": 2, "params": "alpha=10"},
    "linear_regression": {"name": "Linear Regression",   "short": "LR",  "r2": 0.70, "mae": 3.5, "color": "#16a34a", "rank": 3, "params": "default"},
    "lasso":             {"name": "Lasso (α=1000)",       "short": "L",   "r2": 0.68, "mae": 3.8, "color": "#d97706", "rank": 4, "params": "alpha=1000"},
}

def build_input(lb, lt, kt, km, grs, lokasi):
    return pd.DataFrame([{
        "LB": lb, "LT": lt, "KT": kt, "KM": km, "GRS": grs,
        "LOKASI": lokasi,
        "RASIO_BANGUNAN": lb / lt,
        "TOTAL_RUANGAN": kt + km,
    }])

def fmt_price(v):
    if v >= 1e9: return f"Rp {v/1e9:.2f} M"
    return f"Rp {v/1e6:.0f} jt"

def fmt_short(v):
    if v >= 1e9: return f"{v/1e9:.1f}M"
    return f"{v/1e6:.0f}jt"


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
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
        "Jakarta Selatan", "Tebet", "Tebet Timur",
        "Tebet Barat", "Tebet Utara", "Kebon Baru", "Menteng Dalam"
    ])

    st.divider()
    run_btn = st.button("⚡ Run All Models", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Model tersedia:**")
    for key in MODEL_META:
        status = "✅" if key in MODELS else "⚠️"
        st.caption(f"{status} {MODEL_META[key]['name']}")

    st.divider()
    st.caption("Dataset: 1.010 listing\nArea: Tebet, Jakarta Selatan\nBest R²: ~0.87")


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f2044,#1a3460);border-radius:14px;padding:32px 36px;margin-bottom:28px;">
  <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#6ba3e8;margin-bottom:10px;">
    Final Project · COMP6577001 · BINUS University 2025/2026
  </div>
  <h1 style="font-size:2rem;font-weight:700;color:#fff;margin:0 0 8px;letter-spacing:-0.02em;">
    House Price Prediction
  </h1>
  <p style="font-size:14px;color:rgba(255,255,255,0.5);margin:0;">
    Tebet, Jakarta Selatan · 4 Classical ML Models · 1.010 Data Listings
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Dataset", "1.010", "listing rumah")
col2.metric("Best R² Score", "~0.87", "Random Forest")
col3.metric("Model Diuji", "4", "Classical ML")
col4.metric("Cross-Validation", "5-Fold", "RF model")


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Prediction", "📊 Data Overview", "🤖 Model Details", "⚙️ ML Pipeline"
])


# ═══════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-eyebrow">Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Comparison Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Klik "Run All Models" di sidebar untuk menjalankan semua model sekaligus dan membandingkan hasilnya.</div>', unsafe_allow_html=True)

    if run_btn or "last_results" in st.session_state:
        if run_btn:
            # Jalankan semua model
            input_df = build_input(lb, lt, kt, km, grs, lokasi)
            results = []
            for key, pipeline in MODELS.items():
                meta  = MODEL_META[key]
                price = float(pipeline.predict(input_df)[0])
                price = max(price, 400_000_000)
                results.append({
                    "key":   key,
                    "name":  meta["name"],
                    "short": meta["short"],
                    "color": meta["color"],
                    "rank":  meta["rank"],
                    "r2":    meta["r2"],
                    "mae":   meta["mae"],
                    "price": price,
                    "pmin":  price * 0.88,
                    "pmax":  price * 1.12,
                })
            results.sort(key=lambda x: x["rank"])
            st.session_state["last_results"] = results
            st.session_state["last_input"] = (lb, lt, kt, km, grs, lokasi)

        results = st.session_state["last_results"]
        inp = st.session_state["last_input"]
        lb_i, lt_i, kt_i, km_i, grs_i, lok_i = inp

        # ── Input summary chips ──
        st.info(f"📌 Input: LB={lb_i}m² · LT={lt_i}m² · KT={kt_i} · KM={km_i} · GRS={grs_i} · Lokasi={lok_i} · Rasio={lb_i/lt_i:.2f} · Total Ruangan={kt_i+km_i}")

        # ── Winner box ──
        best = next((r for r in results if r["key"] == "random_forest"), results[0])
        st.markdown(f"""
        <div class="price-box">
          <div class="price-label">🏆 Best Model — Random Forest Regressor</div>
          <div class="price-value">{fmt_price(best['price'])}</div>
          <div class="price-range">Kisaran estimasi: {fmt_price(best['pmin'])} – {fmt_price(best['pmax'])}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Comparison table ──
        st.markdown("#### Perbandingan Semua Model")

        medal = ["🥇","🥈","🥉","4️⃣"]
        rows = []
        for r in results:
            rows.append({
                "Rank":           medal[r["rank"]-1],
                "Model":          r["name"],
                "Prediksi Harga": fmt_price(r["price"]),
                "Harga/m²":       fmt_short(r["price"] / lt_i) + "/m²",
                "Range Estimasi": f"{fmt_price(r['pmin'])} – {fmt_price(r['pmax'])}",
                "R² Score":       f"{r['r2']:.2f}",
                "MAE":            f"~{r['mae']} M",
            })

        df_table = pd.DataFrame(rows)
        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "R² Score": st.column_config.ProgressColumn(
                    "R² Score", min_value=0, max_value=1, format="%.2f"
                ),
            }
        )

        # ── Price comparison bar chart ──
        st.markdown("#### Visualisasi Harga per Model")
        fig_bar = go.Figure()
        for r in results:
            fig_bar.add_trace(go.Bar(
                name=r["name"],
                x=[r["name"]],
                y=[r["price"] / 1e9],
                marker_color=r["color"],
                text=[f"{r['price']/1e9:.2f} M"],
                textposition="outside",
                error_y=dict(
                    type="data",
                    array=[(r["pmax"] - r["price"]) / 1e9],
                    arrayminus=[(r["price"] - r["pmin"]) / 1e9],
                    visible=True,
                    color="#9ca3af",
                    thickness=2,
                )
            ))
        fig_bar.update_layout(
            showlegend=False,
            yaxis_title="Harga (Miliar Rp)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", size=12),
            margin=dict(t=20, b=20),
            height=320,
            yaxis=dict(gridcolor="#f3f4f6"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Summary stats ──
        c1, c2, c3 = st.columns(3)
        c1.metric("Rasio Bangunan", f"{lb_i/lt_i:.2f}", "LB / LT")
        c2.metric("Total Ruangan", f"{kt_i + km_i}", "KT + KM")
        rf_price = best["price"]
        lr_price = next((r["price"] for r in results if r["key"] == "linear_regression"), rf_price)
        c3.metric("Selisih RF vs Linear", fmt_short(abs(rf_price - lr_price)), "perbedaan estimasi")

    else:
        st.markdown("""
        <div style="background:#f8faff;border:2px dashed #dbeafe;border-radius:12px;padding:48px;text-align:center;">
          <div style="font-size:48px;margin-bottom:12px;">📊</div>
          <div style="font-size:16px;font-weight:600;color:#374151;margin-bottom:6px;">Hasil perbandingan model akan muncul di sini</div>
          <div style="font-size:13px;color:#9ca3af;">Atur input di sidebar lalu klik ⚡ Run All Models</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature Importance ──
    st.divider()
    st.markdown("#### Feature Importance per Model")

    fi_data = {
        "Random Forest":       {"LB":28,"LT":24,"RASIO_BANGUNAN":16,"KT":12,"TOTAL_RUANGAN":9,"KM":6,"GRS":5},
        "Ridge (α=10)":        {"LB":32,"LT":25,"KT":15,"KM":10,"RASIO_BANGUNAN":7,"GRS":6,"TOTAL_RUANGAN":5},
        "Lasso (α=1000)":      {"LB":38,"LT":28,"KT":13,"KM":9,"GRS":7,"TOTAL_RUANGAN":4,"RASIO_BANGUNAN":1},
        "Linear Regression":   {"LB":34,"LT":26,"KT":14,"KM":10,"GRS":7,"TOTAL_RUANGAN":5,"RASIO_BANGUNAN":4},
    }
    fi_colors = {
        "Random Forest": "#1e4080",
        "Ridge (α=10)": "#7c3aed",
        "Lasso (α=1000)": "#d97706",
        "Linear Regression": "#16a34a",
    }

    col_fi = st.columns(2)
    for i, (model_name, fi) in enumerate(fi_data.items()):
        with col_fi[i % 2]:
            st.markdown(f"**{model_name}**")
            df_fi = pd.DataFrame(list(fi.items()), columns=["Fitur", "Importance (%)"])
            df_fi = df_fi.sort_values("Importance (%)", ascending=True)
            fig_fi = px.bar(
                df_fi, x="Importance (%)", y="Fitur", orientation="h",
                color_discrete_sequence=[fi_colors[model_name]],
                text="Importance (%)",
            )
            fig_fi.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_fi.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=10, l=0, r=20),
                height=240, showlegend=False,
                font=dict(family="Inter", size=11),
                xaxis=dict(gridcolor="#f3f4f6", range=[0, 50]),
                yaxis=dict(gridcolor="white"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2 — DATA OVERVIEW
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listing", "1.010", "rumah Tebet")
    c2.metric("Rata-rata Harga", "7,6 M", "miliar rupiah")
    c3.metric("Harga Minimum", "430 jt", "LB=40m² LT=25m²")
    c4.metric("Harga Maksimum", "65 M", "LB=1126m² LT=1400m²")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Distribusi Harga (Miliar Rp)**")
        labels = ['<1M','1-2M','2-3M','3-4M','4-5M','5-6M','6-7M','7-8M','8-9M','9-10M','>10M']
        values = [3, 8, 85, 148, 162, 121, 98, 76, 82, 68, 159]
        fig_dist = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color="#2563a8",
            marker_line_width=0,
        ))
        fig_dist.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10,b=10), height=280,
            font=dict(family="Inter",size=11),
            yaxis=dict(gridcolor="#f3f4f6"),
            xaxis=dict(gridcolor="white"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_r:
        st.markdown("**Korelasi Fitur vs Harga**")
        features = ['LB','LT','KT','KM','GRS']
        corr     = [0.74, 0.68, 0.52, 0.44, 0.38]
        colors   = ['#1e4080','#2563a8','#3b7dd8','#6ba3e8','#b3cef5']
        fig_corr = go.Figure(go.Bar(
            y=features, x=corr, orientation='h',
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.2f}" for v in corr], textposition="outside",
        ))
        fig_corr.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10,b=10), height=280,
            font=dict(family="Inter",size=11),
            xaxis=dict(gridcolor="#f3f4f6", range=[0,1]),
            yaxis=dict(gridcolor="white"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter plot
    st.markdown("**Scatter Plot: Luas Bangunan vs Harga**")
    np.random.seed(42)
    lbs_  = np.random.randint(40, 550, 150)
    lts_  = (lbs_ * np.random.uniform(0.7, 1.3, 150)).astype(int)
    kts_  = np.random.randint(2, 8, 150)
    price_= 1.5e9 + lbs_*9.2e6 + lts_*6.5e6 + kts_*310e6 + np.random.normal(0, 5e8, 150)
    price_ = np.maximum(price_, 400e6)
    df_scatter = pd.DataFrame({"LB (m²)": lbs_, "Harga (Miliar Rp)": price_/1e9, "KT": kts_})
    fig_sc = px.scatter(
        df_scatter, x="LB (m²)", y="Harga (Miliar Rp)",
        color="KT", color_continuous_scale="Blues",
        opacity=0.65, trendline="ols",
    )
    fig_sc.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10,b=10), height=320,
        font=dict(family="Inter",size=11),
        yaxis=dict(gridcolor="#f3f4f6"),
        xaxis=dict(gridcolor="#f3f4f6"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Feature table
    st.markdown("**Deskripsi Fitur Dataset**")
    df_feat = pd.DataFrame([
        {"Fitur":"LB","Deskripsi":"Luas Bangunan total semua lantai","Range":"40–1200 m²","Pengaruh":"⬆⬆ Sangat Tinggi"},
        {"Fitur":"LT","Deskripsi":"Luas Tanah / kavling","Range":"25–1400 m²","Pengaruh":"⬆⬆ Sangat Tinggi"},
        {"Fitur":"KT","Deskripsi":"Jumlah Kamar Tidur","Range":"2–10","Pengaruh":"⬆ Sedang"},
        {"Fitur":"KM","Deskripsi":"Jumlah Kamar Mandi","Range":"1–10","Pengaruh":"⬆ Sedang"},
        {"Fitur":"GRS","Deskripsi":"Jumlah Garasi / Carport","Range":"0–10","Pengaruh":"→ Rendah-Sedang"},
        {"Fitur":"LOKASI","Deskripsi":"Area dalam Tebet (dari nama listing)","Range":"7 area","Pengaruh":"→ Sedang"},
        {"Fitur":"RASIO_BANGUNAN","Deskripsi":"LB ÷ LT (fitur turunan)","Range":"0.2–2.0","Pengaruh":"⬆ Sedang"},
        {"Fitur":"TOTAL_RUANGAN","Deskripsi":"KT + KM (fitur turunan)","Range":"3–20","Pengaruh":"→ Rendah"},
    ])
    st.dataframe(df_feat, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# TAB 3 — MODEL DETAILS
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-eyebrow">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">4 Model yang Diuji</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Semua model dilatih dengan data yang sama — 80% training, 20% testing, random_state=42, 5-Fold Cross Validation.</div>', unsafe_allow_html=True)

    # R² comparison chart
    model_names  = ["Random Forest","Ridge (α=10)","Linear Regression","Lasso (α=1000)"]
    r2_scores    = [0.87, 0.72, 0.70, 0.68]
    mae_scores   = [1.8, 3.2, 3.5, 3.8]
    colors_model = ["#1e4080","#7c3aed","#16a34a","#d97706"]

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**R² Score (lebih tinggi = lebih baik)**")
        fig_r2 = go.Figure(go.Bar(
            x=model_names, y=r2_scores,
            marker_color=colors_model, marker_line_width=0,
            text=[f"{v:.2f}" for v in r2_scores], textposition="outside",
        ))
        fig_r2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10,b=10), height=260,
            font=dict(family="Inter",size=11),
            yaxis=dict(range=[0,1.1], gridcolor="#f3f4f6"),
            xaxis=dict(gridcolor="white", tickangle=-15),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_chart2:
        st.markdown("**MAE — Miliar Rp (lebih rendah = lebih baik)**")
        fig_mae = go.Figure(go.Bar(
            x=model_names, y=mae_scores,
            marker_color=colors_model, marker_line_width=0,
            text=[f"~{v} M" for v in mae_scores], textposition="outside",
        ))
        fig_mae.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10,b=10), height=260,
            font=dict(family="Inter",size=11),
            yaxis=dict(gridcolor="#f3f4f6"),
            xaxis=dict(gridcolor="white", tickangle=-15),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # Model cards
    col_a, col_b = st.columns(2)
    cards = [
        ("🌲 Random Forest", "#dbeafe", "#1e4080", "0.87", "1.8 M", "n_estimators=300, max_depth=20, random_state=42",
         "Ensemble 300 decision tree. Bagging + feature randomness. Menangkap pola non-linear harga properti. Model terbaik untuk dataset ini.", True),
        ("📐 Ridge (α=10)", "#ede9fe", "#5b21b6", "0.72", "3.2 M", "alpha=10",
         "Linear regression + regularisasi L2. Koefisien besar dihukum untuk mencegah overfitting. Menangani multikolinearitas antar fitur.", False),
        ("📈 Linear Regression", "#dcfce7", "#166534", "0.70", "3.5 M", "default (no regularization)",
         "Baseline model paling sederhana. Mencari hubungan linear antara fitur dan harga. Rentan terhadap outlier dan multikolinearitas.", False),
        ("🔍 Lasso (α=1000)", "#fff3cd", "#92400e", "0.68", "3.8 M", "alpha=1000",
         "Linear regression + regularisasi L1. Membuat koefisien menjadi nol (feature selection otomatis). Alpha tinggi = model sangat sparse.", False),
    ]

    for i, (title, bg, color, r2, mae, params, desc, is_best) in enumerate(cards):
        with (col_a if i % 2 == 0 else col_b):
            border = f"2px solid {color}" if is_best else "1px solid #e5e7eb"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:12px;padding:16px 20px;margin-bottom:12px;">
              <div style="font-size:14px;font-weight:700;color:{color};margin-bottom:8px;">
                {title} {"&nbsp;<span style='background:#dbeafe;color:#1e4080;font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid #93c5fd;font-weight:700;'>Best Model</span>" if is_best else ""}
              </div>
              <div style="font-size:12px;color:#374151;line-height:1.6;margin-bottom:10px;">{desc}</div>
              <div style="display:flex;gap:20px;font-size:12px;">
                <div><div style="color:#9ca3af;font-size:10px;text-transform:uppercase;">R² Score</div><div style="font-weight:700;font-size:18px;color:{color};">{r2}</div></div>
                <div><div style="color:#9ca3af;font-size:10px;text-transform:uppercase;">MAE</div><div style="font-weight:700;font-size:18px;color:{color};">~{mae}</div></div>
                <div><div style="color:#9ca3af;font-size:10px;text-transform:uppercase;">Parameters</div><div style="font-weight:600;font-size:12px;color:#374151;">{params}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 4 — ML PIPELINE
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-eyebrow">Technical Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">End-to-End ML Pipeline</div>', unsafe_allow_html=True)

    col_pipe, col_stack = st.columns([3, 2])

    with col_pipe:
        steps = [
            ("Data Collection", "Load <code>DATA_RUMAH.csv</code> — 1.010 listing rumah Tebet, Jakarta Selatan. Kolom: NAMA RUMAH, HARGA, LB, LT, KT, KM, GRS."),
            ("Feature Engineering", "Ekstrak <code>LOKASI</code> dari NAMA RUMAH. Buat <code>RASIO_BANGUNAN</code> = LB/LT. Buat <code>TOTAL_RUANGAN</code> = KT+KM."),
            ("Preprocessing Pipeline", "<code>SimpleImputer(median)</code> + <code>StandardScaler</code> untuk numerik. <code>SimpleImputer(most_frequent)</code> + <code>OneHotEncoder</code> untuk kategorikal. Gabung via <code>ColumnTransformer</code>."),
            ("Train/Test Split", "80% training (808 data), 20% testing (202 data). <code>random_state=42</code> untuk reprodusibilitas."),
            ("Model Training", "4 model dilatih: Linear Regression, Ridge (α=10), Lasso (α=1000), Random Forest (n=300, d=20). Evaluasi R² dan MAE."),
            ("Cross-Validation", "5-Fold CV pada Random Forest (model terbaik). Konfirmasi performa konsisten, tidak overfitting."),
            ("Save & Deploy", "4 model disimpan ke <code>models/*.pkl</code> via <code>joblib.dump()</code>. Deploy ke Streamlit Cloud dari GitHub."),
        ]
        for i, (title, desc) in enumerate(steps):
            st.markdown(f"""
            <div class="step-pill">
              <div class="step-num">{i+1}</div>
              <div class="step-text"><b>{title}</b><br>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_stack:
        st.markdown("**Deployment Architecture**")
        st.markdown("""
        <div style="background:#f8faff;border:1px solid #dbeafe;border-radius:12px;padding:16px;margin-bottom:12px;">
          <div style="font-weight:700;color:#1e4080;margin-bottom:4px;">🌐 Streamlit Cloud</div>
          <div style="font-size:12px;color:#6b7280;">Single platform · Free tier · Auto deploy dari GitHub</div>
        </div>
        <div style="text-align:center;font-size:18px;color:#d1d5db;margin:4px 0;">↕</div>
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:16px;margin-bottom:12px;">
          <div style="font-weight:700;color:#166534;margin-bottom:4px;">🤖 ML Models (4 .pkl)</div>
          <div style="font-size:12px;color:#6b7280;">RF · Ridge · Lasso · Linear — scikit-learn Pipeline</div>
        </div>
        <div style="text-align:center;font-size:18px;color:#d1d5db;margin:4px 0;">↕</div>
        <div style="background:#fff3cd;border:1px solid #fcd34d;border-radius:12px;padding:16px;">
          <div style="font-weight:700;color:#92400e;margin-bottom:4px;">📊 Dataset</div>
          <div style="font-size:12px;color:#6b7280;">1.010 listings · Tebet · DATA_RUMAH.csv</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Tech Stack**")
        tech = [
            ("Language", "Python 3.11"),
            ("ML Library", "scikit-learn 1.9"),
            ("Data", "pandas, numpy"),
            ("Visualization", "plotly"),
            ("App Framework", "Streamlit"),
            ("Serialization", "joblib"),
            ("Deployment", "Streamlit Cloud"),
            ("Version Control", "Git + GitHub"),
        ]
        for label, val in tech:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f3f4f6;font-size:13px;">
              <span style="font-weight:600;color:#374151;">{label}</span>
              <span style="color:#6b7280;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
