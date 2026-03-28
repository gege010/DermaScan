"""
streamlit_app/app.py
────────────────────────────────────────────────────────────────────────────
DermaScan — Skin Health Analyzer
Streamlit frontend, stateless, compatible with FastAPI /predict schema v1.0.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DermaScan — AI Skin Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
API_BASE   = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_DIR  = Path(__file__).resolve().parent.parent / "model" / "saved_model"
MAX_FILE_MB = 10

URGENCY_CONFIG: dict[str, dict[str, str]] = {
    "SEGERA":    {"icon": "🔴", "css": "urgent",    "label": "Perlu Perhatian Segera"},
    "PERHATIAN": {"icon": "🟡", "css": "attention", "label": "Perlu Diperhatikan"},
    "NORMAL":    {"icon": "🟢", "css": "normal",    "label": "Kondisi Normal"},
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap');

    /* Base typography */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hero */
    .hero-wrap {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(3rem, 8vw, 5.5rem);
        font-weight: 700;
        color: #C4A882;
        letter-spacing: -1px;
        line-height: 1.05;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.85rem;
        font-weight: 400;
        letter-spacing: 4px;
        text-transform: uppercase;
        opacity: 0.55;
        margin: 6px 0 0;
    }

    /* Section headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(196, 168, 130, 0.4);
        padding-bottom: 0.35rem;
        margin: 1.4rem 0 0.7rem;
        color: inherit;
    }

    /* Urgency badge */
    .condition-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }
    .urgent    { background: rgba(192,57,43,0.15); color: #e74c3c; border: 1px solid rgba(192,57,43,0.4); }
    .attention { background: rgba(212,172,13,0.12); color: #d4ac0d; border: 1px solid rgba(212,172,13,0.35); }
    .normal    { background: rgba(30,132,73,0.12);  color: #27ae60; border: 1px solid rgba(30,132,73,0.35); }

    /* Metric cards */
    .metric-card {
        border: 1px solid rgba(196, 168, 130, 0.25);
        border-radius: 10px;
        padding: 1.1rem 1rem;
        text-align: center;
        background: rgba(196, 168, 130, 0.05);
        height: 100%;
    }
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #C4A882;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.65;
        margin-top: 4px;
    }

    /* Source / reference card */
    .source-card {
        border-left: 3px solid #C4A882;
        border-radius: 0 8px 8px 0;
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
        background: rgba(196, 168, 130, 0.07);
        line-height: 1.4;
    }
    .source-card a { color: #C4A882; text-decoration: none; }
    .source-card a:hover { text-decoration: underline; }

    /* Disclaimer */
    .disclaimer-box {
        background: rgba(249, 231, 159, 0.07);
        border: 1px solid rgba(249, 231, 159, 0.3);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.78rem;
        line-height: 1.6;
        margin-top: 1.2rem;
        opacity: 0.85;
    }

    /* Inference time chip */
    .infer-chip {
        display: inline-block;
        background: rgba(196, 168, 130, 0.12);
        border: 1px solid rgba(196, 168, 130, 0.3);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #C4A882;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* Stagger-fade animation for result panel */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-anim {
        animation: fadeSlideUp 0.45s ease both;
    }
    .result-anim-delay { animation-delay: 0.1s; }

    /* Pill label for active ingredient */
    .ingredient-pill {
        display: inline-block;
        background: rgba(196, 168, 130, 0.13);
        border: 1px solid rgba(196, 168, 130, 0.35);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 2px 3px;
        color: #C4A882;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_model_metadata() -> dict:
    """Load model_metadata.json if available (used in Model Performance page)."""
    path = MODEL_DIR / "model_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def base64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64 PNG string to a PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def parse_list_field(value: Any) -> list[str]:
    """Normalize a field that might be a string, list, or None."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).replace("-", "\n").split("\n") if s.strip()]


def urgency_cfg(urgency: str) -> dict[str, str]:
    key = urgency.upper() if urgency else "NORMAL"
    return URGENCY_CONFIG.get(key, URGENCY_CONFIG["NORMAL"])


def results_to_json(data: dict) -> str:
    """Sanitize result dict (drop base64 heatmap) and return as JSON string."""
    export = {k: v for k, v in data.items() if k != "gradcam_heatmap_base64"}
    return json.dumps(export, ensure_ascii=False, indent=2)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🔬 DermaScan")
        st.markdown("*AI-powered skin health analyzer*")
        st.divider()

        page = st.radio(
            "Navigasi",
            ["🏠 Analisis Kulit", "📊 Performa Model", "ℹ️ Tentang"],
            label_visibility="collapsed",
        )
        st.divider()

        st.markdown("**Status Sistem**")
        _render_health_status()
        st.divider()

        meta = load_model_metadata()
        st.caption(
            f"Model: EfficientNetB0  \n"
            f"Dataset: HAM10000 + SD101  \n"
            f"Kelas: {meta.get('num_classes', 16)}"
        )

    return page


def _render_health_status():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        h = r.json() if r.ok else {}
        
        # Gunakan blok if-else standar untuk menghindari Streamlit Magic mencetak objek
        if r.ok:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
            
        if h.get("model_loaded"):
            st.success("✅ Model Loaded")
        else:
            st.warning("⚠️ Model belum dimuat")
            
    except Exception:
        st.error("❌ API tidak dapat dijangkau")
        st.caption("Jalankan: `uvicorn api.main:app --port 8000`")


# ─── Page: Analisis Kulit ─────────────────────────────────────────────────────

def page_analyze():
    # Hero
    st.markdown("""
    <div class="hero-wrap">
        <h1 class="hero-title">DermaScan</h1>
        <p class="hero-subtitle">Skin Health Analyzer · Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col_left, col_right = st.columns([1, 1.6], gap="large")

    # ── Left: Upload ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### 📸 Upload Gambar Kulit")
        uploaded = st.file_uploader(
            "Pilih foto kulit atau lesi",
            type=["jpg", "jpeg", "png", "webp"],
            help=f"Format: JPG, PNG, WEBP · Maks: {MAX_FILE_MB}MB",
        )

        if uploaded is None:
            st.session_state.pop("result", None)
            _render_empty_conditions()
            return

        # Validate file size
        if uploaded.size > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File terlalu besar. Maks {MAX_FILE_MB}MB.")
            return

        img = Image.open(uploaded)
        st.image(img, caption=f"📁 {uploaded.name}", width="stretch")
        st.caption(f"{uploaded.size / 1024:.1f} KB  ·  {img.size[0]}×{img.size[1]} px")

        if st.button("🔬 Analisis Sekarang", type="primary", width="stretch"):
            _run_analysis(uploaded)

    # ── Right: Results ────────────────────────────────────────────────────────
    with col_right:
        if "result" in st.session_state:
            _render_results(st.session_state["result"])
        else:
            st.info("👈 Upload gambar dan klik **Analisis Sekarang** untuk memulai.")


def _run_analysis(uploaded):
    """Call the FastAPI /predict endpoint and store result in session state."""
    with st.spinner("Menganalisis kondisi kulit..."):
        steps = [
            (20,  "🔄 Memproses gambar..."),
            (45,  "🧠 CNN mengklasifikasi kondisi kulit..."),
            (70,  "💬 Groq merangkum analisis medis..."),
            (90,  "🌐 Mencari referensi terkini..."),
        ]
        prog = st.progress(0)
        status = st.empty()

        for pct, msg in steps:
            prog.progress(pct)
            status.markdown(f"<small>{msg}</small>", unsafe_allow_html=True)
            time.sleep(0.3)

        try:
            uploaded.seek(0)
            response = requests.post(
                f"{API_BASE}/predict",
                files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                timeout=120,
            )
            prog.progress(100)
            prog.empty()
            status.empty()

            if response.ok:
                st.session_state["result"] = response.json()
                st.success("✅ Analisis selesai!")
            else:
                detail = response.json().get("detail", response.text)
                st.error(f"❌ Error dari API: {detail}")

        except requests.exceptions.ConnectionError:
            prog.empty()
            status.empty()
            st.error("❌ Tidak dapat terhubung ke API. Pastikan backend berjalan di port 8000.")
        except requests.exceptions.Timeout:
            prog.empty()
            status.empty()
            st.error("❌ Request timeout. Coba lagi atau periksa koneksi.")
        except Exception as exc:
            prog.empty()
            status.empty()
            st.error(f"❌ Error tidak terduga: {exc}")


def _render_results(data: dict):
    """Render the full prediction result panel."""

    # ── Core prediction fields ────────────────────────────────────────────────
    pred_class   = data.get("predicted_class", "Unknown")
    confidence   = data.get("confidence", 0.0)
    urgency      = data.get("urgency_level", "NORMAL")
    ai_analysis  = data.get("ai_analysis", {})
    references   = data.get("references", [])
    top3         = data.get("top_3_predictions", [])
    heatmap_b64  = data.get("gradcam_heatmap_base64")
    infer_ms     = data.get("inference_time_ms")

    cfg = urgency_cfg(urgency)

    # Urgency badge + class name
    st.markdown(
        f'<div class="result-anim">'
        f'<span class="condition-badge {cfg["css"]}">'
        f'{cfg["icon"]} {cfg["label"]}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    infer_chip = (
        f'<span class="infer-chip">⚡ {infer_ms:.0f} ms</span>'
        if infer_ms else ""
    )
    st.markdown(
        f'<div class="result-anim result-anim-delay">'
        f'<span style="font-family:\'Playfair Display\',serif;font-size:1.6rem;font-weight:600;">'
        f'{pred_class}</span>{infer_chip}</div>',
        unsafe_allow_html=True,
    )

    # Confidence metric
    conf_pct = confidence * 100
    c1, c2 = st.columns(2)
    c1.metric("Keyakinan Model", f"{conf_pct:.1f}%")
    c2.metric("Tingkat Urgensi", urgency.capitalize())

    st.markdown("---")

    # ── Tabs: Chart · Grad-CAM · AI Analysis · References · Download ─────────
    tab_labels = ["📊 Probabilitas", "👁️ Grad-CAM", "💡 Analisis AI", "🌐 Referensi", "⬇️ Export"]
    tabs = st.tabs(tab_labels)

    # Tab 1: Probability chart
    with tabs[0]:
        _render_probability_chart(top3)

    # Tab 2: Grad-CAM
    with tabs[1]:
        _render_gradcam(heatmap_b64)

    # Tab 3: AI Analysis
    with tabs[2]:
        _render_ai_analysis(ai_analysis)

    # Tab 4: References
    with tabs[3]:
        _render_references(references)

    # Tab 5: Export
    with tabs[4]:
        _render_export(data, pred_class)

    # Disclaimer
    st.markdown(
        '<div class="disclaimer-box">'
        '⚠️ <strong>Disclaimer:</strong> Hasil ini adalah estimasi dari model AI dan '
        '<strong>bukan diagnosis medis resmi</strong>. Selalu konsultasikan kondisi kulit '
        'Anda kepada dokter spesialis kulit (Dermatolog) terdaftar.'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_probability_chart(top3: list[dict]):
    if not top3:
        st.info("Data probabilitas tidak tersedia.")
        return

    # Support both old and new API field names
    names = [p.get("class") or p.get("class_name") or "?" for p in top3]
    probs = [p.get("probability", 0) * 100 for p in top3]

    df = pd.DataFrame({"Kondisi": names, "Probabilitas (%)": probs})
    df = df.sort_values("Probabilitas (%)", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Probabilitas (%)"],
        y=df["Kondisi"],
        orientation="h",
        marker=dict(
            color=df["Probabilitas (%)"],
            colorscale=[[0, "rgba(196,168,130,0.35)"], [1, "#C4A882"]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in df["Probabilitas (%)"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=55, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Probabilitas (%)", range=[0, 115], showgrid=False),
        yaxis=dict(title=""),
        font=dict(family="DM Sans", size=12),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Top-{len(top3)} prediksi tertinggi dari model.")


def _render_gradcam(heatmap_b64: str | None):
    st.markdown("**Grad-CAM — Area yang Diperhatikan Model**")
    st.caption(
        "Heatmap menunjukkan region gambar yang paling mempengaruhi keputusan model. "
        "Warna merah/kuning = area paling berpengaruh."
    )
    if heatmap_b64:
        heatmap_img = base64_to_pil(heatmap_b64)
        st.image(heatmap_img, caption="Grad-CAM Heatmap Overlay", width="stretch")
    else:
        st.info(
            "Heatmap Grad-CAM tidak tersedia untuk prediksi ini. "
            "Pastikan backend mendukung Grad-CAM (`utils/gradcam.py`)."
        )


def _render_ai_analysis(ai: dict):
    if not ai or "error" in ai:
        st.warning("Analisis AI tidak tersedia." + (f" ({ai.get('error')})" if ai else ""))
        return

    # Generic renderer — handles both old and new field name conventions
    field_map = {
        "cause":               ("🔍 Penyebab", False),
        "penyebab":            ("🔍 Penyebab", False),
        "severity":            ("⚠️ Tingkat Keparahan", False),
        "keparahan":           ("⚠️ Tingkat Keparahan", False),
        "penjelasan_singkat":  ("📋 Penjelasan", False),
        "treatment":           ("💊 Penanganan", True),
        "rekomendasi_perawatan": ("💊 Penanganan", True),
        "langkah_penanganan":  ("💊 Penanganan", True),
        "recommendation":      ("📌 Rekomendasi", False),
        "skincare_ingredients":("🧪 Bahan Aktif Disarankan", True),
        "bahan_aktif_disarankan": ("🧪 Bahan Aktif Disarankan", True),
    }

    rendered_labels: set[str] = set()

    for key, value in ai.items():
        if key not in field_map or not value:
            continue
        label, is_list = field_map[key]
        if label in rendered_labels:   # skip duplicate labels
            continue
        rendered_labels.add(label)

        st.markdown(f'<p class="section-header">{label}</p>', unsafe_allow_html=True)

        items = parse_list_field(value) if is_list else [str(value)]

        if "Bahan Aktif" in label:
            pills = "".join(f'<span class="ingredient-pill">{i}</span>' for i in items)
            st.markdown(pills, unsafe_allow_html=True)
        elif is_list:
            for item in items:
                if item:
                    st.markdown(f"• {item}")
        else:
            st.markdown(items[0] if items else "")


def _render_references(ref_data):
    # Mengambil data dari format Dictionary Tavily yang baru
    if isinstance(ref_data, dict):
        hasil = ref_data.get("hasil_pencarian", [])
        ringkasan = ref_data.get("ringkasan_ai", "")
    else:
        hasil = ref_data if isinstance(ref_data, list) else []
        ringkasan = ""

    # Tampilkan Ringkasan AI jika ada
    if ringkasan and "dinonaktifkan" not in ringkasan:
        st.markdown(f"**💡 Ringkasan dari Web:** {ringkasan}")
        st.write("")

    if not hasil:
        st.info("Tidak ada referensi web yang ditemukan.")
        return

    st.markdown(f"Ditemukan **{len(hasil)}** referensi terkait:")
    for ref in hasil:
        if isinstance(ref, dict):
            url = ref.get("url", "#")
            title = ref.get("title", url)
            snippet = ref.get("snippet", "")
        else:
            url = str(ref)
            title = url
            snippet = ""

        # Kartu UI yang lebih cantik dengan cuplikan (snippet)
        st.markdown(
            f'<div class="source-card">'
            f'📄 <a href="{url}" target="_blank"><strong>{title}</strong></a><br>'
            f'<span style="font-size: 0.82rem; opacity: 0.75;">{snippet[:150]}...</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_export(data: dict, pred_class: str):
    st.markdown("**Download Hasil Analisis**")
    st.caption("Ekspor hasil prediksi + analisis AI (tanpa gambar heatmap) sebagai JSON.")

    json_str = results_to_json(data)
    filename = f"dermascan_{pred_class.lower().replace(' ', '_')}.json"

    st.download_button(
        label="⬇️ Download JSON",
        data=json_str,
        file_name=filename,
        mime="application/json",
        width="stretch",
    )

    st.code(json_str[:800] + ("\n..." if len(json_str) > 800 else ""), language="json")


def _render_empty_conditions():
    """Show list of detectable conditions when no image is uploaded."""
    meta = load_model_metadata()
    display_names = meta.get("class_display_names", {})
    descriptions  = meta.get("class_descriptions", {})

    if not display_names:
        st.markdown("---")
        st.markdown("**16 Kondisi yang Dapat Dideteksi**")
        default_conditions = [
            ("Actinic Keratosis", "Lesi pra-kanker akibat paparan UV"),
            ("Basal Cell Carcinoma", "Jenis kanker kulit paling umum"),
            ("Melanoma", "Kanker kulit berbahaya yang berasal dari melanosit"),
            ("Benign Keratosis", "Pertumbuhan kulit jinak non-kanker"),
            ("Dermatofibroma", "Tumor kulit jinak berbentuk nodul"),
            ("Melanocytic Nevi", "Tahi lalat biasa (jinak)"),
            ("Vascular Lesion", "Lesi pada pembuluh darah kulit"),
            ("SJS/TEN", "Reaksi alergi kulit yang parah"),
            ("Psoriasis (Nail)", "Psoriasis yang menyerang kuku"),
            ("Vitiligo", "Hilangnya pigmen warna kulit"),
            ("Acne", "Peradangan folikel rambut dan kelenjar minyak"),
            ("Hyperpigmentation", "Bercak kulit lebih gelap akibat melanin berlebih"),
            ("Oily Skin", "Produksi sebum berlebihan"),
            ("Dry Skin", "Kekurangan kelembapan alami kulit"),
            ("Combination Skin", "Gabungan area kering dan berminyak"),
            ("Normal Skin", "Keseimbangan kelembapan yang baik"),
        ]
        c1, c2 = st.columns(2)
        half = len(default_conditions) // 2
        for col, items in [(c1, default_conditions[:half]), (c2, default_conditions[half:])]:
            with col:
                for name, desc in items:
                    st.markdown(f"**{name}**")
                    st.caption(f"_{desc}_")
        return

    st.markdown("---")
    st.markdown("**Kondisi yang Dapat Dideteksi**")
    c1, c2 = st.columns(2)
    items = list(display_names.items())
    half = len(items) // 2 + len(items) % 2
    for col, chunk in [(c1, items[:half]), (c2, items[half:])]:
        with col:
            for key, name in chunk:
                desc = descriptions.get(key, "")
                st.markdown(f"**{name}**")
                if desc:
                    st.caption(f"_{desc}_")


# ─── Page: Model Performance ──────────────────────────────────────────────────

def page_model():
    meta = load_model_metadata()

    st.markdown(
        '<p style="font-family:\'Playfair Display\',serif;font-size:2.4rem;font-weight:700;margin-bottom:4px">'
        'Performa Model</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"*EfficientNetB0 Transfer Learning — {meta.get('num_classes', 16)} Kelas*"
    )
    st.divider()

    # Metrics from saved JSON
    metrics_path = MODEL_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        st.markdown("### 🎯 Metrik Test Set")
        cols = st.columns(4)
        for col, (key, label) in zip(cols, [
            ("accuracy", "Accuracy"),
            ("f1_score", "F1 Score (Weighted)"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ]):
            val = metrics.get(key, 0)
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val*100:.1f}%</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.write("")
    else:
        # Fallback hardcoded values (from README)
        st.markdown("### 🎯 Metrik Test Set (Hardcoded — jalankan training untuk auto-update)")
        c1, c2, c3, c4 = st.columns(4)
        for col, (val, label) in zip([c1, c2, c3, c4], [
            ("74.2%", "Accuracy"),
            ("76.3%", "F1 Score"),
            ("83.0%", "Precision"),
            ("74.2%", "Recall"),
        ]):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.write("")

    # Training plots
    col_h, col_c = st.columns(2)
    history_img = MODEL_DIR / "training_history.png"
    cm_img      = MODEL_DIR / "confusion_matrix.png"

    with col_h:
        st.markdown("### 📉 Training History")
        if history_img.exists():
            st.image(str(history_img), width="stretch")
        else:
            st.info(
                "File `training_history.png` belum tersedia.  \n"
                "Jalankan `python model/train.py` untuk men-generate grafik ini."
            )

    with col_c:
        st.markdown("### 🔲 Confusion Matrix")
        if cm_img.exists():
            st.image(str(cm_img), width="stretch")
        else:
            st.info(
                "File `confusion_matrix.png` belum tersedia.  \n"
                "Jalankan `python model/train.py` untuk men-generate grafik ini."
            )

    # Model architecture summary
    st.divider()
    st.markdown("### 🏗️ Arsitektur & Training Strategy")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
**Backbone:** EfficientNetB0 (ImageNet pretrained)

**Phase 1 — Feature Adaptation**
- Frozen base, train classification head only
- Optimizer: Adam · LR: 1e-3
- Epochs: 20 · Batch: 32

**Phase 2 — Fine-tuning**
- Unfreeze top 20 layers
- Optimizer: Adam · LR: 1e-5
- Epochs: 30 · Batch: 16
        """)
    with a2:
        st.markdown("""
**Classification Head**
- GlobalAveragePooling2D
- Dense(256, relu)
- Dropout(0.3)
- Dense(16, softmax)

**Regularization**
- Class weighting (balanced)
- Data augmentation (flip, rotate, zoom)
- Early stopping + ReduceLROnPlateau
        """)


# ─── Page: About ──────────────────────────────────────────────────────────────

def page_about():
    st.markdown(
        '<p style="font-family:\'Playfair Display\',serif;font-size:2.4rem;font-weight:700">'
        'Tentang DermaScan</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("""
**DermaScan** adalah sistem analisis kesehatan kulit berbasis AI yang memadukan tiga lapisan kecerdasan secara *end-to-end*:

---

### 🧠 Lapisan 1 — Computer Vision (CNN)
Menggunakan arsitektur **EfficientNetB0** melalui *Transfer Learning* dua fase untuk mengklasifikasi 16 kondisi kulit — mulai dari lesi kanker klinis (dari dataset HAM10000) hingga kondisi kulit wajah sehari-hari (dari dataset SD101).

### 💬 Lapisan 2 — Generative AI (LLM)
Hasil klasifikasi diinterpretasi oleh **Llama-3.3-70b** via Groq API. Model menghasilkan penjelasan medis terstruktur: penyebab, tingkat keparahan, langkah penanganan, dan rekomendasi bahan aktif skincare dalam format JSON yang ketat.

### 🌐 Lapisan 3 — Real-time Web Search
**Tavily Search API** digunakan untuk menyilang-rujuk hasil AI dengan artikel dan sumber dermatologi terkini, memberikan tautan referensi yang dapat dipercaya.

### 👁️ Explainability — Grad-CAM
Setiap prediksi disertai **heatmap Grad-CAM** yang menunjukkan area gambar mana yang paling mempengaruhi keputusan model — meningkatkan transparansi dan kepercayaan terhadap sistem AI.

---

### ⚙️ Tech Stack
| Komponen | Teknologi |
|---|---|
| Deep Learning | TensorFlow / Keras (EfficientNetB0) |
| Backend API | FastAPI + Uvicorn |
| GenAI | Groq API · llama-3.3-70b-versatile |
| Web Search | Tavily API |
| Explainability | Grad-CAM (tf.GradientTape) |
| Frontend | Streamlit |
| Visualisasi | Plotly |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

### ⚠️ Penafian Medis
Aplikasi ini dikembangkan **murni untuk tujuan riset dan portofolio Machine Learning**. Output yang dihasilkan **bukan diagnosis medis yang sah**. Selalu konsultasikan masalah kesehatan kulit Anda kepada dokter spesialis kulit (Dermatolog) yang terdaftar.
    """)


# ─── Main Router ──────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if "🏠" in page:
        page_analyze()
    elif "📊" in page:
        page_model()
    elif "ℹ️" in page:
        page_about()


if __name__ == "__main__":
    main()