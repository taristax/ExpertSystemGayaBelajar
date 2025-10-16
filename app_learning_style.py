# app_learning_style.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import glob
import os
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎯 Expert System - Learning Style Prediction",
                   page_icon="🎓", layout="centered")

# ---------------- HELPERS ----------------
def find_latest(pattern):
    files = glob.glob(pattern)
    return sorted(files)[-1] if files else None

def yesno(x): 
    return 1.0 if str(x).lower().startswith("y") else 0.0

EXPLANATIONS = {
    "Visual": "Kamu lebih cepat memahami informasi melalui gambar, diagram, peta konsep, dan catatan tertulis.",
    "Auditory": "Kamu lebih cepat belajar dengan mendengar penjelasan, berdiskusi, dan mendengarkan rekaman.",
    "Kinesthetic": "Kamu lebih cepat memahami lewat praktik, eksperimen, atau gerakan fisik.",
}

# ---------------- LOAD MODEL ----------------
MODEL_PATTERN = "learning_style_model_*.pkl"
SCALER_PATTERN = "scaler_*.pkl"

model_file = find_latest(MODEL_PATTERN)
scaler_file = find_latest(SCALER_PATTERN)

if not (model_file and scaler_file):
    st.error("❌ Model atau scaler belum ditemukan. Jalankan script pelatihan dulu (generate_rules_from_data_final.py).")
    st.stop()

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:16px;padding:10px;border-radius:10px;
                background:linear-gradient(90deg,#262730,#3b3c4a);box-shadow:0 4px 10px rgba(0,0,0,.2)">
      <div style="width:56px;height:56px;border-radius:50%;background:white;display:flex;align-items:center;justify-content:center;padding:6px;">
        🎓
      </div>
      <div style="color:white;">
        <div style="font-size:18px;font-weight:600">🎓 Expert System: Learning Style Prediction</div>
        <div style="font-size:12px;opacity:0.8">Sistem pakar sederhana untuk mengidentifikasi gaya belajar</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")  # spacing

# ---------------- NAVIGATION (TABS) ----------------
tab_home, tab_prediksi, tab_tentang = st.tabs(["🏠 Beranda", "🎯 Prediksi", "ℹ️ Tentang"])

# ---------------- HOME TAB ----------------
with tab_home:
    st.header("Selamat Datang 👋")
    st.markdown(
        """
        Selamat datang di **Sistem Pakar Gaya Belajar**.  
        Aplikasi ini membantu kamu mengetahui gaya belajar dominan (Visual, Auditory, atau Kinesthetic)
        berdasarkan jawaban sederhana.
        """)
    st.markdown("---")
    st.markdown("Klik tab **Prediksi** untuk mulai mengisi kuesioner dan melihat hasil analisis gaya belajarmu.")

# ---------------- PREDIKSI TAB ----------------
with tab_prediksi:
    st.header("🎯 Prediksi Gaya Belajar Kamu")

    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("🧩 Jawab Pertanyaan Berikut")
        col1, col2 = st.columns(2)

        with col1:
            catatan = st.slider("📝 Seberapa sering kamu membuat catatan/mind-map?", 1, 5, 3)
            diagram = st.slider("📊 Seberapa mudah kamu memahami materi lewat diagram/gambar/warna?", 1, 5, 3)
            baca = st.slider("📚 Seberapa sering kamu belajar dengan membaca teks/buku?", 1, 5, 3)
            rekaman = st.slider("🎧 Seberapa sering kamu memanfaatkan rekaman suara/podcast untuk belajar?", 1, 5, 3)
            hadir = st.slider("🏫 Seberapa sering kamu hadir di kelas atau pertemuan?", 1, 5, 3)

        with col2:
            mendengarkan = st.slider("👂 Seberapa mudah kamu memahami penjelasan guru/dosen secara lisan?", 1, 5, 3)
            diskusi = st.radio("💬 Apakah kamu suka berdiskusi atau aktif berbicara di kelas?", ["Ya", "Tidak"], index=1)
            praktik = st.radio("🧪 Apakah kamu suka belajar dengan praktik langsung/eksperimen?", ["Ya", "Tidak"], index=1)
            mencoba = st.slider("🔧 Seberapa sering kamu merasa lebih paham setelah mencoba sendiri/praktik?", 1, 5, 3)
            bosan = st.slider("😐 Seberapa sering kamu merasa bosan duduk diam saat belajar?", 1, 5, 3)
            aktif = st.slider("🙋 Seberapa aktif kamu berpartisipasi dalam kegiatan kelas/kelompok?", 1, 5, 3)

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        AcademicScore = np.mean([(catatan-1)/4, (diagram-1)/4, (baca-1)/4])
        CourseParticipation = np.mean([yesno(diskusi), (aktif-1)/4])
        AttendanceRate = (hadir-1)/4
        PhysicalActivity = np.mean([yesno(praktik), (mencoba-1)/4, (bosan-1)/4])
        EmotionalEngagement = np.mean([(mendengarkan-1)/4, (rekaman-1)/4, (aktif-1)/4])

        user_df = pd.DataFrame([{
            "AcademicScore": AcademicScore,
            "CourseParticipation": CourseParticipation,
            "AttendanceRate": AttendanceRate,
            "PhysicalActivity": PhysicalActivity,
            "EmotionalEngagement": EmotionalEngagement
        }])

        user_scaled = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)
        proba = model.predict_proba(user_scaled)[0]
        classes = model.classes_

        probs_df = pd.DataFrame({"Gaya Belajar": classes, "Probabilitas": proba * 100})
        probs_df = probs_df.sort_values("Probabilitas", ascending=False)

        fig = px.pie(probs_df, names="Gaya Belajar", values="Probabilitas",
                     color="Gaya Belajar", color_discrete_sequence=px.colors.qualitative.Dark2,
                     hole=0.35)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Probabilitas Detail")
        for idx, row in probs_df.iterrows():
            st.write(f"- **{row['Gaya Belajar']}** — {row['Probabilitas']:.2f}%")

        max_p = max(proba)
        tol = 1e-9
        top_styles = [cls for cls, p in zip(classes, proba) if abs(p - max_p) <= tol]

        st.markdown("---")
        if len(top_styles) == 1:
            st.success(f"✨ Gaya belajar yang dominan: **{top_styles[0]}**")
            st.info(EXPLANATIONS.get(top_styles[0], ""))
        else:
            st.warning("⚖️ Gaya belajar kamu seimbang antara beberapa tipe:")
            st.markdown(", ".join([f"**{s}**" for s in top_styles]))
            for s in top_styles:
                st.caption(f"{s}: {EXPLANATIONS.get(s,'')}")

        # tampilkan gaya belajar tanpa ikon
        cols = st.columns(len(top_styles))
        for ccol, style in zip(cols, top_styles):
            with ccol:
                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.04); padding:14px; border-radius:12px; text-align:center;">
                        <div style="font-weight:600; font-size:16px;">{style}</div>
                        <div style="font-size:13px; opacity:0.85; margin-top:6px;">{EXPLANATIONS.get(style,'')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")
        if st.button("🔄 Ulangi Prediksi"):
            st.experimental_rerun()

# ---------------- TENTANG TAB ----------------
with tab_tentang:
    st.header("ℹ️ Tentang Sistem")
    st.markdown(
        """
        Sistem ini membantu menentukan gaya belajar dominan (Visual/Auditory/Kinesthetic)
        berdasarkan jawaban pengguna. Model dilatih dari dataset yang tersedia kemudian
        diintegrasikan dengan antarmuka interaktif ini.
        """)
    st.markdown("**Penjelasan singkat tiap gaya:**")
    st.write("- **Visual:** Belajar efektif lewat gambar, diagram, warna, dan catatan.")
    st.write("- **Auditory:** Belajar efektif lewat mendengarkan, diskusi, dan rekaman.")
    st.write("- **Kinesthetic:** Belajar efektif lewat praktik langsung dan pengalaman.")
    st.markdown("---")
    st.write("🔧 Pengembang: Taristax")
