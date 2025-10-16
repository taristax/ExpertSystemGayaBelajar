# expert_system_learning_style_final.py
"""
Interactive, user-friendly expert system:
- collects answers (mix of Likert 1-5 and ya/tidak),
- maps to 5 features (no overlap),
- loads the latest model+scaler and predicts,
- displays probabilities + explanation,
- logs the user's input + result to Excel.
"""

import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import datetime

# CONFIG
MODEL_PATTERN = "learning_style_model_*.pkl"
SCALER_PATTERN = "scaler_*.pkl"
LOGFILE = "hasil_prediksi_user.xlsx"

# Explanations for each class
EXPLANATIONS = {
    "Visual": "Kamu lebih cepat memahami informasi melalui gambar, diagram, peta konsep, dan catatan tertulis.",
    "Auditory": "Kamu lebih cepat belajar dengan mendengar penjelasan, diskusi, dan mendengarkan rekaman.",
    "Kinesthetic": "Kamu lebih cepat memahami lewat praktik, eksperimen, atau gerakan fisik."
}

def find_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return sorted(files)[-1]

def ask_scale(prompt):
    while True:
        ans = input(prompt + " (1= sangat rendah ... 5= sangat tinggi): ").strip()
        try:
            v = int(ans)
            if 1 <= v <= 5:
                # convert to 0-1 using (val-1)/4 mapping (1->0,5->1)
                return (v - 1) / 4.0
            else:
                print("Masukkan angka antara 1 dan 5.")
        except ValueError:
            print("Masukkan angka 1-5.")

def ask_yesno(prompt):
    while True:
        ans = input(prompt + " (ya/tidak): ").strip().lower()
        if ans in ("ya", "y"):
            return 1.0
        if ans in ("tidak", "t", "no", "n"):
            return 0.0
        print("Jawab 'ya' atau 'tidak'.")

def load_latest_model_and_scaler():
    model_file = find_latest(MODEL_PATTERN)
    scaler_file = find_latest(SCALER_PATTERN)
    if model_file is None or scaler_file is None:
        print("⚠️ Model atau scaler belum ditemukan. Jalankan 'generate_rules_from_data_final.py' terlebih dahulu.")
        return None, None
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    print(f"🔁 Memuat model: {model_file}")
    return model, scaler

def main():
    print("🎯 Sistem Pakar Gaya Belajar — Versi Final (Friendly)\n")
    print("Silakan jawab pertanyaan berikut. Jawaban skala 1–5 atau ya/tidak sesuai instruksi.\n")

    # VISUAL-related questions (map to AcademicScore)
    catatan = ask_scale("1) Seberapa sering kamu membuat catatan atau mind-map saat belajar?")
    diagram = ask_scale("2) Seberapa mudah kamu memahami materi dari diagram/gambar/warna?")
    baca = ask_scale("3) Seberapa sering kamu belajar dengan membaca teks atau buku sebagai sumber utama?")

    # AUDITORY-related questions (map to EmotionalEngagement)
    mendengarkan = ask_scale("4) Seberapa mudah kamu memahami penjelasan guru/dosen saat mendengarkan?")
    diskusi = ask_yesno("5) Apakah kamu suka berdiskusi atau aktif berbicara di kelas?")
    rekaman = ask_scale("6) Seberapa sering kamu memanfaatkan rekaman suara/podcast untuk belajar?")

    # KINESTHETIC-related questions (map to PhysicalActivity)
    praktik = ask_yesno("7) Apakah kamu suka belajar dengan praktik langsung/eksperimen?")
    mencoba = ask_scale("8) Seberapa sering kamu merasa lebih paham setelah mencoba sendiri/praktik?")
    bosan = ask_scale("9) Seberapa sering kamu merasa bosan duduk diam saat belajar?")

    # ACADEMIC HABITS (attendance & participation)
    hadir = ask_scale("10) Seberapa sering kamu hadir di kelas atau pertemuan belajar?")
    aktif = ask_scale("11) Seberapa aktif kamu berpartisipasi dalam kegiatan kelas/kelompok?")

    # Build features (no overlap and interpretable)
    AcademicScore = np.mean([catatan, diagram, baca])
    CourseParticipation = np.mean([diskusi, aktif])
    AttendanceRate = hadir
    PhysicalActivity = np.mean([praktik, mencoba, bosan])
    EmotionalEngagement = np.mean([mendengarkan, rekaman, aktif])

    user_df = pd.DataFrame([{
        "AcademicScore": AcademicScore,
        "CourseParticipation": CourseParticipation,
        "AttendanceRate": AttendanceRate,
        "PhysicalActivity": PhysicalActivity,
        "EmotionalEngagement": EmotionalEngagement
    }])

    model, scaler = load_latest_model_and_scaler()
    if model is None or scaler is None:
        return

    user_scaled = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)
    proba = model.predict_proba(user_scaled)[0]
    classes = model.classes_

    # 🔹 Tampilkan probabilitas dalam persen
    probs_sorted = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    print("\n📊 Hasil prediksi (probabilitas):")
    for c, p in probs_sorted:
        print(f" - {c}: {p*100:.2f}%")

    # 🔹 Tangani kasus hasil imbang (50:50)
    max_proba = max(proba)
    top_styles = [cls for cls, p in zip(classes, proba) if abs(p - max_proba) < 1e-6]

    if len(top_styles) == 1:
        predicted = top_styles[0]
        print(f"\n✅ Gaya belajar yang paling dominan: {predicted}\n")
        print("🧾 Penjelasan singkat:")
        print(EXPLANATIONS.get(predicted, "Tidak ada penjelasan spesifik."))
    else:
        print("\n⚖️ Gaya belajar kamu seimbang antara beberapa tipe:")
        for s in top_styles:
            print(f" - {s}")
            print(f"   💡 {EXPLANATIONS.get(s, 'Tidak ada penjelasan spesifik.')}")

    # 🔹 Log hasil ke Excel
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "catatan": catatan,
        "diagram": diagram,
        "baca": baca,
        "mendengarkan": mendengarkan,
        "diskusi": diskusi,
        "rekaman": rekaman,
        "praktik": praktik,
        "mencoba": mencoba,
        "bosan": bosan,
        "hadir": hadir,
        "aktif": aktif,
        "AcademicScore": AcademicScore,
        "CourseParticipation": CourseParticipation,
        "AttendanceRate": AttendanceRate,
        "PhysicalActivity": PhysicalActivity,
        "EmotionalEngagement": EmotionalEngagement,
        "predicted": ", ".join(top_styles)
    }

    rec_df = pd.DataFrame([record])
    if os.path.exists(LOGFILE):
        existing = pd.read_excel(LOGFILE)
        combined = pd.concat([existing, rec_df], ignore_index=True)
    else:
        combined = rec_df

    combined.to_excel(LOGFILE, index=False)
    print(f"\n💾 Hasil wawancara & prediksi disimpan ke: {LOGFILE}")

if __name__ == "__main__":
    main()
