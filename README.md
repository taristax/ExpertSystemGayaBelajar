# 🧩 Expert System Gaya Belajar

### (Learning Style Expert System)

Sistem pakar ini membantu menentukan **gaya belajar dominan** seseorang — **Visual, Auditory, atau Kinesthetic** — berdasarkan jawaban pengguna.  
Model dilatih dari dataset yang tersedia (`data_gaya_belajar.xlsx`) dan diintegrasikan dengan antarmuka interaktif menggunakan **Streamlit**.

This expert system helps identify a person’s **dominant learning style** — **Visual, Auditory, or Kinesthetic** — based on user responses.  
The model is trained from an existing dataset (`data_gaya_belajar.xlsx`) and integrated with an interactive **Streamlit** interface.

---

## 🚀 Cara Menjalankan / How to Run

### 1️⃣ Clone repository

```bash
git clone https://github.com/username/ExpertSystemGayaBelajar.git
cd ExpertSystemGayaBelajar
```

##(Opsional) Buat virtual environment

Direkomendasikan untuk membuat environment terpisah agar dependensi tidak bentrok.
It’s recommended to create a separate environment to avoid dependency conflicts.

python -m venv venv

##Aktifkan virtual environment
Windows (PowerShell):
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

##Install semua dependensi
Pastikan Anda berada di folder proyek yang sama dengan file requirements.txt.
Make sure you are inside the project folder where requirements.txt is located.

pip install -r requirements.txt

##Jalankan aplikasi Streamlit
Perintah ini akan membuka antarmuka sistem pakar di browser Anda.
This command will open the expert system interface in your browser.

streamlit run app_learning_style.py

Jika berhasil, terminal akan menampilkan alamat seperti:
If successful, the terminal will show an address like:

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501

Klik atau buka alamat tersebut di browser.

##🧠 Tentang Sistem / About the System

Sistem ini menggunakan pendekatan berbasis aturan (rule-based system) dan model pembelajaran mesin sederhana untuk memprediksi gaya belajar dominan pengguna.
This system uses a rule-based approach combined with a simple machine learning model to predict the user’s dominant learning style.

##📊 Fitur / Features
Prediksi gaya belajar: Visual, Auditory, atau Kinesthetic
Penjelasan hasil dalam bahasa Indonesia yang mudah dipahami
Antarmuka pengguna interaktif menggunakan Streamlit
Dapat dilatih ulang menggunakan dataset baru

📁 Struktur Folder / Folder Structure
ExpertSystemGayaBelajar/
├── app*learning_style.py # Main Streamlit app
├── expert_system_learning_style_final.py # Expert system logic
├── generate_rules_from_data_final.py # Training & rule generation script
├── data_gaya_belajar.xlsx # Dataset sumber
├── learning_style_model*\_.pkl # File model hasil pelatihan
├── scaler\_\_.pkl # Scaler untuk preprocessing data
├── requirements.txt # Daftar dependensi Python
├── README.md # Dokumentasi proyek
└── .gitignore # File untuk pengecualian Git

🧩 Teknologi yang Digunakan / Technologies Used

Python 3.10+
Streamlit — untuk antarmuka web interaktif
Scikit-learn — untuk pelatihan model
Pandas — untuk manipulasi data
Joblib — untuk menyimpan model dan scaler

##🧾 Lisensi / License
Proyek ini bersifat open-source dan dapat digunakan untuk tujuan edukasi.
This project is open-source and free to use for educational purposes.

##✨ Kontributor / Contributors
taristax

##📬 Kontak / Contact
Jika Anda memiliki pertanyaan, saran, atau masalah, silakan hubungi:
If you have questions, suggestions, or issues, please contact:
📧 Email: [alviantaristaputra.com
]
🌐 GitHub: https://github.com/taristax
