# 🌿 LeafySense

Aplikasi web untuk mengklasifikasikan penyakit daun tanaman dari sebuah foto. Pengguna mengunggah gambar daun atau memotretnya lewat kamera, lalu aplikasi menampilkan tiga prediksi teratas beserta penjelasan singkat dan saran penanganan umum.

Model yang dipakai adalah MobileNetV3Small (TensorFlow/Keras) yang dilatih dengan dataset PlantVillag (38 kelas, 14 jenis tanaman). Antarmuka dibuat dengan Streamlit.

## Struktur proyek

```
Leaf-Plants-Classification/
├── .streamlit/
│   └── config.toml                                 # Tema aplikasi
├── demo/
│   ├── Apple Black Rot/
│   │   └── 0139bc6d-391c-4fd1-bcae-cc74dabfddd7___JR_FrgE.S 2734.jpg
│   ├── Corn Cercospora Leaf Spot (Gray Leaf Spot)/
│   │   └── 038464d1-47a9-4169-afb1-72c87e568a95___RS_GLSp 4480.jpg
│   └── Grape Esca (Black Measles)/
│       └── 0279ec2c-5470-47e8-b1ba-79442bd323bb___FAM_B.Msls 4396.jpg
├── models/
│   ├── leaflens_mobilenetv3_small.keras            # wajib
│   ├── class_names.json                            # wajib
│   ├── metrics.json                                # opsional
│   ├── classification_report.csv                   # opsional
│   ├── confusion_matrix.png                        # opsional
│   ├── training_curves.png                         # opsional
│   └── class_distribution.png                      # opsional
├── notebook/
│   └── leaf-plants-classification.ipynb            # Notebook pelatihan model
├── app.py                                          # Streamlit
├── requirements.txt
└── README.md

```

File `.keras` dan `class_names.json` wajib ada. File opsional membuat halaman "Tentang model" menampilkan angka eksak, grafik F1 per kelas, dan visualisasi evaluasi. Semua file tersebut berasal dari output notebook pelatihan.

## Instalasi

Prasyarat: Python 3.12 atau lebih baru.

```bash
# 1. Buat virtual environment (disarankan)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Pasang dependensi
pip install -r requirements.txt
```

Salin file model dari output Kaggle ke folder `models/` sesuai struktur di atas.

## Menjalankan aplikasi

```bash
streamlit run app.py
```

Lalu buka `http://localhost:8501`.

## Dataset

[PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## Deployment 

Web juga dapat diakses melalui https://leafysense.streamlit.app/

## License

CC BY-NC-SA 4.0
