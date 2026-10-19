# LeafLens

Struktur folder:

```
leaflens/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
└── models/
    ├── leaflens_mobilenetv3_small.keras   (wajib)
    ├── class_names.json                   (wajib)
    ├── metrics.json                       (opsional, angka eksak + latensi)
    ├── classification_report.csv          (opsional, grafik F1 per kelas)
    ├── confusion_matrix.png               (opsional)
    ├── training_curves.png                (opsional)
    └── class_distribution.png             (opsional)
```

Semua file di folder `models/` berasal dari output notebook (`leaflens_tensorflow_bundle.zip`).

Jalankan: `streamlit run app.py`
