from __future__ import annotations

import contextlib
import hashlib
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ──────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "leafysense_mobilenetv3_small.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"

# Dihasilkan notebook
METRICS_PATH = MODEL_DIR / "metrics.json"  # opsional
REPORT_PATH = MODEL_DIR / "classification_report.csv"  # opsional
CM_PATH = MODEL_DIR / "confusion_matrix.png"  # opsional
CURVES_PATH = MODEL_DIR / "training_curves.png"  # opsional
DIST_PATH = MODEL_DIR / "class_distribution.png"  # opsional

IMAGE_SIZE = (224, 224)
DATASET_URL = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"

# Nilai cadangan bila models/metrics.json tidak ditemukan.
FALLBACK_METRICS = {
    "test_accuracy": 0.9735,
    "test_macro_f1": 0.9668,
    "train_images": 38013,
    "validation_images": 8146,
    "test_images": 8146,
}

st.set_page_config(
    page_title="LeafySense | Klasifikasi Penyakit Daun",
    page_icon="🌿",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Kelas PlantVillage
# ──────────────────────────────────────────────────────────────────────────────
ALL_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

CROP_ID = {
    "Apple": "Apple", "Blueberry": "Blueberry", "Cherry": "Cherry", "Corn": "Corn",
    "Grape": "Grape", "Orange": "Orange", "Peach": "Peach", "Pepper": "Bell pepper",
    "Potato": "Potato", "Raspberry": "Raspberry", "Soybean": "Soybean", "Squash": "Squash",
    "Strawberry": "Strawberry", "Tomato": "Tomato",
}

BASE_ADVICE = {
    "Jamur": (
        "Buang bagian tanaman yang terinfeksi, jaga sirkulasi udara, hindari menyiram dari atas "
        "daun, dan lakukan rotasi tanaman. Tanyakan pilihan fungisida yang sesuai ke penyuluh "
        "pertanian setempat."
    ),
    "Bakteri": (
        "Pisahkan tanaman yang terinfeksi, gunakan benih atau bibit sehat, hindari bekerja saat "
        "daun basah, dan bersihkan alat pangkas. Tanyakan opsi pengendalian ke penyuluh pertanian "
        "setempat."
    ),
    "Virus": (
        "Infeksi virus tidak dapat disembuhkan. Cabut dan musnahkan tanaman yang terinfeksi, "
        "kendalikan vektor, dan gunakan benih atau bibit bebas virus."
    ),
    "Hama": (
        "Periksa sisi bawah daun secara rutin, jaga kelembapan lingkungan, dan terapkan "
        "pengendalian hama terpadu sebelum menggunakan pestisida."
    ),
}


def _k(raw: str) -> str:
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def _d(raw, name, cat, desc, tip="", advice=None):
    return _k(raw), {
        "name": name,
        "cat": cat,
        "desc": desc,
        "advice": advice or (BASE_ADVICE[cat] + (" " + tip if tip else "")),
    }


DISEASES = dict([
    _d("Apple___Apple_scab", "Apple scab", "Jamur",
       "Disebabkan jamur Venturia inaequalis. Muncul bercak zaitun-kecokelatan berpermukaan "
       "seperti beludru pada daun dan buah.",
       "Kumpulkan dan musnahkan daun gugur karena menjadi sumber infeksi musim berikutnya."),
    _d("Apple___Black_rot", "Black rot", "Jamur",
       "Disebabkan jamur Botryosphaeria obtusa. Daun menunjukkan bercak cokelat berpola "
       "'mata katak', dan buah dapat membusuk hitam.",
       "Pangkas cabang mati atau berkanker dan singkirkan buah mumi."),
    _d("Apple___Cedar_apple_rust", "Cedar apple rust", "Jamur",
       "Disebabkan jamur Gymnosporangium. Bercak kuning-oranye cerah muncul di daun; siklus "
       "hidupnya memerlukan inang lain berupa tanaman juniper/sedar.",
       "Jauhkan dari inang alternatif bila memungkinkan."),
    _d("Cherry_(including_sour)___Powdery_mildew", "Powdery mildew", "Jamur",
       "Lapisan putih seperti tepung pada permukaan daun akibat jamur Podosphaera; daun muda "
       "dapat mengerut atau melengkung."),
    _d("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Cercospora leaf spot (Gray leaf spot)", "Jamur",
       "Bercak persegi panjang berwarna abu-abu hingga cokelat muda yang sejajar tulang daun, "
       "umumnya pada cuaca hangat dan lembap.",
       "Kelola sisa tanaman dan pilih varietas yang lebih tahan."),
    _d("Corn_(maize)___Common_rust_", "Common rust", "Jamur",
       "Pustul kecil berwarna cokelat kemerahan pada kedua sisi daun yang pecah dan "
       "mengeluarkan spora seperti serbuk karat.",
       "Tanam varietas tahan dan pantau sejak fase awal pertumbuhan."),
    _d("Corn_(maize)___Northern_Leaf_Blight", "Northern leaf blight", "Jamur",
       "Lesi panjang berbentuk cerutu berwarna abu-abu kehijauan hingga cokelat yang dapat "
       "menyatu dan mengeringkan daun.",
       "Rotasi tanaman dan pengelolaan sisa panen membantu menekan penyakit."),
    _d("Grape___Black_rot", "Black rot", "Jamur",
       "Disebabkan jamur Guignardia bidwellii. Bercak cokelat bermargin gelap pada daun dengan "
       "titik-titik hitam kecil di tengahnya; buah dapat mengerut menghitam.",
       "Buang buah mumi dan jaga kanopi agar cepat kering."),
    _d("Grape___Esca_(Black_Measles)", "Esca (Black measles)", "Jamur",
       "Kompleks penyakit jamur pada batang. Daun menampilkan pola 'belang macan' berupa area "
       "kuning atau cokelat di antara tulang daun, dan buah dapat berbintik gelap.",
       "Pangkas bagian batang yang sakit pada waktu kering dan lindungi luka pangkasan."),
    _d("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Leaf blight (Isariopsis leaf spot)", "Jamur",
       "Bercak tidak beraturan berwarna cokelat tua hingga hitam pada daun, terutama menjelang "
       "akhir musim tanam."),
    _d("Orange___Haunglongbing_(Citrus_greening)", "Huanglongbing (Citrus greening)", "Bakteri",
       "Disebabkan bakteri Candidatus Liberibacter yang ditularkan kutu loncat jeruk. Daun "
       "menguning belang tidak simetris dan buah kecil, asimetris, serta pahit.",
       advice=(
           "Belum ada obat untuk penyakit ini. Laporkan dugaan kasus ke dinas pertanian atau "
           "penyuluh setempat, kendalikan kutu loncat, dan gunakan bibit bersertifikat."
       )),
    _d("Peach___Bacterial_spot", "Bacterial spot", "Bakteri",
       "Disebabkan Xanthomonas. Bercak kecil bersudut berwarna gelap pada daun yang dapat "
       "berlubang seperti terkena tembakan (shot-hole)."),
    _d("Pepper,_bell___Bacterial_spot", "Bacterial spot", "Bakteri",
       "Disebabkan Xanthomonas. Bercak kecil basah berwarna cokelat pada daun yang dapat "
       "menguning dan rontok, serta bercak kasar pada buah."),
    _d("Potato___Early_blight", "Early blight", "Jamur",
       "Disebabkan jamur Alternaria solani. Bercak cokelat dengan cincin konsentris seperti "
       "papan sasaran, biasanya muncul pada daun tua lebih dulu.",
       "Jaga nutrisi dan penyiraman tanaman tetap seimbang agar tidak stres."),
    _d("Potato___Late_blight", "Late blight", "Jamur",
       "Disebabkan Phytophthora infestans (oomycete, 'jamur air'). Bercak basah gelap yang "
       "cepat meluas, terutama pada cuaca sejuk dan lembap; dapat menghancurkan pertanaman "
       "dalam hitungan hari.",
       "Tindak cepat karena penyebarannya sangat cepat."),
    _d("Squash___Powdery_mildew", "Powdery mildew", "Jamur",
       "Lapisan putih seperti tepung pada permukaan daun akibat jamur Podosphaera xanthii; "
       "daun dapat menguning lalu mengering."),
    _d("Strawberry___Leaf_scorch", "Leaf scorch", "Jamur",
       "Disebabkan jamur Diplocarpon earlianum. Bercak ungu kecil yang membesar dan membuat "
       "tepi daun tampak hangus."),
    _d("Tomato___Bacterial_spot", "Bacterial spot", "Bakteri",
       "Disebabkan Xanthomonas. Bercak kecil basah kemudian gelap pada daun, batang, dan "
       "buah; menyebar cepat saat hujan atau penyiraman dari atas."),
    _d("Tomato___Early_blight", "Early blight", "Jamur",
       "Disebabkan jamur Alternaria solani. Bercak cokelat bercincin konsentris pada daun tua "
       "yang dikelilingi halo kuning."),
    _d("Tomato___Late_blight", "Late blight", "Jamur",
       "Disebabkan Phytophthora infestans (oomycete). Bercak abu-abu kehijauan basah yang "
       "cepat meluas pada kondisi sejuk dan lembap, juga menyerang buah.",
       "Tindak cepat karena dapat menyebar sangat cepat."),
    _d("Tomato___Leaf_Mold", "Leaf mold", "Jamur",
       "Bercak kuning pada sisi atas daun dengan lapisan kapang zaitun di sisi bawahnya; "
       "berkembang pada kelembapan tinggi.",
       "Kurangi kelembapan, terutama di rumah kaca atau tunnel."),
    _d("Tomato___Septoria_leaf_spot", "Septoria leaf spot", "Jamur",
       "Banyak bercak bulat kecil dengan pusat keabu-abuan dan tepi gelap, biasanya dimulai "
       "dari daun bagian bawah."),
    _d("Tomato___Spider_mites Two-spotted_spider_mite", "Spider mites (Two-spotted spider mite)", "Hama",
       "Hama Tetranychus urticae. Daun tampak berbintik kekuningan (stippling) dan pada "
       "serangan berat terlihat jaring halus; umum pada cuaca panas dan kering."),
    _d("Tomato___Target_Spot", "Target spot", "Jamur",
       "Disebabkan jamur Corynespora cassiicola. Bercak cokelat bercincin pada daun yang dapat "
       "meluas dan menyebabkan daun gugur."),
    _d("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato yellow leaf curl virus", "Virus",
       "Virus yang ditularkan kutu kebul (whitefly). Daun menguning, menggulung ke atas, "
       "mengecil, dan pertumbuhan tanaman terhambat.",
       "Gunakan jaring pelindung dan varietas tahan untuk menekan kutu kebul."),
    _d("Tomato___Tomato_mosaic_virus", "Tomato mosaic virus", "Virus",
       "Daun menunjukkan pola mosaik hijau muda–tua, keriput, atau menyempit. Virus ini mudah "
       "menular lewat sentuhan, alat, dan benih.",
       "Cuci tangan dan sterilkan alat setiap berpindah tanaman."),
])

HEALTHY_INFO = {
    "name": "Healthy",
    "cat": "Sehat",
    "desc": "Model tidak menemukan pola penyakit yang dikenalnya pada daun ini.",
    "advice": (
        "Tetap pantau tanaman secara rutin, jaga nutrisi serta penyiraman yang seimbang, dan "
        "periksa kembali bila muncul bercak atau perubahan warna."
    ),
}

CAT_CLASS = {"Sehat": "ok", "Jamur": "fungal", "Bakteri": "bact", "Virus": "virus", "Hama": "pest"}


def parse_label(raw: str) -> dict:
    """Ubah nama folder PlantVillage menjadi informasi yang mudah dibaca."""
    crop_raw, sep, cond_raw = raw.partition("___")
    if not sep:
        crop_raw, cond_raw = "", raw
    crop_en = re.split(r"[_,(]", crop_raw)[0].strip() if crop_raw else ""
    crop_id = CROP_ID.get(crop_en, crop_en)

    healthy = cond_raw.strip("_").lower() == "healthy"
    info = HEALTHY_INFO if healthy else DISEASES.get(_k(raw))
    if info is None:
        fallback_name = re.sub(r"\s+", " ", cond_raw.replace("_", " ")).strip()
        info = {
            "name": fallback_name or raw,
            "cat": "Lainnya",
            "desc": "Informasi untuk kelas ini belum tersedia.",
            "advice": "Konsultasikan dengan penyuluh pertanian setempat untuk diagnosis lanjutan.",
        }
    return {
        "raw": raw,
        "crop_en": crop_en,
        "crop_id": crop_id,
        "healthy": healthy,
        "title": f"{crop_id} — {info['name']}" if crop_id else info["name"],
        **info,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CSS Styling
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#F6F8F3; --surface:#FFFFFF; --ink:#17261D; --muted:#5E6E63; --line:#E1E8DD;
  --green:#2F6B3F; --green-d:#1F4A2C; --green-soft:#E6EFE2; --sage:#7FA284;
  --amber:#9A6412; --amber-soft:#FBF1DC;
  --brown:#8A5A2B; --brown-soft:#F4EADF;
  --red:#A63D2A; --red-soft:#FBE9E4;
  --violet:#5B4B86; --violet-soft:#ECE8F4;
  --teal:#23706F; --teal-soft:#E0F0EF;
}
html, body, .stApp, [class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:var(--bg);}
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"]{display:none !important;}
.block-container{max-width:1080px; padding:2.2rem 1.5rem 4rem;}
p, li{color:var(--ink);}

/* Hero */
.hero{padding:.4rem 0 1.4rem;}
.eyebrow{display:inline-flex; align-items:center; gap:.45rem; font-size:.78rem; font-weight:600;
  letter-spacing:.06em; text-transform:uppercase; color:var(--green);
  background:var(--green-soft); padding:.35rem .75rem; border-radius:999px;}
.hero-title{font-weight:700; font-size:3rem; line-height:1.08;
  margin:.9rem 0 .5rem; color:var(--green-d); letter-spacing:-.03em;}
.hero-sub{max-width:640px; color:var(--muted); font-size:1.04rem; line-height:1.6; margin:0;}
.chips{display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.1rem;}
.chip{font-size:.8rem; color:var(--muted); background:var(--surface); border:1px solid var(--line);
  padding:.3rem .7rem; border-radius:999px;}
.chip b{color:var(--ink); font-weight:600;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:.25rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:2.8rem; padding:0 1rem; color:var(--muted); font-weight:500;
  background:transparent;}
.stTabs [aria-selected="true"]{color:var(--green-d) !important; font-weight:600;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--green) !important; height:2px;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.5rem;}

/* Kartu */
.card{background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:1.4rem 1.5rem;
  margin-bottom:1rem;}
.card.soft{background:var(--green-soft); border-color:transparent;}
.kicker{font-size:.74rem; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--muted);}
.crop{font-size:.95rem; color:var(--muted); margin-top:.5rem;}
.crop .en{opacity:.7;}
.cond{font-weight:600; font-size:2rem; line-height:1.15;
  color:var(--green-d); margin:.15rem 0 .75rem; letter-spacing:-.01em;}
.sec-title{font-size:.95rem; font-weight:600; color:var(--ink); margin:0 0 .35rem;}
.body{font-size:.94rem; line-height:1.65; color:var(--muted); margin:0;}

/* Pill */
.pill{display:inline-block; font-size:.75rem; font-weight:600; padding:.22rem .65rem; border-radius:999px;}
.pill.ok{background:var(--green-soft); color:var(--green);}
.pill.fungal{background:var(--amber-soft); color:var(--amber);}
.pill.bact{background:var(--red-soft); color:var(--red);}
.pill.virus{background:var(--violet-soft); color:var(--violet);}
.pill.pest{background:var(--teal-soft); color:var(--teal);}
.pill.other{background:#EEF0EC; color:var(--muted);}

/* Meter */
.meter{height:8px; background:#EDF1EA; border-radius:999px; overflow:hidden;}
.meter span{display:block; height:100%; background:linear-gradient(90deg, var(--sage), var(--green));
  border-radius:999px;}
.meter-meta{display:flex; justify-content:space-between; font-size:.82rem; color:var(--muted); margin-top:.45rem;}
.meter-meta b{color:var(--ink);}

/* Top-3 */
.row{display:grid; grid-template-columns:1.6rem 1fr auto; gap:.2rem .75rem; align-items:center; padding:.55rem 0;}
.row + .row{border-top:1px solid var(--line);}
.rank{width:1.5rem; height:1.5rem; border-radius:50%; background:var(--green-soft); color:var(--green);
  font-size:.75rem; font-weight:700; display:flex; align-items:center; justify-content:center;}
.row .name{font-size:.92rem; font-weight:500; color:var(--ink);}
.row .pct{font-size:.88rem; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums;}
.row .meter{grid-column:2 / 4; height:6px;}

/* Statistik */
.stats{display:grid; grid-template-columns:repeat(4, 1fr); gap:.8rem; margin-bottom:1rem;}
.stat{background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1rem 1.1rem;}
.stat .v{font-size:1.6rem; font-weight:600; letter-spacing:-.01em; color:var(--green-d);}
.stat .l{font-size:.8rem; color:var(--muted); margin-top:.1rem;}
.mini{display:grid; grid-template-columns:repeat(3, 1fr); gap:.6rem; margin-top:.2rem;}
.mini div{background:var(--bg); border-radius:14px; padding:.65rem .8rem;}
.mini b{display:block; font-size:1rem; color:var(--ink);}
.mini span{font-size:.75rem; color:var(--muted);}

/* Notice */
.notice{border-radius:16px; padding:.85rem 1rem; font-size:.88rem; line-height:1.55; margin-bottom:1rem;}
.notice.warn{background:var(--amber-soft); color:#6E4609;}
.notice.mid{background:var(--brown-soft); color:#5E3C17;}
.notice.info{background:var(--green-soft); color:var(--green-d);}

/* Empty state */
.empty{text-align:center; padding:2.6rem 1.5rem;}
.empty .ico{font-size:2.4rem;}
.empty h4{margin:.6rem 0 .3rem; font-size:1.05rem; color:var(--ink);}
.empty p{color:var(--muted); font-size:.92rem; margin:0 auto; max-width:360px; line-height:1.6;}

/* Grid kelas */
.grid{display:grid; grid-template-columns:repeat(2, 1fr); gap:.9rem;}
.cls{background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem;}
.cls .top{display:flex; justify-content:space-between; align-items:flex-start; gap:.6rem;}
.cls .crop-name{font-size:.78rem; color:var(--muted); font-weight:500;}
.cls .dn{font-weight:600; font-size:1.02rem; color:var(--ink); margin:.1rem 0 .45rem;}
.cls p{font-size:.86rem; line-height:1.6; color:var(--muted); margin:0 0 .5rem;}
.cls .tip{font-size:.82rem; color:var(--green-d); background:var(--green-soft); border-radius:12px;
  padding:.55rem .7rem; line-height:1.55;}

/* Langkah */
.steps{counter-reset:s; display:grid; gap:.75rem;}
.step{display:grid; grid-template-columns:2rem 1fr; gap:.8rem; align-items:start;}
.step::before{counter-increment:s; content:counter(s); width:2rem; height:2rem; border-radius:50%;
  background:var(--green); color:#fff; font-weight:600; font-size:.85rem; display:flex;
  align-items:center; justify-content:center;}
.step b{display:block; color:var(--ink); font-size:.95rem;}
.step span{color:var(--muted); font-size:.88rem; line-height:1.6;}

.kv{display:grid; grid-template-columns:1fr auto; gap:.1rem 1rem; font-size:.9rem;}
.kv div{padding:.5rem 0; border-bottom:1px solid var(--line);}
.kv div:nth-child(odd){color:var(--muted);}
.kv div:nth-child(even){color:var(--ink); font-weight:500; text-align:right;}

.footer{margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid var(--line); color:var(--muted);
  font-size:.82rem; line-height:1.65;}
.footer a, .body a{color:var(--green); text-decoration:none; border-bottom:1px solid var(--sage);}

/* Widget Streamlit */
[data-testid="stFileUploaderDropzone"]{background:var(--surface); border:1.5px dashed #BBD0BB;
  border-radius:18px; padding:1.6rem;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--green);}
[data-testid="stImage"] img{border-radius:18px; border:1px solid var(--line);}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img{border-radius:18px;}
.stDownloadButton button, .stButton button{border-radius:999px; border:1px solid var(--line);
  background:var(--surface); color:var(--green-d); font-weight:500; padding:.45rem 1.1rem;}
.stDownloadButton button:hover, .stButton button:hover{border-color:var(--green); color:var(--green);}
[data-testid="stExpander"]{background:var(--surface); border:1px solid var(--line) !important;
  border-radius:16px; margin-bottom:.6rem;}
[data-testid="stExpander"] details{border:none;}

@media (max-width: 760px){
  .hero-title{font-size:2.3rem;}
  .stats{grid-template-columns:repeat(2, 1fr);}
  .grid{grid-template-columns:1fr;}
  .cond{font-size:1.6rem;}
}
"""

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utilitas
# ──────────────────────────────────────────────────────────────────────────────
def html_block(s: str) -> str:
    """Rapatkan HTML agar tidak dianggap blok kode oleh Markdown."""
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())


def md(s: str) -> None:
    st.markdown(html_block(s), unsafe_allow_html=True)


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def pct_text(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def num_text(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def pill(cat: str) -> str:
    return f'<span class="pill {CAT_CLASS.get(cat, "other")}">{esc(cat)}</span>'


def meter(value: float) -> str:
    return f'<div class="meter"><span style="width:{max(0.0, min(100.0, value)):.1f}%"></span></div>'



def show_image(image, caption: str | None = None) -> None:
    """st.image lebar penuh yang kompatibel dengan versi Streamlit lama maupun baru."""
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


def show_dataframe(df) -> None:
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def show_bar_chart(df, height: int = 720) -> None:
    try:
        st.bar_chart(df, horizontal=True, color="#2F6B3F", height=height)
    except TypeError:
        st.bar_chart(df, height=height)


@st.cache_data
def load_metrics() -> tuple[dict, bool]:
    """Baca metrics.json dari notebook; gunakan nilai cadangan bila tidak ada."""
    if METRICS_PATH.exists():
        try:
            with METRICS_PATH.open("r", encoding="utf-8") as f:
                return {**FALLBACK_METRICS, **json.load(f)}, True
        except (OSError, json.JSONDecodeError):
            pass
    return dict(FALLBACK_METRICS), False


@contextlib.contextmanager
def _ignore_quantization_config(keras):
    """Sementara abaikan 'quantization_config' pada konfigurasi layer.

    Model yang disimpan dengan Keras lebih baru dapat menyertakan argumen ini, sedangkan
    Keras yang lebih lama menolaknya saat memuat model.
    """
    saved = []

    def make(base_fn):
        def from_config(cls_, config, *args, **kwargs):
            config = dict(config)
            config.pop("quantization_config", None)
            return base_fn(cls_, config, *args, **kwargs)

        return classmethod(from_config)

    for name in ("Dense", "EinsumDense", "Embedding"):
        cls = getattr(keras.layers, name, None)
        if cls is None:
            continue
        had_own = "from_config" in cls.__dict__
        original = cls.__dict__.get("from_config")
        setattr(cls, "from_config", make(cls.from_config.__func__))
        saved.append((cls, had_own, original))
    try:
        yield
    finally:
        for cls, had_own, original in saved:
            if had_own:
                setattr(cls, "from_config", original)
            else:
                delattr(cls, "from_config")


def load_keras_model(keras, path):
    """Muat model .keras; bila versi Keras lokal lebih lama, coba muat dengan penyesuaian."""
    try:
        return keras.models.load_model(path, compile=False)
    except (TypeError, ValueError) as exc:
        if "quantization_config" not in str(exc):
            raise
    with _ignore_quantization_config(keras):
        return keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. Salin file .keras dari output Kaggle ke folder models/."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Pemetaan kelas tidak ditemukan: {LABELS_PATH}. Salin class_names.json dari output Kaggle ke models/."
        )

    try:
        import tensorflow as tf  # impor ditunda agar tampilan awal cepat
    except Exception as exc:  # DLL gagal dimuat, paket rusak, dsb.
        first_line = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
        raise RuntimeError(
            "TensorFlow gagal dimuat di komputer ini, sehingga model belum bisa dipakai. "
            "Penyebab umum di Windows: Microsoft Visual C++ Redistributable belum terpasang atau usang, "
            "atau CPU tidak mendukung instruksi AVX/AVX2. "
            f"Detail: {type(exc).__name__}: {first_line}"
        ) from exc

    model = load_keras_model(tf.keras, MODEL_PATH)
    # Pemanasan satu kali agar prediksi pertama tidak terpengaruh overhead tracing.
    _ = model(np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32), training=False).numpy()
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_names.json harus berisi daftar 'class_names' yang tidak kosong.")
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model memiliki {model.output_shape[-1]} kelas, tetapi class_names.json berisi {len(class_names)} label."
        )
    return model, class_names


def prepare_image(image: Image.Image) -> np.ndarray:
    """Sesuai pelatihan: RGB, resize 224x224, float32 dengan rentang [0, 255].

    MobileNetV3Small pada notebook memakai include_preprocessing=True, jadi
    jangan menambahkan normalisasi /255 di sini.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def confidence_level(top: float, margin: float) -> str:
    if top >= 0.85 and margin >= 0.30:
        return "high"
    if top >= 0.60:
        return "mid"
    return "low"


metrics, metrics_from_file = load_metrics()
acc = float(metrics["test_accuracy"]) * 100
f1 = float(metrics["test_macro_f1"]) * 100

# ──────────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────────
md(f"""
<div class="hero">
<span class="eyebrow">🌿 Klasifikasi penyakit daun berbasis AI</span>
<div class="hero-title">LeafySense</div>
<p class="hero-sub">Unggah atau ambil foto daun untuk melihat prediksi klasifikasi 
penyakitnya dari model MobileNetV3Small - PlantVillage. Hasil dilengkapi dengan penjelasan 
singkat dan panduan penanganan umum.</p>
<div class="chips">
<span class="chip"><b>{len(ALL_CLASSES)}</b> kelas</span>
<span class="chip"><b>{len(CROP_ID)}</b> jenis tanaman</span>
<span class="chip">Akurasi uji <b>{pct_text(acc)}%</b></span>
<span class="chip">Macro F1 <b>{pct_text(f1)}%</b></span>
<span class="chip">TensorFlow / Keras</span>
</div>
</div>
""")

tab_predict, tab_classes, tab_model, tab_guide = st.tabs(
    ["Klasifikasi", "Daftar kelas", "Tentang model", "Panduan & batasan"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 - Klasifikasi
# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    col_in, col_out = st.columns([1, 1.15], gap="large")

    with col_in:
        source = st.radio(
            "Sumber gambar", ["Unggah file", "Gunakan kamera"],
            horizontal=True, label_visibility="collapsed",
        )
        if source == "Unggah file":
            uploaded_file = st.file_uploader(
                "Pilih gambar daun", type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                help="Gunakan satu gambar daun yang jelas dan hindari gambar kolase.",
            )
            if uploaded_file is None:
                st.caption("Format JPG, PNG, atau WEBP. Satu daun per gambar.")
        else:
            uploaded_file = st.camera_input("Ambil foto daun", label_visibility="collapsed")

        pil_image = None
        if uploaded_file is not None:
            try:
                pil_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            except (UnidentifiedImageError, OSError):
                st.error("File tidak dapat dibaca sebagai gambar. Coba file JPG, PNG, atau WEBP lain.")
                st.stop()
            show_image(pil_image)
            w, h = pil_image.size
            st.caption(f"{w} × {h} piksel · akan diubah ukurannya menjadi 224 × 224 untuk model")
            if min(w, h) < 128:
                st.caption("⚠️ Resolusi gambar cukup rendah, hasil mungkin kurang akurat.")

    with col_out:
        if pil_image is None:
            md("""
            <div class="card empty">
            <div class="ico">🍃</div>
            <h4>Belum ada gambar</h4>
            <p>Hasil prediksi akan tampil di sini. Untuk hasil terbaik, pastikan daun memenuhi
            sebagian besar bingkai, cahaya merata, dan latar sederhana.</p>
            </div>
            """)
        else:
            try:
                model, class_names = load_artifacts()
            except RuntimeError as exc:
                st.error(str(exc))
                st.info(
                    "Langkah perbaikan: (1) pasang Microsoft Visual C++ Redistributable x64 terbaru "
                    "(aka.ms/vs/17/release/vc_redist.x64.exe) lalu restart komputer; (2) cek dukungan AVX2 "
                    "pada CPU; (3) atau jalankan aplikasi di Linux, misalnya Streamlit Community Cloud atau "
                    "Hugging Face Spaces. Detail lengkap ada di README_DEPLOY.md."
                )
                st.stop()
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
                st.info("Lihat README_DEPLOY.md untuk instruksi menempatkan artefak model.")
                st.stop()

            file_bytes = uploaded_file.getvalue()
            cache_key = hashlib.md5(file_bytes).hexdigest()
            cached = st.session_state.get("last_prediction")
            if cached and cached["key"] == cache_key:
                scores, elapsed_ms = cached["scores"], cached["elapsed_ms"]
            else:
                input_array = prepare_image(pil_image)
                start = time.perf_counter()
                scores = model(input_array, training=False).numpy()[0]
                elapsed_ms = (time.perf_counter() - start) * 1000
                st.session_state["last_prediction"] = {
                    "key": cache_key, "scores": scores, "elapsed_ms": elapsed_ms,
                }

            top_indices = [int(i) for i in np.argsort(scores)[::-1][:3]]
            top = parse_label(class_names[top_indices[0]])
            top_score = float(scores[top_indices[0]])
            margin = top_score - float(scores[top_indices[1]])
            level = confidence_level(top_score, margin)

            md(f"""
            <div class="card">
            <div class="kicker">Prediksi teratas</div>
            <div class="crop">{esc(top['crop_id'])}</div>
            <div class="cond">{esc(top['name'])}</div>
            {pill(top['cat'])}
            <div style="height:1.1rem"></div>
            {meter(top_score * 100)}
            <div class="meter-meta"><span>Skor model</span><b>{pct_text(top_score * 100)}%</b></div>
            </div>
            """)

            if level == "low":
                md("""<div class="notice warn"><b>Skor rendah.</b> Coba foto ulang dengan cahaya lebih baik
                dan fokus pada satu daun. Bisa juga gambar berasal dari tanaman atau kondisi di luar
                kelas yang dikenali model.</div>""")
            elif level == "mid":
                md("""<div class="notice mid"><b>Skor menengah.</b> Bandingkan dengan prediksi kedua dan
                ketiga sebelum menarik kesimpulan.</div>""")

            md(f"""
            <div class="card">
            <div class="sec-title">Tentang kondisi ini</div>
            <p class="body">{esc(top['desc'])}</p>
            <div style="height:.9rem"></div>
            <div class="sec-title">Saran penanganan umum</div>
            <p class="body">{esc(top['advice'])}</p>
            </div>
            """)

            rows = ""
            for rank, idx in enumerate(top_indices, start=1):
                info = parse_label(class_names[idx])
                s = float(scores[idx])
                rows += f"""
                <div class="row">
                <div class="rank">{rank}</div>
                <div class="name">{esc(info['title'])}</div>
                <div class="pct">{pct_text(s * 100)}%</div>
                {meter(s * 100)}
                </div>"""
            md(f"""
            <div class="card">
            <div class="sec-title" style="margin-bottom:.4rem">Tiga prediksi teratas</div>
            {rows}
            </div>
            """)

            md(f"""
            <div class="mini">
            <div><b>{elapsed_ms:.0f} ms</b><span>Waktu inferensi</span></div>
            <div><b>{pct_text(margin * 100, 1)} poin</b><span>Selisih peringkat 1 &amp; 2</span></div>
            <div><b>224 × 224</b><span>Ukuran input</span></div>
            </div>
            """)

            report = {
                "waktu": datetime.now().isoformat(timespec="seconds"),
                "file": getattr(uploaded_file, "name", "kamera"),
                "model": "MobileNetV3Small",
                "waktu_inferensi_ms": round(elapsed_ms, 1),
                "prediksi": [
                    {
                        "peringkat": r,
                        "kelas": class_names[i],
                        "label": parse_label(class_names[i])["title"],
                        "skor": round(float(scores[i]), 6),
                    }
                    for r, i in enumerate(top_indices, start=1)
                ],
                "catatan": "Skor bukan probabilitas terkalibrasi. Bukan diagnosis agronomi.",
            }
            st.write("")
            st.download_button(
                "Unduh hasil (JSON)",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name="LeafySense_hasil.json",
                mime="application/json",
            )

    if pil_image is not None:
        md("""<div class="notice info" style="margin-top:1rem"><b>Catatan:</b> skor model bukan probabilitas
        yang sudah dikalibrasi, dan aplikasi ini adalah demonstrasi machine learning, bukan diagnosis
        agronomi. Model selalu memilih salah satu kelas yang dikenal, bahkan untuk tanaman atau kondisi
        di luar dataset.</div>""")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 - Daftar Kelas
# ──────────────────────────────────────────────────────────────────────────────
with tab_classes:
    parsed = [parse_label(c) for c in ALL_CLASSES]
    n_healthy = sum(p["healthy"] for p in parsed)
    n_disease = len(parsed) - n_healthy

    md(f"""
    <div class="stats">
    <div class="stat"><div class="v">{len(parsed)}</div><div class="l">Total kelas</div></div>
    <div class="stat"><div class="v">{len(CROP_ID)}</div><div class="l">Jenis tanaman</div></div>
    <div class="stat"><div class="v">{n_disease}</div><div class="l">Kondisi penyakit / hama</div></div>
    <div class="stat"><div class="v">{n_healthy}</div><div class="l">Kondisi sehat</div></div>
    </div>
    """)

    f1_col, f2_col, f3_col = st.columns([1, 1, 1.4])
    with f1_col:
        crop_filter = st.selectbox("Tanaman", ["Semua"] + sorted({p["crop_id"] for p in parsed}))
    with f2_col:
        cat_filter = st.selectbox("Kategori", ["Semua", "Sehat", "Jamur", "Bakteri", "Virus", "Hama"])
    with f3_col:
        query = st.text_input("Cari", placeholder="e.g. powdery mildew, tomato, rust")

    def _match(p: dict) -> bool:
        if crop_filter != "Semua" and p["crop_id"] != crop_filter:
            return False
        if cat_filter != "Semua" and p["cat"] != cat_filter:
            return False
        if query:
            hay = f"{p['crop_id']} {p['crop_en']} {p['name']} {p['desc']}".lower()
            return query.lower().strip() in hay
        return True

    shown = [p for p in parsed if _match(p)]
    st.caption(f"Menampilkan {len(shown)} dari {len(parsed)} kelas")

    if shown:
        cards = ""
        for p in shown:
            cards += f"""
            <div class="cls">
            <div class="top">
            <div><div class="crop-name">{esc(p['crop_id'])}</div>
            <div class="dn">{esc(p['name'])}</div></div>
            {pill(p['cat'])}
            </div>
            <p>{esc(p['desc'])}</p>
            <div class="tip">{esc(p['advice'])}</div>
            </div>"""
        md(f'<div class="grid">{cards}</div>')
    else:
        st.info("Tidak ada kelas yang cocok dengan filter.")

    md("""<p class="body" style="margin-top:1rem">Deskripsi dan saran bersifat umum untuk tujuan edukasi.
    Penanganan yang tepat bergantung pada kondisi lokal, tingkat serangan, dan regulasi setempat.</p>""")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Tentang model
# ──────────────────────────────────────────────────────────────────────────────
with tab_model:
    latency = metrics.get("single_image_latency_ms")
    latency_dev = metrics.get("latency_device", "")
    latency_txt = f"{float(latency):.0f} ms" if latency else "—"
    md(f"""
    <div class="stats">
    <div class="stat"><div class="v">{pct_text(acc)}%</div><div class="l">Akurasi (test split)</div></div>
    <div class="stat"><div class="v">{pct_text(f1)}%</div><div class="l">Macro F1 (test split)</div></div>
    <div class="stat"><div class="v">{len(ALL_CLASSES)}</div><div class="l">Kelas keluaran</div></div>
    <div class="stat"><div class="v">{latency_txt}</div>
    <div class="l">Latensi 1 gambar{f' ({esc(latency_dev)})' if latency and latency_dev else ''}</div></div>
    </div>
    """)

    left, right = st.columns(2, gap="large")

    with left:
        md("""
        <div class="card">
        <div class="sec-title" style="margin-bottom:.9rem">Alur pelatihan</div>
        <div class="steps">
        <div class="step"><div><b>Pembagian data</b><span>Data dibagi menjadi 70% data latih, 15% validasi, 
        dan 15% uji.</span></div></div>
        <div class="step"><div><b>Preprocessing dam augmentasi</b><span>Resize 224 × 224, flip horizontal,
        rotasi acak ±4%, zoom acak 10%. Normalisasi piksel ditangani lapisan bawaan MobileNetV3.</span></div></div>
        <div class="step"><div><b>Tahap 1 - Transfer Learning</b><span>Mempertahankan bobot (frozen) pada backbone ImageNet, 
        model dilatih berfokus pada GlobalAveragePooling, Dropout (0,2), dan Dense softmax selama maksimal 5 epoch..</span></div></div>
        <div class="step"><div><b>Tahap 2 - Fine-Tuning</b><span>Sebanyak 30 lapisan teratas dari backbone dibuka (unfrozen) 
        untuk dilatih ulang menggunakan optimizer Adam dengan learning rate 1e-5 selama maksimal 10 epoch.</span></div></div>
        <div class="step"><div><b>Seleksi &amp; evaluasi</b><span>Checkpoint model terbaik ditentukan berdasarkan 
        metrik akurasi validasi yang selanjutnya dievaluasi pada data uji.</span></div></div>
        <div class="step"><div><b>Ekspor</b><span>Model akhir disimpan dalam format .keras untuk deployment.</span></div></div>
        </div>
        </div>
        """)

    with right:
        tr = metrics.get("train_images")
        va = metrics.get("validation_images")
        te = metrics.get("test_images")
        prefix = "" if metrics_from_file else "≈ "
        md(f"""
        <div class="card">
        <div class="sec-title" style="margin-bottom:.4rem">Spesifikasi</div>
        <div class="kv">
        <div>Arsitektur</div><div>MobileNetV3Small</div>
        <div>Bobot awal</div><div>ImageNet</div>
        <div>Input</div><div>224 × 224 RGB (0–255)</div>
        <div>Loss</div><div>Sparse categorical crossentropy</div>
        <div>Batch size</div><div>32</div>
        <div>Callback</div><div>Checkpoint, EarlyStopping, ReduceLROnPlateau</div>
        <div>Data latih</div><div>{prefix}{num_text(tr)}</div>
        <div>Data validasi</div><div>{prefix}{num_text(va)}</div>
        <div>Data uji</div><div>{prefix}{num_text(te)}</div>
        </div>
        </div>
        <div class="card">
        <div class="sec-title">Dataset</div>
        <p class="body">PlantVillage (varian berwarna) berisi sekitar 54 ribu gambar daun dalam 38 kelas dan 14
        jenis tanaman. Unduh di
        <a href="{DATASET_URL}" target="_blank" rel="noopener">Kaggle</a>. Dataset awal dipublikasikan oleh
        Hughes &amp; Salathé (2015).</p>
        </div>
        """)

    md("""
    <div class="card soft">
    <div class="sec-title">Cara membaca metrik</div>
    <p class="body"><b>Akurasi</b> adalah persentase gambar uji yang kelasnya ditebak benar. <b>Macro F1</b> adalah
    rata-rata F1 per kelas tanpa memperhitungkan jumlah gambar, sehingga kelas kecil sama pentingnya dengan
    kelas besar.</p>
    </div>
    <div class="notice warn">
    <b>Perlu diingat:</b> Gambar PlantVillage diambil dengan latar terkontrol dan pembagian acak
    memungkinkan gambar dari daun yang sama atau sangat mirip muncul di data latih maupun uji. Akibatnya,
    angka uji cenderung optimis dibanding performa aktual di lapangan.
    </div>
    """)

    if not metrics_from_file:
        st.caption(
            "Nilai jumlah data masih perkiraan. Letakkan metrics.json dari output notebook di folder "
            "models/ untuk menampilkan angka eksak, termasuk latensi."
        )

    # Artefak evaluasi opsional dari notebook
    if REPORT_PATH.exists():
        try:
            rep = pd.read_csv(REPORT_PATH, index_col=0)
            rep = rep.drop(index=[i for i in ["accuracy", "macro avg", "weighted avg"] if i in rep.index])
            rep = rep[["precision", "recall", "f1-score", "support"]]
            st.markdown("#### F1-Score per Kelas")
            chart_df = rep[["f1-score"]].sort_values("f1-score")
            show_bar_chart(chart_df)
            st.markdown("##### Lima Kelas dengan F1 Terendah")
            low = rep.sort_values("f1-score").head(5).copy()
            low.index.name = "Kelas"
            show_dataframe(
                low.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"})
            )
        except (OSError, ValueError, KeyError):
            pass

    imgs = [(CURVES_PATH, "Learning Curves"), (CM_PATH, "Confusion Matrix (Test Split)"),
            (DIST_PATH, "Distribusi Data Latih per Kelas")]
    imgs = [(p, c) for p, c in imgs if p.exists()]
    if imgs:
        st.markdown("#### Visualisasi Evaluasi")
        for path, caption in imgs:
            show_image(str(path), caption)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 - Panduan dan Batasan
# ──────────────────────────────────────────────────────────────────────────────
with tab_guide:
    g1, g2 = st.columns(2, gap="large")
    with g1:
        md("""
        <div class="card">
        <div class="sec-title" style="margin-bottom:.5rem">📷 Tips Memotret Daun</div>
        <p class="body">
        • Pastikan Anda hanya memotret satu helai daun yang posisinya memenuhi sekitar 70 persen dari bingkai kamera.<br>
        • Gunakan latar belakang polos yang kontras seperti selembar kertas putih agar objek daun menonjol dan terlihat jelas.<br>
        • Manfaatkan cahaya alami yang merata serta hindari bayangan gelap dan penggunaan lampu kilat kamera.<br>
        • Arahkan fokus kamera agar gambar terlihat tajam terutama pada bagian daun yang menunjukkan gejala kelainan.<br>
        • Hindari menggunakan gambar berupa kolase, tumpukan banyak daun sekaligus, atau daun basah yang memantulkan cahaya.<br>
        • Ambil gambar tepat dari atas secara tegak lurus menghadap permukaan daun.
        </p>
        </div>
        """)
    with g2:
        md("""
        <div class="card">
        <div class="sec-title" style="margin-bottom:.5rem">⚠️ Batasan model</div>
        <p class="body">
        • Hanya mengenali 14 jenis tanaman dan 38 kelas PlantVillage.<br>
        • Hanya memproses bagian daun sehingga buah, batang, maupun akar sama sekali tidak akan dianalisis.<br>
        • Tidak dirancang mendeteksi kekurangan nutrisi, kerusakan fisik, atau residu pestisida.<br>
        • Dilatih pada gambar berlatar terkontrol, jadi foto lapangan bisa kurang akurat.<br>
        • Selalu memilih salah satu kelas, termasuk saat gambar bukan daun.<br>
        • Hasil deteksi ini murni sebagai referensi awal dan tidak bisa dijadikan pengganti diagnosis dari ahli.
        </p>
        </div>
        """)

    st.markdown("#### Pertanyaan umum")
    with st.expander("Mengapa daun sehat saya terdeteksi sakit (atau sebaliknya)?"):
        st.write(
            "Model mempelajari pola visual dari dataset berlatar terkontrol. Bayangan, pantulan cahaya, "
            "debu, atau latar yang ramai dapat disalahartikan sebagai gejala. Cek peringkat kedua dan "
            "ketiga, coba foto ulang dengan cahaya lebih baik, dan konfirmasi ke ahli bila ragu."
        )
    with st.expander("Tanaman saya (mis. cabai rawit, padi, singkong) tidak ada di daftar. Bagaimana?"):
        st.write(
            "Model hanya dilatih pada 14 jenis tanaman. "
            "Untuk tanaman di luar daftar, model tetap memberi jawaban dari kelas terdekat, "
            "tetapi hasilnya tidak dapat diandalkan."
        )
    with st.expander("Apa arti 'skor model'? Apakah itu peluang sakitnya tanaman?"):
        st.write(
            "Skor adalah keluaran softmax. Angka ini menunjukkan seberapa kuat model condong ke suatu "
            "kelas dibanding kelas lain, bukan peluang sakitnya tanaman. Skor 99% tidak berarti "
            "99% pasti benar, terutama untuk gambar di luar distribusi data latih."
        )
    with st.expander("Apakah foto saya disimpan?"):
        st.write(
            "Aplikasi ini hanya memproses gambar di memori selama sesi berjalan dan tidak menyimpannya "
            "ke disk. Tombol 'Unduh hasil' hanya berisi teks prediksi, bukan gambar."
        )
    with st.expander("Apa yang sebaiknya saya lakukan setelah mendapat hasil?"):
        st.write(
            "Anggap hasil sebagai petunjuk awal. Bandingkan dengan gejala yang terlihat langsung, "
            "periksa beberapa daun lain, lalu konsultasikan ke penyuluh pertanian atau dinas pertanian "
            "setempat sebelum tindakan lebih lanjut."
        )

    md(f"""
    <div class="card soft" style="margin-top:1rem">
    <div class="sec-title">Sumber &amp; rujukan</div>
    <p class="body">
    • Dataset: <a href="{DATASET_URL}" target="_blank" rel="noopener">PlantVillage Dataset (Kaggle)</a><br>
    • Hughes &amp; Salathé (2015), <i>An open access repository of images on plant health to enable the
    development of mobile disease diagnostics</i>, arXiv:1511.08060<br>
    </p>
    </div>
    """)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
md(f"""
<div class="footer">
<b>LeafySense</b> · Proyek portofolio · TensorFlow/Keras · PlantVillage<br>
Model ini dilatih pada dataset PlantVillage yang terkontrol, 
sehingga bukan jaminan performa aktual di lapangan.
</div>
""")