import json
import time
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input as xception_prep
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep

# Konfigurasi Halaman
st.set_page_config(page_title="ChromaLab — Color Classifier", page_icon="🎨", layout="centered")

# CSS Minimalis agar tampilan ringkas dan bersih seperti desain sebelumnya
st.markdown("""
<style>
    .stApp { background-color: #FAF9F5; }
    .block-container { max-width: 520px; padding-top: 2.5rem; padding-bottom: 2.5rem; }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E0D8;
        border-radius: 12px;
        padding: 18px;
        margin-top: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .swatch-circle {
        width: 46px;
        height: 46px;
        border-radius: 50%;
        border: 1.5px solid rgba(0,0,0,0.15);
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache model agar hanya dimuat 1x & sudah warm-up sejak awal
@st.cache_resource
def load_assets():
    knn = joblib.load("model_knn.pkl")
    rf = joblib.load("model_randomforest.pkl")
    xception = tf.keras.models.load_model("model_xception.keras")
    resnet = tf.keras.models.load_model("model_resnet50.keras")

    # Warm-up TensorFlow agar klik pertama langsung cepat
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    xception(dummy, training=False)
    resnet(dummy, training=False)

    with open("label_warna.json", "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    return knn, rf, xception, resnet, labels

model_knn, model_rf, model_xception, model_resnet, label_map = load_assets()

# Header & Dropdown Pilihan Model (Sejajar seperti desain sebelumnya)
col_title, col_model = st.columns([1.1, 1])
with col_title:
    st.markdown("<h3 style='margin:0; padding-top:6px; letter-spacing:1px; color:#141414;'>CHROMALAB</h3>", unsafe_allow_html=True)
with col_model:
    selected_model = st.selectbox(
        "Model AI",
        ["ResNet50 (DL)", "Random Forest (ML)", "KNN (ML)", "Xception (DL)"],
        label_visibility="collapsed",
    )

# Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar sampel", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    # Mulai hitung waktu running
    start_time = time.perf_counter()

    # Hitung rata-rata RGB & kode HEX aktual
    arr_64 = np.array(img.resize((64, 64)))
    mean_rgb = arr_64.mean(axis=(0, 1)).astype(int)
    r, g, b = int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])
    hex_code = f"#{r:02X}{g:02X}{b:02X}"

    # Proses Klasifikasi sesuai model yang dipilih
    if "(ML)" in selected_model:
        rgb_norm = arr_64.mean(axis=(0, 1)) / 255.0
        hsv_norm = np.array(img.resize((64, 64)).convert("HSV")).mean(axis=(0, 1)) / 255.0
        fitur = np.concatenate([rgb_norm, hsv_norm]).reshape(1, -1)

        clf = model_knn if "KNN" in selected_model else model_rf
        pred_label = str(clf.predict(fitur)[0])
        confidence = float(np.max(clf.predict_proba(fitur)) * 100)
    else:
        arr_224 = np.expand_dims(np.array(img.resize((224, 224)), dtype=np.float32), axis=0)
        if "Xception" in selected_model:
            arr_prep = xception_prep(arr_224)
            dl_model = model_xception
        else:
            arr_prep = resnet_prep(arr_224)
            dl_model = model_resnet

        preds = dl_model(arr_prep, training=False).numpy()[0]
        idx = int(np.argmax(preds))
        pred_label = label_map[idx]
        confidence = float(preds[idx] * 100)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    time_str = f"{elapsed_ms:.1f} ms ({elapsed_ms / 1000:.3f} dtk)"

    # Kartu Hasil Klasifikasi Ringkas
    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex; align-items:center; gap:14px;">
            <div class="swatch-circle" style="background-color:{hex_code};"></div>
            <div>
                <div style="font-size:18px; font-weight:800; color:#141414;">{pred_label.upper()}</div>
                <div style="font-size:13px; color:#555555;">
                    Confidence <b>{confidence:.2f}%</b>  •  <code>{hex_code}</code>  •  {time_str}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Panel Detail Spesifikasi (Pengganti Pop-up Modal)
    with st.expander("Lihat Detail Spesifikasi & Salin Kode"):
        st.write(f"**Kategori Warna:** {pred_label.upper()}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"**RGB Value:** `{r}, {g}, {b}`")
        st.write(f"**Model AI:** {selected_model}")
        st.write(f"**Waktu Running:** {time_str}")
        st.caption("Klik ikon salin di kanan kotak kode HEX berikut:")
        st.code(hex_code, language="text")