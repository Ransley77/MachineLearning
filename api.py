import io
import json
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input as xception_prep
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep

app = FastAPI(title="ChromaLab Single-Color API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_pna_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

model_knn = joblib.load("model_knn.pkl")
model_rf = joblib.load("model_randomforest.pkl")
model_xception = tf.keras.models.load_model("model_xception.keras")
model_resnet = tf.keras.models.load_model("model_resnet50.keras")

with open("label_warna.json", "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

@app.post("/predict")
async def predict_color(file: UploadFile = File(...), model_name: str = Form("ResNet50 (DL)")):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    arr_64 = np.array(img.resize((64, 64)))
    mean_rgb = arr_64.mean(axis=(0, 1)).astype(int)
    r, g, b = int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])
    hex_code = f"#{r:02X}{g:02X}{b:02X}"

    if "(ML)" in model_name:
        rgb_norm = arr_64.mean(axis=(0, 1)) / 255.0
        hsv_norm = np.array(img.resize((64, 64)).convert("HSV")).mean(axis=(0, 1)) / 255.0
        fitur = np.concatenate([rgb_norm, hsv_norm]).reshape(1, -1)

        clf = model_knn if "KNN" in model_name else model_rf
        pred_label = str(clf.predict(fitur)[0])
        proba = float(np.max(clf.predict_proba(fitur)) * 100)
    else:
        arr_224 = np.expand_dims(np.array(img.resize((224, 224)), dtype=np.float32), axis=0)
        if "Xception" in model_name:
            arr_prep = xception_prep(arr_224)
            dl_model = model_xception
        else:
            arr_prep = resnet_prep(arr_224)
            dl_model = model_resnet

        preds = dl_model.predict(arr_prep, verbose=0)[0]
        idx = int(np.argmax(preds))
        pred_label = label_map[idx]
        proba = float(preds[idx] * 100)

    return {
        "color_name": pred_label,
        "rgb": [r, g, b],
        "hex": hex_code,
        "confidence": round(proba, 2),
        "model_used": model_name,
    }