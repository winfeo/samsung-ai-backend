import os
import json
import io
import tempfile
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import ClapModel, ClapProcessor
from catboost import CatBoostClassifier
from typing import List, Dict, Any

MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 48000

app = FastAPI(
    title="Mood & Genre Classifier API",
    description="Анализ настроения и жанра аудиотреков с помощью CLAP + CatBoost",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clap_model = None
clap_processor = None
group_models = []
genre_models = []
METADATA = None
CALIBRATORS = None
MOOD_GROUPS = []
GENRE_CATEGORIES = []
MOOD_THRESHOLDS = {}
GENRE_THRESHOLDS = {}
EXPECTED_FEATURES = 782

#Модели
def load_models():
    global clap_model, clap_processor, group_models, genre_models
    global METADATA, CALIBRATORS, MOOD_GROUPS, GENRE_CATEGORIES
    global MOOD_THRESHOLDS, GENRE_THRESHOLDS

    print("=" * 50)
    print("ЗАГРУЗКА МОДЕЛЕЙ")
    print("=" * 50)

    print("1. Загрузка CLAP...")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(DEVICE)
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    clap_model.eval()
    print("   ✓ CLAP загружен")

    print("2. Загрузка метаданных...")
    with open(os.path.join(MODELS_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        METADATA = json.load(f)

    MOOD_GROUPS = METADATA["mood_groups"]
    GENRE_CATEGORIES = METADATA["genre_categories"]
    MOOD_THRESHOLDS = METADATA["mood_thresholds"]
    GENRE_THRESHOLDS = METADATA["genre_thresholds"]
    print(f"   ✓ Настроений: {len(MOOD_GROUPS)}")
    print(f"   ✓ Жанров: {len(GENRE_CATEGORIES)}")

    print("3. Загрузка CatBoost моделей (настроения)...")
    group_models = []
    for i, group in enumerate(MOOD_GROUPS):
        model_path = os.path.join(MODELS_DIR, f"group_{i}.cbm")
        if os.path.exists(model_path):
            model = CatBoostClassifier()
            model.load_model(model_path)
            group_models.append(model)
            print(f"   ✓ {group}")
        else:
            print(f"   ✗ {group} не найден")

    print("4. Загрузка CatBoost моделей (жанры)...")
    genre_models = []
    for i, genre in enumerate(GENRE_CATEGORIES):
        model_path = os.path.join(MODELS_DIR, f"genre_{i}.cbm")
        if os.path.exists(model_path):
            model = CatBoostClassifier()
            model.load_model(model_path)
            genre_models.append(model)
            print(f"   ✓ {genre}")
        else:
            print(f"   ✗ {genre} не найден")

    print("5. Загрузка калибраторов...")
    cal_path = os.path.join(MODELS_DIR, "calibrators.json")
    if os.path.exists(cal_path):
        with open(cal_path, "r", encoding="utf-8") as f:
            CALIBRATORS = json.load(f)
        print("   ✓ Калибраторы загружены")
    else:
        CALIBRATORS = None
        print("   ⚠ Калибраторы не найдены")

    print("=" * 50)
    print("СЕРВЕР ГОТОВ!")
    print("=" * 50)


load_models()


#HC-ПРИЗНАКИ
def extract_hc_features(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    features = []

    #TODO сделать норм признаки потом
    # 1. Tempo (упрощённо)
    features.append(120.0)

    # 2-3. RMS mean и std
    rms = torch.sqrt(torch.mean(waveform ** 2)).item()
    rms_std = torch.std(waveform).item()
    features.extend([rms, rms_std])

    # 4-6. Спектральные признаки (упрощённые)
    features.extend([0.0, 0.0, 0.0])

    # 7. ZCR
    zcr = ((waveform[:, 1:] * waveform[:, :-1]) < 0).float().mean().item()
    features.append(zcr)

    # 8-12. MFCC (заглушки)
    features.extend([0.0] * 5)

    # 13. Chroma (заглушка)
    features.append(0.0)

    # 14. Harmonic ratio (заглушка)
    features.append(0.0)

    features = np.array(features, dtype=np.float32)

    if len(features) < 14:
        features = np.pad(features, (0, 14 - len(features)))
    elif len(features) > 14:
        features = features[:14]

    return features


def apply_calibration(prob: float, group_name: str) -> float:
    if CALIBRATORS is None or group_name not in CALIBRATORS:
        return prob

    cal = CALIBRATORS[group_name]
    x_points = cal.get("x", [])
    y_points = cal.get("y", [])

    if not x_points or not y_points:
        return prob

    if prob <= x_points[0]:
        return y_points[0]
    if prob >= x_points[-1]:
        return y_points[-1]

    for i in range(len(x_points) - 1):
        if x_points[i] <= prob <= x_points[i + 1]:
            t = (prob - x_points[i]) / (x_points[i + 1] - x_points[i])
            return y_points[i] + t * (y_points[i + 1] - y_points[i])

    return prob


# ============================================================
@app.get("/")
async def root():
    return {
        "message": "Mood & Genre Classifier API",
        "mood_groups": MOOD_GROUPS,
        "genre_categories": GENRE_CATEGORIES,
        "status": "ready"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        suffix = os.path.splitext(file.filename)[1]
        if not suffix:
            suffix = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        import librosa
        waveform_np, sample_rate = librosa.load(tmp_path, sr=None, mono=True)

        if sample_rate != TARGET_SR:
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR

        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
        os.unlink(tmp_path)

        audio_array = waveform.squeeze().cpu().numpy()
        inputs = clap_processor(
            audios=[audio_array],
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            clap_emb = clap_model.get_audio_features(**inputs).cpu().numpy()[0]

        hc_features = extract_hc_features(waveform, sample_rate)
        features = np.concatenate([clap_emb, hc_features]).reshape(1, -1)

        if features.shape[1] != EXPECTED_FEATURES:
            if features.shape[1] < EXPECTED_FEATURES:
                features = np.pad(features, ((0, 0), (0, EXPECTED_FEATURES - features.shape[1])))
            else:
                features = features[:, :EXPECTED_FEATURES]

        mood_results = []
        for i, model in enumerate(group_models):
            prob = model.predict_proba(features)[0, 1]
            prob_cal = apply_calibration(prob, MOOD_GROUPS[i])
            threshold = MOOD_THRESHOLDS.get(MOOD_GROUPS[i], 0.5)
            if prob_cal > threshold:
                mood_results.append({
                    "tag": MOOD_GROUPS[i],
                    "probability": round(float(prob_cal), 4)
                })

        genre_results = []
        for i, model in enumerate(genre_models):
            prob = model.predict_proba(features)[0, 1]
            prob_cal = apply_calibration(prob, GENRE_CATEGORIES[i])
            threshold = GENRE_THRESHOLDS.get(GENRE_CATEGORIES[i], 0.5)
            if prob_cal > threshold:
                genre_results.append({
                    "tag": GENRE_CATEGORIES[i],
                    "probability": round(float(prob_cal), 4)
                })

        mood_results.sort(key=lambda x: x["probability"], reverse=True)
        genre_results.sort(key=lambda x: x["probability"], reverse=True)

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "sample_rate": sample_rate,
            "clap_dim": len(clap_emb),
            "hc_dim": len(hc_features),
            "total_dim": features.shape[1],
            "mood": mood_results[:5],
            "genre": genre_results[:5],
            "mood_count": len(mood_results),
            "genre_count": len(genre_results)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )