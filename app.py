#!/usr/bin/env python3
import os
import json
import io
import tempfile
import sys
import traceback
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import aiohttp
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from catboost import CatBoostClassifier
from transformers import ClapModel, ClapProcessor
from hear21passt.base import get_basic_model
import joblib

from hc_features import extract_hc_gpu_enhanced

MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR_CLAP = 48000
TARGET_SR_PASST = 32000
MAX_DURATION_CLAP = 30.0
MAX_DURATION_PASST = 10.0

#настроения
clap_model = None
clap_processor = None
mood_group_models = []
mood_group_calibrators = []
num_mood_groups = 0
mood_groups_names = []

#жанры
passt_model = None
genre_stage1 = None
genre_stage2_rhythmic = None
genre_stage2_melodic = None
genre_cal_rhythmic = []
genre_cal_melodic = []
genre_meta = {}

app = FastAPI(
    title="Mood & Genre Classifier API",
    description="Анализ настроения и жанра аудиотреков (CLAP + PaSST + CatBoost)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#загрузка моделей
def load_all_models():
    global clap_model, clap_processor
    global mood_group_models, mood_group_calibrators, num_mood_groups, mood_groups_names
    global passt_model, genre_stage1, genre_stage2_rhythmic, genre_stage2_melodic
    global genre_cal_rhythmic, genre_cal_melodic, genre_meta

    clap_model = ClapModel.from_pretrained(os.path.join(MODELS_DIR, "clap_model")).to(DEVICE)
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    clap_model.eval()
    for p in clap_model.parameters():
        p.requires_grad = False

    with open(os.path.join(MODELS_DIR, "mood_mapping.json"), "r") as f:
        mood_map = json.load(f)
    mood_groups_names = mood_map["all_mood_groups"]
    num_mood_groups = mood_map["num_mood_groups"]

    MOOD_ENSEMBLE = 3
    mood_group_models = []
    mood_group_calibrators = []
    for i in range(num_mood_groups):
        ensemble = []
        for e in range(MOOD_ENSEMBLE):
            model = CatBoostClassifier()
            model.load_model(os.path.join(MODELS_DIR, "catboost_models", f"group_{i}_ens{e}.cbm"))
            ensemble.append(model)
        mood_group_models.append(ensemble)
        cal = joblib.load(os.path.join(MODELS_DIR, "catboost_models", f"group_cal_{i}.pkl"))
        mood_group_calibrators.append(cal)

    passt_model = get_basic_model(mode="embed_only").to(DEVICE)
    passt_model.eval()
    for p in passt_model.parameters():
        p.requires_grad = False

    stage1 = CatBoostClassifier()
    stage1.load_model(os.path.join(MODELS_DIR, "fma_two_stage", "stage1.cbm"))
    stage2_r = CatBoostClassifier()
    stage2_r.load_model(os.path.join(MODELS_DIR, "fma_two_stage", "stage2_rhythmic.cbm"))
    stage2_m = CatBoostClassifier()
    stage2_m.load_model(os.path.join(MODELS_DIR, "fma_two_stage", "stage2_melodic.cbm"))

    with open(os.path.join(MODELS_DIR, "fma_two_stage", "meta.json"), "r") as f:
        meta = json.load(f)
    rhythmic_indices = meta["rhythmic_indices"]
    melodic_indices = meta["melodic_indices"]
    rhythmic_classes = meta["rhythmic_classes"]
    melodic_classes = meta["melodic_classes"]

    cal_rhythmic = [joblib.load(os.path.join(MODELS_DIR, "fma_two_stage", f"cal_rhythmic_{i}.pkl"))
                    for i in range(len(rhythmic_indices))]
    cal_melodic = [joblib.load(os.path.join(MODELS_DIR, "fma_two_stage", f"cal_melodic_{i}.pkl"))
                   for i in range(len(melodic_indices))]

    with open(os.path.join(MODELS_DIR, "genre_mapping.json"), "r") as f:
        genre_map = json.load(f)
    genre_categories = genre_map["genre_categories"]
    cat_to_idx = genre_map["cat_to_idx"]
    idx_to_cat = {int(k): v for k, v in genre_map["idx_to_cat"].items()}
    num_genre_cats = genre_map["num_genre_cats"]

    genre_stage1 = stage1
    genre_stage2_rhythmic = stage2_r
    genre_stage2_melodic = stage2_m
    genre_cal_rhythmic = cal_rhythmic
    genre_cal_melodic = cal_melodic
    genre_meta = {
        "rhythmic_indices": rhythmic_indices,
        "melodic_indices": melodic_indices,
        "rhythmic_classes": rhythmic_classes,
        "melodic_classes": melodic_classes,
        "all_categories": genre_categories,
        "cat_to_idx": cat_to_idx,
        "idx_to_cat": idx_to_cat,
        "num_genre_cats": num_genre_cats,
        "all_rhythmic_idx": [cat_to_idx[c] for c in rhythmic_classes],
        "all_melodic_idx": [cat_to_idx[c] for c in melodic_classes],
    }

def is_uniform_track(waveform_48khz, window_size=15, variation_thresh=0.15, range_thresh=0.3):
    win_samples = 48000 * window_size
    step = win_samples // 2
    total = len(waveform_48khz)
    if total < win_samples * 2:
        return True
    rms_vals = []
    for start in range(0, total - win_samples, step):
        window = waveform_48khz[start:start + win_samples]
        rms_vals.append(torch.sqrt(torch.mean(window ** 2)).item())
    rms_vals = np.array(rms_vals)
    rms_mean = np.mean(rms_vals)
    if rms_mean == 0:
        return True
    variation = np.std(rms_vals) / rms_mean
    rms_range = (np.max(rms_vals) - np.min(rms_vals)) / rms_mean
    return (variation < variation_thresh) and (rms_range < range_thresh)

def find_climax_window(waveform_48khz, window_size=30):
    win_samples = 48000 * window_size
    step = win_samples // 3
    total = len(waveform_48khz)
    if total <= win_samples:
        return waveform_48khz
    max_energy = 0
    best_start = 0
    for start in range(0, total - win_samples, step):
        window = waveform_48khz[start:start + win_samples]
        energy = torch.sqrt(torch.mean(window ** 2))
        if energy > max_energy:
            max_energy = energy
            best_start = start
    return waveform_48khz[best_start:best_start + win_samples]

def extract_clap_features(audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
    waveform = torch.tensor(audio_array, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    if sample_rate != TARGET_SR_CLAP:
        resampler = T.Resample(sample_rate, TARGET_SR_CLAP).to(DEVICE)
        waveform = resampler(waveform)
    max_samples = int(TARGET_SR_CLAP * MAX_DURATION_CLAP)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    elif waveform.shape[1] < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[1]))
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    inputs = clap_processor(audios=[waveform.squeeze(0).cpu().numpy()], sampling_rate=TARGET_SR_CLAP, return_tensors="pt")
    mel = inputs["input_features"].to(DEVICE)
    with torch.no_grad():
        emb = clap_model.audio_model.audio_encoder(
            mel, is_longer=torch.zeros(1, dtype=torch.long, device=DEVICE)
        ).pooler_output.squeeze(0).cpu().numpy()
    hc = extract_hc_gpu_enhanced(waveform.squeeze(0))
    return np.concatenate([emb, hc])

def extract_passt_features(audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
    waveform = torch.tensor(audio_array, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    if sample_rate != TARGET_SR_CLAP:
        resampler_48 = T.Resample(sample_rate, TARGET_SR_CLAP).to(DEVICE)
        waveform = resampler_48(waveform)
    hc = extract_hc_gpu_enhanced(waveform.squeeze(0))

    resampler_32 = T.Resample(TARGET_SR_CLAP, TARGET_SR_PASST).to(DEVICE)
    waveform_32 = resampler_32(waveform)
    max_samples = int(TARGET_SR_PASST * MAX_DURATION_PASST)
    if waveform_32.shape[1] > max_samples:
        waveform_32 = waveform_32[:, :max_samples]
    elif waveform_32.shape[1] < max_samples:
        waveform_32 = torch.nn.functional.pad(waveform_32, (0, max_samples - waveform_32.shape[1]))
    peak = waveform_32.abs().max()
    if peak > 0:
        waveform_32 = waveform_32 / peak
    with torch.no_grad():
        emb = passt_model(waveform_32).squeeze(0).cpu().numpy()
    return np.concatenate([emb, hc])

#предсказание настроений
def predict_moods(audio_array: np.ndarray, sample_rate: int, min_prob=0.15, top_k=5) -> List[Dict[str, Any]]:
    waveform_48k = torch.tensor(audio_array, dtype=torch.float32)
    if waveform_48k.dim() == 2:
        waveform_48k = waveform_48k.mean(dim=1)
    waveform_48k = T.Resample(sample_rate, 48000)(waveform_48k.unsqueeze(0)).squeeze(0)

    uniform = is_uniform_track(waveform_48k)
    if uniform:
        win_samples = 48000 * 30
        total = len(waveform_48k)
        start_win = waveform_48k[:win_samples]
        mid_start = max(0, total // 2 - win_samples // 2)
        mid_win = waveform_48k[mid_start:mid_start + win_samples]
        prob1 = _predict_mood_window(start_win)
        prob2 = _predict_mood_window(mid_win)
        avg_probs = (prob1 + prob2) / 2
    else:
        climax = find_climax_window(waveform_48k, window_size=30)
        avg_probs = _predict_mood_window(climax)

    sorted_idx = np.argsort(avg_probs)[::-1]
    results = []
    for idx in sorted_idx:
        prob = float(avg_probs[idx])
        if prob < min_prob:
            break
        results.append({"tag": mood_groups_names[idx], "probability": round(prob, 4)})
        if len(results) >= top_k:
            break
    if not results:
        best = int(sorted_idx[0])
        results.append({"tag": mood_groups_names[best], "probability": round(float(avg_probs[best]), 4)})
    return results

def _predict_mood_window(window_48k: torch.Tensor) -> np.ndarray:
    peak = window_48k.abs().max()
    if peak > 0:
        window_48k = window_48k / peak
    target_samples = 48000 * 30
    if len(window_48k) < target_samples:
        window_48k = torch.nn.functional.pad(window_48k, (0, target_samples - len(window_48k)))
    feat = extract_clap_features(window_48k.cpu().numpy(), 48000)
    X = feat.reshape(1, -1)

    probs = np.zeros(num_mood_groups)
    for i in range(num_mood_groups):
        raw = np.array([m.predict_proba(X)[0, 1] for m in mood_group_models[i]]).mean()
        probs[i] = mood_group_calibrators[i].predict(np.array([[raw]]))[0]
    return probs

#предсказание жанра
def predict_genre(audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    waveform_48k = torch.tensor(audio_array, dtype=torch.float32)
    if waveform_48k.dim() == 2:
        waveform_48k = waveform_48k.mean(dim=1)
    waveform_48k = T.Resample(sample_rate, 48000)(waveform_48k.unsqueeze(0)).squeeze(0)
    total = len(waveform_48k)
    win_samples = 48000 * 10

    if total <= win_samples:
        windows = [waveform_48k]
    else:
        windows = [
            waveform_48k[:win_samples],
            waveform_48k[-win_samples:],
            find_climax_window(waveform_48k, window_size=10)
        ]

    feats = []
    for w in windows:
        if len(w) < win_samples:
            w = torch.nn.functional.pad(w, (0, win_samples - len(w)))
        if w.abs().max() > 0:
            w = w / w.abs().max()
        feats.append(extract_passt_features(w.cpu().numpy(), 48000))
    X = np.mean(feats, axis=0).reshape(1, -1)

    is_rhythmic_prob = genre_stage1.predict_proba(X)[0, 1]
    if is_rhythmic_prob > 0.5:
        probs_raw = genre_stage2_rhythmic.predict_proba(X)[0]
        class_indices = genre_meta["all_rhythmic_idx"]
        calibrators = genre_cal_rhythmic
    else:
        probs_raw = genre_stage2_melodic.predict_proba(X)[0]
        class_indices = genre_meta["all_melodic_idx"]
        calibrators = genre_cal_melodic

    full_probs = np.zeros(genre_meta["num_genre_cats"])
    for i, cls_idx in enumerate(class_indices):
        if i < len(probs_raw):
            cal_prob = calibrators[i].predict(np.array([[probs_raw[i]]]))[0]
            full_probs[cls_idx] = max(0.0, min(1.0, float(cal_prob)))

    top_idx = np.argsort(full_probs)[::-1]
    top3 = []
    for idx in top_idx[:3]:
        prob = float(full_probs[idx])
        if prob > 0:
            top3.append({"genre": genre_meta["idx_to_cat"][idx], "probability": round(prob, 4)})
    primary = genre_meta["idx_to_cat"][int(np.argmax(full_probs))] if top3 else "unknown"
    return {"primary": primary, "top3": top3}

#подбор похожих треков
class RecommendRequest(BaseModel):
    moods: List[str] = []
    genre: Optional[str] = None

class TrackInfo(BaseModel):
    title: str
    artist: str
    url: str
    tag: str

class RecommendResponse(BaseModel):
    tracks: List[TrackInfo] = []

MUSIXMATCH_API_URL = "https://api.musixmatch.com/ws/1.1/track.search"
MUSIXMATCH_API_KEY = ""

#API
@app.on_event("startup")
async def startup_event():
    load_all_models()

@app.get("/")
async def root():
    return {
        "message": "Mood & Genre Classifier API",
        "mood_groups": mood_groups_names,
        "genre_categories": genre_meta.get("all_categories", []),
        "status": "ready"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        suffix = os.path.splitext(file.filename)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        waveform, sample_rate = torchaudio.load(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Не удалось прочитать аудио: {str(e)}"})

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_array = waveform.squeeze(0).numpy()

    try:
        moods = predict_moods(audio_array, sample_rate)
        genre = predict_genre(audio_array, sample_rate)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

    return JSONResponse(content={
        "success": True,
        "filename": file.filename,
        "moods": moods,
        "genre": genre
    })


@app.post("/recommend")
async def recommend_tracks(request: RecommendRequest):
    tags = []
    if request.genre:
        tags.append(request.genre)
    tags.extend(request.moods[:3])

    if not tags:
        return RecommendResponse(tracks=[])

    queries = []
    queries.append(" ".join(tags))
    for i in range(len(tags) - 1, 0, -1):
        queries.append(" ".join(tags[:i]))

    async def search_deezer(query: str):
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.deezer.com/search", params={"q": query, "limit": 5}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
        return []

    tracks_data = []
    used_query = ""
    for q in queries:
        tracks_data = await search_deezer(q)
        if tracks_data:
            used_query = q
            break

    if not tracks_data:
        return RecommendResponse(tracks=[])

    tracks = []
    for item in tracks_data[:3]:
        tracks.append(TrackInfo(
            title=item.get("title", "Unknown"),
            artist=item.get("artist", {}).get("name", "Unknown"),
            url=item.get("link", ""),
            tag=used_query
        ))
    return RecommendResponse(tracks=tracks)

#запуск
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")