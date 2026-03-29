import io
import pickle
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Model, Wav2Vec2Processor

BASE_DIR = Path(__file__).resolve().parents[1]
try:
    from src.pipeline_utils import CHUNK_SEC, TARGET_SR, chunk_waveform, GRUSequenceClassifier
except ImportError:
    from pipeline_utils import CHUNK_SEC, TARGET_SR, chunk_waveform, GRUSequenceClassifier

FEATURE_DIR = BASE_DIR / "data" / "features_turn_level"
MODEL_DIR = BASE_DIR / "models"
SCALER_PATH = MODEL_DIR / "inference_scaler.pkl"
HF_CACHE = BASE_DIR / "data" / "hf_cache"
EXTRACT_LAYER = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.53

torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)


@lru_cache(maxsize=1)
def load_artifacts():
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
        local_files_only=HF_CACHE.exists(),
    )

    wav2vec2 = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
        local_files_only=HF_CACHE.exists(),
    ).to(DEVICE).eval()

    for param in wav2vec2.parameters():
        param.requires_grad = False

    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as fh:
            scaler = pickle.load(fh)
    else:
        scaler = StandardScaler()
        feature_files = sorted(FEATURE_DIR.glob("*_chunk_embeddings.csv"))
        if not feature_files:
            raise FileNotFoundError(
                f"No inference scaler found at {SCALER_PATH} and no feature CSVs found in {FEATURE_DIR}."
            )

        for file_path in feature_files:
            df = pd.read_csv(file_path)
            feature_cols = sorted(
                [c for c in df.columns if c.startswith("w2v_")],
                key=lambda c: int(c.split("_")[1]),
            )
            scaler.partial_fit(df[feature_cols].values.astype(np.float32))

    model_paths = sorted(MODEL_DIR.glob("best_bigru_fold*.pt"))
    if not model_paths:
        raise FileNotFoundError(
            f"No trained models found in {MODEL_DIR}. Run src/03_train_sequence.py first."
        )

    return processor, wav2vec2, scaler, tuple(model_paths)


def load_sequence_model(model_path: Path) -> GRUSequenceClassifier:
    model = GRUSequenceClassifier(
        input_dim=768,
        hidden_dim=128,
        num_layers=2,
        dropout=0.35,
    )
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def load_transcript_bytes(raw_bytes: bytes) -> pd.DataFrame:
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    first_line = raw_text.splitlines()[0].strip() if raw_text.splitlines() else ""

    if "start_timestop_timespeakervalue" in first_line.replace(" ", "").lower():
        transcript = pd.read_csv(
            io.StringIO(raw_text),
            sep="\t",
            skiprows=1,
            names=["start_time", "stop_time", "speaker", "value"],
        )
    else:
        try:
            transcript = pd.read_csv(io.StringIO(raw_text), sep="\t")
        except Exception:
            transcript = pd.read_csv(io.StringIO(raw_text), sep=",")

    transcript.columns = [c.lower().strip() for c in transcript.columns]

    if "speaker" not in transcript.columns:
        collapsed_cols = "".join(transcript.columns).replace(" ", "")
        if "start_timestop_timespeakervalue" in collapsed_cols:
            transcript = pd.read_csv(
                io.StringIO(raw_text),
                sep="\t",
                skiprows=1,
                names=["start_time", "stop_time", "speaker", "value"],
            )
            transcript.columns = [c.lower().strip() for c in transcript.columns]

    return transcript


def isolate_participant_audio(waveform: np.ndarray, sr: int, transcript: pd.DataFrame) -> np.ndarray:
    start_col = "start_time" if "start_time" in transcript.columns else "start"
    stop_col = "stop_time" if "stop_time" in transcript.columns else "end_time"

    if "speaker" not in transcript.columns:
        raise ValueError("Transcript does not contain a speaker column.")

    participant_rows = transcript[
        transcript["speaker"].astype(str).str.lower().str.contains("participant", na=False)
    ]

    if participant_rows.empty:
        raise ValueError("Transcript does not contain participant turns.")

    total_len = len(waveform)
    segments = []

    for _, row in participant_rows.iterrows():
        start = int(float(row[start_col]) * sr)
        end = int(float(row[stop_col]) * sr)

        start = max(0, min(start, total_len))
        end = max(0, min(end, total_len))

        if end <= start:
            continue

        seg = waveform[start:end]
        if len(seg) > 0:
            segments.append(seg)

    if not segments:
        raise ValueError("Transcript filtering produced no valid participant audio.")

    return np.concatenate(segments).astype(np.float32)


def load_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    waveform, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    return waveform.astype(np.float32), sr


@torch.no_grad()
def extract_chunk_embedding(chunk: np.ndarray, processor, wav2vec2) -> np.ndarray | None:
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < 1e-4:
        return None

    inputs = processor(
        chunk,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=False,
    )

    input_values = inputs.input_values.to(DEVICE)
    outputs = wav2vec2(input_values)
    layer9 = outputs.hidden_states[EXTRACT_LAYER]
    embedding = layer9.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    if np.isnan(embedding).any() or np.isinf(embedding).any():
        return None

    return embedding


def build_sequence_features(waveform: np.ndarray, sr: int, processor, wav2vec2):
    if sr != TARGET_SR:
        waveform_t = torch.from_numpy(waveform).unsqueeze(0)
        waveform_t = torchaudio.functional.resample(waveform_t, sr, TARGET_SR)
        waveform = waveform_t.squeeze(0).numpy()
        sr = TARGET_SR

    chunks = chunk_waveform(waveform, sr)
    embeddings = []
    skipped = 0

    for chunk in chunks:
        embedding = extract_chunk_embedding(chunk, processor, wav2vec2)
        if embedding is None:
            skipped += 1
            continue
        embeddings.append(embedding)

    if not embeddings:
        raise ValueError("No valid chunks were extracted from the uploaded audio.")

    features = np.vstack(embeddings).astype(np.float32)
    metadata = {
        "total_chunks": len(chunks),
        "valid_chunks": len(embeddings),
        "skipped_chunks": skipped,
        "duration_sec": len(waveform) / sr,
    }

    return features, metadata


@torch.no_grad()
def run_inference(features: np.ndarray, scaler, model_paths, threshold: float = DEFAULT_THRESHOLD) -> dict:
    features = scaler.transform(features).astype(np.float32)

    x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=DEVICE)

    probs = []
    for model_path in model_paths:
        model = load_sequence_model(model_path)
        logit = model(x, lengths)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)
        del model

    probs = np.asarray(probs, dtype=np.float32)
    final_prob = float(probs.mean())

    return {
        "probability": final_prob,
        "prediction": int(final_prob >= threshold),
        "label": "Depression Detected" if final_prob >= threshold else "No Depression Detected",
        "confidence_band": (
            "high"
            if abs(final_prob - threshold) >= 0.20
            else "moderate"
            if abs(final_prob - threshold) >= 0.10
            else "low"
        ),
        "fold_probabilities": probs.tolist(),
        "threshold": threshold,
    }


def predict_from_upload(audio_bytes: bytes, transcript_bytes: bytes | None = None, threshold: float = DEFAULT_THRESHOLD):
    processor, wav2vec2, scaler, model_paths = load_artifacts()

    started_at = time.time()
    waveform, sr = load_audio_bytes(audio_bytes)

    transcript_used = False
    if transcript_bytes:
        transcript = load_transcript_bytes(transcript_bytes)
        waveform = isolate_participant_audio(waveform, sr, transcript)
        transcript_used = True

    features, metadata = build_sequence_features(waveform, sr, processor, wav2vec2)
    result = run_inference(features, scaler, model_paths, threshold=threshold)
    result["metadata"] = metadata
    result["transcript_used"] = transcript_used
    result["processing_time_sec"] = round(time.time() - started_at, 2)

    return result
