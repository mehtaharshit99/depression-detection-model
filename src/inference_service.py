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
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

BASE_DIR = Path(__file__).resolve().parents[1]
try:
    from src.pipeline_utils import CHUNK_SEC, TARGET_SR, chunk_waveform, MultimodalGRUSequenceClassifier
except ImportError:
    from pipeline_utils import CHUNK_SEC, TARGET_SR, chunk_waveform, MultimodalGRUSequenceClassifier

FEATURE_DIR = BASE_DIR / "data" / "features_multimodal"
MODEL_DIR = BASE_DIR / "models"
SCALER_PATH = MODEL_DIR / "multimodal_inference_scaler.pkl"
HF_CACHE = BASE_DIR / "data" / "hf_cache"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384
EXTRACT_LAYER = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.53

torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)


# Loads reusable inference artifacts once per server process.
@lru_cache(maxsize=1)
def load_artifacts():
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
    )

    wav2vec2 = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()

    for param in wav2vec2.parameters():
        param.requires_grad = False

    text_tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_NAME,
        cache_dir=HF_CACHE,
    )

    text_model = AutoModel.from_pretrained(
        TEXT_MODEL_NAME,
        cache_dir=HF_CACHE,
    ).to(DEVICE).eval()

    for param in text_model.parameters():
        param.requires_grad = False

    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as fh:
            scaler_payload = pickle.load(fh)
        if isinstance(scaler_payload, dict):
            scaler = scaler_payload["scaler"]
            audio_dim = int(scaler_payload.get("audio_dim", 768))
            text_dim = int(scaler_payload.get("text_dim", TEXT_DIM))
        else:
            scaler = scaler_payload
            audio_dim = 768
            text_dim = 0
    else:
        scaler = StandardScaler()
        feature_files = sorted(FEATURE_DIR.glob("*_multimodal_embeddings.csv"))
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
            text_cols = sorted(
                [c for c in df.columns if c.startswith("text_")],
                key=lambda c: int(c.split("_")[1]),
            )
            feature_cols = feature_cols + text_cols
            scaler.partial_fit(df[feature_cols].values.astype(np.float32))
        audio_dim = 768
        text_dim = max(0, len(feature_cols) - audio_dim)

    model_paths = sorted(MODEL_DIR.glob("best_multimodal_fold*.pt"))
    if not model_paths:
        raise FileNotFoundError(
            f"No trained models found in {MODEL_DIR}. Run src/03_train_sequence.py first."
        )

    return processor, wav2vec2, text_tokenizer, text_model, scaler, audio_dim, text_dim, tuple(model_paths)


# Rebuilds the sequence model and loads one fold checkpoint.
def load_sequence_model(model_path: Path, audio_dim: int, text_dim: int) -> MultimodalGRUSequenceClassifier:
    model = MultimodalGRUSequenceClassifier(
        input_dim=audio_dim,
        text_dim=text_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.35,
    )
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


# Parses an uploaded transcript CSV from raw bytes.
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


# Joins participant transcript values into one text document.
def build_participant_text(transcript: pd.DataFrame) -> str:
    if "speaker" not in transcript.columns or "value" not in transcript.columns:
        return ""

    participant_rows = transcript[
        transcript["speaker"].astype(str).str.lower().str.contains("participant", na=False)
    ]
    values = participant_rows["value"].dropna().astype(str).tolist()
    return " ".join(value.strip() for value in values if value.strip())


# Converts uploaded transcript text into a fixed-size text embedding.
@torch.no_grad()
def extract_text_embedding(text: str, text_tokenizer, text_model, text_dim: int = TEXT_DIM) -> np.ndarray:
    if text_dim <= 0:
        return np.zeros(0, dtype=np.float32)

    if not text.strip():
        return np.zeros(text_dim, dtype=np.float32)

    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    outputs = text_model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    masked_embeddings = token_embeddings * attention_mask
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1)
    embedding = (summed / counts).squeeze(0).cpu().numpy().astype(np.float32)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    if embedding.shape[0] != text_dim:
        fixed = np.zeros(text_dim, dtype=np.float32)
        copy_dim = min(text_dim, embedding.shape[0])
        fixed[:copy_dim] = embedding[:copy_dim]
        return fixed

    return embedding


# Uses transcript timestamps to keep only participant speech from the waveform.
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


# Loads uploaded audio bytes into a mono float32 waveform.
def load_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    waveform, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    return waveform.astype(np.float32), sr


# Converts one audio chunk into a Wav2Vec2 Layer-9 embedding.
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


# Builds the full participant feature sequence from uploaded audio.
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


# Runs the fold ensemble and converts probabilities into a final label.
@torch.no_grad()
def run_inference(
    audio_features: np.ndarray,
    text_features: np.ndarray,
    scaler,
    audio_dim: int,
    text_dim: int,
    model_paths,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    if text_dim > 0:
        text_repeated = np.tile(text_features.reshape(1, -1), (audio_features.shape[0], 1))
        combined_features = np.hstack([audio_features, text_repeated]).astype(np.float32)
    else:
        combined_features = audio_features.astype(np.float32)

    combined_features = scaler.transform(combined_features).astype(np.float32)
    audio_features = combined_features[:, :audio_dim]
    text_features = (
        combined_features[0, audio_dim: audio_dim + text_dim]
        if text_dim > 0
        else np.zeros(0, dtype=np.float32)
    )

    x = torch.from_numpy(audio_features).unsqueeze(0).to(DEVICE)
    text_x = torch.from_numpy(text_features).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([audio_features.shape[0]], dtype=torch.long, device=DEVICE)

    probs = []
    for model_path in model_paths:
        model = load_sequence_model(model_path, audio_dim=audio_dim, text_dim=text_dim)
        logit = model(x, text_x, lengths)
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


# Handles the complete upload-to-prediction inference workflow.
def predict_from_upload(audio_bytes: bytes, transcript_bytes: bytes | None = None, threshold: float = DEFAULT_THRESHOLD):
    processor, wav2vec2, text_tokenizer, text_model, scaler, audio_dim, text_dim, model_paths = load_artifacts()

    started_at = time.time()
    waveform, sr = load_audio_bytes(audio_bytes)

    transcript_used = False
    text_features = np.zeros(text_dim, dtype=np.float32)
    if transcript_bytes:
        transcript = load_transcript_bytes(transcript_bytes)
        participant_text = build_participant_text(transcript)
        text_features = extract_text_embedding(participant_text, text_tokenizer, text_model, text_dim=text_dim)
        waveform = isolate_participant_audio(waveform, sr, transcript)
        transcript_used = True

    features, metadata = build_sequence_features(waveform, sr, processor, wav2vec2)
    result = run_inference(
        audio_features=features,
        text_features=text_features,
        scaler=scaler,
        audio_dim=audio_dim,
        text_dim=text_dim,
        model_paths=model_paths,
        threshold=threshold,
    )
    result["metadata"] = metadata
    result["transcript_used"] = transcript_used
    result["text_used"] = bool(transcript_bytes and text_dim > 0)
    result["processing_time_sec"] = round(time.time() - started_at, 2)

    return result
