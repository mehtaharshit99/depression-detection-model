import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Model, Wav2Vec2Processor

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import CHUNK_SEC, TARGET_SR, chunk_waveform, GRUSequenceClassifier

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEATURE_DIR = BASE_DIR / "data" / "features_turn_level"
MODEL_DIR = BASE_DIR / "models"
HF_CACHE = BASE_DIR / "data" / "hf_cache"
EXTRACT_LAYER = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.53

st.set_page_config(
    page_title="Depression Detection Demo",
    page_icon=":microphone:",
    layout="wide",
)


# ─────────────────────────────────────────────
# TRAINING STATS / MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading training artifacts...")
def load_artifacts():
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
    )

    wav2vec2 = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
        output_hidden_states=True,
    ).to(DEVICE).eval()

    for param in wav2vec2.parameters():
        param.requires_grad = False

    scaler = StandardScaler()
    feature_files = sorted(FEATURE_DIR.glob("*_chunk_embeddings.csv"))
    if not feature_files:
        raise FileNotFoundError(
            f"No feature CSVs found in {FEATURE_DIR}. Run src/02_extract_features.py first."
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

    models = []
    for model_path in model_paths:
        model = GRUSequenceClassifier(
            input_dim=768,
            hidden_dim=128,
            num_layers=2,
            dropout=0.35,
        )
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        models.append(model)

    return processor, wav2vec2, scaler, models


# ─────────────────────────────────────────────
# TRANSCRIPT HELPERS
# ─────────────────────────────────────────────
def load_transcript_file(uploaded_file) -> pd.DataFrame:
    raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
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


# ─────────────────────────────────────────────
# AUDIO / FEATURE EXTRACTION
# ─────────────────────────────────────────────
def load_audio(uploaded_audio) -> tuple[np.ndarray, int]:
    waveform, sr = sf.read(uploaded_audio, dtype="float32")

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


def build_sequence_features(
    waveform: np.ndarray,
    sr: int,
    processor,
    wav2vec2,
) -> tuple[np.ndarray, dict]:
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


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(features: np.ndarray, scaler, models) -> dict:
    features = scaler.transform(features).astype(np.float32)

    x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=DEVICE)

    probs = []
    for model in models:
        logit = model(x, lengths)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)

    probs = np.asarray(probs, dtype=np.float32)
    final_prob = float(probs.mean())

    return {
        "probability": final_prob,
        "prediction": int(final_prob >= 0.5),
        "fold_probs": probs,
    }


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
threshold = DEFAULT_THRESHOLD

st.title("Depression Detection")
st.caption("Upload an audio file and run a quick screening prediction.")
st.info("This tool is for research and demo purposes only. It is not a medical diagnosis.")

audio_file = st.file_uploader("Upload Audio File", type=["wav"])
transcript_file = st.file_uploader(
    "Optional Transcript CSV",
    type=["csv"],
)

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    if st.button("Run Prediction", type="primary"):
        progress_bar = st.progress(0, text="Starting inference...")
        status_text = st.empty()

        try:
            status_text.info("Loading model artifacts...")
            processor, wav2vec2, scaler, models = load_artifacts()
            progress_bar.progress(15, text="Model artifacts loaded")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Could not load model artifacts: {e}")
            st.stop()

        try:
            t0 = time.time()

            status_text.info("Loading audio...")
            waveform, sr = load_audio(audio_file)
            progress_bar.progress(30, text="Audio loaded")

            participant_waveform = None
            if transcript_file is not None:
                status_text.info("Filtering participant-only audio from transcript...")
                transcript = load_transcript_file(transcript_file)
                participant_waveform = isolate_participant_audio(waveform, sr, transcript)
                progress_bar.progress(45, text="Participant-only audio prepared")
            else:
                progress_bar.progress(45, text="Using full uploaded audio")

            status_text.info("Extracting Wav2Vec2 sequence features...")
            full_features, full_metadata = build_sequence_features(
                waveform,
                sr,
                processor,
                wav2vec2,
            )
            progress_bar.progress(70, text="Full-audio features extracted")

            participant_result = None
            participant_metadata = None
            if participant_waveform is not None:
                participant_features, participant_metadata = build_sequence_features(
                    participant_waveform,
                    sr,
                    processor,
                    wav2vec2,
                )
                progress_bar.progress(85, text="Participant-only features extracted")
                participant_result = run_inference(participant_features, scaler, models)
            else:
                progress_bar.progress(85, text="Sequence features extracted")

            status_text.info("Running ensemble prediction...")
            full_result = run_inference(full_features, scaler, models)
            elapsed = time.time() - t0
            progress_bar.progress(100, text="Inference complete")
            status_text.success("Prediction ready")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Inference failed: {e}")
            st.stop()

        def decide(prob: float) -> str:
            return "Depression Detected" if prob >= threshold else "No Depression Detected"

        full_label = decide(full_result["probability"])

        if participant_result is not None:
            primary_result = participant_result
            primary_label = decide(participant_result["probability"])
        else:
            primary_result = full_result
            primary_label = full_label

        if primary_result["probability"] >= threshold:
            st.error(f"### {primary_label}")
        else:
            st.success(f"### {primary_label}")

        confidence = abs(primary_result["probability"] - threshold)
        confidence_label = "High" if confidence >= 0.20 else "Moderate" if confidence >= 0.10 else "Low"

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction Score", f"{primary_result['probability']:.3f}")
        c2.metric("Confidence", confidence_label)
        c3.metric("Processing Time", f"{elapsed:.1f}s")

        st.caption("If you upload a transcript CSV, the app will try to focus on participant speech only.")
