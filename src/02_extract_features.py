import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from pipeline_utils import CHUNK_SEC, TARGET_SR, chunk_waveform

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "DAIC-WOZ_raw"
OUTPUT_DIR = BASE_DIR / "data" / "features_turn_level"
LABEL_FILE = RAW_DIR / "train_split_Depression_AVEC2017.csv"
HF_CACHE = BASE_DIR / "data" / "hf_cache"

os.environ["HF_HOME"] = str(HF_CACHE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXTRACT_LAYER = 9

print(f"Device       : {DEVICE}")
print(f"Chunk size   : {CHUNK_SEC}s")
print(f"Extract layer: {EXTRACT_LAYER}")

labels_df = pd.read_csv(LABEL_FILE)
label_map = dict(
    zip(
        labels_df["Participant_ID"].astype(str),
        labels_df["PHQ8_Binary"],
    )
)

print("\nLoading Wav2Vec2...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h",
    cache_dir=HF_CACHE,
)

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h",
    cache_dir=HF_CACHE,
    output_hidden_states=True,
).to(DEVICE).eval()

for param in model.parameters():
    param.requires_grad = False

print("Wav2Vec2 loaded and frozen.")


# Loads DAIC-WOZ transcript files while handling common formatting issues.
def load_transcript(transcript_path: Path) -> pd.DataFrame:
    """
    Load DAIC-WOZ transcript CSV robustly.

    Handles:
      - tab-separated transcript files
      - comma-separated transcript files
      - malformed single-line headers
      - variant timestamp column names
    """
    with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()

    if "start_timestop_timespeakervalue" in first_line.replace(" ", "").lower():
        transcript = pd.read_csv(
            transcript_path,
            sep="\t",
            skiprows=1,
            names=["start_time", "stop_time", "speaker", "value"],
        )
    else:
        try:
            transcript = pd.read_csv(transcript_path, sep="\t")
        except Exception:
            transcript = pd.read_csv(transcript_path, sep=",")

    transcript.columns = [c.lower().strip() for c in transcript.columns]

    if "speaker" not in transcript.columns:
        collapsed_cols = "".join(transcript.columns).replace(" ", "")
        if "start_timestop_timespeakervalue" in collapsed_cols:
            transcript = pd.read_csv(
                transcript_path,
                sep="\t",
                skiprows=1,
                names=["start_time", "stop_time", "speaker", "value"],
            )
            transcript.columns = [c.lower().strip() for c in transcript.columns]

    return transcript

# Converts one audio chunk into a Wav2Vec2 Layer-9 embedding.
@torch.no_grad()
def extract_chunk_embedding(chunk: np.ndarray) -> np.ndarray | None:
    """
    Extract a 768-dim embedding from Layer 9 of Wav2Vec2
    for a single 1-D float32 audio chunk.

    Returns:
        numpy array of shape (768,), or None if chunk is silent/invalid.
    """
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
    outputs = model(input_values)

    layer9 = outputs.hidden_states[EXTRACT_LAYER]
    embedding = layer9.mean(dim=1).squeeze(0).cpu().numpy()

    if np.isnan(embedding).any() or np.isinf(embedding).any():
        return None

    return embedding.astype(np.float32)


# Extracts and saves participant-only chunk embeddings for one participant.
def process_participant(folder: Path) -> int:
    """
    For a given participant folder:
      1. Load audio
      2. Filter to participant-only turns (strip interviewer)
      3. Concatenate participant turns
      4. Chunk into CHUNK_SEC windows
      5. Extract Layer-9 Wav2Vec2 embedding per chunk
      6. Save as CSV

    Returns number of chunks saved (0 on skip/error).
    """
    pid = folder.name.split("_")[0]

    if pid not in label_map:
        return 0

    label = label_map[pid]
    out_path = OUTPUT_DIR / f"{pid}_chunk_embeddings.csv"

    if out_path.exists():
        return 0

    audio_path = next(folder.glob("*_AUDIO.wav"), None)
    transcript_path = next(folder.glob("*TRANSCRIPT*.csv"), None)

    if not audio_path or not transcript_path:
        return 0

    try:
        waveform, sr = sf.read(str(audio_path), dtype="float32")

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        if sr != TARGET_SR:
            waveform_t = torch.from_numpy(waveform).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sr, TARGET_SR)
            waveform = waveform_t.squeeze(0).numpy()

        waveform = waveform.astype(np.float32)

        transcript = load_transcript(transcript_path)

        start_col = "start_time" if "start_time" in transcript.columns else "start"
        stop_col = "stop_time" if "stop_time" in transcript.columns else "end_time"

        participant_rows = transcript[
            transcript["speaker"].str.lower().str.contains("participant", na=False)
        ]

        if participant_rows.empty:
            return 0

        total_len = len(waveform)
        segments = []

        for _, row in participant_rows.iterrows():
            start = int(row[start_col] * TARGET_SR)
            end = int(row[stop_col] * TARGET_SR)

            start = max(0, min(start, total_len))
            end = max(0, min(end, total_len))

            if end <= start:
                continue

            seg = waveform[start:end]
            if len(seg) > 0:
                segments.append(seg)

        if not segments:
            return 0

        participant_audio = np.concatenate(segments)

    except Exception as e:
        try:
            waveform, sr = torchaudio.load(audio_path)
            waveform = (
                torchaudio.functional.resample(waveform, sr, TARGET_SR)
                .mean(dim=0)
                .numpy()
                .astype(np.float32)
            )

            transcript = load_transcript(transcript_path)

            start_col = "start_time" if "start_time" in transcript.columns else "start"
            stop_col = "stop_time" if "stop_time" in transcript.columns else "end_time"

            participant_rows = transcript[
                transcript["speaker"].str.lower().str.contains("participant", na=False)
            ]

            if participant_rows.empty:
                return 0

            total_len = len(waveform)
            segments = []

            for _, row in participant_rows.iterrows():
                start = int(row[start_col] * TARGET_SR)
                end = int(row[stop_col] * TARGET_SR)

                start = max(0, min(start, total_len))
                end = max(0, min(end, total_len))

                if end <= start:
                    continue

                seg = waveform[start:end]
                if len(seg) > 0:
                    segments.append(seg)

            if not segments:
                return 0

            participant_audio = np.concatenate(segments)

        except Exception as fallback_e:
            print(f"[WARN] {pid} load/filter failed: {fallback_e}")
            print(f"[WARN] {pid} original loader error: {e}")
            return 0

    chunks = chunk_waveform(participant_audio, TARGET_SR)

    if not chunks:
        return 0

    records = []

    for idx, chunk in enumerate(chunks):
        embedding = extract_chunk_embedding(chunk)

        if embedding is None:
            continue

        rec = {
            "participant_id": pid,
            "chunk_idx": idx,
            "label": int(label),
        }
        rec.update({f"w2v_{j}": float(embedding[j]) for j in range(768)})
        records.append(rec)

    if not records:
        print(f"[WARN] {pid}: all chunks were silent or invalid — skipped.")
        return 0

    pd.DataFrame(records).to_csv(out_path, index=False)

    return len(records)


participants = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())

existing_feature_files = sorted(OUTPUT_DIR.glob("*_chunk_embeddings.csv"))
if existing_feature_files:
    print(
        f"\n[INFO] Found {len(existing_feature_files)} existing feature CSV files in {OUTPUT_DIR}."
    )
    print("[INFO] Existing files are skipped by default.")
    print("[INFO] Delete old CSVs first if you want to regenerate features with the current extractor.\n")

total_chunks = 0

for p in tqdm(participants, desc="Extracting features"):
    total_chunks += process_participant(p)

print(f"\nDone. Total chunks saved: {total_chunks}")

sample_files = list(OUTPUT_DIR.glob("*.csv"))
if sample_files:
    sample_df = pd.read_csv(sample_files[0])
    w2v_cols = [c for c in sample_df.columns if c.startswith("w2v_")]
    print(f"[CHECK] Sample file : {sample_files[0].name}")
    print(f"[CHECK] W2V cols    : {len(w2v_cols)}  (expected 768)")
    print(f"[CHECK] Chunks      : {len(sample_df)}")
    print(f"[CHECK] Label       : {sample_df['label'].iloc[0]}")
