import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

from pipeline_utils import CHUNK_SEC, TARGET_SR, chunk_waveform, clean_transcript_text

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "DAIC-WOZ_raw"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "features_multimodal"
LABEL_FILE = RAW_DIR / "train_split_Depression_AVEC2017.csv"
AUDIT_DIR = BASE_DIR / "data" / "extraction_audits"
HF_CACHE = BASE_DIR / "data" / "hf_cache"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384
EXTRACT_LAYER = 9

SKIP_NOT_IN_SPLIT = "not_in_split"
SKIP_OUTPUT_EXISTS = "output_exists"
SKIP_MISSING_AUDIO = "missing_audio"
SKIP_MISSING_TRANSCRIPT = "missing_transcript"
SKIP_MISSING_BOTH = "missing_audio_and_transcript"
SKIP_METADATA_COLLISION = "hidden_metadata_file_collision"
SKIP_TRANSCRIPT_PARSE_FAILURE = "transcript_parse_failure"
SKIP_NO_PARTICIPANT_ROWS = "no_participant_rows"
SKIP_NO_USABLE_SEGMENTS = "no_usable_segments"
SKIP_LOAD_OR_FILTER_FAILURE = "load_or_filter_failure"
SKIP_NO_VALID_CHUNKS = "no_valid_chunks"
SKIP_ALL_CHUNKS_SILENT = "all_chunks_silent_or_invalid"
STATUS_SUCCESS = "success"

os.environ["HF_HOME"] = str(HF_CACHE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated feature CSVs. Defaults to data/features_multimodal.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else BASE_DIR / path


def visible_glob(folder: Path, pattern: str) -> list[Path]:
    return sorted(f for f in folder.glob(pattern) if not f.name.startswith("._"))


def has_hidden_metadata_collision(folder: Path, pattern: str) -> bool:
    visible_names = {f.name for f in visible_glob(folder, pattern)}
    for hidden in folder.glob("._*"):
        visible_name = hidden.name[2:]
        if visible_name in visible_names:
            return True
    return False


def load_transcript(transcript_path: Path) -> pd.DataFrame:
    with open(transcript_path, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline().strip()

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


def build_participant_text(participant_rows: pd.DataFrame) -> str:
    if "value" not in participant_rows.columns:
        return ""
    values = participant_rows["value"].dropna().astype(str).tolist()
    raw = " ".join(value.strip() for value in values if value.strip())
    return clean_transcript_text(raw)


@torch.no_grad()
def extract_text_embedding(text: str, tokenizer, model, device) -> np.ndarray:
    if not text.strip():
        return np.zeros(TEXT_DIM, dtype=np.float32)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    masked_embeddings = token_embeddings * attention_mask
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1)
    embedding = (summed / counts).squeeze(0).cpu().numpy().astype(np.float32)
    return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def extract_chunk_embedding(chunk: np.ndarray, processor, model, device) -> np.ndarray | None:
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < 1e-4:
        return None

    inputs = processor(
        chunk,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=False,
    )
    outputs = model(inputs.input_values.to(device))
    layer9 = outputs.hidden_states[EXTRACT_LAYER]
    embedding = layer9.mean(dim=1).squeeze(0).cpu().numpy()

    if np.isnan(embedding).any() or np.isinf(embedding).any():
        return None
    return embedding.astype(np.float32)


def get_participant_segments(waveform: np.ndarray, sample_rate: int, transcript_path: Path):
    try:
        transcript = load_transcript(transcript_path)
    except Exception as exc:
        raise ValueError(f"{SKIP_TRANSCRIPT_PARSE_FAILURE}: {exc}") from exc

    if "speaker" not in transcript.columns:
        raise ValueError(f"{SKIP_TRANSCRIPT_PARSE_FAILURE}: missing speaker column")

    start_col = "start_time" if "start_time" in transcript.columns else "start"
    stop_col = "stop_time" if "stop_time" in transcript.columns else "end_time"
    if start_col not in transcript.columns or stop_col not in transcript.columns:
        raise ValueError(f"{SKIP_TRANSCRIPT_PARSE_FAILURE}: missing timestamp columns")

    participant_rows = transcript[
        transcript["speaker"].astype(str).str.lower().str.contains("participant", na=False)
    ]
    if participant_rows.empty:
        raise ValueError(SKIP_NO_PARTICIPANT_ROWS)

    participant_text = build_participant_text(participant_rows)
    audio_len = len(waveform)
    segments = []

    for _, row in participant_rows.iterrows():
        try:
            start = int(float(row[start_col]) * sample_rate)
            end = int(float(row[stop_col]) * sample_rate)
        except (TypeError, ValueError):
            continue

        start = max(0, min(start, audio_len))
        end = max(0, min(end, audio_len))
        if end > start:
            segments.append(waveform[start:end])

    if not segments:
        raise ValueError(SKIP_NO_USABLE_SEGMENTS)

    return np.concatenate(segments), participant_text


def load_filtered_audio(audio_path: Path, transcript_path: Path):
    try:
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sr != TARGET_SR:
            waveform_t = torch.from_numpy(waveform).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sr, TARGET_SR)
            waveform = waveform_t.squeeze(0).numpy()
        waveform = waveform.astype(np.float32)
        return get_participant_segments(waveform, TARGET_SR, transcript_path)
    except Exception as primary_exc:
        primary_text = str(primary_exc)
        for known_skip in (
            SKIP_TRANSCRIPT_PARSE_FAILURE,
            SKIP_NO_PARTICIPANT_ROWS,
            SKIP_NO_USABLE_SEGMENTS,
        ):
            if known_skip in primary_text:
                raise

        try:
            waveform, sr = torchaudio.load(audio_path)
            waveform = (
                torchaudio.functional.resample(waveform, sr, TARGET_SR)
                .mean(dim=0)
                .numpy()
                .astype(np.float32)
            )
            return get_participant_segments(waveform, TARGET_SR, transcript_path)
        except Exception as fallback_exc:
            fallback_text = str(fallback_exc)
            for known_skip in (
                SKIP_TRANSCRIPT_PARSE_FAILURE,
                SKIP_NO_PARTICIPANT_ROWS,
                SKIP_NO_USABLE_SEGMENTS,
            ):
                if known_skip in fallback_text:
                    raise
            print(f"[WARN] {audio_path.parent.name} load/filter failed: {fallback_exc}")
            print(f"[WARN] {audio_path.parent.name} original loader error: {primary_exc}")
            raise ValueError(SKIP_LOAD_OR_FILTER_FAILURE) from fallback_exc


def process_participant(
    folder: Path,
    label_map: dict,
    output_dir: Path,
    encoders,
) -> tuple[int, str]:
    pid = folder.name.split("_")[0]
    if pid not in label_map:
        return 0, SKIP_NOT_IN_SPLIT

    out_path = output_dir / f"{pid}_multimodal_embeddings.csv"
    if out_path.exists():
        return 0, SKIP_OUTPUT_EXISTS

    audio_files = visible_glob(folder, "*_AUDIO.wav")
    transcript_files = visible_glob(folder, "*TRANSCRIPT*.csv")
    audio_path = audio_files[0] if audio_files else None
    transcript_path = transcript_files[0] if transcript_files else None

    if not audio_path and not transcript_path:
        return 0, SKIP_MISSING_BOTH
    if not audio_path:
        return 0, SKIP_MISSING_AUDIO
    if not transcript_path:
        return 0, SKIP_MISSING_TRANSCRIPT

    try:
        participant_audio, participant_text = load_filtered_audio(audio_path, transcript_path)
    except Exception as exc:
        err_text = str(exc)
        for known_skip in (
            SKIP_TRANSCRIPT_PARSE_FAILURE,
            SKIP_NO_PARTICIPANT_ROWS,
            SKIP_NO_USABLE_SEGMENTS,
            SKIP_LOAD_OR_FILTER_FAILURE,
        ):
            if known_skip in err_text:
                return 0, known_skip
        return 0, SKIP_LOAD_OR_FILTER_FAILURE

    chunks = chunk_waveform(participant_audio, TARGET_SR)
    if not chunks:
        return 0, SKIP_NO_VALID_CHUNKS

    processor, wav_model, text_tokenizer, text_model, device = encoders
    text_embedding = extract_text_embedding(participant_text, text_tokenizer, text_model, device)
    records = []

    for idx, chunk in enumerate(chunks):
        embedding = extract_chunk_embedding(chunk, processor, wav_model, device)
        if embedding is None:
            continue

        rec = {
            "participant_id": pid,
            "chunk_idx": idx,
            "label": int(label_map[pid]),
        }
        rec.update({f"w2v_{j}": float(embedding[j]) for j in range(768)})
        rec.update({f"text_{j}": float(text_embedding[j]) for j in range(TEXT_DIM)})
        records.append(rec)

    if not records:
        return 0, SKIP_ALL_CHUNKS_SILENT

    pd.DataFrame(records).to_csv(out_path, index=False)
    return len(records), STATUS_SUCCESS


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device              : {device}")
    print(f"Chunk size          : {CHUNK_SEC}s")
    print(f"Extract layer       : {EXTRACT_LAYER}")
    print(f"Output dir          : {output_dir}")

    labels_df = pd.read_csv(LABEL_FILE)
    label_map = dict(zip(labels_df["Participant_ID"].astype(str), labels_df["PHQ8_Binary"]))

    print("\nLoading Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
    )
    wav_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=HF_CACHE,
        output_hidden_states=True,
    ).to(device).eval()
    for param in wav_model.parameters():
        param.requires_grad = False
    print("Wav2Vec2 loaded and frozen.")

    print("\nLoading text encoder...")
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, cache_dir=HF_CACHE)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, cache_dir=HF_CACHE).to(device).eval()
    for param in text_model.parameters():
        param.requires_grad = False
    print("Text encoder loaded and frozen.")

    participants = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and not p.name.startswith("._"))
    existing_feature_files = sorted(output_dir.glob("*_multimodal_embeddings.csv"))
    if existing_feature_files:
        print(f"\n[INFO] Found {len(existing_feature_files)} existing feature CSV files in {output_dir}.")
        print("[INFO] Existing files are skipped by default.")
        print("[INFO] Use --output_dir for isolated experiment features.\n")

    encoders = (processor, wav_model, text_tokenizer, text_model, device)
    total_chunks = 0
    audit_records = []

    for participant in tqdm(participants, desc="Extracting features"):
        chunks_saved, reason = process_participant(
            participant,
            label_map,
            output_dir,
            encoders,
        )
        total_chunks += chunks_saved
        audit_records.append({
            "participant_id": participant.name.split("_")[0],
            "chunks_saved": chunks_saved,
            "status": reason,
        })

    audit_df = pd.DataFrame(audit_records)
    audit_path = AUDIT_DIR / f"{output_dir.name}_extraction_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    print(f"\nDone. Total chunks saved: {total_chunks}")
    print(f"Extraction audit log saved to: {audit_path}")
    print("\n--- Skip-reason summary ---")
    for reason, count in audit_df["status"].value_counts().items():
        print(f"  {reason:<42} : {count}")

    sample_files = [f for f in output_dir.glob("*.csv") if "audit" not in f.name]
    if sample_files:
        sample_df = pd.read_csv(sample_files[0])
        w2v_cols = [c for c in sample_df.columns if c.startswith("w2v_")]
        text_cols = [c for c in sample_df.columns if c.startswith("text_")]
        print(f"\n[CHECK] Sample file : {sample_files[0].name}")
        print(f"[CHECK] W2V cols    : {len(w2v_cols)}  (expected 768)")
        print(f"[CHECK] Text cols   : {len(text_cols)}  (expected {TEXT_DIM})")
        print(f"[CHECK] Chunks      : {len(sample_df)}")
        print(f"[CHECK] Label       : {sample_df['label'].iloc[0]}")


if __name__ == "__main__":
    main()
