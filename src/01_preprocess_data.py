"""
01_preprocess_data.py — DAIC-WOZ Dataset Preprocessing
=======================================================
Steps:
  1. Extract participant ZIPs (if raw_zips/ folder exists)
  2. Parse & clean transcripts (handles malformed headers, tab/comma separators)
  3. Filter to participant-only speech
  4. Copy audio files to processed_audio/
  5. Save metadata.csv

Preprocessing changes:
  - File discovery uses find_participant_files() from pipeline_utils so the
    same deterministic, ._*-ignoring logic is shared with 02_extract_features.py.
    The old fallback that constructed a guessed path (which could silently pick
    up a macOS ghost file) is replaced with an explicit missing-file check.
  - Every skip branch now uses the canonical skip-reason constants and also
    calls `continue` so execution never falls through into transcript parsing
    when a required file is absent.
  - A structured audit CSV (data/preprocessing_audit.csv) is written at the
    end, mirroring the extraction_audit.csv produced by 02_extract_features.py.

Run this first before any other script.
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from pipeline_utils import (
    find_participant_files,
    SKIP_MISSING_AUDIO,
    SKIP_MISSING_TRANSCRIPT,
    SKIP_TRANSCRIPT_PARSE_FAILURE,
    SKIP_NO_PARTICIPANT_ROWS,
    SUCCESS,
)

BASE_DIR              = Path(__file__).resolve().parents[1]
RAW_ZIP_DIR           = BASE_DIR / "data" / "raw_zips"
RAW_DATA_DIR          = BASE_DIR / "data" / "DAIC-WOZ_raw"
OUTPUT_AUDIO_DIR      = BASE_DIR / "data" / "processed_audio"
OUTPUT_TRANSCRIPT_DIR = BASE_DIR / "data" / "processed_transcripts"
OUTPUT_METADATA       = BASE_DIR / "data" / "metadata.csv"
OUTPUT_AUDIT          = BASE_DIR / "data" / "preprocessing_audit.csv"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — ZIP extraction
# ---------------------------------------------------------------------------
if RAW_ZIP_DIR.exists():
    zip_files = sorted(RAW_ZIP_DIR.glob("*.zip"))
    if zip_files:
        print(f"\nExtracting {len(zip_files)} ZIP files...")
        for zip_path in tqdm(zip_files, desc="Extracting ZIPs"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    extract_dir = RAW_DATA_DIR / zip_path.stem
                    if not extract_dir.exists():
                        zf.extractall(extract_dir)
            except Exception as e:
                print(f"  [WARN] Error extracting {zip_path.name}: {e}")
    else:
        print("raw_zips/ folder is empty — skipping extraction.")
else:
    print("No raw_zips/ folder found — skipping extraction.")

# ---------------------------------------------------------------------------
# Step 2-5 — Per-participant processing
# ---------------------------------------------------------------------------
metadata      = []
audit_records = []

# ignore ._* participant directories, sort for determinism
participant_dirs = sorted(
    p for p in RAW_DATA_DIR.glob("*_P") if not p.name.startswith("._")
)
print(f"\nFound {len(participant_dirs)} participant folders.")

for folder in tqdm(participant_dirs, desc="Processing participants"):
    participant_id = folder.name.split("_")[0]

    # single deterministic discovery call, no fallback path guessing
    audio_file, transcript_file = find_participant_files(folder)

    if transcript_file is None or not transcript_file.exists():
        print(f"  [{SKIP_MISSING_TRANSCRIPT}] {participant_id}")
        audit_records.append({"participant_id": participant_id, "status": SKIP_MISSING_TRANSCRIPT})
        continue  # was missing before

    if audio_file is None or not audio_file.exists():
        print(f"  [{SKIP_MISSING_AUDIO}] {participant_id}")
        audit_records.append({"participant_id": participant_id, "status": SKIP_MISSING_AUDIO})
        continue  # was missing before

    # ── Transcript parsing ────────────────────────────────────────────────
    try:
        with open(transcript_file, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()

        if "start_timestop_timespeakervalue" in first_line.replace(" ", "").lower():
            df = pd.read_csv(
                transcript_file, sep="\t", skiprows=1,
                names=["start_time", "stop_time", "speaker", "value"],
            )
        else:
            try:
                df = pd.read_csv(transcript_file, sep="\t")
            except Exception:
                df = pd.read_csv(transcript_file, sep=",")
    except Exception as e:
        print(f"  [{SKIP_TRANSCRIPT_PARSE_FAILURE}] {participant_id}: {e}")
        audit_records.append({"participant_id": participant_id, "status": SKIP_TRANSCRIPT_PARSE_FAILURE})
        continue

    required = {"speaker", "value"}
    df.columns = [c.strip().lower() for c in df.columns]
    if not required.issubset(df.columns):
        print(f"  [{SKIP_TRANSCRIPT_PARSE_FAILURE}] {participant_id}: bad columns {list(df.columns)}")
        audit_records.append({"participant_id": participant_id, "status": SKIP_TRANSCRIPT_PARSE_FAILURE})
        continue

    df = df.dropna(subset=["speaker", "value"])

    participant_df = df[df["speaker"].astype(str).str.lower().str.startswith("participant")]
    if participant_df.empty:
        print(f"  [{SKIP_NO_PARTICIPANT_ROWS}] {participant_id}")
        audit_records.append({"participant_id": participant_id, "status": SKIP_NO_PARTICIPANT_ROWS})
        continue

    # ── Save cleaned transcript ───────────────────────────────────────────
    cleaned_path = OUTPUT_TRANSCRIPT_DIR / f"{participant_id}_cleaned.csv"
    participant_df.to_csv(cleaned_path, index=False)

    # ── Copy audio ────────────────────────────────────────────────────────
    out_audio = OUTPUT_AUDIO_DIR / f"{participant_id}.wav"
    if not out_audio.exists():
        try:
            out_audio.write_bytes(audio_file.read_bytes())
        except Exception as e:
            print(f"  [WARN] Error copying audio for {participant_id}: {e}")
            audit_records.append({"participant_id": participant_id, "status": "audio_copy_failure"})
            continue

    # ── Metadata record ───────────────────────────────────────────────────
    full_text = " ".join(participant_df["value"].astype(str).tolist())
    metadata.append(
        {
            "Participant_ID":     participant_id,
            "Audio_Path":         str(out_audio),
            "Transcript_Path":    str(cleaned_path),
            "Num_Turns":          len(participant_df),
            "Transcript_Length":  len(full_text.split()),
            "Transcript_Preview": (full_text[:200] + "...") if len(full_text) > 200 else full_text,
        }
    )
    audit_records.append({"participant_id": participant_id, "status": SUCCESS})

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
if metadata:
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(OUTPUT_METADATA, index=False)
    print(f"\n✅ Done.")
    print(f"   Participants processed : {len(metadata)}")
    print(f"   Metadata saved to      : {OUTPUT_METADATA}")
else:
    print("\n⚠️  No metadata generated — check your raw dataset structure.")
    print("    Expected: data/DAIC-WOZ_raw/<PID>_P/<PID>_TRANSCRIPT.csv + <PID>_AUDIO.wav")

# structured audit output (mirrors extraction_audit.csv from 02_extract_features.py)
audit_df = pd.DataFrame(audit_records)
audit_df.to_csv(OUTPUT_AUDIT, index=False)

print(f"\nSkip-reason summary:")
for reason, count in audit_df["status"].value_counts().items():
    marker = "✅" if reason == SUCCESS else "⚠️ "
    print(f"  {marker}  {reason:<40} {count}")
print(f"\nAudit log saved to: {OUTPUT_AUDIT}")