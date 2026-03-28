# Depression Detection — Audio Sequence Pipeline

Audio-only depression detection on DAIC-WOZ using participant-only speech, Wav2Vec2 chunk embeddings, and participant-level sequence modeling.

## Project Structure

```text
dd_p/
├── data/
│   ├── DAIC-WOZ_raw/
│   ├── features_turn_level/
│   ├── hf_cache/
│   ├── processed_audio/
│   └── features_all.parquet
├── models/
├── src/
│   ├── 01_preprocess_data.py
│   ├── 02_extract_features.py
│   ├── 03_train_sequence.py
│   └── pipeline_utils.py
├── depvenv/
├── requirements.txt
└── README.md
```

## Run Order

```powershell
# 1. Optional dataset preprocessing / cleanup
python src\01_preprocess_data.py

# 2. Extract chunk-level Wav2Vec2 features
python src\02_extract_features.py

# 3. Train participant-level sequence classifier
python src\03_train_sequence.py
```
