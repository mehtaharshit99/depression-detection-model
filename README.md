# Depression Detection

Multimodal depression screening prototype built on the DAIC-WOZ interview dataset. The system combines acoustic speech features from interview audio with linguistic features from transcript text, then exposes inference through a Flask API with a React + Vite frontend.

> This is a research/demo project. It is not a clinical diagnostic tool.

## Architecture At A Glance

```text
Audio branch:
  Raw .wav audio
    -> transcript-based participant filtering
    -> 12-second audio chunks
    -> Wav2Vec2 base, Layer 9
    -> 768-dimensional embedding per chunk
    -> BiGRU + attention audio representation

Text branch:
  Participant transcript text
    -> sentence-transformers/all-MiniLM-L6-v2
    -> participant-level text embedding

Fusion:
  audio representation + text embedding
    -> classifier
    -> 5-fold ensemble average
    -> depression probability and final label
```

The transcript is now used in two ways: timestamps identify participant speech in the audio, and the participant's spoken text is encoded as a real text modality for prediction.

## Current Result

Previous cross-validation result from the audio-only participant-filtered pipeline:

- Macro F1: `0.6007`
- UAR: `0.6300`
- AUC: `0.6067`
- Accuracy: `0.6571`

The default decision threshold is `0.53`, selected by validation threshold tuning for best Macro F1. A threshold of `0.50` gave the best UAR.

After regenerating multimodal features and retraining, these metrics should be updated with the new audio+text results.

## Stack

- Python
- PyTorch
- Torchaudio
- Transformers
- SoundFile
- Pandas
- NumPy
- scikit-learn
- Flask
- Flask-CORS
- Gunicorn
- React
- Vite

## Project Structure

```text
dd_p/
|-- data/
|   |-- DAIC-WOZ_raw/
|   |-- features_turn_level/
|   |-- features_multimodal/
|   |-- hf_cache/
|   `-- processed_audio/
|-- models/
|   |-- best_bigru_fold0.pt
|   |-- best_multimodal_fold0.pt
|   |-- inference_scaler.pkl
|   |-- multimodal_inference_scaler.pkl
|   |-- cv_results_sequence.csv
|   `-- cv_results_multimodal.csv
|-- src/
|   |-- 01_preprocess_data.py
|   |-- 02_extract_features.py
|   |-- 03_train_sequence.py
|   |-- inference_service.py
|   `-- pipeline_utils.py
|-- web/
|   |-- backend/
|   |   `-- api.py
|   `-- frontend/
|       |-- package.json
|       |-- vite.config.js
|       |-- .env.example
|       `-- src/
|           |-- App.jsx
|           |-- main.jsx
|           `-- styles.css
|-- requirements.txt
|-- render.yaml
|-- .gitignore
`-- README.md
```

## Main Files

### `src/01_preprocess_data.py`

Optional preprocessing helper. It can extract participant ZIPs, parse DAIC-WOZ transcripts, keep participant-only transcript rows, copy audio into `data/processed_audio/`, and write metadata.

This script is useful for dataset cleanup, but the current extraction script can also work directly from the raw DAIC-WOZ folders.

### `src/02_extract_features.py`

Feature extraction script. It:

- loads DAIC-WOZ audio and labels
- reads transcript timestamps robustly
- removes interviewer speech by keeping participant turns
- joins participant transcript text into one text document
- resamples audio to 16 kHz
- splits participant audio into 12-second chunks
- runs frozen `facebook/wav2vec2-base-960h`
- extracts Layer 9 hidden states
- mean-pools each chunk into one 768-dimensional vector
- runs frozen `sentence-transformers/all-MiniLM-L6-v2`
- stores participant-level transcript-text embeddings
- saves one feature CSV per participant in `data/features_multimodal/`

Each output row represents one chunk and contains:

- `participant_id`
- `chunk_idx`
- `label`
- `w2v_0` through `w2v_767`
- `text_0` through `text_383`

Existing feature CSVs are skipped by default. Delete old feature files before regenerating features with changed extraction logic.

The previous audio-only feature files in `data/features_turn_level/` are left untouched. The multimodal extractor writes separate files named `*_multimodal_embeddings.csv`.

### `src/pipeline_utils.py`

Shared ML utilities:

- `CHUNK_SEC = 12`
- `TARGET_SR = 16000`
- `chunk_waveform(...)`
- optional prosody feature extraction
- `ParticipantSequenceDataset`
- `collate_fn(...)` for variable-length sequences
- `GRUSequenceClassifier`
- `MultimodalGRUSequenceClassifier`

The multimodal classifier uses a BiGRU + attention audio branch and a text projection branch, then fuses both representations before classification.

### `src/03_train_sequence.py`

Training script. It:

- loads all chunk embedding CSVs
- groups chunks into participant sequences
- runs stratified K-fold cross-validation
- applies fold-wise `StandardScaler` fitting on train folds only for audio and text features
- trains a multimodal BiGRU + attention fusion classifier
- uses weighted `BCEWithLogitsLoss` for class imbalance
- saves the best model for each fold
- writes validation predictions and cross-validation metrics

Saved outputs include:

- `models/best_bigru_fold*.pt`
- `models/val_predictions_fold*.csv`
- `models/cv_results_sequence.csv`
- `models/inference_scaler.pkl`
- `models/best_multimodal_fold*.pt`
- `models/val_predictions_multimodal_fold*.csv`
- `models/cv_results_multimodal.csv`
- `models/multimodal_inference_scaler.pkl`

### `src/inference_service.py`

Shared inference logic used by the Flask backend. It:

- lazily loads Wav2Vec2, the saved scaler, and fold checkpoints
- lazily loads the transcript text encoder
- reads uploaded audio bytes
- filters participant-only speech using transcript timestamps when transcript bytes are provided
- encodes participant transcript text as the text modality
- chunks audio and extracts Wav2Vec2 Layer 9 embeddings
- standardizes features using `models/multimodal_inference_scaler.pkl`
- runs the 5 fold checkpoints
- averages fold probabilities
- applies the threshold and returns prediction metadata

### `web/backend/api.py`

Flask API with:

- `GET /api/health`
- `POST /api/predict`
- `OPTIONS /api/predict`

The backend accepts:

- required `audio` file
- recommended `transcript` CSV for multimodal prediction
- optional `threshold`

The API uses lazy inference imports so the server can bind quickly during deployment.

### `web/frontend/src/App.jsx`

Main React UI. It:

- accepts `.wav` upload
- accepts transcript CSV for audio filtering and text-modality inference
- exposes a decision-threshold slider
- sends `FormData` to `/api/predict`
- shows staged progress
- displays outcome and prediction score

Threshold slider behavior:

- lower threshold = more sensitive
- higher threshold = more conservative

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
cd C:\Users\harsh\Desktop\dd_p
python -m venv depvenv
.\depvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd C:\Users\harsh\Desktop\dd_p\web\frontend
npm install
```

## Training Pipeline

Run from the project root:

```powershell
# Optional dataset cleanup
python src\01_preprocess_data.py

# Extract participant-only audio embeddings and transcript-text embeddings
python src\02_extract_features.py

# Train multimodal BiGRU + attention fusion model
python src\03_train_sequence.py
```

## Local Website

Backend:

```powershell
cd C:\Users\harsh\Desktop\dd_p
.\depvenv\Scripts\Activate.ps1
python web\backend\api.py
```

Frontend:

```powershell
cd C:\Users\harsh\Desktop\dd_p\web\frontend
npm run dev
```

Default local URLs:

- Flask API: `http://127.0.0.1:5000`
- React app: `http://127.0.0.1:5173`

Health check:

```text
http://127.0.0.1:5000/api/health
```

## API

### `GET /api/health`

Returns:

```json
{
  "status": "ok",
  "service": "depression-detection-api"
}
```

### `POST /api/predict`

Multipart form fields:

- `audio`: required `.wav` file
- `transcript`: recommended transcript CSV
- `threshold`: optional float between `0.0` and `1.0`

Example response:

```json
{
  "probability": 0.61,
  "prediction": 1,
  "label": "Depression Detected",
  "confidence_band": "moderate",
  "fold_probabilities": [0.58, 0.62, 0.60, 0.65, 0.59],
  "threshold": 0.53,
  "metadata": {
    "total_chunks": 10,
    "valid_chunks": 9,
    "skipped_chunks": 1,
    "duration_sec": 122.4
  },
  "transcript_used": false,
  "text_used": false,
  "processing_time_sec": 34.2
}
```

## Important Model Notes

- Macro F1 is not accuracy.
- Accuracy for the current run is `0.6571`.
- Macro F1 for the current run is `0.6007`.
- These metrics are from the previous audio-only baseline and should be refreshed after multimodal retraining.
- The transcript text is now used as a linguistic modality.
- Transcript timing is also used to isolate participant speech.
- For true multimodal inference, upload both the audio file and the matching transcript CSV.
