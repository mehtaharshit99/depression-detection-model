# Depression Detection

Audio-only depression screening prototype built on the DAIC-WOZ interview dataset. The system extracts Wav2Vec2 speech embeddings from participant audio, models the full interview as a chunk sequence with a BiGRU + attention classifier, and exposes inference through a Flask API with a React + Vite frontend.

> This is a research/demo project. It is not a clinical diagnostic tool.

## Architecture At A Glance

```text
Raw .wav audio
  -> optional transcript-based participant filtering
  -> 12-second audio chunks
  -> Wav2Vec2 base, Layer 9
  -> 768-dimensional embedding per chunk
  -> participant sequence: num_chunks x 768
  -> BiGRU + attention classifier
  -> 5-fold ensemble average
  -> depression probability and final label
```

The model is audio-only. Transcript CSV files are not used as text features; they are only used, when available, to remove interviewer speech and keep participant turns.

## Current Result

Cross-validation on the current participant-filtered audio pipeline:

- Macro F1: `0.6007`
- UAR: `0.6300`
- AUC: `0.6067`
- Accuracy: `0.6571`

The default decision threshold is `0.53`, selected by validation threshold tuning for best Macro F1. A threshold of `0.50` gave the best UAR.

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
|   |-- hf_cache/
|   `-- processed_audio/
|-- models/
|   |-- best_bigru_fold0.pt
|   |-- best_bigru_fold1.pt
|   |-- best_bigru_fold2.pt
|   |-- best_bigru_fold3.pt
|   |-- best_bigru_fold4.pt
|   |-- inference_scaler.pkl
|   `-- cv_results_sequence.csv
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
- resamples audio to 16 kHz
- splits participant audio into 12-second chunks
- runs frozen `facebook/wav2vec2-base-960h`
- extracts Layer 9 hidden states
- mean-pools each chunk into one 768-dimensional vector
- saves one feature CSV per participant in `data/features_turn_level/`

Each output row represents one chunk and contains:

- `participant_id`
- `chunk_idx`
- `label`
- `w2v_0` through `w2v_767`

Existing feature CSVs are skipped by default. Delete old feature files before regenerating features with changed extraction logic.

### `src/pipeline_utils.py`

Shared ML utilities:

- `CHUNK_SEC = 12`
- `TARGET_SR = 16000`
- `chunk_waveform(...)`
- optional prosody feature extraction
- `ParticipantSequenceDataset`
- `collate_fn(...)` for variable-length sequences
- `GRUSequenceClassifier`

The classifier is a BiGRU + attention model. It reads a participant sequence of chunk embeddings and outputs one participant-level logit.

### `src/03_train_sequence.py`

Training script. It:

- loads all chunk embedding CSVs
- groups chunks into participant sequences
- runs stratified K-fold cross-validation
- applies fold-wise `StandardScaler` fitting on train folds only
- trains a BiGRU + attention classifier
- uses weighted `BCEWithLogitsLoss` for class imbalance
- saves the best model for each fold
- writes validation predictions and cross-validation metrics

Saved outputs include:

- `models/best_bigru_fold*.pt`
- `models/val_predictions_fold*.csv`
- `models/cv_results_sequence.csv`

### `src/inference_service.py`

Shared inference logic used by the Flask backend. It:

- lazily loads Wav2Vec2, the saved scaler, and fold checkpoints
- reads uploaded audio bytes
- optionally filters participant-only speech using transcript bytes
- chunks audio and extracts Wav2Vec2 Layer 9 embeddings
- standardizes features using `models/inference_scaler.pkl`
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
- optional `transcript` CSV
- optional `threshold`

The API uses lazy inference imports so the server can bind quickly during deployment. CORS is configured for browser access from the React/Vercel frontend.

### `web/frontend/src/App.jsx`

Main React UI. It:

- accepts `.wav` upload
- accepts optional transcript CSV
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

# Extract participant-only Wav2Vec2 chunk embeddings
python src\02_extract_features.py

# Train BiGRU + attention sequence model
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
- `transcript`: optional transcript CSV
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
  "processing_time_sec": 34.2
}
```

## Deployment

The project is designed for split deployment:

- Backend: Flask API on Render or another Python host
- Frontend: React/Vite app on Vercel

### Backend

Render build command:

```bash
pip install -r requirements.txt
```

Render start command:

```bash
gunicorn web.backend.api:app
```

Optional safer start command for slow model startup or long inference:

```bash
gunicorn web.backend.api:app --workers 1 --threads 2 --timeout 300
```

The deployed backend should expose:

```text
https://<backend-url>/api/health
```

### Frontend

Set Vercel root directory to:

```text
web/frontend
```

Set the Vercel environment variable:

```text
VITE_API_BASE_URL=https://<backend-url>
```

Do not include a trailing slash in `VITE_API_BASE_URL`.

Correct:

```text
https://depression-detection-model-juew.onrender.com
```

Incorrect:

```text
https://depression-detection-model-juew.onrender.com/
```

### CORS

The backend supports `ALLOWED_ORIGINS` as a comma-separated environment variable. During development or temporary preview deployments, this can be:

```text
ALLOWED_ORIGINS=*
```

For a stricter deployment, use specific frontend URLs:

```text
ALLOWED_ORIGINS=https://your-vercel-domain.vercel.app
```

## Deployment Notes

- `/api/health` is lightweight and should work even before heavy inference is used.
- `/api/predict` loads and runs Wav2Vec2, which is memory-heavy on CPU.
- Small free-tier instances may fail or kill the worker during inference.
- If health checks work but prediction fails, the likely cause is backend memory/runtime limits.
- Consider a higher-memory host, Google Cloud Run, Fly.io, Railway paid tier, or a lighter speech model for stable production inference.

## Important Model Notes

- Macro F1 is not accuracy.
- Accuracy for the current run is `0.6571`.
- Macro F1 for the current run is `0.6007`.
- The transcript is not used as text input.
- Transcript timing is only used to isolate participant speech.
- The website can run without transcript upload, but transcript upload better matches the participant-only training setup.

## Limitations

- This is not a diagnostic system.
- Results are based on DAIC-WOZ-style interview audio.
- Generalization to casual phone recordings, noisy environments, or non-DAIC speech is not guaranteed.
- Wav2Vec2 inference can be slow and memory-intensive on small servers.

## Future Work

- Deploy on a stronger backend suitable for Wav2Vec2 inference.
- Test lighter or optimized speech models for faster serving.
- Evaluate on more realistic and diverse audio.
- Improve calibration, confidence reporting, and interpretability.
