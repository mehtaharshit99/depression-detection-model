# Depression Detection

Multimodal depression-screening prototype built on the DAIC-WOZ interview dataset. The system combines participant speech audio features with participant transcript-text features, then exposes prediction through a Flask API and React frontend.

> This is a research/demo project. It is not a clinical diagnostic tool.

## Architecture

```text
Raw interview audio + transcript
  -> keep participant speech using transcript timestamps
  -> split participant audio into 12-second chunks
  -> Wav2Vec2 base, Layer 9 audio embedding per chunk
  -> clean participant transcript text
  -> MiniLM sentence-transformer text embedding
  -> 3-head attention pooling over audio chunks
  -> fuse pooled audio context with transcript-text context
  -> classifier
  -> 5-fold checkpoint ensemble for inference
```

The transcript is used in two ways: timestamps isolate participant audio, and participant text provides a separate linguistic modality.

## Current Cross-Validation Result

Current default model: multimodal 3-head attention pooling, seed `99`, 5-fold cross-validation.

Fold-mean metrics at the standard `0.50` decision threshold:

| Metric | Score |
| --- | ---: |
| Accuracy | `0.7931` |
| Macro F1 | `0.7152` |
| Precision | `0.7375` |
| Recall | `0.7138` |
| UAR / Macro Recall | `0.7138` |
| AUC | `0.6904` |

Out-of-fold predictions gave the best validation accuracy and Macro F1 at threshold `0.51`. That threshold is used as the app default and gives: Accuracy `0.8113`, Macro F1 `0.7383`, Precision `0.7878`, UAR `0.7171`.

## Stack

Python, PyTorch, Torchaudio, Transformers, SoundFile, Pandas, NumPy, scikit-learn, Flask, Flask-CORS, Gunicorn, React, Vite.

## Project Structure

```text
dd_p/
|-- data/
|   |-- DAIC-WOZ_raw/
|   |-- features_multimodal/
|   |-- hf_cache/
|   `-- processed_audio/
|-- models/
|   |-- best_multimodal_fold*.pt
|   |-- val_predictions_multimodal_fold*.csv
|   |-- cv_results_multimodal.csv
|   `-- multimodal_inference_scaler.pkl
|-- src/
|   |-- 01_preprocess_data.py
|   |-- 02_extract_features.py
|   |-- 03_train_sequence.py
|   |-- inference_service.py
|   `-- pipeline_utils.py
|-- web/
|   |-- backend/api.py
|   `-- frontend/
|-- requirements.txt
|-- render.yaml
|-- .gitignore
`-- README.md
```

## Main Files

### `src/01_preprocess_data.py`

Optional dataset cleanup helper. It can extract participant ZIPs, parse DAIC-WOZ transcripts, keep participant-only transcript rows, copy audio into `data/processed_audio/`, and write metadata.

### `src/02_extract_features.py`

Feature extraction script. It loads audio and labels, parses transcripts, isolates participant speech, chunks audio, extracts Wav2Vec2 Layer 9 embeddings, cleans participant transcript text, extracts MiniLM text embeddings, and saves one CSV per participant in `data/features_multimodal/`.

Each feature row contains one audio chunk plus repeated participant-level text features:

- `participant_id`
- `chunk_idx`
- `label`
- `w2v_0` through `w2v_767`
- `text_0` through `text_383`

Existing feature CSVs are skipped by default. Delete old feature files before regenerating features after extraction-code changes.

### `src/pipeline_utils.py`

Shared utilities and model components:

- audio constants and chunking helpers
- participant-level sequence dataset
- variable-length batch collation
- legacy BiGRU multimodal classifier
- current multimodal 3-head attention-pooling classifier

### `src/03_train_sequence.py`

Main training script. By default it trains the current best model: multimodal 3-head attention pooling with seed `99`.

It performs stratified 5-fold cross-validation, standardizes features fold-wise using train data only, trains with weighted `BCEWithLogitsLoss`, saves the best checkpoint for each fold, and writes validation metrics.

Default outputs:

- `models/best_multimodal_fold*.pt`
- `models/val_predictions_multimodal_fold*.csv`
- `models/cv_results_multimodal.csv`
- `models/multimodal_inference_scaler.pkl`

Useful options:

```powershell
# Current best model
python src\03_train_sequence.py

# Legacy BiGRU comparison, if needed
python src\03_train_sequence.py --model gru --seed 88 --dropout 0.35 --lr 0.001 --epochs 30 --patience 6
```

### `src/inference_service.py`

Shared inference logic used by the Flask backend. It lazily loads Wav2Vec2, MiniLM, the saved scaler, and the fold checkpoints; extracts audio/text features from uploads; standardizes them; averages fold probabilities; and returns the final prediction.

## Setup

```powershell
cd C:\Users\harsh\Desktop\projects\dd_p
python -m venv depvenv
.\depvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Frontend setup:

```powershell
cd C:\Users\harsh\Desktop\projects\dd_p\web\frontend
npm install
```

## Training Pipeline

Run from the project root:

```powershell
python src\01_preprocess_data.py
python src\02_extract_features.py
python src\03_train_sequence.py
```

`01_preprocess_data.py` is optional if the raw DAIC-WOZ folder is already organized for extraction.

## Local App

Backend:

```powershell
cd C:\Users\harsh\Desktop\projects\dd_p
.\depvenv\Scripts\Activate.ps1
python web\backend\api.py
```

Frontend:

```powershell
cd C:\Users\harsh\Desktop\projects\dd_p\web\frontend
npm run dev
```

Default local URLs:

- Flask API: `http://127.0.0.1:5000`
- React app: `http://127.0.0.1:5173`

## API

### `GET /api/health`

Returns service health.

### `POST /api/predict`

Multipart form fields:

- `audio`: required `.wav` file
- `transcript`: recommended transcript CSV
- `threshold`: optional float between `0.0` and `1.0`

For true multimodal inference, upload both the audio file and its matching transcript CSV.
