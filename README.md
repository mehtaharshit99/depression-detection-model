# Depression Detection

Multimodal depression screening prototype built on DAIC-WOZ interviews. The selected methodology combines participant-only speech audio with cleaned participant transcript text, then trains a concat-fusion BiGRU sequence classifier.

> This is a research/demo project. It is not a clinical diagnostic tool.

## Methodology

The pipeline uses the transcript timestamps to isolate participant speech from the interview audio. Participant-only audio is resampled to 16 kHz, split into fixed 12-second non-overlapping chunks, and encoded with frozen `facebook/wav2vec2-base-960h` Layer 9 embeddings.

The participant transcript text is cleaned with the shared text-cleaning utility, removing transcript artifacts and non-semantic control tokens before embedding with `sentence-transformers/all-MiniLM-L6-v2`.

For each participant, the model treats the audio chunks as a sequence and uses a BiGRU with attention to produce an audio representation. That representation is concatenated with the participant-level text embedding, then passed through a classifier. Cross-validation showed this concat multimodal setup was the best-performing option among the tested variants.

## Current Selected Result

Best cross-validation result from the cleaned multimodal concat model:

- Accuracy: `0.7260`
- Macro F1: `0.6589`
- UAR: `0.6571`
- AUC: `0.6050`

The CV decision threshold is `0.50`. Runtime inference still exposes a threshold parameter for the app UI, with `0.53` retained as the default application threshold from earlier validation.

## Project Structure

```text
dd_p/
|-- data/
|   |-- DAIC-WOZ_raw/
|   |-- features_multimodal/
|   |-- hf_cache/
|   `-- processed_audio/
|-- models/
|   |-- best_multimodal_concat_fold*.pt
|   |-- multimodal_inference_scaler.pkl
|   `-- cv_results_multimodal.csv
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
`-- README.md
```

## Main Scripts

`src/02_extract_features.py`

- reads DAIC-WOZ audio, transcripts, and labels
- ignores hidden metadata files during discovery
- keeps participant-only audio using transcript timestamps
- cleans participant transcript text
- extracts Wav2Vec2 audio embeddings and MiniLM text embeddings
- writes one `*_multimodal_embeddings.csv` file per participant
- writes extraction audit CSVs to `data/extraction_audits/`

`src/03_train_sequence.py`

- loads multimodal participant feature CSVs
- groups chunks into participant-level sequences
- runs stratified 5-fold cross-validation
- trains the selected concat-fusion multimodal classifier
- saves fold checkpoints as `models/best_multimodal_concat_fold*.pt`
- writes `models/cv_results_multimodal.csv`
- writes `models/multimodal_inference_scaler.pkl`

`src/inference_service.py`

- loads the concat fold checkpoints and scaler
- extracts audio/text features from uploaded files
- averages fold probabilities
- returns prediction metadata for the Flask API

## Setup

```powershell
cd C:\Users\harsh\Desktop\dd_p
python -m venv depvenv
.\depvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Frontend:

```powershell
cd C:\Users\harsh\Desktop\dd_p\web\frontend
npm install
```

## Training

Run from the project root:

```powershell
python src\02_extract_features.py
python src\03_train_sequence.py
```

To regenerate features without touching the default feature directory:

```powershell
python src\02_extract_features.py --output_dir data\features_multimodal_experiment
python src\03_train_sequence.py --feature_dir data\features_multimodal_experiment
```

## Local App

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

Default URLs:

- Flask API: `http://127.0.0.1:5000`
- React app: `http://127.0.0.1:5173`

## API

`POST /api/predict` accepts:

- `audio`: required `.wav` file
- `transcript`: recommended transcript CSV
- `threshold`: optional float between `0.0` and `1.0`

The response includes probability, label, fold probabilities, threshold, and feature-extraction metadata.
