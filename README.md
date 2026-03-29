# Depression Detection

Audio-based depression screening project built on the DAIC-WOZ dataset using Wav2Vec2 embeddings, participant-level sequence modeling, a Flask API, and a React frontend.

## Current Result

Cross-validation on the current audio-only pipeline:
- Macro F1: `0.6007`
- UAR: `0.6300`
- AUC: `0.6067`
- Accuracy: `0.6571`

## Stack

- Python
- PyTorch
- Transformers
- Flask
- React + Vite

## Project Structure

```text
dd_p/
├── data/
│   ├── DAIC-WOZ_raw/
│   ├── features_turn_level/
│   └── hf_cache/
├── models/
│   ├── best_bigru_fold0.pt
│   ├── best_bigru_fold1.pt
│   ├── best_bigru_fold2.pt
│   ├── best_bigru_fold3.pt
│   ├── best_bigru_fold4.pt
│   └── cv_results_sequence.csv
├── src/
│   ├── 01_preprocess_data.py
│   ├── 02_extract_features.py
│   ├── 03_train_sequence.py
│   ├── inference_service.py
│   └── pipeline_utils.py
├── web/
│   ├── backend/
│   │   └── api.py
│   └── frontend/
│       ├── package.json
│       ├── vite.config.js
│       └── src/
│           ├── App.jsx
│           ├── main.jsx
│           └── styles.css
├── requirements.txt
└── README.md
```

## Training Pipeline

Run in this order:

```powershell
# 1. Optional preprocessing / dataset cleanup
python src\01_preprocess_data.py

# 2. Extract participant-only chunk embeddings
python src\02_extract_features.py

# 3. Train the participant-level sequence model
python src\03_train_sequence.py
```

## Run The Website

Backend:

```powershell
.\depvenv\Scripts\Activate.ps1
python web\backend\api.py
```

Frontend:

```powershell
cd web\frontend
npm install
npm run dev
```

Default ports:
- Flask API: `http://127.0.0.1:5000`
- React app: `http://127.0.0.1:5173`

## Deployment Notes

The project is now prepared for a split deployment:

- Backend: Flask API on Render
- Frontend: React/Vite app on Vercel

Backend deployment assets now include:
- `models/best_bigru_fold*.pt`
- `models/inference_scaler.pkl`

### Backend on Render

- Use the root of the repository
- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
gunicorn web.backend.api:app
```

The repository also includes `render.yaml` with this configuration.

### Frontend on Vercel

Set the frontend root directory to:

```text
web/frontend
```

Set the environment variable:

```text
VITE_API_BASE_URL=https://your-backend-url.onrender.com
```

Then build normally with Vite.

## Website Flow

- Upload a `.wav` audio file
- Optionally upload the matching transcript CSV
- The backend extracts Wav2Vec2 Layer 9 chunk embeddings
- If a transcript is provided, participant-only speech is isolated before inference
- The ensemble of trained BiGRU models returns the final result

## Main Files

- `src/02_extract_features.py`
  Extracts participant-only Wav2Vec2 chunk embeddings from DAIC-WOZ interview audio.

- `src/03_train_sequence.py`
  Trains the BiGRU + attention classifier with weighted BCE, fold-wise standardization, and cross-validation.

- `src/inference_service.py`
  Shared prediction logic used by the Flask API.

- `web/backend/api.py`
  Flask API for health checks and inference requests.

- `web/frontend/src/App.jsx`
  React frontend for the website.

## Notes

- This is a research/demo project, not a clinical diagnostic system.
- Best behavior is expected on DAIC-style interview audio.
- Uploading the transcript CSV helps the app match training conditions more closely by focusing on participant speech.
