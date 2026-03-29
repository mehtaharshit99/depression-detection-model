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
