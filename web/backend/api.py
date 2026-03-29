from pathlib import Path
from functools import lru_cache
import os
import sys

from flask import Flask, jsonify, request # type: ignore
from flask_cors import CORS # pyright: ignore[reportMissingModuleSource]

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__)
CORS(app)
DEFAULT_THRESHOLD = 0.53


@lru_cache(maxsize=1)
def get_inference_service():
    from src.inference_service import DEFAULT_THRESHOLD as service_threshold, predict_from_upload

    global DEFAULT_THRESHOLD
    DEFAULT_THRESHOLD = service_threshold
    return predict_from_upload


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "depression-detection-api",
        }
    )


@app.post("/api/predict")
def predict():
    audio_file = request.files.get("audio")
    transcript_file = request.files.get("transcript")

    if audio_file is None or audio_file.filename == "":
        return jsonify({"error": "Audio file is required."}), 400

    threshold_raw = request.form.get("threshold", str(DEFAULT_THRESHOLD))
    try:
        threshold = float(threshold_raw)
    except ValueError:
        return jsonify({"error": "Threshold must be a valid number."}), 400

    if threshold < 0.0 or threshold > 1.0:
        return jsonify({"error": "Threshold must be between 0.0 and 1.0."}), 400

    try:
        predict_from_upload = get_inference_service()
        result = predict_from_upload(
            audio_bytes=audio_file.read(),
            transcript_bytes=transcript_file.read() if transcript_file and transcript_file.filename else None,
            threshold=threshold,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
