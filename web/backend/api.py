from pathlib import Path
from functools import lru_cache
import os
import sys
import logging

from flask import Flask, jsonify, request # type: ignore
from flask_cors import CORS # pyright: ignore[reportMissingModuleSource]

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
DEFAULT_THRESHOLD = 0.53

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

CORS(
    app,
    resources={r"/api/*": {"origins": ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else "*"}},
    supports_credentials=False,
)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")

    if ALLOWED_ORIGINS == ["*"]:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"

    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response


@lru_cache(maxsize=1)
def get_inference_service():
    from src.inference_service import DEFAULT_THRESHOLD as service_threshold, predict_from_upload

    global DEFAULT_THRESHOLD
    DEFAULT_THRESHOLD = service_threshold
    return predict_from_upload


@app.get("/api/health")
def health():
    app.logger.info("Health check received")
    return jsonify(
        {
            "status": "ok",
            "service": "depression-detection-api",
        }
    )


@app.route("/api/predict", methods=["OPTIONS"])
def predict_options():
    return ("", 204)


@app.post("/api/predict")
def predict():
    app.logger.info(
        "Prediction request received from origin=%s audio_present=%s transcript_present=%s",
        request.headers.get("Origin"),
        "audio" in request.files,
        "transcript" in request.files,
    )

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
        app.logger.exception("Prediction request failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
