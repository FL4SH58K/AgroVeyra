"""AgroVeyra Backend - FastAPI application for plant disease detection."""

import json
import os

import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.predict import preprocess_image
from utils.severity import estimate_stage
from utils.weather import get_weather_risk

# Load environment variables
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/agroveyra_model.tflite")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "models/class_names.json")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

# Below this top-1 probability the classifier is guessing rather than reading the leaf. Re-measured
# for the leak-free model on the 16 real field photos (ml/confidence_gate_analysis.txt): 43.8%
# precision ungated, 54.5% at the old 0.70 default, 85.7% at 0.90 and 100% at 0.95 - where it also
# keeps every correct answer 0.70 kept, so /predict flags the answer instead of returning a
# diagnosis the caller cannot stand behind. Costs ~7 points of in-domain coverage
# (merged_clean/val 91.7% -> 84.4%). Override with LOW_CONFIDENCE_THRESHOLD in the env.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.95"))

# Global variables for model and class names
interpreter = None
class_names = []


def load_model():
    """Load TFLite model and class names at startup."""
    global interpreter, class_names

    try:
        print(f"[main.py] Loading model from: {MODEL_PATH}")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print(f"[main.py] Model loaded successfully")

        print(f"[main.py] Loading class names from: {CLASS_NAMES_PATH}")
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        print(f"[main.py] Loaded {len(class_names)} class names")
    except Exception as e:
        print(f"[main.py] Startup error: {e}")
        interpreter = None
        class_names = []


# Load resources at startup
load_model()

app = FastAPI(title="AgroVeyra Backend", version="1.0.0")

# CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WeatherRequest(BaseModel):
    lat: float
    lon: float
    disease: str


@app.get("/")
async def health_check():
    """Health check endpoint."""
    try:
        print("[main.py] Health check called")
        return {"status": "running", "app": "AgroVeyra Backend"}
    except Exception as e:
        print(f"[main.py] Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from an uploaded plant leaf image."""
    try:
        print(f"[main.py] Prediction request received: {file.filename}")

        # Read and preprocess image
        image_bytes = await file.read()
        print(f"[main.py] Image size: {len(image_bytes)} bytes")

        input_array = preprocess_image(image_bytes)

        # Run inference
        if interpreter is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], input_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # Process results
        probabilities = output[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        disease_name = class_names[class_index] if class_names else f"class_{class_index}"
        is_healthy = "healthy" in disease_name.lower()

        confidence_pct = round(confidence * 100, 1)
        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
        print(f"[main.py] Prediction: {disease_name} ({confidence_pct}%), low_confidence={low_confidence}")

        response = {
            "disease": disease_name,
            "confidence": confidence_pct,
            "is_healthy": is_healthy,
            "low_confidence": low_confidence,
        }
        if low_confidence:
            # Stays HTTP 200 so existing callers keep working, but the caller must not treat
            # `disease`/`is_healthy` as a diagnosis while this flag is set.
            response["message"] = (
                f"Low confidence ({confidence_pct}%). Retake the photo with a single leaf "
                "filling the frame in even light."
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"[main.py] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/weather")
async def weather(request: WeatherRequest):
    """Get weather-based disease spread risk assessment."""
    try:
        print(f"[main.py] Weather request: lat={request.lat}, lon={request.lon}, disease={request.disease}")

        if not OPENWEATHERMAP_API_KEY:
            raise HTTPException(status_code=500, detail="OpenWeatherMap API key not configured")

        result = get_weather_risk(request.lat, request.lon, request.disease)

        print(f"[main.py] Weather result: {result['spread_risk']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"[main.py] Weather error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


