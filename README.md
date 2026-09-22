# AgroVeyra

On-device crop-disease (and pest) classification with a supporting backend. The Android app points a phone camera at a leaf, runs a **YOLO11-nano** classifier **locally** on the phone (no upload for inference), and returns the disease, confidence, severity stage and treatment guidance. A FastAPI backend mirrors `/predict` and adds a weather-based disease-spread-risk endpoint.

## What's in it

| component | stack | what it does |
|---|---|---|
| Android app | Kotlin, CameraX, TFLite, Room, Retrofit | capture → on-device inference → disease/confidence → severity + treatment + history |
| Backend | FastAPI + TensorFlow Lite | `GET /`, `POST /predict`, `POST /weather` |
| ML pipeline | Python + Ultralytics | dataset rebuild, training, export, held-out evaluation |

Two trained models, both `yolo11n-cls`:

- **Disease** — 60 classes across 17 crops (PlantVillage). Shipped as `agroveyra_model.tflite`.
- **Pest** — 41 classes (rice, wheat, citrus, grape, corn, peach; IP102). Shipped as `agroveyra_pest_model.tflite`. **Trained, exported and verified; app integration pending.**

## The honest accuracy numbers

| scope | disease model |
|---|---|
| held-out test — 9,372 images, split by source photo | **97.38% top-1 / 99.64% top-5** |
| real field photos — 16 phone photos | **43.75% top-1** |
| pest test — 8,286 held-out images | **73.53% top-1 / 92.30% top-5** |

The 97.38% → 43.75% drop is **domain shift, not a defect** — proven (not argued) in `ml/field_failure_analysis.txt`: the model is ~75% right on "is this leaf healthy?" and ~33% on "which disease?", and the gap is unchanged by preprocessing, decode path, or removing leakage. At the shipped **0.95 confidence gate** the model shows only answers it can stand behind (in-domain 93.68% coverage @ 99.77% precision; on field photos 6/16 shown, all correct). Read `ml/disease_test_evaluation.txt` and its caveats before quoting any of this.

## Repo layout

```
android/   Kotlin app (ScanActivity, ResultActivity, HistoryActivity, TFLiteHelper, Room, Retrofit)
backend/   FastAPI + tensorflow-lite (main.py, utils/predict|severity|weather|voice.py, models/, treatment_db.json)
ml/        data pipeline + training + export + evaluation scripts and their reports
```

## Run it

### Backend
```bash
cd backend
pip install -r requirements.txt
copy .env.example .env      # fill OPENWEATHERMAP_API_KEY (see the BOM note in the file)
uvicorn main:app --port 8000
python test_backend.py       # 3/3 expected
```

### Android
```bat
ml\_build_apk.bat            # needs a portable Temurin 17 JDK in .tools\jdk-17
ml\_install_apk.bat
```

### ML evaluation
```bash
ml\.venv\Scripts\python.exe ml\disease_test_evaluation.py   # held-out test + leakage audit
ml\.venv\Scripts\python.exe ml\evaluate_real_world.py        # the 16 field photos
ml\.venv\Scripts\python.exe ml\score_real_world2.py          # supplementary real-world set
```

## Known limitations

- **Disease model is weakest on wheat** (80.15% in-domain; carries 215 of the 246 test errors) and on real field photos (domain shift).
- **Pest model is not yet wired into the app or backend** — it exists only as a verified asset; IP102 images are close-ups *of the insect*, there is no "no pest present" class, and rice/wheat are its weakest crops.
- **Field validation is thin** (16 phone photos, and growing in `ml/real_world_test2/`); the honest field number has a wide interval.
- The backend's OpenWeatherMap key is a placeholder by default (`your_key_here`) — `/weather` degrades to `spread_risk: UNKNOWN` until a real key is set.

## Key docs

- `ml/GPU_RUNBOOK.md` — the full training/export/evaluation runbook (§11 is the held-out test evaluation).
- `ml/field_failure_analysis.txt` — why the field accuracy is 43.75%, decomposed.
- `ml/disease_test_evaluation.txt` — the held-out number, per-class/per-crop, gate sweep, and the leakage audit.

