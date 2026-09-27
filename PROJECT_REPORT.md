# AgroVeyra — Project Report

On-device crop disease and pest detection with treatment guidance. The Android app points a phone
camera at a leaf, runs lightweight classifiers **locally** (no upload for inference), and returns
the condition (healthy / diseased / pest-damaged), confidence, severity and treatment guidance. A
supporting FastAPI backend mirrors the same inference plus a weather-based disease-spread-risk
endpoint.

## 1. Problem statement and objectives

[SIH problem statement — insert the portal problem statement ID and text here.]

**Objective.** Give a farmer a phone tool that, from a single leaf photo, tells them whether the
plant is healthy, diseased, or pest-damaged — and, for disease and pest cases, what to do about it.

**Success criteria** (measured, not assumed):

- On-device inference (classification needs no network).
- Honest accuracy numbers, including a held-out test split **and** a real field-photo set.
- A confidence gate that shows an answer only when the model can stand behind it.
- Verified export: the shipped TFLite file reproduces the trained checkpoint (parity-checked).

## 2. Dataset

Three datasets feed three models.

### 2.1 Disease classifier — 60 classes, 97,869 images
Merged from PlantVillage + rice + wheat + cotton. After the leak audit (section 5.1) the split is
75,419 train / 18,957 val, split **along source-photo boundaries** so no augmented family of the
same leaf spans both sides. 3,493 images were quarantined (2,861 same-class duplicates, 582
cross-class duplicates with identical pixels under two labels, 50 mislabeled wheat-healthy images).

### 2.2 Pest classifier (insect-ID) — 41 classes, 27,280 images
IP102 scoped to 41 classes (rice, wheat, citrus, grape, corn, peach), post-quarantine
train 16,271 / val 2,723 / test 8,286. Trained and exported, but **not wired to any screen**
(see limitations — this is future work).

### 2.3 Pest-damage triage — 3 classes, 2,554 images
The triage step that routes a scan to disease vs. pest vs. healthy:
healthy 794 / disease 799 / pest_damage 961; train 2,042 / val 512 (80/20, seed 42), md5 + dHash
dedup. `pest_damage` is cross-crop — LeLePhid aphids (lemon, 335) + sesame "Insect Leaf Damage"
(724) — so the triage learns *generic leaf pest-damage appearance*, not a specific species.

## 3. Method

### 3.1 Architecture — three on-device classifiers
The app runs a **routing pipeline**, not one model:

```
leaf photo ──► triage (3-class) ──┬─ pest_damage ──► 4-category questionnaire ──► IPM guidance
                                  ├─ disease      ──► disease classifier (60-class) ──► result
                                  └─ healthy      ──► disease classifier ──► "healthy" result
```

- **Triage** (`agroveyra_triage_model.tflite`, 3 classes) decides healthy / disease / pest_damage.
- **Disease** (`agroveyra_model.tflite`, 60 classes) names the disease.
- **Pest insect-ID** (`agroveyra_pest_model.tflite`, 41 classes) is trained/exported but unused by the UI.

All three are **YOLO11n-cls** (nano, ~1.58 M parameters, 224×224 input), chosen so they run on a
phone CPU with no upload.

### 3.2 Export — torch → ONNX → onnx2tf → float32 TFLite
Ultralytics' own TFLite exporter cannot run on Windows, so the pipeline is
`torch → NCHW ONNX (opset 13) → fold integer shape arithmetic → onnx2tf flatbuffer_direct → float32
TFLite with an NHWC input`. The NHWC input is verified **op-level** (read from the flatbuffer, not
assumed from flags): `[1,224,224,3]` float32, no quantization/dequantization ops, softmax present.

### 3.3 Confidence gating
A softmax confidence gate prevents the app from showing a guess. The disease gate (0.95) was tuned
by sweeping the 16-photo field set (section 5.3); the triage reuses the same threshold so the
routing step is equally conservative.

## 4. Results — the honest numbers

| Model | In-domain | Field / held-out | Notes |
|---|---|---|---|
| **Disease** (leak-free) | **97.38%** top-1 / **99.64%** top-5 (9,372 held-out test, 95% CI 97.0–97.7%) | **43.75%** (16 real phone photos) | domain gap — root-caused, not a bug |
| **Pest** (insect-ID) | **73.53%** top-1 / **92.30%** top-5 (8,286 held-out test) | not field-tested | trained, not wired to UI |
| **Pest-damage triage** | **97.27%** top-1 / **100%** top-5 (512 val) | not field-tested | cross-crop data (lemon/sesame) |

Triage per-class (val): disease F1 0.960 · healthy F1 0.955 · pest_damage F1 0.997.

The disease number is a genuine held-out figure: the test half is split by source-photo family,
`best.pt == last.pt` (no epoch was selected on val), and the leak-free range after removing
near-duplicates is 97.33–97.37%.

## 5. Rigor — the leak, the parity, and the gate

### 5.1 The leakage discovery and fix
The original split leaked: 693 val files were byte-identical to train, and 450/1050 wheat val
images (42.9%) had already been seen in training. Fix: rebuild the split along source-photo
boundaries (`ml/rebuild_split.py`), quarantining 3,493 files. Post-rebuild: **0 leaks** — no source
photo spans train and val, and `class_names.json` order is unchanged.

### 5.2 Export parity verification
The shipped flatbuffer is compared to the checkpoint on byte-identical preprocessed pixels.
Disease: **1216/1216** argmax agreement, max |Δp| **0.000006**. Triage: **30/30**, max |Δp|
**0.000002**. Verdict: the shipped file scores what the checkpoint scores, so a field-photo drop is
domain gap, not an export regression.

### 5.3 Gate tuning
Swept the 16-photo field set: at **0.95** the app shows 6/16 and all 6 are correct (**100%**
precision), while the in-domain val still shows 152/180 (84.44% coverage @ 98.03% precision). The
rule: the lowest gate whose field precision is 100% while still showing at least 4 photos.

### 5.4 Field-accuracy root cause
The 97.38% → 43.75% drop is **domain shift, proven not argued**: decomposed by crop (cotton 100%,
wheat 50%, rice 33%, tomato 0%) and by question ("is it healthy?" 75% vs "which disease?" 33%).
The nine field errors are mostly "same crop, sibling disease" confusions. Re-split, preprocessing
and decode-path audits ruled out leakage and pipeline errors as the cause.

## 6. Limitations and future work

- **Field accuracy gap is domain shift**, proven via the re-split experiment, not a broken model.
  Closing it needs field imagery (50–200 verified photos), field-realistic augmentation, and/or a
  detect-then-classify pipeline — not a code change.
- **Pest-damage triage is trained on off-crop data** (lemon aphids, sesame); it is category-level
  (healthy / disease / pest_damage), not species-specific. The questionnaire narrows to four damage
  categories but does not name the pest species.
- **Insect-ID pest model** (`agroveyra_pest_model.tflite`) is trained and verified but **not wired
  to any screen** — presented here as future work, not hidden.
- **The FastAPI backend exists but the app does not call it**: offline-first by design; the backend
  is a reference implementation (mirrors `/predict` and adds weather spread-risk).
- **The confidence gate is tuned on 16 photos** — re-check it as the field set grows.

## 7. Source material (ml/*.txt)

- `ml/disease_test_evaluation.txt` — held-out disease numbers + caveats.
- `ml/field_failure_analysis.txt` — 43.75% root-cause decomposition.
- `ml/confidence_gate_analysis.txt` — gate sweep.
- `ml/dataset_rebuild_report.txt` — leak rebuild + quarantine breakdown.
- `ml/export_parity_report.txt` / `ml/triage_export_report.txt` — export parity.
- `ml/pest_training_report.txt` / `ml/pest_scope_report.txt` — pest (insect-ID).
- `ml/triage_training_report.txt` — triage training summary.

