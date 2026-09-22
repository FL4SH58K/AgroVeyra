# AgroVeyra — System Diagrams

All diagrams are Mermaid. Render with any Mermaid viewer (GitHub, VS Code Markdown preview, mermaid.live).

> Architecture note that matters for a reviewer: the Android app is **fully on-device** for inference and
> calls **OpenWeatherMap directly** (not the backend). The FastAPI backend is a parallel service that
> mirrors `/predict` and `/weather` but is **not consumed by the app** as shipped.

## 1. Architecture

```mermaid
flowchart TB
    subgraph PHONE["Android app (Kotlin)"]
        MA[MainActivity] --> SA[ScanActivity]
        SA --> CAM[CameraX capture]
        SA --> TF[TFLiteHelper]
        TF --> DM[agroveyra_model.tflite<br/>60 classes]
        SA --> RA[ResultActivity]
        RA --> META[DiseaseMetadata + treatment_db.json]
        RA --> SEV[severity stage]
        RA --> NET[NetworkClient / Retrofit]
        NET --> OWM1[OpenWeatherMap API]
        RA --> ROOM[(Room DB<br/>scan_history)]
        ROOM --> HA[HistoryActivity]
    end

    subgraph BACK["Backend (FastAPI) - not consumed by the app"]
        API[main.py] --> P1["POST /predict"]
        API --> W1["POST /weather"]
        W1 --> TF2[TensorFlow Lite]
        W1 --> OWM2[OpenWeatherMap API]
    end

    subgraph ML["ML pipeline (ml/)"]
        PIPE[rebuild split -> train -> export -> evaluate]
        PIPE --> DM
        PIPE --> PM[agroveyra_pest_model.tflite<br/>41 classes - trained, not wired]
    end
```

## 2. Use-case

```mermaid
flowchart LR
    U((Farmer)) --> A[Scan a leaf]
    U --> B[View disease, confidence, severity]
    U --> C[View chemical + organic treatment]
    U --> D[Check disease spread risk (weather)]
    U --> E[View / reopen scan history]
    A -.->|primary flow| APP[AgroVeyra app]
    B -.-> APP
    C -.-> APP
    D -.-> APP
    E -.-> APP
    A -.->|if confidence < 0.95| R[Retake photo prompt]
```

## 3. DFD — level 0 (context)

```mermaid
flowchart LR
    USER[[Farmer]] -->|photo, taps| APP[AgroVeyra App]
    APP -->|diagnosis + treatment + risk| USER
    APP -->|latitude, longitude| OWM[OpenWeatherMap]
    OWM -->|forecast| APP
    APP -->|(alternative mirror)| BE[Backend FastAPI]
```

## 4. DFD — level 1 (app)

```mermaid
flowchart TB
    U[[Farmer]] --> P1[1. Capture leaf photo]
    P1 --> P2[2. Classify disease<br/>TFLite on-device]
    P2 --> P3{confidence >= 0.95?}
    P3 -->|no| P4[4. Ask to retake]
    P3 -->|yes| P5[5. Look up treatment + severity]
    P5 --> P6[6. Fetch weather spread risk]
    P6 --> P7[7. Persist to Room history]
    P7 --> P8[8. Show result]
    P2 -.reads.-> DM[(disease model)]
    P5 -.reads.-> TD[(treatment_db.json)]
    P6 -.calls.-> OWM[OpenWeatherMap]
    P7 -.writes.-> RH[(scan_history)]
```

## 5. Sequence — scan flow

```mermaid
sequenceDiagram
    participant U as Farmer
    participant MA as MainActivity
    participant SA as ScanActivity
    participant CAM as CameraX
    participant TF as TFLiteHelper
    participant RA as ResultActivity
    participant OWM as OpenWeatherMap
    participant DB as Room (scan_history)

    U->>MA: open app
    MA->>SA: tap Scan
    SA->>CAM: capture leaf photo
    CAM-->>SA: bitmap
    SA->>TF: classify(bitmap)
    TF-->>SA: top-1 disease + confidence
    SA->>RA: open result (disease, confidence, image path)
    alt confidence < 0.95
        RA-->>U: prompt "retake photo" (low confidence)
    else confidence >= 0.95
        RA->>RA: severity stage + treatment lookup
        RA->>OWM: GET data/2.5/forecast (lat, lon, appid)
        OWM-->>RA: forecast
        RA->>DB: insert scan_history row
        DB-->>RA: id
        RA-->>U: show disease, severity, treatment, spread risk
    end
```

## 6. ER — persistence

Room table `scan_history` plus the two JSON lookup assets (`treatment_db.json`, `class_names.json`),
which are read-only assets keyed by disease name rather than foreign-keyed tables.

```mermaid
erDiagram
    SCAN_HISTORY {
        long id PK
        string diseaseName
        string displayName
        string crop
        float confidence
        string stage
        boolean isHealthy
        string imagePath
        long dateScanned
        string chemicalTreatment
        string organicTreatment
        string spreadRisk
    }
    TREATMENT_DB {
        string disease PK
        string chemical
        string organic
        string severity
        string prevention
    }
    CLASS_NAMES {
        int index PK
        string name
    }
    SCAN_HISTORY }o--|| CLASS_NAMES : diseaseName
    SCAN_HISTORY }o--|| TREATMENT_DB : diseaseName
```

## 7. Gantt — project timeline

```mermaid
gantt
    title AgroVeyra timeline (2026-09)
    dateFormat YYYY-MM-DD
    section Data & disease model
    Dataset rebuild + leak fix        :done, d1, 2026-09-18, 3d
    Train disease (yolo11n-cls)       :done, d2, 2026-09-19, 2d
    Export + verify TFLite            :done, d3, 2026-09-20, 1d
    Held-out test + leak audit        :done, d4, 2026-09-21, 2d
    section Pest model
    IP102 scope + near-dup quarantine :done, p1, 2026-09-20, 2d
    Train pest (yolo11n-cls)          :done, p2, 2026-09-21, 1d
    Export + gate sweep               :done, p3, 2026-09-21, 1d
    section Presentation
    Docs (README, field analysis)     :done, f1, 2026-09-22, 1d
    Diagrams                          :active, f2, 2026-09-22, 1d
    Demo photos + screenshots         :f3, 2026-09-23, 1d
    Deck (12-14 slides)               :f4, 2026-09-24, 2d
    section Optional
    Pest app integration              :crit, p4, after f4, 2d
```

