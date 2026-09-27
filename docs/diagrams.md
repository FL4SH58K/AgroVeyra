# AgroVeyra — System Diagrams

All diagrams are Mermaid. Render with any Mermaid viewer (GitHub, VS Code Markdown preview, mermaid.live).

> Architecture note that matters for a reviewer: the Android app is **fully on-device** for inference and
> calls **OpenWeatherMap directly** (not the backend). The FastAPI backend is a parallel service that
> mirrors `/predict` and `/weather` but is **not consumed by the app** as shipped.

## 1. Architecture — three on-device models

```mermaid
flowchart TB
    subgraph PHONE["Android app (Kotlin)"]
        MA[MainActivity] --> SA[ScanActivity]
        SA --> CAM[CameraX capture]
        SA --> TRI[Triage<br/>agroveyra_triage_model.tflite<br/>3 classes]
        TRI -->|disease / healthy| TF[TFLiteHelper<br/>agroveyra_model.tflite<br/>60 classes]
        TRI -->|pest_damage| PRA[PestResultActivity]
        PRA --> Q[4-category questionnaire<br/>pest_category_db.json]
        TF --> RA[ResultActivity]
        RA --> META[DiseaseMetadata + treatment_db.json]
        RA --> SEV[severity stage]
        RA --> NET[NetworkClient / Retrofit]
        NET --> OWM1[OpenWeatherMap API]
        RA --> ROOM[(Room DB<br/>scan_history)]
        PRA --> ROOM
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
        PIPE --> DM2[agroveyra_model.tflite<br/>60 classes]
        PIPE --> PM[agroveyra_pest_model.tflite<br/>41 classes - trained, not wired]
        PIPE --> TM[agroveyra_triage_model.tflite<br/>3 classes]
    end
```

## 2. Use-case

```mermaid
flowchart LR
    U((Farmer)) --> A[Scan a leaf]
    U --> B[View disease result + severity]
    U --> C[View pest category result]
    U --> D[Answer damage questionnaire]
    U --> E[View chemical + organic treatment]
    U --> F[Hear voice advisory]
    U --> G[View weather spread risk]
    U --> H[View / reopen history]
    A -.->|primary flow| APP[AgroVeyra app]
    B -.-> APP
    C -.-> APP
    D -.-> APP
    E -.-> APP
    F -.-> APP
    G -.-> APP
    H -.-> APP
    A -.->|if confidence < 0.95| R[Retake photo prompt]
```

## 3. DFD — level 0 (context)

```mermaid
flowchart LR
    USER[[Farmer]] -->|photo, taps| APP[AgroVeyra App<br/>3 on-device models]
    APP -->|diagnosis + treatment + risk| USER
    APP -->|latitude, longitude| OWM[OpenWeatherMap]
    OWM -->|forecast| APP
    APP -->|(alternative mirror)| BE[Backend FastAPI]
```

## 4. DFD — level 1 (app)

```mermaid
flowchart TB
    U[[Farmer]] --> P1[1. Capture leaf photo]
    P1 --> P2[2. Triage: healthy / disease / pest_damage]
    P2 --> P2a{which class?}
    P2a -->|pest_damage| P3[3. Pest questionnaire -> IPM guidance]
    P2a -->|disease / healthy| P4[4. Disease classifier (60-class)]
    P4 --> P5[5. Look up treatment + severity]
    P5 --> P6[6. Fetch weather spread risk]
    P6 --> P7[7. Persist to Room history]
    P7 --> P8[8. Show result]
    P3 --> P7
    P2 -.reads.-> TRI[(triage model)]
    P4 -.reads.-> DM[(disease model)]
    P3 -.reads.-> PCD[(pest_category_db.json)]
    P5 -.reads.-> TD[(treatment_db.json)]
    P6 -.calls.-> OWM[OpenWeatherMap]
    P7 -.writes.-> RH[(scan_history)]
```

## 5. Sequence — scan flow with triage branch

```mermaid
sequenceDiagram
    participant U as Farmer
    participant SA as ScanActivity
    participant TRI as Triage model
    participant DM as Disease model
    participant PRA as PestResultActivity
    participant DB as Room (scan_history)

    U->>SA: capture leaf photo
    SA->>TRI: classify (3-class)
    TRI-->>SA: pest_damage / disease / healthy
    alt pest_damage
        SA->>PRA: open pest result (image, confidence)
        PRA->>U: "What does the damage look like?" (4 options + skip)
        U-->>PRA: pick category (or skip)
        PRA->>DB: insert scan_history (diseaseName = "pest_damage")
        PRA-->>U: show IPM guidance (chemical + organic + prevention)
    else disease / healthy
        SA->>DM: classify (60-class)
        DM-->>SA: disease + confidence
        alt confidence < 0.95
            SA-->>U: retake prompt (low confidence)
        else confidence >= 0.95
            SA->>DB: insert scan_history
            SA-->>U: show disease, severity, treatment, spread risk
        end
    end
```

## 6. ER — persistence

Room table `scan_history` plus three JSON lookup assets (`treatment_db.json`, `pest_category_db.json`,
`class_names.json`), which are read-only assets keyed by name rather than foreign-keyed tables.
Pest entries reuse the same table: `diseaseName = "pest_damage"` and `displayName` holds the category.

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
    PEST_CATEGORY_DB {
        string category PK
        string symptoms
        string chemical
        string organic
        string prevention
    }
    SCAN_HISTORY }o--|| TREATMENT_DB : diseaseName
    SCAN_HISTORY }o--|| PEST_CATEGORY_DB : "diseaseName = pest_damage"
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
    section Pest model (insect-ID)
    IP102 scope + near-dup quarantine :done, p1, 2026-09-20, 2d
    Train pest (yolo11n-cls)          :done, p2, 2026-09-21, 1d
    Export + gate sweep               :done, p3, 2026-09-21, 1d
    section Triage (pest-damage)
    Build triage dataset              :done, t1, 2026-09-26, 1d
    Train triage (3-class)            :done, t2, 2026-09-27, 1d
    Export + parity + install         :done, t3, 2026-09-27, 1d
    section App integration
    Triage routing + questionnaire    :done, a1, 2026-09-27, 1d
    Pest history integration          :done, a2, 2026-09-27, 1d
    section Presentation
    Docs (README, report, diagrams)   :active, f1, 2026-09-27, 1d
    Demo photos + screenshots         :f3, 2026-09-28, 1d
    Deck (12-14 slides)               :f4, 2026-09-28, 2d
    section Optional
    Pest insect-ID UI                 :crit, p4, after f4, 2d
```