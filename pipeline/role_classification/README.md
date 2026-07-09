# Module Reference: Speaker Role Classification and Hybrid Router

This module is responsible for predicting participant organizational roles based on conversational turn transcripts and metadata features. It implements a hybrid classification architecture utilizing local machine learning models and large language model fallback interfaces.

---

## 1. Module Overview

The `role_classification` module sits within the intermediate stage of the SpeechInSight analysis pipeline. It is executed after speech transcription and semantic segmentation, but before post-inference target resolution and behavioural scoring.

```
                      +-----------------------------+
                      |     Semantic Segments       |
                      +--------------+--------------+
                                     |
                                     ▼
                      +-----------------------------+
                      |  role_classification Module |
                      |  - TF-IDF Vectorization     |
                      |  - XGBoost Inference        |
                      |  - Gemini Fallback Router   |
                      +--------------+--------------+
                                     |
                                     ▼
                      +-----------------------------+
                      |   Target Resolution Engine  |
                      +-----------------------------+
```

---

## 2. Responsibilities

### What the Module Owns
* **Feature Extraction:** Constructing lexical vectors from aggregated speaker transcripts.
* **Direct Inference:** Running the local XGBoost model to produce initial role probabilities.
* **Agentic Routing:** Evaluating confidence and triggering the Gemini fallback API when class confidence falls below specified thresholds.
* **Normalization:** Mapping raw predictions to canonical role labels: `Leader`, `HR`, `Junior`, `Other`.
* **Provenance Compilation:** Recording complete prediction history (including raw XGBoost confidence, Gemini status, and decision sources) in the output.

### What the Module Deliberately Does NOT Own
* **Semantic Segmentation:** This module does not parse raw conversation logs; it consumes pre-grouped turns.
* **Target Speaker Election:** It does not decide which participant is the primary evaluation target. This is handled independently by the `pipeline/lead_speaker` module.
* **Vocal Emotion Analysis:** Emotion profiling is handled separately by the `pipeline/emotion` module.

---

## 3. Execution Flow

The sequence diagram below shows the processing lifecycle of a single speaker profile within this module.

```mermaid
sequenceDiagram
    participant Orchestrator as Pipeline Orchestrator
    participant Predictor as RolePredictor (predictor.py)
    +Predictor-->>Predictor: get_speaker_features()
    participant XGB as XGBoost Model (models/)
    participant Gem as Gemini Client (agentic_router.py)

    Orchestrator->>Predictor: predict_role(speaker_name, turns)
    Predictor->>XGB: predict_proba(features)
    XGB-->>Predictor: Probability Distribution
    alt Max Probability >= 0.8
        Predictor->>Predictor: Mark Source as XGBoost
    else Max Probability < 0.8
        Predictor->>Gem: route_fallback(transcript, metadata)
        Gem->>Gem: Call Gemini API
        alt Gemini API Success
            Gem-->>Predictor: Valid Role JSON
            Predictor->>Predictor: Mark Source as Gemini
        else Gemini API Failure
            Gem-->>Predictor: Exception / Timeout
            Predictor->>Predictor: Mark Source as XGBoost (Fallback)
        end
    end
    Predictor->>Predictor: Map to Canonical Label
    Predictor-->>Orchestrator: Return Provenance Dict
```

---

## 4. Public Interfaces

The primary interface for this module is located in `pipeline/role_classification/predictor.py`.

### `predict_role(speaker_name: str, transcript_data: dict, model_path: str) -> dict`
* **Inputs:**
  * `speaker_name`: String identifier of the participant (e.g., `"SPEAKER_00"`).
  * `transcript_data`: Dict containing conversational turns associated with the speaker.
  * `model_path`: Directory path pointing to model binary storage.
* **Outputs:** Returns a dictionary matching the provenance schema.
* **Expected Contracts:**
  * Must return normalized canonical roles (`Leader`, `HR`, `Junior`, or `Other`).
  * Must never throw an exception due to external API timeouts; fallbacks must fail gracefully back to XGBoost outputs.
  * Must include complete `xgboost` probability distributions.

---

## 5. Internal Architecture

The module is structured as follows:

```
role_classification/
│
├── __init__.py           # Package exports
├── feature_pipeline.py   # TF-IDF vectorization and text cleaning methods
├── agentic_router.py     # Gemini API integration and fallback prompt layouts
└── predictor.py          # Main entry point coordinating models and routers
```

* **`feature_pipeline.py`**: Handles character cleansing, vectorization scaling, and dimensional transforms.
* **`agentic_router.py`**: Coordinates communication with external LLM models and handles structured JSON response parsing.
* **`predictor.py`**: Handles classification routing, validating confidence limits, and mapping output fields.

---

## 6. Data Structures

The module returns a speaker provenance payload for each participant, which is serialized and stored in the job's `speaker_roles` block.

### Example JSON Payload
```json
{
  "speaker": "SPEAKER_01",
  "predicted_role": "Leader",
  "confidence": 0.85,
  "xgboost": {
    "role": "Leader",
    "confidence": 0.85,
    "probabilities": {
      "Leader": 0.85,
      "HR": 0.05,
      "Junior": 0.05,
      "Other": 0.05
    }
  },
  "gemini": {
    "used": false,
    "role": null
  },
  "final_role": "Leader",
  "prediction_source": "xgboost"
}
```

---

## 7. Design Decisions

* **XGBoost as Primary Classifier:** XGBoost operates locally with low latency and minimal resource consumption. This enables fast initial assessments without incurring API costs.
* **Gemini as Fallback Only:** Relying on Gemini as a fallback limit API request costs, avoids rate-limiting, and protects the system against external network outages.
* **Canonical Role Normalization:** Mapping all outputs to standard labels (`Leader`, `HR`, `Junior`, `Other`) decouples classifier variations from the downstream scoring and resolution engines.
* **Decoupled Resolver:** By separating classification from evaluation target resolution, the NLP models can be updated independently without affecting target selection logic.

---

## 8. Error Handling

The module implements fallback logic to ensure continuous processing when external dependencies fail.

```
                      Start Fallback Evaluation
                                  │
                       Trigger Gemini API Call
                                  │
                     ┌────────────┴────────────┐
             Gemini Succeeds?          Gemini Fails?
                     │             (Timeout, API Error, Bad JSON)
                     ▼                         ▼
              Use Gemini output       Log warning to console
                     │                Set prediction_source = "xgboost"
                     │                Use original XGBoost prediction
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                        Normalize Role Output
```

---

## 9. Extension Guide

### Replacing the XGBoost Classifier
1. Train a new model (e.g., Random Forest or a small local transformer).
2. Save the model files to `pipeline/role_classification/models/`.
3. Update `pipeline/role_classification/predictor.py` to load and run the new model.
4. Ensure the new model outputs a probability distribution dictionary.

### Modifying the Gemini Fallback Prompt
1. Open `pipeline/role_classification/agentic_router.py`.
2. Locate the prompt string template inside the routing module.
3. Modify the system instructions while ensuring the output remains valid JSON matching the expected keys.

### Adding a New Role Label
1. Update `mapRoleLabel` in `speech-insight-frontend/src/utils/speakerUtils.js`.
2. Update the canonical role normalization mapping in `predictor.py`.
3. Update the candidate filters in the downstream `lead_speaker/__init__.py` resolver.

---

## 10. Configuration

Configurable parameters are stored in environment variables or within system config blocks:

* `CONFIDENCE_THRESHOLD`: Probability score threshold below which the Gemini fallback is triggered (default: `0.80`).
* `GEMINI_FALLBACK_ENABLED`: Toggle to enable or disable Gemini API fallback logic (default: `True`).
* `MODEL_PATH`: Directory path pointing to local model weights storage.

---

## 11. Developer Notes

* **Invariant 1:** Never change the core canonical labels (`Leader`, `HR`, `Junior`, `Other`) without updating both the frontend utilities and the downstream target resolver.
* **Invariant 2:** The target resolver must never call LLMs or rebuild feature arrays; it must operate on finalized outputs.
* **Invariant 3:** Do not bypass provenance logging, as the frontend uses this metadata to explain classification paths in developer mode.

---

## 12. Future Improvements

### Engineering
* **Batch Fallback Processing:** Group low-confidence speakers into a single bulk Gemini request to reduce network latency and token costs.
* **Local Transformer Fallback:** Replace Gemini with a local lightweight transformer model (e.g., Llama-3-8B-Instruct) to run the fallback loop entirely on-premise.

### Research
* **Sequence-Aware Classification:** Incorporate conversation turn progression and timing features into XGBoost vector arrays.
* **Active Learning Pipeline:** Implement a pipeline to export low-confidence turns to labeling platforms, allowing continuous model retraining.