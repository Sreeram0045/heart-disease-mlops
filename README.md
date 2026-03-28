```mermaid
%%{init: {'theme': 'neutral'}}%%
graph TD
    subgraph PRODUCTION_PIPELINE [" "]
        direction TB
        UserQuery[/"Valid User Clinical Vitals Submission"/]:::query --> RenderAPI[("RENDER.COM API (Docker Container)")]:::api

        Storage[("Persistent Artifact Storage")]:::storage -.->|"robust_scaler.joblib<br/>(Robust Scaler)<br/>champion_model.joblib<br/>(Final XGBoost Model)"| FASTAPI

        subgraph FASTAPI ["FastAPI Route Processing"]
            direction TB
            RenderAPI -->|"POST /predict"| APIKey{"Security Check"}:::key
            
            APIKey -->|"Access Granted"| PydanticCheck("Pydantic Validation"):::key
            PydanticCheck -->|"FAIL"| HttpError["HTTP 422 Error"]:::fail
            
            PydanticCheck -->|"PASS"| DataMapping("Map Input to 15 Dimensions<br/>Pad missing with Zeros"):::scaler
            DataMapping -->|"Apply transform"| ScalerGate("<b>robust_scaler.joblib</b></br>Robust Scaler"):::scaler
            ScalerGate -->|"Slice to WOA features"| SlicedData("Scaled & Sliced Query"):::scaler
        end

        subgraph INFERENCE ["Inference"]
            direction TB
            XGBoost_joblib("<b>champion_model.joblib</b><br/>XGBoost Ensemble Model"):::brainxg
            XGBoost_joblib -->|"Pass ML Probability & Raw Cholesterol"| Fuzzy_py("<b>fuzzy_translator.py</b><br/>Written Rules"):::brainfuzzy
            Fuzzy_py -->|"Combine metrics"| JsonOutput("JSON Format Response"):::merged
        end

        %% By moving this connection OUTSIDE the subgraphs, 
        %% Mermaid will route it cleanly without striking through the heading.
        SlicedData --> XGBoost_joblib
    end

    %% Colorful Styling definitions
    classDef storage fill:#475569,stroke:#94a3b8,stroke-width:2px,color:#fff;
    classDef query fill:#2563eb,stroke:#60a5fa,stroke-width:2px,color:#fff;
    classDef api fill:#4f46e5,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef key fill:#ca8a04,stroke:#fde047,stroke-width:2px,color:#fff;
    classDef fail fill:#e11d48,stroke:#fda4af,stroke-width:2px,color:#fff;
    classDef scaler fill:#0891b2,stroke:#67e8f9,stroke-width:2px,color:#fff;
    classDef brainxg fill:#059669,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef brainfuzzy fill:#7c3aed,stroke:#a78bfa,stroke-width:2px,color:#fff;
    classDef merged fill:#0f766e,stroke:#5eead4,stroke-width:2px,color:#fff;
```
