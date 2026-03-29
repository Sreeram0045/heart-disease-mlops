# 🫀 Predictive Heart Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Next.js](https://img.shields.io/badge/Next.js-14+-000000.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)

A Full-Stack, highly resilient Clinical AI Platform. This project transforms raw cardiovascular data into actionable clinical insights using a modular Machine Learning pipeline, a deterministic Fuzzy Logic safety net, and a Generative AI failover circuit breaker.

---

## 🏗️ System Architecture

This system is built on a strict "Separation of Concerns" philosophy, divided into three core pipelines:

### 1. The MLOps & Training Pipeline (Offline)
Replaced messy Jupyter Notebooks with a highly modular Python architecture:
* **Dimensionality Reduction (`woa.py`):** Utilizes the meta-heuristic Whale Optimization Algorithm (WOA) to aggressively filter 15 clinical features down to the 5 most critical predictive signals.
* **Hyperparameter Tuning (`optimize.py`):** Uses Optuna (Bayesian Search) to minimize a custom blended loss function (balancing AUC, F1, and feature cost).
* **Orchestration (`run_pipeline.py`):** Automates the ingestion, preprocessing (Robust Scaling), tuning, and training of the XGBoost ensemble model.
* **Tracking:** Fully integrated with **MLflow** to log parameters, metrics, and model binaries locally.

### 2. The Production Inference Engine (FastAPI & Docker)
A lightweight, containerized Python backend deployed via Render.com:
* **Defense in Depth:** Uses **Pydantic** to enforce strict mathematical boundaries on incoming clinical vitals, preventing "Garbage In, Garbage Out".
* **Dual-Brain Inference:** 1.  Calculates exact disease probability using the tuned **XGBoost** model.
    2.  Passes the probability and raw cholesterol through a **Scikit-Fuzzy** expert system. This deterministic logic engine overrides the ML model with a "Warning" or "Critical" score if biologically dangerous outliers are detected.

## 🛠️ Tech Stack

* **Machine Learning:** XGBoost, Scikit-Learn, Scikit-Fuzzy, pyMetaheuristic (WOA), Optuna
* **MLOps:** MLflow, Joblib, Pandas, NumPy
* **Backend:** FastAPI, Uvicorn, Pydantic, Python 3.10-slim
* **Infrastructure:** Docker, Render.com

---

## 🚀 Local Installation & Setup

### Prerequisites
* Python 3.10+
* Docker Desktop (Optional, for containerized running)
* Node.js 18+ (For Frontend)


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
```mermaid
%%{init: {'theme': 'neutral'}}%%
graph TD
    subgraph WEB_APP [" "]
        direction TB
        UserInteraction[/"USER SUBMITS FORM"/]:::query --> Vercel_UX("VERCEL FRONTEND UX (Client Check)"):::ux
        
        Vercel_UX -->|"Invalid Input"| RedUI["Input Validation Error<br/>(Highlight Fields RED)"]:::fail
        Vercel_UX -->|"Valid Data"| BackendCall("POST Request to Render"):::route
        
        BackendCall --> RenderAPI{{"RENDER.COM API /predict"}}:::api
        RenderAPI -->|"Returns Merged JSON"| BackendCall

        subgraph LLM_LOOP ["Next.js LLM API ROUTE"]
            direction TB
            LLM_Route("Next.js Request to LLM"):::route
            PromptMandate["Prompt given with certain rules"]:::mandate
            
            BackendCall -->|"Accepts the response from local model"| LLM_Route
            LLM_Route --> PromptMandate
            
            PromptMandate --> Attempt1("<b>ATTEMPT 1:</b><br/>Google SDK Call<br/><b>Gemini 3 Flash</b>"):::gemini
            
            Attempt1 -->|"Success"| SuccessRes["Human-Readable Summary"]:::emerald
            Attempt1 -.->|"Catch RateLimit / Timeout Error"| CatchBlock("Error Interceptor (Silent Failover)"):::fail
            
            CatchBlock --> Attempt2("<b>ATTEMPT 2:</b><br/>Vercel AI SDK Call<br/>OpenRouter baseURL<br/><b>Stepfun 3.5 Flash</b>"):::stepfun
            Attempt2 -->|"Success"| SuccessRes
        end

        %% The dual-data stream into the final UI
        SuccessRes -->|"Streams LLM text"| Display["FINAL UI DISPLAY<br/>(Shows ML Probabilities + LLM Summary)"]:::emerald
        BackendCall -->|"Passes JSON directly to UI"| Display
    end

    %% Colorful Styling definitions
    classDef query fill:#2563eb,stroke:#60a5fa,stroke-width:2px,color:#fff;
    classDef ux fill:#4f46e5,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef fail fill:#e11d48,stroke:#fda4af,stroke-width:2px,color:#fff;
    classDef route fill:#0891b2,stroke:#22d3ee,stroke-width:2px,color:#fff;
    classDef api fill:#ea580c,stroke:#fdba74,stroke-width:2px,color:#fff;
    classDef mandate fill:#475569,stroke:#94a3b8,stroke-width:2px,color:#fff;
    classDef gemini fill:#7c3aed,stroke:#c4b5fd,stroke-width:2px,color:#fff;
    classDef stepfun fill:#d97706,stroke:#fcd34d,stroke-width:2px,color:#fff;
    classDef emerald fill:#059669,stroke:#6ee7b7,stroke-width:2px,color:#fff;
```
```mermaid
%%{init: {'theme': 'neutral'}}%%
graph TD
    subgraph LANE_1 [" "]
        direction TB
        A[/"Raw Clinical CSV Data (15 Features)"/]:::data

        Orchestrator{{"<b>run_pipeline.py</b><br/>(Master Orchestrator)<br/>Imports and executes all modules sequentially"}}:::orchest

        subgraph MODULES ["Python Modules (Separation of Concerns)"]
            direction TB
            Config("<b>config.py</b><br/>- Centralized Base Parameters<br/>- CPU/GPU toggles"):::config
            Preprocess("<b>preprocess.py</b><br/>- Data Cleaning & Outlier Handling<br/>- Encode categorical data<br/>- RobustScaler fit/transform"):::preprocess
            WOA("<b>woa.py</b><br/>- Whale Optimization Algorithm<br/>- Narrows 11 concept features down to 5"):::woa
            Optimize("<b>optimize.py</b><br/>- Optuna Bayesian Search (50 trials)<br/>- Blended Loss Minimization"):::optimize
            Model("<b>model.py</b><br/>- XGBoost Ensemble Model"):::model
        end

        %% Orchestrator Control Flow (Dashed lines)
        Orchestrator -.->|"Executes"| Preprocess
        Orchestrator -.->|"Executes"| WOA
        Orchestrator -.->|"Executes"| Optimize
        Orchestrator -.->|"Executes"| Model

        %% Centralized Config Feeding into modules
        Config -.->|"Injects Params"| WOA
        Config -.->|"Injects Params"| Model

        %% True Data Flow (Solid lines)
        A -->|"Read CSV"| Preprocess
        Preprocess -->|"Fitted Scaler & Cleaned Data"| WOA
        WOA -->|"Selected Features List"| Optimize
        Optimize -->|"Optimized Hyperparameters"| Model

        subgraph MLOPS ["MLflow Tracking & Local Storage"]
            direction TB
            MLflow[/"MLflow Tracking Server"/]:::mlops
            WOA -.->|"log_param (features_list)"| MLflow
            Optimize -.->|"log_params (best hyperparameters)"| MLflow
            Model -.->|"log_metrics (Accuracy, Precision, Recall)<br/>log_model (XGBoost binary)"| MLflow
            
            E[("Local Directory (/models)")]:::storage
            Model -->|"champion_model.joblib"| E
            Preprocess -->|"robust_scaler.joblib"| E
        end
    end

    %% Colorful Styling definitions
    classDef data fill:#2563eb,stroke:#60a5fa,stroke-width:2px,color:#fff;
    classDef config fill:#64748b,stroke:#94a3b8,stroke-width:2px,color:#fff;
    classDef preprocess fill:#7c3aed,stroke:#a78bfa,stroke-width:2px,color:#fff;
    classDef woa fill:#db2777,stroke:#f472b6,stroke-width:2px,color:#fff;
    classDef optimize fill:#d97706,stroke:#fbbf24,stroke-width:2px,color:#fff;
    classDef model fill:#059669,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef orchest fill:#0284c7,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef mlops fill:#0f766e,stroke:#2dd4bf,stroke-width:2px,color:#fff;
    classDef storage fill:#475569,stroke:#94a3b8,stroke-width:2px,color:#fff;
```
