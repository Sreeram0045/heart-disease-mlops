import os
import sys

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# Add the root directory to the path so we can import our fuzzy logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import fuzzy_translator

# ==========================================
# 1. SETUP & SECURITY
# ==========================================
app = FastAPI(
    title="Secure Heart Disease API",
    description="A highly optimized, secure AI diagnostic endpoint.",
    version="1.0.0",
)

API_KEY = os.getenv("HEART_API_KEY", "password")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")
    return api_key


# ==========================================
# 2. LOAD RESOURCES
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/champion_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/robust_scaler.joblib")

try:
    champion_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Could not find model files. Did you run src/run_pipeline.py?")

WOA_FEATURES = [
    "Cholesterol",
    "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA",
    "ChestPainType_NAP",
    "ChestPainType_TA",
    "ST_Slope_Flat",
    "ST_Slope_Up",
]


# ==========================================
# 3. LEAN INCOMING JSON SCHEMA
# ==========================================
# We ONLY ask for the 5 traits that WOA actually cares about!
class PatientData(BaseModel):
    Sex: str  # 'M' or 'F'
    ChestPainType: str  # 'TA', 'ATA', 'NAP', 'ASY'
    Cholesterol: int
    Oldpeak: float
    ST_Slope: str  # 'Up', 'Flat', 'Down'


# ==========================================
# 4. THE SECURE API ENDPOINT
# ==========================================
@app.post("/predict")
def predict_heart_disease(patient: PatientData, api_key: str = Depends(verify_api_key)):
    try:
        data = patient.dict()

        # Step A: Build the full 15-column format, padding the unneeded columns with 0s
        processed_data = {
            "Age": 0,  # Padded
            "RestingBP": 0,  # Padded
            "Cholesterol": data["Cholesterol"],
            "FastingBS": 0,  # Padded
            "MaxHR": 0,  # Padded
            "Oldpeak": data["Oldpeak"],
            "Sex_M": 1 if data["Sex"] == "M" else 0,
            "ExerciseAngina_Y": 0,  # Padded
            "ChestPainType_ATA": 1 if data["ChestPainType"] == "ATA" else 0,
            "ChestPainType_NAP": 1 if data["ChestPainType"] == "NAP" else 0,
            "ChestPainType_TA": 1 if data["ChestPainType"] == "TA" else 0,
            "RestingECG_Normal": 0,  # Padded
            "RestingECG_ST": 0,  # Padded
            "ST_Slope_Flat": 1 if data["ST_Slope"] == "Flat" else 0,
            "ST_Slope_Up": 1 if data["ST_Slope"] == "Up" else 0,
        }

        # Convert to a single-row Pandas DataFrame
        df_patient = pd.DataFrame([processed_data])

        # Step B: Scale in-place (The robust Pandas way!)
        cols_to_scale = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        df_patient[cols_to_scale] = scaler.transform(df_patient[cols_to_scale]).astype(
            "float32"
        )

        # Step C: Filter down to ONLY the 8 features WOA selected and force float type
        X_final = df_patient[WOA_FEATURES].astype(float)

        # Step D: Predict Probabilities
        prob_safe = float(champion_model.predict_proba(X_final)[0, 0])
        prob_disease = float(champion_model.predict_proba(X_final)[0, 1])

        # Step E: Fuzzy Logic Translation
        raw_cholesterol = float(data["Cholesterol"])
        final_verdict = fuzzy_translator.generate_linguistic_inference(
            prob_disease, raw_cholesterol
        )

        # Step F: The Context-Rich Response
        return {
            "status": "success",
            "patient_profile": data,
            "ml_probabilities": {
                "safe_chance": round(prob_safe, 3),
                "disease_chance": round(prob_disease, 3),
            },
            "fuzzy_risk_score": final_verdict["fuzzy_risk_score"],
            "fuzzy_verdict": final_verdict["fuzzy_verdict"],
            "driving_features": final_verdict["driving_features"],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def health_check():
    return {"status": "Secure API is running. Requires X-API-Key header."}
