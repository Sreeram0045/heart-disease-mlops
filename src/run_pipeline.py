import os

import joblib
import mlflow
import pandas as pd

from model import model_creation
from optimize import bayesian_optimization

# Import all our modular pieces!
from preprocess import clean_and_optimize_data, encode_and_scale_data, read_data
from woa import run_woa_feature_selection


def main():
    print("🚀 STARTING MLOPS PIPELINE 🚀\n")

    # 1. Create the models directory if it doesn't exist yet
    os.makedirs("../models", exist_ok=True)

    # 2. Setup MLflow Tracking (Absolute Path to avoid the Ghost Folder bug)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_path = os.path.join(project_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Heart_Disease_Production_Pipeline")

    # ==========================================
    # PHASE 1: DATA INGESTION & PREPROCESSING
    # ==========================================
    print("📦 PHASE 1: Loading and Preprocessing Data...")
    df = read_data("../data/heart.csv")
    if df is None:
        print("❌ CRITICAL ERROR: Could not find heart.csv")
        return

    df = clean_and_optimize_data(df)

    # NOTE: Your preprocess.py already saves the robust_scaler.joblib automatically here!
    X_train, X_test, y_train, y_test = encode_and_scale_data(df)
    print(f"✅ Data preprocessed successfully. Training shape: {X_train.shape}\n")

    # ==========================================
    # PHASE 2: FEATURE SELECTION (WOA)
    # ==========================================
    print("🐋 PHASE 2: Running Whale Optimization Algorithm...")
    selected_features = run_woa_feature_selection(X_train, y_train, use_gpu=False)

    # Slice the datasets down to ONLY the winning features
    X_train_woa = X_train[selected_features]
    X_test_woa = X_test[selected_features]
    print(f"✅ Selected {len(selected_features)} optimal features.\n")
    print(f"✅ Selected Features are: {selected_features}.\n")

    # ==========================================
    # PHASE 3: HYPERPARAMETER TUNING (OPTUNA)
    # ==========================================
    print("🧠 PHASE 3: Running Bayesian Optimization...")
    # We pass the WOA-sliced data so it tunes specifically for those features
    best_params = bayesian_optimization(
        X_train_woa, y_train, use_gpu=False, n_trials=50
    )
    print("✅ Hyperparameters optimized.\n")

    # ==========================================
    # PHASE 4: TRAIN CHAMPION MODEL & SAVE
    # ==========================================
    print("🏆 PHASE 4: Training Final Champion Model...")

    with mlflow.start_run(run_name="Production_Champion_Model"):
        # Log our winning features
        mlflow.log_param("woa_features", ", ".join(selected_features))
        mlflow.log_text("\n".join(selected_features), "selected_features.txt")

        # Train the model using the Optuna parameters
        champion_model, final_metrics = model_creation(
            X_train_woa,
            X_test_woa,
            y_train,
            y_test,
            model_name="Champion_Model",
            use_gpu=False,
            custom_params=best_params,
        )

        # Log to MLflow
        mlflow.xgboost.log_model(champion_model, "champion_model")

        # 🔥 THE CRITICAL STEP: Save the actual file to disk for FastAPI!
        model_path = "../models/champion_model.joblib"
        joblib.dump(champion_model, model_path)
        print(f"\n💾 Champion Model physically saved to: {model_path}")

    print("\n🎉 PIPELINE COMPLETE! 🎉")
    print("Your models/ folder now contains both the Scaler and the XGBoost Model.")
    print("You are ready to build the Web API.")


if __name__ == "__main__":
    main()
