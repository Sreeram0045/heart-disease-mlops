import os

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score


def bayesian_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_gpu: bool = False,
    random_state: int = 42,
    n_trials: int = 50,
) -> dict:
    """
    Runs Optuna to find the absolute best XGBoost hyperparameters for the given data.
    Returns a dictionary of the best parameters.
    """
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 0.5, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # Fixed parameters
            "scale_pos_weight": ratio,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": random_state,
        }

        if use_gpu:
            params["device"] = "cuda"

        clf = xgb.XGBClassifier(**params)

        # Use 5-Fold CV to evaluate how good these parameters are
        scores = cross_val_score(
            clf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    print("Starting Bayesian Optimization (Finding the best hyperparameters)... 🧠")

    # Mute Optuna's print statements so it doesn't flood your terminal
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Optimization Complete! Best ROC-AUC Score: {study.best_value:.4f}")

    # Reconstruct the absolute best parameter dictionary
    best_params = study.best_params

    # We must add back the fixed parameters that Optuna wasn't searching for
    best_params.update(
        {
            "scale_pos_weight": ratio,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": random_state,
        }
    )

    if use_gpu:
        best_params["device"] = "cuda"

    return best_params


# ==========================================
# TESTING ZONE
# ==========================================
if __name__ == "__main__":
    from model import model_creation
    from preprocess import clean_and_optimize_data, encode_and_scale_data, read_data
    from woa import run_woa_feature_selection

    df = read_data("../data/heart.csv")

    if df is not None:
        df = clean_and_optimize_data(df)
        X_train, X_test, y_train, y_test = encode_and_scale_data(df)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        db_path = os.path.join(project_root, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        mlflow.set_experiment("Heart_Disease_Model_Comparison")

        # 1. Get the WOA Features (We don't need to log this run here, just get the data)
        print("\nRunning Whale Optimization for feature selection...")
        selected_features = run_woa_feature_selection(X_train, y_train, use_gpu=False)

        # Slice the data down to just the best features
        X_train_woa = X_train[selected_features]
        X_test_woa = X_test[selected_features]

        # 2. RUN BAYESIAN OPTIMIZATION
        # We pass the SLICED data so it optimizes specifically for those features!
        best_optuna_params = bayesian_optimization(
            X_train_woa, y_train, use_gpu=False, n_trials=50
        )

        # 3. TRAIN THE FINAL CHAMPION MODEL
        print("\nTraining CHAMPION Model with WOA features and Bayesian Parameters...")
        with mlflow.start_run(run_name="Champion_WOA_Bayesian"):
            # Log the WOA features explicitly
            mlflow.log_param("woa_features", ", ".join(selected_features))
            mlflow.log_text("\n".join(selected_features), "selected_features.txt")

            # Pass our custom Optuna dictionary into the model creator!
            model_champ, metrics_champ = model_creation(
                X_train_woa,
                X_test_woa,
                y_train,
                y_test,
                model_name="Champion_Model",
                use_gpu=False,
                custom_params=best_optuna_params,  # <--- The Magic Link
            )

            mlflow.xgboost.log_model(model_champ, "champion_model")

        print("\nTesting Complete! Check the MLflow UI.")
