import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import get_xgb_params


def save_and_log_figures(model, X_test, y_test, model_name):
    """
    Generates the plots, saves them locally with unique names,
    and uploads them to the active MLflow run.
    """
    os.makedirs("../reports/figures", exist_ok=True)

    # 1. Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm, cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    cm_path = f"../reports/figures/{model_name}_cm.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close(fig_cm)

    # 2. ROC-AUC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    plt.title(f"{model_name} - ROC Curve")
    roc_path = f"../reports/figures/{model_name}_roc.png"
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)
    plt.close(fig_roc)


def model_creation(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name="Baseline",
    use_gpu: bool = False,
    custom_params: dict | None = None,
):

    # If custom parameters (like Optuna's best params) are provided, use them!
    # Otherwise, fallback to the default config parameters.
    if custom_params:
        xgb_params = custom_params
    else:
        xgb_params = get_xgb_params(y_train, use_gpu=use_gpu)

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    eval_metric = {
        "Accuracy": accuracy_score(y_true=y_test, y_pred=y_pred),
        "Precision": precision_score(y_true=y_test, y_pred=y_pred),
        "Recall": recall_score(y_true=y_test, y_pred=y_pred),
        "F1": f1_score(y_true=y_test, y_pred=y_pred),
        "ROC-AUC": roc_auc_score(y_true=y_test, y_score=y_pred_proba),
    }

    # Print nicely to the terminal
    print(f"\n--- {model_name} EVALUATION METRICS ---")
    for key, value in eval_metric.items():
        print(f"{key:<12}: {value:.4f}")

    print(
        f"\nClassification Report for {model_name}: \n{classification_report(y_true=y_test, y_pred=y_pred)}"
    )

    # Log EVERYTHING to the current active MLflow run
    mlflow.log_metrics(eval_metric)
    mlflow.log_params(xgb_params)
    save_and_log_figures(model, X_test, y_test, model_name)

    return model, eval_metric


# ==========================================
# TESTING ZONE
# ==========================================
if __name__ == "__main__":
    from preprocess import clean_and_optimize_data, encode_and_scale_data, read_data
    from woa import run_woa_feature_selection

    df = read_data("../data/heart.csv")

    if df is not None:
        df = clean_and_optimize_data(df)
        X_train, X_test, y_train, y_test = encode_and_scale_data(df)

        # Force the database to ALWAYS be created in the root project folder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        db_path = os.path.join(project_root, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        mlflow.set_experiment("Heart_Disease_Model_Comparison")

        # --- RUN 1: ALL FEATURES ---
        print("\nTraining Model with ALL features...")
        with mlflow.start_run(run_name="All_Features_Baseline"):
            model_all, metrics_all = model_creation(
                X_train,
                X_test,
                y_train,
                y_test,
                model_name="Baseline_All_Features",
                use_gpu=False,
            )
            # Log the actual model file to MLflow
            mlflow.xgboost.log_model(model_all, "model")

        # --- RUN 2: WOA FEATURES ---
        print("\nRunning Whale Optimization for feature selection...")
        with mlflow.start_run(run_name="WOA_Selected_Features"):
            selected_features = run_woa_feature_selection(
                X_train, y_train, use_gpu=False
            )

            print("\nTraining Model with SELECTED features...")
            model_woa, metrics_woa = model_creation(
                X_train[selected_features],
                X_test[selected_features],
                y_train,
                y_test,
                model_name="WOA_Features",
                use_gpu=False,
            )
            # Log the actual model file to MLflow
            mlflow.xgboost.log_model(model_woa, "model")

        print("\nTesting Complete! Check the MLflow UI for figures and metrics.")
