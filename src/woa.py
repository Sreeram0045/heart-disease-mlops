import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from pyMetaheuristic.algorithm import whale_optimization_algorithm
from sklearn.model_selection import cross_validate


def run_woa_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_gpu: bool = False,
    random_state: int = 42,
) -> list[str]:
    """
    Runs WOA to select the best logical features.
    use_gpu: Set to False for local testing, True for Colab/Kaggle.
    """
    # 1. Reproducibility
    np.random.seed(random_state)

    # 2. Calculate Imbalance Ratio dynamically
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    # 3. Setup XGBoost Parameters (CPU vs GPU)
    xgb_params = {
        "objective": "binary:logistic",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 4,
        "gamma": 1.0,
        "reg_lambda": 10.0,
        "reg_alpha": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": ratio,
        "random_state": random_state,
        "eval_metric": "logloss",
    }

    if use_gpu:
        xgb_params["tree_method"] = "hist"
        xgb_params["device"] = "cuda"
    else:
        xgb_params["tree_method"] = "hist"  # Hist runs efficiently on standard CPU

    # 4. Define Logical Feature Groups
    feature_groups = [
        [0],  # 0: Age
        [1],  # 1: RestingBP
        [2],  # 2: Cholestrol
        [3],  # 3: FastingBS
        [4],  # 4: MaxHR
        [5],  # 5: Oldpeak
        [6],  # 6: Sex_M
        [7, 8, 9],  # 7: ChestPaintype (ATA, NAP, TA)
        [10, 11],  # 8: RestingECG (Normal, ST)
        [12],  # 9: ExerciseAngina_Y
        [13, 14],  # 10: ST_Slope (Flat, Up)
    ]

    num_logical_features = len(feature_groups)

    # Convert to numpy for fast slicing
    X_train_np = X_train.to_numpy(dtype="float32")
    y_train_np = y_train.to_numpy(dtype="int32")

    # 5. Define the Fitness Function (Closure)
    def fitness_function(variables):
        probs = np.array(variables)
        selected_logical_mask = probs > 0.5
        num_selected_logical = selected_logical_mask.sum()

        # HARD CONSTRAINT: Force the algorithm to pick exactly 5 or 6 logical features
        if num_selected_logical < 5 or num_selected_logical > 6:
            return 10.0

        # Expand logical choices to actual column indices
        selected_col_indices = []
        for i, is_selected in enumerate(selected_logical_mask):
            if is_selected:
                selected_col_indices.extend(feature_groups[i])

        X_subset = X_train_np[:, selected_col_indices]
        clf = xgb.XGBClassifier(**xgb_params)
        scoring = {"auc": "roc_auc", "f1": "f1"}

        try:
            scores = cross_validate(
                clf, X_subset, y_train_np, cv=5, scoring=scoring, n_jobs=-1
            )
            error_auc = 1.0 - scores["test_auc"].mean()
            error_f1 = 1.0 - scores["test_f1"].mean()
        except Exception:
            return 10.0

        # Continuous Feature Cost (Smooth Penalty)
        feature_cost = num_selected_logical / num_logical_features

        # Blended Loss Function
        combined_loss = (0.45 * error_auc) + (0.45 * error_f1) + (0.10 * feature_cost)
        return combined_loss

    # 6. WOA Parameters
    woa_parameters = {
        "hunting_party": 50,  # Reduced slightly for faster local testing
        "iterations": 50,
        "min_values": [0.0] * num_logical_features,
        "max_values": [1.0] * num_logical_features,
        "spiral_param": 1.0,
        "verbose": True,
        "start_init": None,
        "target_value": 0.0,
    }

    print(f"Starting WOA Feature Selection (GPU Enabled: {use_gpu})... 🐋")
    solution = whale_optimization_algorithm(
        target_function=fitness_function, **woa_parameters
    )

    # 7. Decode the Best Solution
    best_variables = solution[:-1]
    final_logical_mask = np.array(best_variables) > 0.5

    selected_col_indices = []
    for i, is_selected in enumerate(final_logical_mask):
        if is_selected:
            selected_col_indices.extend(feature_groups[i])

    all_column_names = X_train.columns.tolist()
    selected_features = [all_column_names[i] for i in selected_col_indices]

    # 8. Log to MLflow if a run is active
    if mlflow.active_run():
        mlflow.log_param("woa_logical_features_kept", int(final_logical_mask.sum()))
        mlflow.log_param("woa_total_columns_kept", len(selected_features))
        mlflow.log_text(str(selected_features), "selected_features.txt")

    return selected_features


# ==========================================
# TESTING ZONE
# ==========================================
if __name__ == "__main__":
    from preprocess import clean_and_optimize_data, encode_and_scale_data, read_data

    print("Testing woa.py locally...")

    # 1. Get the preprocessed data
    df = read_data("../data/heart.csv")
    if df is not None:
        df = clean_and_optimize_data(df)
        X_train, X_test, y_train, y_test = encode_and_scale_data(df)

        # 2. Start an MLflow run strictly for testing
        mlflow.set_experiment("WOA_Local_Testing")
        with mlflow.start_run():
            # Set use_gpu=False so it runs on your CPU laptop
            best_features = run_woa_feature_selection(X_train, y_train, use_gpu=False)

            print("\n" + "=" * 40)
            print("TEST SUCCESSFUL!")
            print(f"Features Selected: {len(best_features)}")
            print(f"Feature List: {best_features}")
            print("=" * 40)
