import numpy as np
import pandas as pd


def get_xgb_params(
    y_train: pd.Series, use_gpu: bool = False, random_state: int = 42
) -> dict:
    """
    Centralized XGBoost parameters.
    Import this into both woa.py and model.py.
    """
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

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
        xgb_params["tree_method"] = "hist"

    return xgb_params
