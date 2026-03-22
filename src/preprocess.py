import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def encode_and_scale_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=["HeartDisease"])
    y = df_encoded["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    cols_to_scale = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

    scaler = RobustScaler().set_output(transform="pandas")

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale]).astype(
        "float32"
    )
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale]).astype(
        "float32"
    )

    joblib.dump(scaler, "../models/robust_scaler.joblib")

    return X_train_scaled, X_test_scaled, y_train, y_test


# function used to clean and optimize the data
# optimizing the data means some columns have values which doesn't reflect the real world so we change it
# to reflect the real world
# In this case the RestingBP and Cholestrol shouldn't be 0 at any point as it means that the person would be dead in the real world
def clean_and_optimize_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.duplicated().sum() != 0:
        df.drop_duplicates(inplace=True)

    # 1. Calculate the true mean (exactly as you did)
    mean_bp = df["RestingBP"].replace(0, np.nan).mean()
    mean_cholesterol = df["Cholesterol"].replace(0, np.nan).mean()

    # 2. Replace the 0s directly with the calculated means
    df["RestingBP"] = df["RestingBP"].replace(0, mean_bp)
    df["Cholesterol"] = df["Cholesterol"].replace(0, mean_cholesterol)

    df = df.fillna(df.median(numeric_only=True))
    return df


# function used to read the csv data to which the path to data file is passed
def read_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print("Path to the file is wrong")
        return None


if __name__ == "__main__":
    print("Starting preprocessing data...")

    filepath = "../data/heart.csv"

    df = read_data(filepath)

    if df is not None:
        df = clean_and_optimize_data(df)
        X_train_scaled, X_test_scaled, y_train, y_test = encode_and_scale_data(df)
        print("Success! Data cleaned, scaled, and split.")
        print(f"X_train shape: {X_train_scaled.shape}")
        print(f"X_test shape: {X_test_scaled.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
