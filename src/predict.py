"""Inference script for City General Hospital 30-day readmission prediction.

Usage:
  python src/predict.py --input data/test.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import train_test_split

KPA_TO_MMHG = 7.50062
DEFAULT_THRESHOLD = 0.70
DROPOUT_RATE = 0.3


@dataclass
class PreprocessConfig:
    bp_kpa_threshold: float = 50.0
    date_column: str = "admission_date"
    target_column: str = "readmitted_30d"
    id_column: str = "patient_id"


def fix_blood_pressure(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Convert likely kPa values (< threshold) to mmHg in-place."""
    if "blood_pressure_systolic" not in df.columns:
        return df

    bp = df["blood_pressure_systolic"].astype(float)
    mask = bp < cfg.bp_kpa_threshold
    # Convert only the low values; the rest are assumed to already be mmHg.
    df.loc[mask, "blood_pressure_systolic"] = bp.loc[mask] * KPA_TO_MMHG
    return df


def parse_admission_date(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Parse mixed-format admission dates and expand into year/month/day."""
    if cfg.date_column not in df.columns:
        return df

    dates = pd.to_datetime(df[cfg.date_column], errors="coerce")
    df["admission_year"] = dates.dt.year
    df["admission_month"] = dates.dt.month
    df["admission_day"] = dates.dt.day
    df.drop(columns=[cfg.date_column], inplace=True)
    return df


def prepare_features(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Apply all deterministic preprocessing steps."""
    df = df.copy()
    df = fix_blood_pressure(df, cfg)
    df = parse_admission_date(df, cfg)
    # Treat impossible ages as missing (e.g., 999 sentinel).
    if "age" in df.columns:
        df.loc[df["age"] > 120, "age"] = np.nan

    # Treat coded numeric categories as categorical strings.
    for col in ["admission_type", "discharge_destination"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str)

    return df


def compute_outlier_bounds(df: pd.DataFrame, exclude_cols: List[str]) -> dict:
    """Compute 1st/99th percentile caps for numeric columns."""
    bounds = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            bounds[col] = (lower, upper)
    return bounds


def apply_outlier_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def split_features_target(
    df: pd.DataFrame, cfg: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[cfg.target_column].astype(int)
    X = df.drop(columns=[cfg.target_column])
    return X, y


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    return preprocessor


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            # tf.keras.layers.Dense(32, activation="relu"),
            # tf.keras.layers.Dropout(DROPOUT_RATE),
            # tf.keras.layers.Dense(16, activation="relu"),
            # tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def infer_feature_types(X: pd.DataFrame, cfg: PreprocessConfig) -> Tuple[List[str], List[str]]:
    categorical_cols = [
        c
        for c in X.columns
        if c != cfg.id_column
        and (
            pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
            or isinstance(X[c].dtype, pd.CategoricalDtype)
        )
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols + [cfg.id_column]]
    return categorical_cols, numeric_cols


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on validation data."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def balance_training_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Randomly oversample the minority class to match majority count."""
    data = X.copy()
    data["_target"] = y.values

    majority = data[data["_target"] == 0]
    minority = data[data["_target"] == 1]

    if len(minority) == 0 or len(majority) == 0:
        return X, y

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )

    balanced = pd.concat([majority, minority_upsampled]).sample(frac=1.0, random_state=42)
    y_bal = balanced.pop("_target").astype(int)
    return balanced, y_bal


def train_for_inference(train_path: str, cfg: PreprocessConfig):
    train_df = pd.read_csv(train_path)
    train_df = prepare_features(train_df, cfg)
    X, y = split_features_target(train_df, cfg)

    categorical_cols, numeric_cols = infer_feature_types(X, cfg)
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    # Outlier capping (train-derived percentiles).
    bounds = compute_outlier_bounds(
        X,
        exclude_cols=[cfg.id_column] + categorical_cols,
    )

    # ----- Evaluation split (metrics printed as JSON) -----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train = apply_outlier_bounds(X_train, bounds)
    X_val = apply_outlier_bounds(X_val, bounds)

    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
    X_train_bal = X_train_bal.drop(columns=[cfg.id_column])
    X_val = X_val.drop(columns=[cfg.id_column])

    X_train_proc = preprocessor.fit_transform(X_train_bal)
    X_val_proc = preprocessor.transform(X_val)
    if sp.issparse(X_train_proc):
        X_train_proc = X_train_proc.toarray()
    if sp.issparse(X_val_proc):
        X_val_proc = X_val_proc.toarray()

    tf.random.set_seed(42)
    model = build_model(input_dim=X_train_proc.shape[1])
    model.fit(
        X_train_proc,
        y_train_bal.values,
        epochs=30,
        batch_size=64,
        verbose=0,
        validation_split=0.1,
    )

    val_probs = model.predict(X_val_proc, verbose=0).ravel()
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    val_preds = (val_probs >= best_t).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, val_probs)),
        "pr_auc": float(average_precision_score(y_val, val_probs)),
        "f1": float(best_f1),
        "precision": float(precision_score(y_val, val_preds)),
        "recall": float(recall_score(y_val, val_preds)),
        "threshold": float(best_t),
    }
    print(json.dumps({"validation_metrics": metrics}, indent=2))

    # ----- Final fit on full data -----
    X_full = apply_outlier_bounds(X, bounds)
    X_full_bal, y_full_bal = balance_training_data(X_full, y)
    X_full_bal = X_full_bal.drop(columns=[cfg.id_column])

    X_full_proc = preprocessor.fit_transform(X_full_bal)
    if sp.issparse(X_full_proc):
        X_full_proc = X_full_proc.toarray()

    model = build_model(input_dim=X_full_proc.shape[1])
    model.fit(
        X_full_proc,
        y_full_bal.values,
        epochs=30,
        batch_size=64,
        verbose=0,
        validation_split=0.1,
    )

    preprocessor.outlier_bounds = bounds
    return preprocessor, model


def predict_file(
    preprocessor: ColumnTransformer,
    model: tf.keras.Model,
    input_path: str,
    output_path: str,
    cfg: PreprocessConfig,
) -> None:
    test_df = pd.read_csv(input_path)
    test_df = prepare_features(test_df, cfg)
    if hasattr(preprocessor, "outlier_bounds"):
        test_df = apply_outlier_bounds(test_df, preprocessor.outlier_bounds)

    ids = test_df[cfg.id_column]
    X_test = test_df.drop(columns=[cfg.id_column])

    X_proc = preprocessor.transform(X_test)
    if sp.issparse(X_proc):
        X_proc = X_proc.toarray()
    probs = model.predict(X_proc, verbose=0).ravel()
    preds = (probs >= DEFAULT_THRESHOLD).astype(int)
    out = pd.DataFrame({"patient_id": ids, "readmitted_30d": preds})
    out.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate readmission predictions.")
    parser.add_argument("--input", required=True, help="Path to input CSV (test set).")
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to output predictions CSV.",
    )
    parser.add_argument(
        "--train",
        default="data/train.csv",
        help="Path to training CSV.",
    )
    args = parser.parse_args()

    cfg = PreprocessConfig()
    preprocessor, model = train_for_inference(args.train, cfg)
    predict_file(preprocessor, model, args.input, args.output, cfg)


if __name__ == "__main__":
    main()
