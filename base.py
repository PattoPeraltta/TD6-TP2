import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

# Adjust this path if needed
COMPETITION_PATH = "./competition_data"


def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    """
    Load train and test datasets, optionally sample a fraction of the training set,
    concatenate, and reset index.
    """
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "train_data.txt")
    test_file = os.path.join(data_dir, "test_data.txt")

    # Load training data and optionally subsample
    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    # Load test data
    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    # Concatenate and reset index
    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    return combined

def cast_column_types(df):
    """
    Cast columns to efficient dtypes and parse datetime fields.
    """
    print("Casting column types and parsing datetime fields...")
    dtype_map = {
        "platform": "category",
        "conn_country": "category",
        "ip_addr": "category",
        "master_metadata_track_name": "category",
        "master_metadata_album_artist_name": "category",
        "master_metadata_album_album_name": "category",
        "reason_end": "category",
        "username": "category",
        "spotify_track_uri": "string",
        "episode_name": "string",
        "episode_show_name": "string",
        "spotify_episode_uri": "string",
        "audiobook_title": "string",
        "audiobook_uri": "string",
        "audiobook_chapter_uri": "string",
        "audiobook_chapter_title": "string",
        "shuffle": bool,
        "offline": bool,
        "incognito_mode": bool,
        "obs_id": int,
    }

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(
        df["offline_timestamp"], unit="s", errors="coerce", utc=True
    )
    df = df.astype(dtype_map)
    print("  → Column types cast successfully.")
    return df

def split_train_test(X, y, test_mask):
    """
    Split features and labels into train/test based on mask.
    """
    print("Splitting data into train/test sets...")
    
    X_train = X
    X_test = X
    y_train = y
    y_test = y

    print(f"  → Training set: {X_train.shape[0]} rows")
    print(f"  → Test set:     {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train, params=None):
    """
    Train a Classifier 
    """
    print("Training model...")
    
    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
        "bootstrap": True,
    }
    rf_params = default_params.copy()
    if params:
        rf_params.update(params)

    model = RandomForestClassifier(**rf_params)

    print("  → Fitting RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("  → Model training complete.")
    return model

def main():
    print("=== empezando ===")

    # 1) cargar y castear
    df = load_competition_datasets(COMPETITION_PATH, sample_frac=0.2, random_state=1234)
    df = cast_column_types(df)

    # 2) ordenar y crear una feature simple (sumá más después)
    df = df.sort_values(["username", "ts"])
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1
    df = df.sort_values("obs_id")

    # 3) target e indicador de test
    print("creando target e is_test...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()

    # 4) guardar labels para mergear después (solo donde hay label)
    labels_df = df.loc[~df["is_test"], ["obs_id", "reason_end"]].copy()
    labels_df["objetivo"] = (labels_df["reason_end"] == "fwdbtn").astype(int)

    # 5) definir features limpias 
    feature_cols = ["user_order"]  # agregá más columnas del dataset después
    X_all = df[feature_cols].copy()
    y_all = df["target"].to_numpy()

    # 6) separar filas con y (train/valid) de las sin y (test real kaggle)
    idx_labeled = df.index[~df["is_test"]]
    idx_test = df.index[df["is_test"]]

    X_labeled = X_all.loc[idx_labeled]
    y_labeled = y_all[~df["is_test"].to_numpy()]

    X_test = X_all.loc[idx_test]
    test_obs_ids = df.loc[idx_test, "obs_id"].to_numpy()

    # 7) split para evaluar sin fugas (20% valid estratificado)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, random_state=42
    )

    # 8) entrenar
    model = train_classifier(X_tr, y_tr)

    # 9) evaluar en valid
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    )
    val_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (val_proba >= 0.5).astype(int)

    print("metricas en valid (sin humo):")
    print("  roc auc:", roc_auc_score(y_val, val_proba))
    print("  accuracy:", accuracy_score(y_val, y_pred))
    print("  precision:", precision_score(y_val, y_pred))
    print("  recall:", recall_score(y_val, y_pred))
    print("  f1:", f1_score(y_val, y_pred))
    
    # 10) reentrenar en TODO el labeled para predecir kaggle test (opcional pero prolijo)
    model_full = train_classifier(X_labeled, y_labeled)

    # 11) predecir test real y exportar para kaggle
    test_proba = model_full.predict_proba(X_test)[:, 1]
    preds_kaggle = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": test_proba})
    preds_kaggle.to_csv("modelo_benchmark.csv", index=False)

    print("→ exportados:")
    print("   - modelo_benchmark.csv (formato kaggle)")

    print("=== listo ===")


if __name__ == "__main__":
    main()
