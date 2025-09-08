# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


pd.set_option("display.max_columns", None)

# ======= Ajustá este path si hace falta =======
COMPETITION_PATH = "Datos"

# Ventanas temporales clave
VAL_START = pd.Timestamp("2024-08-01", tz="UTC")
VAL_END   = pd.Timestamp("2024-09-01", tz="UTC")  # [ago1, sep1)

# ------------------------------------------------
# 1) LOADERS & CASTING
# ------------------------------------------------
def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    """
    Load train and test datasets, optionally sample a fraction of the training set,
    concatenate, and reset index.
    """
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "train_data.txt")
    test_file = os.path.join(data_dir, "test_data.txt")

    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  -> Concatenated DataFrame: {combined.shape[0]:,} rows")
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
        "shuffle": "boolean",
        "offline": "boolean",
        "incognito_mode": "boolean",
        "obs_id": "int64",
    }

    # Parse timestamps
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # offline_timestamp puede venir en epoch/segundos ya redondeado
    df["offline_timestamp"] = pd.to_datetime(
        df["offline_timestamp"], unit="s", errors="coerce", utc=True
    )

    # Cast rest
    for col, typ in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(typ)
            except Exception:
                # Si falla el cast (por valores raros), dejamos como está
                pass

    print("  -> Column types cast successfully.")
    return df


# ------------------------------------------------
# 2) FEATURE ENGINEERING (BÁSICO Y SEGURO)
# ------------------------------------------------
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features sin leakage y baratas:
      - Orden por usuario (user_order)
      - Componentes temporales: hora, día de semana, finde, mes
      - Flags de tipo de contenido (song/podcast/audiobook)
      - Señales de modo/ctx: shuffle, offline, incognito, platform, conn_country
    """
    print("Generating basic features...")

    # Orden por usuario: requiere ordenar por username+ts
    df = df.sort_values(["username", "ts"])
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1
    df = df.sort_values(["obs_id"])

    # Componentes temporales (UTC, consistentes con ts)
    df["hour"] = df["ts"].dt.hour.astype("Int16")
    df["dow"] = df["ts"].dt.dayofweek.astype("Int16")  # 0=Mon
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("Int8")
    df["month"] = df["ts"].dt.month.astype("Int16")

    # Tipo de contenido (mutuamente excluyentes en la práctica)
    is_song = df["spotify_track_uri"].notna()
    is_episode = df["spotify_episode_uri"].notna()
    is_audiobook = df["audiobook_uri"].notna()
    df["is_song"] = is_song.astype("Int8")
    df["is_episode"] = is_episode.astype("Int8")
    df["is_audiobook"] = is_audiobook.astype("Int8")

    # Modo/ctx (ya son boolean, pero aseguramos Int8 para ML)
    for col in ["shuffle", "offline", "incognito_mode"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype("Int8")

    return df


# ------------------------------------------------
# 3) MATRIZ DE DISEÑO (NUM + OHE CATEG)
# ------------------------------------------------
def build_design_matrix(df: pd.DataFrame):
    """
    Devuelve X (sparse CSR), y, feature_names y máscara de test.
    One-hot encoding esparso para categóricas.
    """
    # Crear target y máscara test
    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype("Int8")
    df["is_test"] = df["reason_end"].isna()

    # Columnas base a conservar para splits/predicción
    keep_cols = [
        "obs_id", "username", "ts", "target", "is_test",
        # numéricas
        "user_order", "hour", "dow", "is_weekend", "month",
        "is_song", "is_episode", "is_audiobook",
        "shuffle", "offline", "incognito_mode",
        # categóricas
        "platform", "conn_country"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Definir numéricas y categóricas
    numeric_cols = [
        "user_order", "hour", "dow", "is_weekend", "month",
        "is_song", "is_episode", "is_audiobook",
        "shuffle", "offline", "incognito_mode",
    ]
    categorical_cols = ["platform", "conn_country"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # One-hot esparso
    print("Building sparse design matrix (numeric + OHE)...")
    X_num = df[numeric_cols].astype("float32")

    if len(categorical_cols) > 0:
        X_cat = pd.get_dummies(
            df[categorical_cols],
            dummy_na=True,
            sparse=True
        )
        # Concatenar (sparse)
        X = sparse.hstack(
            [sparse.csr_matrix(X_num.values), sparse.csr_matrix(X_cat.sparse.to_coo())],
            format="csr"
        )
        feature_names = list(numeric_cols) + list(X_cat.columns.astype(str))
    else:
        X = sparse.csr_matrix(X_num.values)
        feature_names = list(numeric_cols)

    y = df["target"].astype("float32").to_numpy()
    test_mask = df["is_test"].to_numpy()

    # Guardar referencias clave para splits y submit
    refs = df[["obs_id", "username", "ts", "is_test"]].copy()

    print(f"  -> X shape: {X.shape}, nnz: {X.nnz:,}")
    return X, y, feature_names, test_mask, refs


# ------------------------------------------------
# 4) VALIDATION SPLITS
# ------------------------------------------------
def make_validation_split(refs: pd.DataFrame,
                          mode: str = "warm",
                          valid_user_frac: float = 0.5,
                          seed: int = 42,
                          min_train_rows: int = 1000):
    """
    Devuelve máscaras booleanas para TRAIN y VALID dentro de las filas etiquetadas (is_test == False).

    mode:
      - "warm": train = ts < 2024-08-01, valid = 2024-08  (usuarios pueden repetirse)
      - "cold": group split por usuario **submuestreando** usuarios de agosto para VALID.
                (usuarios disjuntos; si el train queda vacío o muy chico, se reduce la fracción)
    """
    assert mode in ("warm", "cold"), "mode must be 'warm' or 'cold'"

    not_test = ~refs["is_test"]
    in_aug = (refs["ts"] >= VAL_START) & (refs["ts"] < VAL_END)
    before_aug = refs["ts"] < VAL_START

    if mode == "warm":
        train_mask = not_test & before_aug
        valid_mask = not_test & in_aug

        tr_users = refs.loc[train_mask, "username"].nunique()
        va_users = refs.loc[valid_mask, "username"].nunique()
        overlap = len(set(refs.loc[train_mask, "username"]) & set(refs.loc[valid_mask, "username"]))
        print("\n=== VALIDATION SPLIT: WARM-START (Temporal) ===")
        print(f"TRAIN rows: {train_mask.sum():,} | TRAIN users: {tr_users:,}")
        print(f"VALID rows: {valid_mask.sum():,} | VALID users: {va_users:,}")
        print(f"User overlap (esperado > 0 en warm): {overlap:,}")
        return train_mask, valid_mask

    # ---- COLD-START ROBUSTO (subset de usuarios de agosto) ----
    users_in_aug = pd.Index(refs.loc[not_test & in_aug, "username"].dropna().unique())
    rng = np.random.default_rng(seed)

    if len(users_in_aug) == 0:
        print("\n[WARN] No hay usuarios con reproducciones en agosto. Fallback a WARM.")
        return make_validation_split(refs, mode="warm", seed=seed)

    # Tomamos un subset de usuarios de agosto para VALID
    valid_frac = float(valid_user_frac)
    nn = max(1, int(np.ceil(valid_frac * len(users_in_aug))))
    valid_users = set(rng.choice(users_in_aug, size=nn, replace=False))

    is_user_in_valid = refs["username"].isin(valid_users)
    valid_mask = not_test & is_user_in_valid
    train_mask = not_test & (~is_user_in_valid)  # incluye usuarios que NO están en el subset

    # Si por el sample queda train muy chico o vacío, reducimos valid_frac iterativamente
    while (train_mask.sum() < min_train_rows) and (valid_frac > 0.05):
        valid_frac *= 0.5
        nn = max(1, int(np.ceil(valid_frac * len(users_in_aug))))
        valid_users = set(rng.choice(users_in_aug, size=nn, replace=False))
        is_user_in_valid = refs["username"].isin(valid_users)
        valid_mask = not_test & is_user_in_valid
        train_mask = not_test & (~is_user_in_valid)

    # Si aun así train quedó vacío, fallback a WARM
    if train_mask.sum() == 0:
        print("\n[WARN] TRAIN quedó vacío con cold-start incluso tras reducir fracción. Fallback a WARM.")
        return make_validation_split(refs, mode="warm", seed=seed)

    tr_users = refs.loc[train_mask, "username"].nunique()
    va_users = refs.loc[valid_mask, "username"].nunique()
    overlap = len(set(refs.loc[train_mask, "username"]) & set(refs.loc[valid_mask, "username"]))
    print("\n=== VALIDATION SPLIT: COLD-START (Usuarios disjuntos, subset agosto) ===")
    print(f"valid_user_frac usado: {valid_frac:.3f}  |  usuarios_agosto_total: {len(users_in_aug):,}")
    print(f"TRAIN rows: {train_mask.sum():,} | TRAIN users: {tr_users:,}")
    print(f"VALID rows: {valid_mask.sum():,} | VALID users: {va_users:,}")
    print(f"User overlap (debe ser 0): {overlap:,}")
    return train_mask, valid_mask


# ------------------------------------------------
# 5) MODEL
# ------------------------------------------------
def train_classifier(X_train, y_train, params=None, progress=True, progress_step=50):
    """
    Entrena un RandomForest. Si progress=True, entrena en tandas con warm_start
    y muestra el % completado (árboles entrenados sobre total).
    """
    print("Training RandomForest...")
    default_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
        "bootstrap": True,
        # 'verbose': 2,  # alternativa: logs internos de sklearn (no en % exacto)
    }
    rf_params = default_params if params is None else {**default_params, **params}

    if not progress:
        model = RandomForestClassifier(**rf_params)
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"  -> Model trained. Elapsed: {time.time() - t0:.1f}s")
        return model

    # ---- Entrenamiento con progreso ----
    total = int(rf_params.get("n_estimators", 300))
    step = max(1, int(progress_step))
    step = min(step, total)

    # empezamos con pocos árboles y vamos sumando
    rf_params["n_estimators"] = step
    rf_params["warm_start"] = True
    rf_params["verbose"] = 0  # usamos nuestro propio logger
    model = RandomForestClassifier(**rf_params)

    trained = 0
    t0 = time.time()
    while trained < total:
        trained = min(trained + step, total)
        model.set_params(n_estimators=trained)
        model.fit(X_train, y_train)  # warm_start: agrega árboles nuevos
        pct = 100.0 * trained / total
        print(f"[RF] {trained}/{total} árboles ({pct:.1f}%) | elapsed {time.time() - t0:.1f}s")

    print(f"  -> Model trained. Total elapsed: {time.time() - t0:.1f}s")
    return model



# ------------------------------------------------
# 6) MAIN
# ------------------------------------------------
def main(
    validation_mode: str = "cold",   # "warm" o "cold"
    sample_frac: float | None = 0.2, # podés subir esto cuando ya esté estable
    random_state: int = 1234
):
    print("=== Starting pipeline ===")

    # 1) Cargar y castear
    df = load_competition_datasets(COMPETITION_PATH, sample_frac=sample_frac, random_state=random_state)
    df = cast_column_types(df)

    # 2) Features básicas
    df = add_basic_features(df)

    # 3) Matriz de diseño (X es esparsa)
    X, y, feature_names, test_mask, refs = build_design_matrix(df)

    # 4) Split de validación elegido
    train_mask_lbl, valid_mask_lbl = make_validation_split(
    refs,
    mode=validation_mode,
    valid_user_frac=0.5,   # probá 0.5; si el train queda muy chico, la función se autoajusta
    seed=random_state,
    min_train_rows=2000    # umbral razonable para entrenar con sample=0.2
    )

    # Asegurar que evaluamos sólo sobre filas etiquetadas
    labelled_mask = ~refs["is_test"].to_numpy()

    # Indices
    idx_train = np.where(train_mask_lbl.to_numpy())[0]
    idx_valid = np.where(valid_mask_lbl.to_numpy())[0]
    idx_test  = np.where(test_mask)[0]

    # 5) Entrenar en TRAIN y evaluar en VALID
    X_train, y_train = X[idx_train], y[idx_train]
    X_valid, y_valid = X[idx_valid], y[idx_valid]

    model = train_classifier(X_train, y_train, progress=True, progress_step=50)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    print(f"\n>>> ROC-AUC on VALID [{validation_mode}]: {auc:.5f}")

    # 6) Re-entrenar modelo final con TODO lo etiquetado (opción estándar para submit)
    print("\nTraining final model on ALL labelled data...")
    X_full, y_full = X[labelled_mask], y[labelled_mask]
    final_model = train_classifier(X_full, y_full, progress=True, progress_step=50)

    # 7) Predecir en TEST y guardar submit
    print("\nGenerating TEST predictions...")
    X_test_mat = X[idx_test]
    test_obs_ids = refs.iloc[idx_test]["obs_id"].to_numpy()
    test_pred = final_model.predict_proba(X_test_mat)[:, 1]

    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": test_pred})
    preds_df.to_csv("modelo_benchmark.csv", index=False)
    print("  -> Predictions written to 'modelo_benchmark.csv'")

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    # Elegí "warm" (temporal) o "cold" (usuarios completos). Recomendado: probar ambos.
    main(validation_mode="cold", sample_frac=0.2, random_state=1234)
