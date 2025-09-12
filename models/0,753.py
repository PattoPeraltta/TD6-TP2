import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# configuración de pandas para mejor visualización
pd.set_option("display.max_columns", None)

# ruta de los datos
DATA_PATH = "./clean_data"

def cargar_datos():
    """carga y procesa los datos de entrenamiento y test"""
    print("cargando datos...")
    
    # cargar datos limpios
    train_df = pd.read_csv(f"{DATA_PATH}/train_data_cleaned.csv", low_memory=False)
    test_df = pd.read_csv(f"{DATA_PATH}/test_data_cleaned.csv", low_memory=False)
    
    # concatenar para procesamiento conjunto
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # procesar timestamps offline si existe
    if 'offline_timestamp' in df.columns:
        df['offline_timestamp'] = pd.to_datetime(df['offline_timestamp'], errors='coerce', utc=True)
    
    # crear target y flag de test
    df['target'] = (df['reason_end'] == 1).astype(int)  # 1 = fwdbtn en datos limpios
    df['is_test'] = df['reason_end'].isna()
    
    print(f"datos cargados: {len(df)} filas, {df['username'].nunique()} usuarios únicos")
    print(f"usuarios únicos: {sorted(df['username'].unique())}")
    print(f"años en datos: {sorted(df['year'].unique())}")
    print(f"distribución por año:")
    print(df.groupby(['year', 'is_test']).size().unstack(fill_value=0))
    
    return df

def crear_features_robustas(df):
    """crea features robustas sin data leakage"""
    print("creando features robustas sin data leakage...")
    
    # ordenar por usuario y luego por año, mes, día y obs_id para mantener orden temporal
    # esto es CRÍTICO para evitar data leakage
    df = df.sort_values(['username', 'year', 'month', 'day', 'obs_id']).reset_index(drop=True)
    
    # =============================================================================
    # ONE-HOT ENCODING PARA USUARIOS (FEATURE MÁS IMPORTANTE)
    # =============================================================================
    print("implementando one-hot encoding para usuarios...")
    
    # verificar número de usuarios únicos
    unique_users = df['username'].nunique()
    print(f"número de usuarios únicos: {unique_users}")
    
    # crear one-hot encoding para usuarios
    user_dummies = pd.get_dummies(df['username'], prefix='user')
    df = pd.concat([df, user_dummies], axis=1)
    
    user_cols = [col for col in df.columns if col.startswith('user_')]
    print(f"columnas de usuarios creadas: {len(user_cols)}")
    
    # =============================================================================
    # FEATURES BÁSICAS DE SECUENCIA DEL USUARIO (SIN DATA LEAKAGE)
    # =============================================================================
    print("creando features de secuencia temporal...")
    
    # contar sesiones previas del usuario hasta cada observación (sin incluir la actual)
    df['user_sessions_so_far'] = df.groupby('username').cumcount()  # empezará en 0
    df['is_first_user_session'] = (df['user_sessions_so_far'] == 0).astype(int)
    
    # calcular total de sesiones por usuario (esto se puede saber sin data leakage)
    user_total_sessions = df.groupby('username').size()
    df['user_total_sessions'] = df['username'].map(user_total_sessions)
    
    # progreso del usuario (session actual / total sessions)
    df['user_session_progress'] = (df['user_sessions_so_far'] + 1) / df['user_total_sessions']
    df['is_last_user_session'] = (df['user_sessions_so_far'] == df['user_total_sessions'] - 1).astype(int)
    
    # =============================================================================
    # FEATURES TEMPORALES MEJORADAS
    # =============================================================================
    print("procesando features temporales...")
    
    # usar features temporales ya procesadas
    df['time_of_day_sin'] = df['time_of_day_sin'].fillna(0)
    df['time_of_day_cos'] = df['time_of_day_cos'].fillna(0)
    
    # features de fecha
    df['year'] = df['year'].fillna(2020)
    df['month'] = df['month'].fillna(1)
    df['day'] = df['day'].fillna(1)
    
    # crear features de día de la semana y mes
    df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # estaciones del año
    df['is_spring'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(int)
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_autumn'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
    df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)
    
    # =============================================================================
    # FEATURES CATEGÓRICAS CODIFICADAS
    # =============================================================================
    print("procesando features categóricas...")
    
    # label encoding para features categóricas
    le_platform = LabelEncoder()
    le_country = LabelEncoder()
    le_album_type = LabelEncoder()
    
    df['platform_encoded'] = le_platform.fit_transform(df['platform'].astype(str))
    df['conn_country_encoded'] = le_country.fit_transform(df['conn_country'].astype(str))
    df['album_type_encoded'] = le_album_type.fit_transform(df['album_type'].astype(str))
    
    # =============================================================================
    # FEATURES DE CONTENIDO DISPONIBLE
    # =============================================================================
    df['has_track_name'] = df['master_metadata_track_name'].notna().astype(int)
    df['has_artist_name'] = df['master_metadata_album_artist_name'].notna().astype(int)
    df['has_episode_name'] = df['episode_name'].notna().astype(int)
    df['has_album_name'] = df['master_metadata_album_album_name'].notna().astype(int)
    df['has_episode_show_name'] = df['episode_show_name'].notna().astype(int)
    df['has_artist_uri'] = df['artist_uri'].notna().astype(int)
    
    # content type detection
    df['is_music'] = (df['has_track_name'] & df['has_artist_name']).astype(int)
    df['is_podcast'] = (df['has_episode_name'] & df['has_episode_show_name']).astype(int)
    
    # =============================================================================
    # FEATURES DE COMPORTAMIENTO DEL USUARIO
    # =============================================================================
    df['shuffle'] = df['shuffle'].astype(int)
    df['offline'] = df['offline'].astype(int)
    df['incognito_mode'] = df['incognito_mode'].astype(int)
    
    # =============================================================================
    # FEATURES DE SPOTIFY MEJORADAS
    # =============================================================================
    # normalizar features numéricas
    df['duration_normalized'] = df['duration_normalized'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0)
    df['track_number'] = df['track_number'].fillna(0)
    df['available_markets_count'] = df['available_markets_count'].fillna(0)
    df['song_age'] = df['song_age'].fillna(0)
    df['explicit'] = df['explicit'].fillna(False).astype(int)
    
    # features derivadas de spotify
    df['has_spotify_features'] = (df['duration_normalized'] > 0).astype(int)
    df['is_popular_track'] = (df['popularity'] > 50).astype(int)  # threshold fijo más interpretable
    df['is_recent_track'] = (df['song_age'] < 5).astype(int)  # canciones de menos de 5 años
    df['is_widely_available'] = (df['available_markets_count'] > 50).astype(int)
    
    # categorías de duración más interpretables
    df['is_very_short_track'] = (df['duration_normalized'] < 0.2).astype(int)  # < 20%
    df['is_short_track'] = ((df['duration_normalized'] >= 0.2) & (df['duration_normalized'] < 0.4)).astype(int)
    df['is_medium_track'] = ((df['duration_normalized'] >= 0.4) & (df['duration_normalized'] < 0.7)).astype(int)
    df['is_long_track'] = (df['duration_normalized'] >= 0.7).astype(int)  # >= 70%
    
    # =============================================================================
    # FEATURES DE TASA DE SKIP DEL USUARIO (SIN DATA LEAKAGE)
    # =============================================================================
    print("creando features de tasa de skip sin data leakage...")
    
    # CRÍTICO: Estas features deben calcularse usando SOLO observaciones anteriores
    # Inicializar todas las columnas con 0
    df['user_skip_rate_historical'] = 0.0
    df['user_skip_rate_shuffle_historical'] = 0.0
    df['user_skip_rate_offline_historical'] = 0.0
    df['user_skip_count_historical'] = 0
    df['user_total_count_historical'] = 0
    df['user_shuffle_count_historical'] = 0
    df['user_offline_count_historical'] = 0
    
    # Calcular para cada usuario por separado

    
    # Progress bar: try tqdm, otherwise print every 100 users

    
    usernames = df['username'].unique()

    
    try:

    
        from tqdm import tqdm  # type: ignore

    
        _iter_users = enumerate(tqdm(usernames, desc="Skip-rate (no leakage)", unit="user"))

    
        _use_fallback = False

    
    except Exception:
        _iter_users = enumerate(usernames)
        _use_fallback = True

    # for _i_user, username in _iter_users:
    #     if _use_fallback and (_i_user % 100 == 0 or _i_user == len(usernames) - 1):
    #         print(f"[skip-rate] processed {_i_user + 1}/{len(usernames)} users "
    #               f"({(_i_user + 1) / len(usernames):.1%})")
    #     user_mask = df['username'] == username
    #     user_indices = df[user_mask].index.tolist()
        
    #     # Variables acumulativas para el usuario
    #     total_observations = 0
    #     total_skips = 0
    #     shuffle_observations = 0
    #     shuffle_skips = 0
    #     offline_observations = 0
    #     offline_skips = 0
        
    #     for i, idx in enumerate(user_indices):
    #         # IMPORTANTE: usar solo observaciones ANTERIORES (i > 0)
    #         if i > 0:  # a partir de la segunda observación
    #             # calcular tasas usando observaciones anteriores
    #             df.loc[idx, 'user_skip_rate_historical'] = total_skips / total_observations if total_observations > 0 else 0.0
    #             df.loc[idx, 'user_skip_rate_shuffle_historical'] = shuffle_skips / shuffle_observations if shuffle_observations > 0 else 0.0
    #             df.loc[idx, 'user_skip_rate_offline_historical'] = offline_skips / offline_observations if offline_observations > 0 else 0.0
    #             df.loc[idx, 'user_skip_count_historical'] = total_skips
    #             df.loc[idx, 'user_total_count_historical'] = total_observations
    #             df.loc[idx, 'user_shuffle_count_historical'] = shuffle_observations
    #             df.loc[idx, 'user_offline_count_historical'] = offline_observations
            
    #         # actualizar contadores para la PRÓXIMA iteración (solo si no es test)
    #         if not df.loc[idx, 'is_test']:
    #             current_target = df.loc[idx, 'target']
    #             current_shuffle = df.loc[idx, 'shuffle']
    #             current_offline = df.loc[idx, 'offline']
                
    #             # actualizar contadores generales
    #             total_observations += 1
    #             total_skips += current_target
                
    #             # actualizar contadores de shuffle
    #             if current_shuffle == 1:
    #                 shuffle_observations += 1
    #                 shuffle_skips += current_target
                
    #             # actualizar contadores de offline
    #             if current_offline == 1:
    #                 offline_observations += 1
    #                 offline_skips += current_target


    # Asegura orden temporal por usuario (ajusta el nombre de columnas si hace falta)
    user_col = next((c for c in ["username", "user", "user_id", "userid"] if c in df.columns), None)
    if user_col is None:
        raise KeyError("No encontré columna de usuario (username/user/user_id/userid).")

    time_col = next((c for c in ["ts", "timestamp", "start_time", "ms_played_start", "play_ts", "event_time"] if c in df.columns), None)

    if time_col is not None:
        df = df.sort_values([user_col, time_col], kind="mergesort")
    else:
        print("[warn] No timestamp column found; preserving current order within user.")
        df = df.sort_values([user_col], kind="mergesort")

    # Detecta indicador de skip
    if "skip" in df.columns:
        is_skip = df["skip"].astype(int)
    elif "skipped" in df.columns:
        is_skip = df["skipped"].astype(int)
    elif "fwdbtn" in df.columns:
        is_skip = (df["fwdbtn"] == 1).astype(int)
    elif "reason_end" in df.columns:
        _re = pd.to_numeric(df["reason_end"], errors="coerce")
        is_skip = (_re == 1).astype(int)
    else:
        # Si no existe indicador en este DF (ej. test), usar 0 (no leakage, solo evita fallas)
        is_skip = pd.Series(0, index=df.index, dtype=int)
    
    df["_is_skip_"] = is_skip.astype("int8")

    # Interacciones previas por usuario (0 en primera fila de cada usuario)
    df["usr_hist_cnt"] = df.groupby(user_col).cumcount()

    # Skips previos (cumsum excluyendo la fila actual)
    cum_skips = df.groupby(user_col)["_is_skip_"].cumsum().astype("int32")
    df["usr_hist_skip"] = (cum_skips - df["_is_skip_"]).clip(lower=0)

    # Tasa histórica previa
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = df["usr_hist_skip"] / df["usr_hist_cnt"]
    df["usr_skip_rate"] = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    # Rolling means previas (ej.: w5, w20, w50)
    shifted = df.groupby(user_col)["_is_skip_"].shift(1)  # excluye actual
    for w in (5, 20, 50):
        roll = (
            shifted.groupby(df[user_col])
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"usr_skip_rate_w{w}"] = roll.fillna(0.0).astype("float32")

    # Limpieza
    df.drop(columns=["_is_skip_"], inplace=True)
    
    # features adicionales basadas en histórico
    df['user_has_skip_history'] = (df['user_total_count_historical'] > 0).astype(int)
    df['user_has_shuffle_history'] = (df['user_shuffle_count_historical'] > 0).astype(int)
    df['user_has_offline_history'] = (df['user_offline_count_historical'] > 0).astype(int)
    
    # experience level del usuario
    df['user_experience_level'] = np.minimum(df['user_total_count_historical'] / 100, 1.0)  # normalizado a [0,1]
    
    # =============================================================================
    # FEATURES DE INTERACCIÓN MEJORADAS
    # =============================================================================
    # interacciones con el comportamiento histórico del usuario
    df['shuffle_x_historical_skip_rate'] = df['shuffle'] * df['user_skip_rate_historical']
    df['offline_x_historical_skip_rate'] = df['offline'] * df['user_skip_rate_historical']
    
    # interacciones con contenido
    df['popularity_x_user_experience'] = df['popularity'] * df['user_experience_level']
    df['duration_x_user_skip_rate'] = df['duration_normalized'] * df['user_skip_rate_historical']
    
    # interacción temporal
    df['is_weekend'] = ((df['time_of_day_sin']**2 + df['time_of_day_cos']**2) > 0.9).astype(int)  # aproximación
    df['weekend_x_shuffle'] = df['is_weekend'] * df['shuffle']
    
    print("features de interacción creadas")
    
    # =============================================================================
    # FEATURES DE CONTEXTO TEMPORAL AVANZADAS
    # =============================================================================
    # tiempo desde primera sesión del usuario (aproximado usando orden)
    df['user_session_sequence'] = df['user_sessions_so_far'] + 1
    df['user_session_sequence_log'] = np.log1p(df['user_session_sequence'])
    
    # features de momentum (cambio en comportamiento reciente)
    # estas solo se pueden calcular para observaciones que tienen suficiente historia
    df['user_momentum_window'] = 0.0
    for username in df['username'].unique():
        user_mask = df['username'] == username
        user_data = df[user_mask].copy()
        
        momentum = []
        for i in range(len(user_data)):
            if i < 5:  # necesitamos al menos 5 observaciones
                momentum.append(0.0)
            else:
                # tasa de skip en últimas 5 vs primeras observaciones
                recent_data = user_data.iloc[max(0, i-5):i]  # últimas 5 observaciones
                if len(recent_data[~recent_data['is_test']]) > 0:
                    recent_skip_rate = recent_data[~recent_data['is_test']]['target'].mean()
                    overall_rate = df.loc[user_data.index[i], 'user_skip_rate_historical']
                    momentum.append(recent_skip_rate - overall_rate)
                else:
                    momentum.append(0.0)
        
        df.loc[user_mask, 'user_momentum_window'] = momentum
    
    print(f"features de contexto temporal creadas")
    
    # contar features finales
    excluded_cols = [ 
        'username', 'obs_id', 'reason_end', 'target', 'is_test', 'offline_timestamp', 
        'master_metadata_track_name', 'master_metadata_album_artist_name', 
        'master_metadata_album_album_name', 'episode_name', 'episode_show_name', 
        'artist_uri', 'album_type', 'platform', 'conn_country'
    ]
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    print(f"total de features creadas: {len(feature_cols)}")
    
    return df

def preparar_datos_entrenamiento(df):
    """prepara los datos para entrenamiento con validación temporal"""
    print("preparando datos para entrenamiento...")
    
    # separar train y test
    df_train = df[~df['is_test']].copy()
    df_test = df[df['is_test']].copy()
    
    print(f"datos de entrenamiento: {len(df_train)} filas")
    print(f"datos de test: {len(df_test)} filas")
    print(f"años en entrenamiento: {sorted(df_train['year'].unique())}")
    print(f"años en test: {sorted(df_test['year'].unique())}")
    
    # seleccionar features (excluir columnas no relevantes)
    excluded_cols = [
        'username', 'obs_id', 'reason_end', 'target', 'is_test', 'offline_timestamp', 
        'master_metadata_track_name', 'master_metadata_album_artist_name', 
        'master_metadata_album_album_name', 'episode_name', 'episode_show_name', 
        'artist_uri', 'album_type', 'platform', 'conn_country'
    ]
    
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    # preparar datos de entrenamiento
    X_train = df_train[feature_cols].fillna(0).astype(float)
    y_train = df_train['target']
    
    # preparar datos de test
    X_test = df_test[feature_cols].fillna(0).astype(float)
    test_obs_ids = df_test['obs_id']
    
    # verificar que no haya data leakage
    print("\n=== verificación de data leakage ===")
    print(f"features de skip rate en train - min: {X_train['user_skip_rate_historical'].min():.4f}, "
          f"max: {X_train['user_skip_rate_historical'].max():.4f}")
    print(f"features de skip rate en test - min: {X_test['user_skip_rate_historical'].min():.4f}, "
          f"max: {X_test['user_skip_rate_historical'].max():.4f}")
    
    print(f"\ndatos finales:")
    print(f"entrenamiento: {len(X_train)} filas, {len(feature_cols)} features")
    print(f"test: {len(X_test)} filas")
    print(f"distribución del target: {y_train.value_counts().to_dict()}")
    print(f"proporción de skip: {y_train.mean():.4f}")
    
    return X_train, y_train, X_test, test_obs_ids, feature_cols

def entrenar_xgboost_optimizado(X_train, y_train, feature_cols):
    """entrena modelo xgboost optimizado con validación temporal"""
    print("entrenando modelo xgboost optimizado...")
    
    # usar validación temporal: últimos 20% de datos como validación
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    print(f"entrenamiento: {len(X_tr)} filas")
    print(f"validación: {len(X_val)} filas")
    print(f"proporción skip en entrenamiento: {y_tr.mean():.4f}")
    print(f"proporción skip en validación: {y_val.mean():.4f}")
    
    # parámetros optimizados del modelo
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.08,
        'n_estimators': 1000,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 3,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # crear y entrenar modelo
    model = xgb.XGBClassifier(**params)
    # Compat capa para early stopping según versión XGBoost
    _fit_kwargs = dict(eval_set=[(X_tr, y_tr), (X_val, y_val)])
    try:
        model.fit(X_tr, y_tr, early_stopping_rounds=100, verbose=False, **_fit_kwargs)
    except TypeError:
        try:
            from xgboost.callback import EarlyStopping
            model.fit(X_tr, y_tr, callbacks=[EarlyStopping(rounds=100, save_best=True)], verbose=False, **_fit_kwargs)
        except Exception:
            model.fit(X_tr, y_tr, verbose=False, **_fit_kwargs)
    
    
    # evaluar en validación
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"\nauc en validación: {val_auc:.4f}")
    
    # estadísticas de predicciones
    print(f"estadísticas de predicciones en validación:")
    print(f"  min: {val_pred.min():.4f}, max: {val_pred.max():.4f}, mean: {val_pred.mean():.4f}")
    print(f"  > 0.5: {(val_pred > 0.5).sum()}, > 0.3: {(val_pred > 0.3).sum()}, > 0.1: {(val_pred > 0.1).sum()}")
    
    # análisis de importancia de features
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== top 15 features más importantes ===")
    for i, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:30s}: {row['importance']:.4f}")
    
    return model

def generar_predicciones(model, X_test, test_obs_ids):
    """genera predicciones para el dataset de test"""
    print("\ngenerando predicciones...")
    
    # predecir probabilidades
    predictions = model.predict_proba(X_test)[:, 1]
    
    # estadísticas de predicciones
    print(f"estadísticas de predicciones:")
    print(f"  min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}")
    print(f"  std: {predictions.std():.4f}")
    print(f"  percentiles: 5%={np.percentile(predictions, 5):.4f}, "
          f"25%={np.percentile(predictions, 25):.4f}, "
          f"75%={np.percentile(predictions, 75):.4f}, "
          f"95%={np.percentile(predictions, 95):.4f}")
    print(f"  > 0.5: {(predictions > 0.5).sum()}, > 0.3: {(predictions > 0.3).sum()}, > 0.1: {(predictions > 0.1).sum()}")
    
    # crear archivo de submission
    submission = pd.DataFrame({
        'obs_id': test_obs_ids,
        'target': predictions
    })
    
    # verificar formato
    print(f"\nverificación del archivo de submission:")
    print(f"  filas: {len(submission)}")
    print(f"  columnas: {list(submission.columns)}")
    print(f"  obs_id únicos: {submission['obs_id'].nunique()}")
    print(f"  duplicados: {submission.duplicated().sum()}")
    
    # guardar submission
    filename = 'submission_improved.csv'
    submission.to_csv(filename, index=False)
    print(f"submission guardada: {filename}")
    
    return submission

def main():
    """función principal del pipeline de machine learning mejorado"""
    print("=== modelo mejorado de predicción de skip ===")
    print("mejoras implementadas:")
    print("- one-hot encoding para usuarios (10 usuarios únicos)")
    print("- features de skip rate sin data leakage")
    print("- validación temporal")
    print("- features adicionales de contexto e interacción")
    print("- análisis de importancia detallado")
    
    # pipeline principal
    print("\n" + "="*50)
    
    # 1. cargar y procesar datos limpios
    df = cargar_datos()
    
    # 2. crear features robustas sin data leakage
    df = crear_features_robustas(df)
    
    # 3. preparar datos para entrenamiento
    X_train, y_train, X_test, test_obs_ids, feature_cols = preparar_datos_entrenamiento(df)
    
    # 4. entrenar modelo xgboost
    model = entrenar_xgboost_optimizado(X_train, y_train, feature_cols)
    
    # 5. generar predicciones
    submission = generar_predicciones(model, X_test, test_obs_ids)
    
    print(f"\n{'='*50}")
    print(f"proceso completado exitosamente")
    print(f"archivo generado: submission_improved_short.csv")
    print(f"total de features utilizadas: {len(feature_cols)}")

if __name__ == "__main__":
    main()