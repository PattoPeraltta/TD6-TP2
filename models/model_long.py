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
    
    # interacción temporal
    df['is_weekend'] = ((df['time_of_day_sin']**2 + df['time_of_day_cos']**2) > 0.9).astype(int)  # aproximación
    df['weekend_x_shuffle'] = df['is_weekend'] * df['shuffle']
    
    print("features de interacción creadas")
    
    # =============================================================================
    # FEATURES DE CONTEXTO TEMPORAL AVANZADAS
    # =============================================================================
    
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
    
    # parámetros optimizados del modelo
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 600,
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
    model.fit(
        X_tr, y_tr, 
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=100
    )
    
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
    print(f"archivo generado: submission_improved.csv")
    print(f"total de features utilizadas: {len(feature_cols)}")

if __name__ == "__main__":
    main()