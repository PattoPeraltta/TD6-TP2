import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# configuración de pandas para mejor visualización
pd.set_option("display.max_columns", None)

# ruta de los datos
DATA_PATH = "./competition_data"

# =============================================================================
# CONFIGURACIÓN DE FECHAS PARA ENTRENAMIENTO
# =============================================================================
# modifica estas variables para cambiar el rango de fechas de entrenamiento
# formato: YYYY-MM
YEAR_FROM = 2020    # año de inicio
MONTH_FROM = 1      # mes de inicio (1-12)
YEAR_TO = 2024      # año de fin
MONTH_TO = 12       # mes de fin (1-12)

# ejemplos de configuraciones:
# solo 2024: YEAR_FROM=2024, MONTH_FROM=1, YEAR_TO=2024, MONTH_TO=12
# solo últimos 2 años: YEAR_FROM=2022, MONTH_FROM=1, YEAR_TO=2024, MONTH_TO=12
# solo 2023: YEAR_FROM=2023, MONTH_FROM=1, YEAR_TO=2023, MONTH_TO=12
# solo últimos 6 meses de 2024: YEAR_FROM=2024, MONTH_FROM=7, YEAR_TO=2024, MONTH_TO=12
# =============================================================================

def cargar_datos():
    """carga y procesa los datos de entrenamiento y test"""
    print("cargando datos...")
    
    # cargar datos
    train_df = pd.read_csv(f"{DATA_PATH}/train_data.txt", sep="\t", low_memory=False)
    test_df = pd.read_csv(f"{DATA_PATH}/test_data.txt", sep="\t", low_memory=False)
    
    # procesar timestamps
    train_df['ts'] = pd.to_datetime(train_df['zts'], utc=True)
    test_df['ts'] = pd.to_datetime(test_df['ts'], utc=True)
    
    # concatenar para procesamiento conjunto
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # procesar timestamps adicionales
    df['offline_timestamp'] = pd.to_datetime(df['offline_timestamp'], unit='s', errors='coerce', utc=True)
    
    # crear target y flag de test
    df['target'] = (df['reason_end'] == 'fwdbtn').astype(int)
    df['is_test'] = df['reason_end'].isna()
    
    print(f"datos cargados: {len(df)} filas, {df['username'].nunique()} usuarios únicos")
    print(f"años en datos: {sorted(df['ts'].dt.year.unique())}")
    
    return df

def crear_features_robustas(df):
    """crea features robustas sin data leakage"""
    print("creando features robustas...")
    
    # ordenar por usuario y timestamp
    df = df.sort_values(['username', 'ts']).reset_index(drop=True)
    
    # features básicas de secuencia
    df['user_order'] = df.groupby('username').cumcount() + 1
    df['user_total_sessions'] = df.groupby('username')['username'].transform('count')
    
    # features temporales
    df['hour'] = df['ts'].dt.hour
    df['day_of_week'] = df['ts'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['ts'].dt.month
    df['day_of_month'] = df['ts'].dt.day
    
    # features de sesión
    df['session_progress'] = df['user_order'] / df['user_total_sessions']
    df['is_first_session'] = (df['user_order'] == 1).astype(int)
    df['is_last_session'] = (df['user_order'] == df['user_total_sessions']).astype(int)
    
    # features categóricas codificadas
    df['platform_encoded'] = df['platform'].astype('category').cat.codes
    df['conn_country_encoded'] = df['conn_country'].astype('category').cat.codes
    
    # features de contenido
    df['has_track_name'] = df['master_metadata_track_name'].notna().astype(int)
    df['has_artist_name'] = df['master_metadata_album_artist_name'].notna().astype(int)
    df['has_episode_name'] = df['episode_name'].notna().astype(int)
    df['has_audiobook_title'] = df['audiobook_title'].notna().astype(int)
    
    # features de comportamiento
    df['shuffle'] = df['shuffle'].astype(int)
    df['offline'] = df['offline'].astype(int)
    df['incognito_mode'] = df['incognito_mode'].astype(int)
    
    # features de tiempo entre sesiones
    df['time_since_last'] = df.groupby('username')['ts'].diff().dt.total_seconds() / 3600  # horas
    df['time_since_last'] = df['time_since_last'].fillna(0)
    
    # features de duración de sesión (aproximada)
    df['session_duration_approx'] = df.groupby('username')['ts'].diff().dt.total_seconds() / 60  # minutos
    df['session_duration_approx'] = df['session_duration_approx'].fillna(0)
    
    # features de frecuencia de uso
    df['user_avg_sessions_per_day'] = df.groupby('username')['ts'].apply(lambda x: x.dt.date.nunique())
    df['user_avg_sessions_per_day'] = df['user_avg_sessions_per_day'] / df['user_avg_sessions_per_day'].max()
    
    # features de horario preferido
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    
    print(f"features creadas: {len([col for col in df.columns if col not in ['ts', 'username', 'obs_id', 'reason_end', 'target', 'is_test', 'offline_timestamp', 'master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name', 'spotify_track_uri', 'episode_name', 'episode_show_name', 'spotify_episode_uri', 'audiobook_title', 'audiobook_uri', 'audiobook_chapter_uri', 'audiobook_chapter_title', 'ip_addr']])} features")
    
    return df

def preparar_datos_entrenamiento(df, year_from=YEAR_FROM, month_from=MONTH_FROM, year_to=YEAR_TO, month_to=MONTH_TO):
    """prepara los datos para entrenamiento con rango de fechas configurable"""
    print("preparando datos para entrenamiento...")
    
    # crear máscara de fechas
    start_date = pd.Timestamp(f"{year_from}-{month_from:02d}-01", tz='UTC')
    end_date = pd.Timestamp(f"{year_to}-{month_to:02d}-28", tz='UTC')  # usar día 28 para evitar problemas con febrero
    
    # filtrar por rango de fechas
    date_mask = (df['ts'] >= start_date) & (df['ts'] <= end_date)
    df_train = df[date_mask & (~df['is_test'])]
    
    print(f"usando datos desde {year_from}-{month_from:02d} hasta {year_to}-{month_to:02d} para entrenamiento...")
    print(f"datos de entrenamiento filtrados: {len(df_train)} filas")
    print(f"años en entrenamiento: {sorted(df_train['ts'].dt.year.unique())}")
    print(f"meses en entrenamiento: {sorted(df_train['ts'].dt.month.unique())}")
    
    # seleccionar features
    feature_cols = [
        'user_order', 'user_total_sessions', 'hour', 'day_of_week', 'is_weekend',
        'month', 'day_of_month', 'session_progress', 'is_first_session', 'is_last_session',
        'platform_encoded', 'conn_country_encoded', 'has_track_name', 'has_artist_name',
        'has_episode_name', 'has_audiobook_title', 'shuffle', 'offline', 'incognito_mode',
        'time_since_last', 'session_duration_approx', 'user_avg_sessions_per_day',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night'
    ]
    
    # preparar datos de entrenamiento
    X_train = df_train[feature_cols].fillna(0).astype(float)
    y_train = df_train['target']
    
    # preparar datos de test
    X_test = df.loc[df['is_test'], feature_cols].fillna(0).astype(float)
    test_obs_ids = df.loc[df['is_test'], 'obs_id']
    
    print(f"datos de entrenamiento: {len(X_train)} filas, {len(feature_cols)} features")
    print(f"datos de test: {len(X_test)} filas")
    print(f"distribución del target: {y_train.value_counts().to_dict()}")
    
    return X_train, y_train, X_test, test_obs_ids, feature_cols

def entrenar_xgboost_optimizado(X_train, y_train):
    """entrena modelo xgboost optimizado para datos recientes"""
    print("entrenando modelo xgboost optimizado...")
    
    # split para validación
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # parámetros optimizados para datos recientes
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # crear y entrenar modelo
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, verbose=100)
    
    # evaluar en validación
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"auc en validación: {val_auc:.4f}")
    
    # mostrar estadísticas de predicciones en validación
    print(f"predicciones en validación:")
    print(f"  min: {val_pred.min():.4f}, max: {val_pred.max():.4f}, mean: {val_pred.mean():.4f}")
    print(f"  > 0.5: {(val_pred > 0.5).sum()}, > 0.3: {(val_pred > 0.3).sum()}, > 0.1: {(val_pred > 0.1).sum()}")
    
    return model

def generar_predicciones(model, X_test, test_obs_ids, year_from=YEAR_FROM, month_from=MONTH_FROM, year_to=YEAR_TO, month_to=MONTH_TO):
    """genera predicciones para el dataset de test"""
    print("generando predicciones...")
    
    # predecir probabilidades
    predictions = model.predict_proba(X_test)[:, 1]
    
    # mostrar estadísticas de predicciones
    print(f"estadísticas de predicciones:")
    print(f"  min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}")
    print(f"  > 0.5: {(predictions > 0.5).sum()}, > 0.3: {(predictions > 0.3).sum()}, > 0.1: {(predictions > 0.1).sum()}")
    
    # crear submission
    submission = pd.DataFrame({
        'obs_id': test_obs_ids,
        'target': predictions
    })
    
    # crear nombre de archivo con fechas
    filename = f'submission_{year_from}{month_from:02d}_to_{year_to}{month_to:02d}.csv'
    
    # guardar submission
    submission.to_csv(filename, index=False)
    print(f"submission guardada: {len(submission)} predicciones en {filename}")
    
    return submission

def main():
    """función principal"""
    print(f"=== modelo con datos desde {YEAR_FROM}-{MONTH_FROM:02d} hasta {YEAR_TO}-{MONTH_TO:02d} ===")
    
    # cargar datos
    df = cargar_datos()
    
    # crear features robustas
    df = crear_features_robustas(df)
    
    # preparar datos con rango de fechas configurable
    X_train, y_train, X_test, test_obs_ids, feature_cols = preparar_datos_entrenamiento(
        df, YEAR_FROM, MONTH_FROM, YEAR_TO, MONTH_TO
    )
    
    # entrenar modelo
    model = entrenar_xgboost_optimizado(X_train, y_train)
    
    # generar predicciones
    submission = generar_predicciones(
        model, X_test, test_obs_ids, YEAR_FROM, MONTH_FROM, YEAR_TO, MONTH_TO
    )
    
    # mostrar importancia de features
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== top 10 features más importantes ===")
    print(feature_importance.head(10))
    
    print(f"\n=== proceso completado ===")
    print(f"archivo generado: submission_{YEAR_FROM}{MONTH_FROM:02d}_to_{YEAR_TO}{MONTH_TO:02d}.csv")

if __name__ == "__main__":
    main()
