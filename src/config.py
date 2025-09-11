from pathlib import Path
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR/"train_data.txt"
TEST_PATH  = DATA_DIR/"test_data.txt"
SPOTIFY_API_DIR = DATA_DIR/"Spotify_api_data"

SEED = 420
TARGET = "reason_end"

# parámetros por defecto del modelo XGBoost, refinados posteriormente por tuning
XGB_DEFAULT = dict(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", random_state=SEED, eval_metric="auc"
)