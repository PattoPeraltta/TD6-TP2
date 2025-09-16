from pathlib import Path
DATA_DIR = "competition_data/"
TRAIN_PATH = DATA_DIR + "train_data.txt"
TEST_PATH = DATA_DIR + "test_data.txt"
SPOTIFY_API_DIR = DATA_DIR + "spotify_api_data"
SEED = 420
TARGET = "reason_end"

# parámetros por defecto del modelo XGBoost, refinados posteriormente por tuning
XGB_DEFAULT = dict(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", random_state=SEED, eval_metric="auc"
)

# Hyperparameter tuning configuration
TUNING_CONFIG = {
    'search_type': 'random',  # 'grid' or 'random'
    'n_trials': 100,  # Number of trials for random search
    'cv_folds': 5,  # Number of cross-validation folds
    'early_stopping_rounds': 150,
    'verbose': True,
    'save_best_model': True,
    'best_model_path': 'models/best_xgboost_model.pkl'
}

# Hyperparameter search space with stronger regularization
XGB_PARAM_GRID = {
    'n_estimators': [300, 500, 800],  # Further reduced range
    'learning_rate': [0.01, 0.03, 0.05],  # Lower learning rates
    'max_depth': [3, 4, 5],  # Shallower trees for better generalization
    'min_child_weight': [5, 10, 15],  # Higher min_child_weight for regularization
    'subsample': [0.7, 0.8],  # Reduced subsample for regularization
    'colsample_bytree': [0.7, 0.8],  # Reduced colsample for regularization
    'reg_alpha': [1.0, 2.0, 3.0],  # Stronger L1 regularization
    'reg_lambda': [2.0, 5.0, 10.0]  # Stronger L2 regularization
}

# Fixed parameters that won't be tuned
XGB_FIXED_PARAMS = {
    'tree_method': "hist",
    'random_state': SEED,
    'eval_metric': "auc"
}