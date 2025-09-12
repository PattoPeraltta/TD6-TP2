#!/usr/bin/env python3
"""
Working pipeline that uses your cleaned data directly.
This script bypasses the complex data loading and focuses on hyperparameter tuning.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from tuning import HyperparameterTuner, create_submission_file
from model import create_final_submission
from config import TUNING_CONFIG, TRAIN_PATH, TEST_PATH, SPOTIFY_API_DIR
from data import make_data_with_features
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data():
    """Load and prepare data using the unified make_data_with_features() function from data.py."""
    print("📊 Loading and preparing data using unified make_data_with_features()...")
    
    try:
        # Use make_data_with_features() with config paths - this now includes all feature engineering
        X_train, y_train, X_val, y_val, X_test = make_data_with_features(
            train_path=str(TRAIN_PATH),
            test_path=str(TEST_PATH), 
            track_data_path=str(SPOTIFY_API_DIR)
        )
        
        return X_train, y_train, X_val, y_val, X_test
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def encode_features(X, X_test, X_val=None):
    """Encode categorical features."""
    print("🔧 Encoding categorical features...")
    
    X_encoded = X.copy()
    X_test_encoded = X_test.copy()
    
    # Find categorical columns
    categorical_cols = X.select_dtypes(include=['object', "category"]).columns
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Combine all unique values to avoid unseen labels
        if X_val is not None:
            all_values = pd.concat([X[col], X_test[col], X_val[col]]).astype(str).unique()
        else:
            all_values = pd.concat([X[col], X_test[col]]).astype(str).unique()
        le.fit(all_values)
        
        # Transform
        X_encoded[col] = le.transform(X[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))
        if X_val is not None:
            X_val_encoded = X_val.copy() if 'X_val_encoded' not in locals() else X_val_encoded
            X_val_encoded[col] = le.transform(X_val[col].astype(str))
    
    # Convert to float
    X_encoded = X_encoded.astype(float)
    X_test_encoded = X_test_encoded.astype(float)
    if X_val is not None:
        X_val_encoded = X_val_encoded.astype(float)
    
    print(f"✅ Encoding completed!")
    print(f"   Final training features: {X_encoded.shape}")
    print(f"   Final test features: {X_test_encoded.shape}")
    if X_val is not None:
        print(f"   Final validation features: {X_val_encoded.shape}")
        return X_encoded, X_test_encoded, X_val_encoded
    
    return X_encoded, X_test_encoded

def run_hyperparameter_tuning(X, y, X_test):
    """Run hyperparameter tuning."""
    print("🎛️  Starting hyperparameter tuning...")
    print("-" * 50)
    
    # Create train/validation split -> 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=420, stratify=y
    )
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    
    # Custom tuning configuration for faster execution
    tuning_config = TUNING_CONFIG.copy()
    tuning_config['n_trials'] = 15  # Reduced for faster execution
    tuning_config['cv_folds'] = 5
    
    print(f"Tuning configuration:")
    print(f"   Search type: {tuning_config['search_type']}")
    print(f"   Number of trials: {tuning_config['n_trials']}")
    print(f"   CV folds: {tuning_config['cv_folds']}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(tuning_config)
    
    # Run tuning
    tuning_results = tuner.tune_hyperparameters(
        X_train, y_train, X_val, y_val
    )
    
    print(f"\n🏆 Tuning completed!")
    print(f"   Best CV AUC: {tuning_results['best_score']:.4f}")
    print(f"   Best parameters: {tuning_results['best_params']}")
    
    return tuning_results, X_train, X_val, y_train, y_val

def train_final_model_and_create_submission(tuning_results, X_train, X_val, y_train, y_val, X_test, test_obs_ids):
    """Train final model and create submission."""
    print("\n🤖 Training final model and creating submission...")
    print("-" * 50)
    
    # Train final model with best parameters
    best_model = XGBClassifier(**tuning_results['best_params'])
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Evaluate on validation set
    from sklearn.metrics import roc_auc_score
    y_val_pred = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    print(f"Final validation AUC: {val_auc:.4f}")
    
    # Create submission with correct obs_ids
    submission_df = create_final_submission(
        best_model, 
        X_test, 
        "final_submission.csv",
        test_obs_ids
    )
    
    # Save model
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_xgboost_model.pkl")
    print(f"Model saved to: models/best_xgboost_model.pkl")
    
    return best_model, submission_df

def main():
    """Main function to run the unified pipeline."""
    
    print("🚀 SPOTIFY SKIP PREDICTION - UNIFIED PIPELINE")
    print("=" * 60)
    
    try:
        # 1ro: cargar data. (esto ya unifica la data de la api y agrega las features en features.py)
        X_train, y_train, X_val, y_val, X_test = load_and_prepare_data()
        if X_train is None:
            return False
        
        # 2do: codificar features (para que todo sea numerico)
        X_train_encoded, X_test_encoded, X_val_encoded = encode_features(X_train, X_test, X_val)
        
        # (para ver las features finales)
        print("TRAIN DATA:",X_train_encoded.head(1).T)
        print("TEST DATA:",X_test_encoded.head(1).T)

        # 3ro: correr el tuning de hiperparametros
        tuning_results, X_train_split, X_val_split, y_train_split, y_val_split = run_hyperparameter_tuning(
            X_train_encoded, y_train, X_test_encoded
        )
        
        # 4to: entrenar modelo final y generar archivo para kaggle
        test_df_raw = pd.read_csv(str(TEST_PATH), sep='\t')
        test_obs_ids = test_df_raw['obs_id'].copy()
        
        best_model, submission_df = train_final_model_and_create_submission(
            tuning_results, X_train_split, X_val_split, y_train_split, y_val_split, X_test_encoded, test_obs_ids
        )
    
        # 5to: resumen de todo
        print("\n" + "=" * 60)
        print("🎉 UNIFIED PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Best model performance:")
        print(f"   Validation AUC: {tuning_results['best_score']:.4f}")
        print(f"   Tuning time: {tuning_results['tuning_time']:.2f} seconds")
        print(f"   Total trials tested: {len(tuning_results['tuning_results'])}")
        
        print(f"\n📁 Output files:")
        print(f"   - final_submission.csv (main submission file)")
        print(f"   - models/best_xgboost_model.pkl (best trained model)")
        print(f"   - tuning_results/ (tuning results and visualizations)")
        
        print(f"\n🔍 Top 3 parameter combinations:")
        tuner = HyperparameterTuner()
        tuner.tuning_results = tuning_results['tuning_results']
        top_3 = tuner.get_top_parameters(3)
        for i, result in enumerate(top_3, 1):
            print(f"   {i}. AUC: {result['mean_cv_score']:.4f}")
        
        return True
            
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All done! Your submission file is ready.")
    else:
        print("\n💥 Pipeline failed. Please check the errors above.")
        sys.exit(1)
