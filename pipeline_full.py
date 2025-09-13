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

def encode_features_properly(X_train, X_val, X_test):
    """Encode categorical features WITHOUT test data leakage."""
    print("🔧 Encoding categorical features (NO LEAKAGE)...")
    
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    
    # Find categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    print(f"Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
    
    # Encode categorical variables WITHOUT test data leakage
    for col in categorical_cols:
        print(f"Encoding column: {col}")
        le = LabelEncoder()
        
        # CRITICAL: Only fit on training data
        le.fit(X_train[col].astype(str))
        
        # Transform training data
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        
        # Handle unseen categories in validation and test
        val_mask = X_val[col].astype(str).isin(le.classes_)
        test_mask = X_test[col].astype(str).isin(le.classes_)
        
        # Count unseen categories
        val_unseen = (~val_mask).sum()
        test_unseen = (~test_mask).sum()
        
        if val_unseen > 0:
            print(f"     Warning: {val_unseen} unseen categories in validation set")
        if test_unseen > 0:
            print(f"     Warning: {test_unseen} unseen categories in test set")
        
        # Handle validation set - only transform seen categories
        if val_unseen > 0:
            # Initialize with unseen category value
            X_val_encoded[col] = np.full(len(X_val), len(le.classes_), dtype=int)
            # Transform only the seen categories
            seen_val_data = X_val[col][val_mask].astype(str)
            if len(seen_val_data) > 0:
                X_val_encoded.loc[val_mask, col] = le.transform(seen_val_data)
        else:
            # No unseen categories, safe to transform directly
            X_val_encoded[col] = le.transform(X_val[col].astype(str))
        
        # Handle test set - only transform seen categories  
        if test_unseen > 0:
            # Initialize with unseen category value
            X_test_encoded[col] = np.full(len(X_test), len(le.classes_), dtype=int)
            # Transform only the seen categories
            seen_test_data = X_test[col][test_mask].astype(str)
            if len(seen_test_data) > 0:
                X_test_encoded.loc[test_mask, col] = le.transform(seen_test_data)
        else:
            # No unseen categories, safe to transform directly
            X_test_encoded[col] = le.transform(X_test[col].astype(str))
    
    # Convert to float
    X_train_encoded = X_train_encoded.astype(float)
    X_val_encoded = X_val_encoded.astype(float)
    X_test_encoded = X_test_encoded.astype(float)
    
    print(f"✅ Encoding completed (NO LEAKAGE)!")
    print(f"   Final training features: {X_train_encoded.shape}")
    print(f"   Final validation features: {X_val_encoded.shape}")
    print(f"   Final test features: {X_test_encoded.shape}")
    
    return X_train_encoded, X_val_encoded, X_test_encoded

def run_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test):
    """Run hyperparameter tuning with proper temporal validation."""
    print("🎛️  Starting hyperparameter tuning (TEMPORAL VALIDATION)...")
    print("-" * 50)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    
    # Custom tuning configuration with stronger regularization
    tuning_config = TUNING_CONFIG.copy()
    tuning_config['n_trials'] = 35  # Increased for better search
    tuning_config['cv_folds'] = 5  # Reduced for temporal validation
    
    print(f"Tuning configuration:")
    print(f"   Search type: {tuning_config['search_type']}")
    print(f"   Number of trials: {tuning_config['n_trials']}")
    print(f"   CV folds: {tuning_config['cv_folds']}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(tuning_config)
    
    # Run tuning with temporal validation
    tuning_results = tuner.tune_hyperparameters(
        X_train, y_train, X_val, y_val
    )
    
    print(f"\n🏆 Tuning completed!")
    print(f"   Best CV AUC: {tuning_results['best_score']:.4f}")
    print(f"   Best parameters: {tuning_results['best_params']}")
    
    return tuning_results

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
    
    # Create submission with correct obs_ids and automatic naming
    submission_df, submission_path = create_final_submission(
        best_model, 
        X_test, 
        submission_path=None,  # Will generate automatic name
        test_obs_ids=test_obs_ids,
        auc_score=val_auc,
        pipeline_type="full"
    )
    
    # Save model
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_xgboost_model.pkl")
    print(f"Model saved to: models/best_xgboost_model.pkl")
    
    return best_model, submission_df, submission_path

def main():
    """Main function to run the unified pipeline."""
    
    print("🚀 SPOTIFY SKIP PREDICTION - UNIFIED PIPELINE")
    print("=" * 60)
    
    try:
        # 1ro: cargar data. (esto ya unifica la data de la api y agrega las features en features.py)
        X_train, y_train, X_val, y_val, X_test = load_and_prepare_data()
        if X_train is None:
            return False
        
        # 2do: codificar features (para que todo sea numerico) - SIN LEAKAGE
        X_train_encoded, X_val_encoded, X_test_encoded = encode_features_properly(X_train, X_val, X_test)
        
        # (para ver las features finales)
        print("TRAIN DATA:",X_train_encoded.head().T)
        print("TEST DATA:",X_test_encoded.head().T)

        # 3ro: correr el tuning de hiperparametros con validacion temporal
        tuning_results = run_hyperparameter_tuning(
            X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded
        )
        
        # 4to: entrenar modelo final y generar archivo para kaggle
        test_df_raw = pd.read_csv(str(TEST_PATH), sep='\t')
        test_obs_ids = test_df_raw['obs_id'].copy()
        
        best_model, submission_df, submission_path = train_final_model_and_create_submission(
            tuning_results, X_train_encoded, X_val_encoded, y_train, y_val, X_test_encoded, test_obs_ids
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
        print(f"   - {submission_path} (main submission file)")
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
