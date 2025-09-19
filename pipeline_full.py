#!/usr/bin/env python3
"""
Working pipeline that uses your cleaned data directly.
This script bypasses the complex data loading and focuses on hyperparameter tuning.
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from src.tuning import HyperoptTuner, run_complete_hyperopt_pipeline
from src.model import create_final_submission
from src.config import TUNING_CONFIG, TRAIN_PATH, TEST_PATH, SPOTIFY_API_DIR
from src.data import make_data_with_features
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

def encode_features(X_train, X_val, X_test):
    """Encode categorical features using ColumnTransformer WITHOUT test data leakage."""
    print("Encoding categorical features with ColumnTransformer...")
    
    # Find categorical and numerical columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    
    print(f"Found {len(cat_cols)} categorical columns: {cat_cols}")
    
    if len(cat_cols) == 0:
        print("No categorical columns found, returning original data")
        return X_train, X_val, X_test
    
    # Numeric pipeline
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical pipeline
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore",   # unseen categories at inference won’t crash
            sparse_output=False,       # dense array output
            min_frequency=0.01         # group rare categories together (1% cutoff)
        )),
    ])
    
    # Combine into one ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    
    # CRITICAL: Only fit on training data to prevent leakage
    print("Fitting ColumnTransformer on training data only...")
    preprocessor.fit(X_train)
    
    # Transform all datasets
    print("Transforming datasets...")
    X_train_encoded = preprocessor.transform(X_train)
    X_val_encoded = preprocessor.transform(X_val)
    X_test_encoded = preprocessor.transform(X_test)
    
    # Convert back to DataFrames with proper feature names from ColumnTransformer
    # Get the actual feature names after transformation
    feature_names = preprocessor.get_feature_names_out()
    
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_val_encoded = pd.DataFrame(X_val_encoded, columns=feature_names, index=X_val.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)
    
    # Convert to float
    X_train_encoded = X_train_encoded.astype(float)
    X_val_encoded = X_val_encoded.astype(float)
    X_test_encoded = X_test_encoded.astype(float)
    
    print(f"ColumnTransformer encoding completed!")
    print(f"  Final training features: {X_train_encoded.shape}")
    print(f"  Final validation features: {X_val_encoded.shape}")
    print(f"  Final test features: {X_test_encoded.shape}")
    
    return X_train_encoded, X_val_encoded, X_test_encoded

def train_final_model_and_create_submission(params, X_train, X_val, y_train, y_val, X_test, test_obs_ids):
    """Train final model and create submission."""
    print("\nStep 4: Training final model and creating submission")
    print("-" * 50)
    
    # Train final model with best parameters
    best_model = XGBClassifier(**params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False  # Reduce verbose output
    )
    
    # Evaluate on validation set
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
        pipeline_type="full",
        verbose=False  # Enable verbose output
    )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_xgboost_model.pkl")
    print(f"Model saved to: models/best_xgboost_model.pkl")
    
    return best_model, submission_path

def print_columns(df: pd.DataFrame):
    print(f"\nFeature Columns:")
    feature_names = list(df.columns)
    for i, col in enumerate(feature_names, 1):
        if 'user_' in col and col.startswith('user_'):
            # Group user features
            if i == 1 or not any('user_' in feature_names[j] for j in range(i-1)):
                print(f"  {i}. One-hot encoding of users (10 features)")
        else:
            print(f"  {i}. {col}")

def main():
    """Main function to run the unified pipeline."""
    
    print("SPOTIFY SKIP PREDICTION - UNIFIED PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        print("\nStep 1: Loading and preparing data")
        print("-" * 50)
        X_train, y_train, X_val, y_val, X_test = make_data_with_features(train_path=str(TRAIN_PATH),
            test_path=str(TEST_PATH), 
            track_data_path=str(SPOTIFY_API_DIR),
            verbose=False  # Enable verbose output
        )
        if X_train is None:
            return False
        
        # Step 2: Encode features
        print("\nStep 2: Encoding categorical features")
        print("-" * 50)
        X_train_encoded, X_val_encoded, X_test_encoded = encode_features(X_train, X_val, X_test)
        
        # Display feature information
        print(f"\nFeature Summary:")
        print(f"  Total features: {X_train_encoded.shape[1]}")
        print(f"  Training samples: {X_train_encoded.shape[0]:,}")
        print(f"  Validation samples: {X_val_encoded.shape[0]:,}")
        print(f"  Test samples: {X_test_encoded.shape[0]:,}")
        
        # Show feature columns
        print_columns(X_train_encoded)

        # Step 3: Hyperparameter tuning
        print("\nStep 3: Hyperparameter tuning")
        print("-" * 50)
        
        # Configure tuning
        tuning_config = {
            'n_trials': 50,  # Adjust based on computational resources
            'verbose': True
        }
        
        # Run hyperparameter tuning
        tuning_results = run_complete_hyperopt_pipeline(
            X_train_encoded, y_train, X_val_encoded, y_val, tuning_config
        )
        
        # Step 4: Train final model and create submission
        print("\nStep 4: Preparing test data")
        print("-" * 50)
        test_df_raw = pd.read_csv(str(TEST_PATH), sep='\t')
        test_obs_ids = test_df_raw['obs_id'].copy()
        
        # Train final single model with best parameters
        best_model, submission_path = train_final_model_and_create_submission(
            tuning_results, X_train_encoded, X_val_encoded, y_train, y_val, X_test_encoded, test_obs_ids
        )
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        else:
            feature_importance = None
        
        if feature_importance is not None:
            importance_df = pd.DataFrame({
                'feature': X_train_encoded.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
        else:
            importance_df = None
    
        # Final summary
        print("\n" + "=" * 60)
        print("UNIFIED PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Best Model Performance:")
        print(f"  Validation AUC: {tuning_results['best_score']:.4f}")
        print(f"  Tuning time: {tuning_results['tuning_time']:.2f} seconds")
        print(f"  Total trials tested: {len(tuning_results['tuning_results'])}")
        print(f"  Training samples: {X_train_encoded.shape[0]:,}")
        print(f"  Validation samples: {X_val_encoded.shape[0]:,}")
        print(f"  Test predictions: {len(test_obs_ids):,}")
        
        if importance_df is not None:
            print(f"\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        else:
            print(f"\nFeature importance not available for this model type.")
        
        print(f"\nTop 3 Parameter Combinations:")
        tuner = HyperoptTuner()
        tuner.tuning_results = tuning_results['tuning_results']
        top_3 = tuner.get_top_parameters(3)
        for i, result in enumerate(top_3, 1):
            print(f"  {i}. AUC: {result['mean_cv_score']:.4f}")
        
        print(f"\nOutput Files:")
        print(f"  - {submission_path} (main submission file)")
        print(f"  - models/best_xgboost_model.pkl (best trained model)")
        print(f"  - tuning_results/ (tuning results and visualizations)")
        
        return True
            
    except Exception as e:
        print(f"ERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nAll done! Your submission file is ready.")
    else:
        print("\nPipeline failed. Please check the errors above.")
        sys.exit(1)
