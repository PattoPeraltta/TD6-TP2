#!/usr/bin/env python3
"""
Test script for ensemble hyperparameter tuning.
This shows how to plug ensemble tuning into your existing pipeline.
"""

from pipeline_full import *
from src.tuning import run_ensemble_hyperopt_pipeline
from src.model import create_final_submission
import pandas as pd
import numpy as np

def test_ensemble_hyperopt():
    """Test ensemble hyperparameter tuning."""
    
    print("ENSEMBLE HYPERPARAMETER TUNING TEST")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data (same as before)
        print("\nStep 1: Loading and preparing data")
        print("-" * 50)
        X_train, y_train, X_val, y_val, X_test = load_and_prepare_data()
        if X_train is None:
            return False
        
        # Step 2: Encode features (same as before)
        print("\nStep 2: Encoding categorical features")
        print("-" * 50)
        X_train_encoded, X_val_encoded, X_test_encoded = encode_features(X_train, X_val, X_test)
        
        # Step 3: Load test observation IDs
        print("\nStep 3: Preparing test data")
        print("-" * 50)
        test_df_raw = pd.read_csv(str(TEST_PATH), sep='\t')
        test_obs_ids = test_df_raw['obs_id'].copy()
        
        # Step 4: Ensemble Hyperparameter Tuning
        print("\nStep 4: Ensemble Hyperparameter Tuning")
        print("-" * 50)
        
        # Configure tuning (fewer trials for testing)
        tuning_config = {
            'n_trials': 5,  # Very small number for testing
            'verbose': True
        }
        
        # Run ensemble hyperparameter tuning
        best_ensemble, tuning_results = run_ensemble_hyperopt_pipeline(
            X_train_encoded, y_train, X_val_encoded, y_val, tuning_config
        )
        
        # Step 5: Create submission with tuned ensemble
        print("\nStep 5: Creating submission with tuned ensemble")
        print("-" * 50)
        
        submission_df, submission_path = create_final_submission(
            best_ensemble, 
            X_test_encoded, 
            submission_path=None,
            test_obs_ids=test_obs_ids,
            auc_score=tuning_results['best_score'],
            pipeline_type="ensemble_tuned",
            verbose=True
        )
        
        # Results summary
        print("\n" + "=" * 60)
        print("ENSEMBLE HYPERPARAMETER TUNING COMPLETED")
        print("=" * 60)
        print(f"Best Ensemble AUC: {tuning_results['best_score']:.4f}")
        print(f"Tuning time: {tuning_results['tuning_time']:.2f} seconds")
        print(f"Total trials: {len(tuning_results['tuning_results'])}")
        
        print(f"\nBest Parameters:")
        for model_name, params in tuning_results['best_model_params'].items():
            print(f"  {model_name.upper()}:")
            for param, value in params.items():
                print(f"    {param}: {value}")
        
        print(f"\nOutput Files:")
        print(f"  - {submission_path} (tuned ensemble submission)")
        print(f"  - tuning_results/ensemble_hyperopt_tuning_results.json")
        print(f"  - tuning_results/ensemble_hyperopt_best_params_summary.txt")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Ensemble hyperopt test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Ensemble Hyperparameter Tuning Test...")
    print("This will tune XGBoost + LightGBM + CatBoost simultaneously")
    print("Note: This is computationally intensive - using only 5 trials for testing")
    print()
    
    success = test_ensemble_hyperopt()
    if success:
        print("\n✅ Ensemble hyperparameter tuning test completed successfully!")
        print("Check the results above and tuning_results/ folder for detailed results.")
    else:
        print("\n❌ Ensemble hyperparameter tuning test failed. Check the errors above.")
