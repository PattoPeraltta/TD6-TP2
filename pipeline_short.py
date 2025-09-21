from pipeline_full import *
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os


def main():
    """Main function to run the short pipeline."""
    
    print("SPOTIFY SKIP PREDICTION - SHORT PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        print("\nStep 1: Loading and preparing data")
        print("-" * 50)
        X_train, y_train, X_val, y_val, X_test = make_data_with_features(
            train_path=str(TRAIN_PATH),
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
        
        # Step 3: Load test observation IDs
        print("\nStep 3: Preparing test data")
        print("-" * 50)
        test_df_raw = pd.read_csv(str(TEST_PATH), sep='\t')
        test_obs_ids = test_df_raw['obs_id'].copy()

        params = {
                    'n_estimators': 900,
                    'learning_rate': 0.03,
                    'max_depth': 5,
                    'min_child_weight': 5,
                    'subsample': 0.7,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 1.0,
                    'reg_lambda': 5.0,
                    'tree_method': 'hist',
                    'random_state': 420,
                    'eval_metric': 'auc'
                }
        
        best_model, submission_path = train_final_model_and_create_submission(
            params, X_train_encoded, X_val_encoded, y_train, y_val, X_test_encoded, test_obs_ids, pipeline_type="short"
        )
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        else:
            feature_importance = None
        
        if feature_importance is not None:
            feature_names = X_train_encoded.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
        else:
            importance_df = None
    
        # Final summary
        print("\n" + "=" * 60)
        print("SHORT PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model Performance:")
        print(f"  Validation AUC: {roc_auc_score(y_val, best_model.predict_proba(X_val_encoded)[:, 1]):.4f}")
        print(f"  Parameters: Fixed hyperparameters (no tuning)")
        print(f"  Training samples: {X_train_encoded.shape[0]:,}")
        print(f"  Validation samples: {X_val_encoded.shape[0]:,}")
        print(f"  Test predictions: {len(test_obs_ids):,}")
        
        if importance_df is not None:
            print(f"\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        else:
            print(f"\nFeature importance not available for this model type.")
        
        print(f"\nOutput Files:")
        print(f"  - {submission_path} (main submission file)")
        print(f"  - models/best_xgboost_model.pkl (trained model)")
        
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
