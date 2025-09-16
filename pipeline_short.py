from pipeline_full import *
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

def train_model_with_fixed_params(params, X_train, X_val, y_train, y_val, X_test, test_obs_ids):
    """Train model with fixed parameters and create submission."""
    print("\nStep 4: Training model with fixed parameters")
    print("-" * 50)
    
    # Train model with fixed parameters
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False  # Reduce verbose output
    )

    
    # Evaluate on validation set
    y_val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Create submission with correct obs_ids and automatic naming
    from src.model import create_final_submission
    submission_df, submission_path = create_final_submission(
        model, 
        X_test, 
        submission_path=None,  # Will generate automatic name
        test_obs_ids=test_obs_ids,
        auc_score=val_auc,
        pipeline_type="short",
        verbose=False  # Enable verbose output
    )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_xgboost_model.pkl")
    print(f"Model saved to: models/best_xgboost_model.pkl")
    
    return model, submission_df, submission_path

def main():
    """Main function to run the short pipeline."""
    
    print("SPOTIFY SKIP PREDICTION - SHORT PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        print("\nStep 1: Loading and preparing data")
        print("-" * 50)
        X_train, y_train, X_val, y_val, X_test = load_and_prepare_data()
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
        print(f"\nFeature Columns:")
        feature_names = list(X_train_encoded.columns)
        for i, col in enumerate(feature_names, 1):
            if 'user_' in col and col.startswith('user_'):
                # Group user features
                if i == 1 or not any('user_' in feature_names[j] for j in range(i-1)):
                    print(f"  {i}. One-hot encoding of users (10 features)")
            else:
                print(f"  {i}. {col}")
        
        # Step 3: Load test observation IDs
        print("\nStep 3: Preparing test data")
        print("-" * 50)
        test_df_raw = pd.read_csv(str(TEST_PATH), sep='\t')
        test_obs_ids = test_df_raw['obs_id'].copy()

        params = {
                    'n_estimators': 200,
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
        
        best_model, submission_df, submission_path = train_model_with_fixed_params(
            params, X_train_encoded, X_val_encoded, y_train, y_val, X_test_encoded, test_obs_ids
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
