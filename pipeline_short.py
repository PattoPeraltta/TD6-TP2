from pipeline_full import *
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

def train_model_with_fixed_params(params, X_train, X_val, y_train, y_val, X_test, test_obs_ids):
    """Train model with fixed parameters and create submission."""
    print("\n🤖 Training model with fixed parameters and creating submission...")
    print("-" * 50)
    
    # Train model with fixed parameters
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
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
        pipeline_type="short"
    )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_xgboost_model.pkl")
    print(f"Model saved to: models/best_xgboost_model.pkl")
    
    return model, submission_df, submission_path

def main():
    """Main function to run the short pipeline."""
    
    print("🚀 SPOTIFY SKIP PREDICTION - SHORT PIPELINE")
    print("=" * 60)
    
    try:
        # 1ro: cargar data. (esto ya unifica la data de la api y agrega las features en features.py)
        X_train, y_train, X_val, y_val, X_test = load_and_prepare_data()
        if X_train is None:
            return False
        
        # 2do: codificar features (para que todo sea numerico) - SIN LEAKAGE
        X_train_encoded, X_val_encoded, X_test_encoded = encode_features(X_train, X_val, X_test)
        
        # (para ver las features finales)
        print("TRAIN DATA:",X_train_encoded.head().T)
        print("TEST DATA:",X_test_encoded.head().T)

        # # 3ro: correr el tuning de hiperparametros con validacion temporal
        # tuning_results = run_hyperparameter_tuning(
        #     X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded
        # )
        
        # 4to: entrenar modelo final y generar archivo para kaggle
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
    
        # 5to: resumen de todo
        print("\n" + "=" * 60)
        print("🎉 SHORT PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Model performance:")
        print(f"   Using fixed hyperparameters (no tuning performed)")
        print(f"   Parameters: {params}")
        
        print(f"\n📁 Output files:")
        print(f"   - {submission_path} (main submission file)")
        print(f"   - models/best_xgboost_model.pkl (trained model)")
        
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
