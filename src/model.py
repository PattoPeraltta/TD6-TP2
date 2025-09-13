import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from typing import Tuple, Dict, Any
import joblib
import os
from pathlib import Path
from datetime import datetime

def train_xgboost_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    params: Dict[str, Any] = None
) -> XGBClassifier:
    """
    Train an XGBoost model with the given parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: XGBoost parameters (uses default if None)
    
    Returns:
        Trained XGBoost classifier
    """
    if params is None:
        # Default parameters from config
        params = {
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'tree_method': "hist",
            'random_state': 420,
            'eval_metric': "auc"
        }
    
    print("Training XGBoost model...")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create and train the model
    model = XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    
    return model

def evaluate_model(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """
    Evaluate the trained model on validation data.
    
    Args:
        model: Trained XGBoost model
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    print(f"Validation AUC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return {
        'auc': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def get_feature_importance(model: XGBClassifier, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """
    Get feature importance from the trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(importance_df.head(top_n))
    
    return importance_df

def save_model(model: XGBClassifier, filepath: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained XGBoost model
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str) -> XGBClassifier:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded XGBoost model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def make_predictions(model: XGBClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on test data.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
    
    Returns:
        Array of predicted probabilities
    """
    predictions = model.predict_proba(X_test)[:, 1]
    print(f"Made predictions for {len(X_test)} test samples")
    return predictions

def train_and_evaluate_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    X_test: pd.DataFrame,
    params: Dict[str, Any] = None,
    save_model_path: str = None
) -> Tuple[XGBClassifier, Dict[str, float], np.ndarray]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        params: XGBoost parameters
        save_model_path: Path to save the model (optional)
    
    Returns:
        Tuple of (trained_model, evaluation_metrics, test_predictions)
    """
    # Train model
    model = train_xgboost_model(X_train, y_train, X_val, y_val, params)
    
    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val)
    
    # Get feature importance
    get_feature_importance(model, X_train.columns.tolist())
    
    # Make test predictions
    test_predictions = make_predictions(model, X_test)
    
    # Save model if path provided
    if save_model_path:
        save_model(model, save_model_path)
    
    return model, metrics, test_predictions

def load_best_model_from_tuning(model_path: str = "models/best_xgboost_model.pkl") -> XGBClassifier:
    """
    Load the best model from hyperparameter tuning.
    
    Args:
        model_path: Path to the saved best model
    
    Returns:
        Loaded XGBoost model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}. Run hyperparameter tuning first.")
    
    model = joblib.load(model_path)
    print(f"Best model loaded from {model_path}")
    return model

def generate_submission_filename(auc_score: float, pipeline_type: str = "full", custom_suffix: str = None) -> str:
    """
    Generate a unique submission filename with timestamp and AUC score.
    
    Args:
        auc_score: Validation AUC score
        pipeline_type: Type of pipeline ("full" or "short")
        custom_suffix: Optional custom suffix to add to filename
    
    Returns:
        Generated filename string
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Format AUC score to 4 decimal places
    auc_str = f"{auc_score:.4f}"
    
    # Create base filename
    filename_parts = [
        "submission",
        pipeline_type,
        f"auc_{auc_str}",
        timestamp
    ]
    
    # Add custom suffix if provided
    if custom_suffix:
        filename_parts.append(custom_suffix)
    
    # Join parts with underscores
    filename = "_".join(filename_parts) + ".csv"
    
    return filename

def create_final_submission(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    submission_path: str = None,
    test_obs_ids: pd.Series = None,
    auc_score: float = None,
    pipeline_type: str = "full"
) -> Tuple[pd.DataFrame, str]:
    """
    Create the final submission file for the competition.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        submission_path: Path to save the final submission (if None, generates automatic name)
        test_obs_ids: Actual obs_id values from test data (if None, uses sequential IDs)
        auc_score: Validation AUC score for filename generation
        pipeline_type: Type of pipeline for filename generation
    
    Returns:
        Tuple of (DataFrame with final predictions, submission file path)
    """
    print("Creating final submission file...")
    
    # Make predictions
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Use actual obs_ids if provided, otherwise use sequential IDs
    if test_obs_ids is not None:
        obs_ids = test_obs_ids.values
        print(f"Using actual obs_ids from test data (range: {obs_ids.min()} - {obs_ids.max()})")
    else:
        obs_ids = range(len(predictions))
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'obs_id': obs_ids,
        'target': predictions
    })
    
    # Generate submission path if not provided
    if submission_path is None:
        if auc_score is not None:
            filename = generate_submission_filename(auc_score, pipeline_type)
            # Create submissions directory
            os.makedirs("submissions", exist_ok=True)
            submission_path = os.path.join("submissions", filename)
        else:
            submission_path = "last_submission.csv"
    
    # Save submission file
    submission_df.to_csv(submission_path, index=False)
    print(f"Final submission saved to: {submission_path}")
    return submission_df, submission_path
