import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from typing import Tuple, Dict, Any
import joblib
import os
from pathlib import Path
from datetime import datetime
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
    pipeline_type: str = "full",
    verbose: bool = False
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
        verbose: Whether to print verbose output
    
    Returns:
        Tuple of (DataFrame with final predictions, submission file path)
    """
    if verbose:
        print("Creating final submission file...")
    
    # Make predictions
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Use actual obs_ids if provided, otherwise use sequential IDs
    if test_obs_ids is not None:
        obs_ids = test_obs_ids.values
        if verbose:
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
    if verbose:
        print(f"Final submission saved to: {submission_path}")
    return submission_df, submission_path


# =============================================================================
# ENSEMBLE METHODS - Quick plug-and-play implementation
# =============================================================================

class SimpleEnsemble:
    """
    Minimalistic ensemble that combines XGBoost, LightGBM, and CatBoost.
    Drop-in replacement for single XGBoost model.
    """
    
    def __init__(self, use_lightgbm=True, use_catboost=True, verbose=False):
        """
        Initialize ensemble with available models.
        
        Args:
            use_lightgbm: Whether to include LightGBM (if available)
            use_catboost: Whether to include CatBoost (if available)
            verbose: Whether to print verbose output
        """
        self.models = {}
        self.verbose = verbose
        
        # XGBoost (always available) - using your best tuned parameters
        self.models['xgb'] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            tree_method='hist',
            random_state=420,
            eval_metric='auc'
        )
        
        # LightGBM (if available) - optimized parameters
        if use_lightgbm:
            self.models['lgb'] = LGBMClassifier(
                n_estimators=300,  # More trees for better performance
                learning_rate=0.02,  # Lower learning rate
                max_depth=6,  # Slightly deeper
                min_child_samples=15,  # Adjusted for your data size
                subsample=0.8,  # Higher subsample
                colsample_bytree=0.8,
                reg_alpha=0.5,  # Lower regularization
                reg_lambda=3.0,  # Lower regularization
                random_state=420,
                verbose=-1
            )
        elif use_lightgbm:
            if verbose:
                print("Skipping LightGBM - not available")
        
        # CatBoost (if available) - optimized parameters
        if use_catboost:
            self.models['cat'] = CatBoostClassifier(
                iterations=300,  # More iterations
                learning_rate=0.02,  # Lower learning rate
                depth=6,  # Slightly deeper
                subsample=0.8,  # Higher subsample
                colsample_bylevel=0.8,
                reg_lambda=3.0,  # Lower regularization
                random_seed=420,
                verbose=False
            )
        elif use_catboost:
            if verbose:
                print("Skipping CatBoost - not available")
        
        if verbose:
            print(f"Ensemble initialized with {len(self.models)} models: {list(self.models.keys())}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in the ensemble."""
        if self.verbose:
            print(f"Training ensemble with {len(self.models)} models...")
        
        # Clean feature names for LightGBM compatibility
        X_train_clean = X_train.copy()
        X_val_clean = X_val.copy() if X_val is not None else None
        
        if hasattr(X_train, 'columns'):
            # Replace special characters that LightGBM doesn't like
            clean_columns = [col.replace(':', '_').replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                           for col in X_train.columns]
            X_train_clean.columns = clean_columns
            if X_val_clean is not None:
                X_val_clean.columns = clean_columns
        
        for name, model in self.models.items():
            if self.verbose:
                print(f"  Training {name}...")
            
            # Different models have different fit signatures
            if name == 'xgb':
                model.fit(X_train_clean, y_train, eval_set=[(X_val_clean, y_val)] if X_val_clean is not None else None, verbose=False)
            elif name == 'lgb':
                model.fit(X_train_clean, y_train, eval_set=[(X_val_clean, y_val)] if X_val_clean is not None else None)
            elif name == 'cat':
                model.fit(X_train_clean, y_train, eval_set=[(X_val_clean, y_val)] if X_val_clean is not None else None, verbose=False)
        
        if self.verbose:
            print("Ensemble training completed!")
    
    def predict_proba(self, X):
        """Make ensemble predictions by averaging probabilities."""
        predictions = []
        
        # Clean feature names for LightGBM compatibility
        X_clean = X.copy()
        if hasattr(X, 'columns'):
            clean_columns = [col.replace(':', '_').replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                           for col in X.columns]
            X_clean.columns = clean_columns
        
        for name, model in self.models.items():
            pred = model.predict_proba(X_clean)[:, 1]  # Get probability of positive class
            predictions.append(pred)
        
        # Simple average of all model predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Return in sklearn format (2D array with both classes)
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def predict(self, X):
        """Make binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def get_feature_importances(self, normalized=True):
        """Get average feature importance across all models."""
        importances = []
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        
        if importances:
            avg_importance = np.mean(importances, axis=0)
            if normalized:
                # Normalize to sum to 1
                return avg_importance / avg_importance.sum()
            else:
                return avg_importance
        else:
            return None


def train_ensemble_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    use_lightgbm: bool = True,
    use_catboost: bool = True,
    verbose: bool = True
) -> SimpleEnsemble:
    """
    Train an ensemble model - drop-in replacement for train_xgboost_model().
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        use_lightgbm: Whether to include LightGBM
        use_catboost: Whether to include CatBoost
        verbose: Whether to print verbose output
    
    Returns:
        Trained ensemble model
    """
    if verbose:
        print("Training ensemble model...")
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
    
    # Create and train ensemble
    ensemble = SimpleEnsemble(use_lightgbm=use_lightgbm, use_catboost=use_catboost, verbose=verbose)
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    return ensemble


def evaluate_ensemble_model(ensemble: SimpleEnsemble, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    """
    Evaluate the trained ensemble model - drop-in replacement for evaluate_model().
    
    Args:
        ensemble: Trained ensemble model
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
    y_pred = ensemble.predict(X_val)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    print(f"Ensemble Validation AUC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return {
        'auc': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def save_ensemble_model(ensemble: SimpleEnsemble, filepath: str) -> None:
    """
    Save the trained ensemble model to disk.
    
    Args:
        ensemble: Trained ensemble model
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(ensemble, filepath)
    print(f"Ensemble model saved to {filepath}")


def load_ensemble_model(filepath: str) -> SimpleEnsemble:
    """
    Load a trained ensemble model from disk.
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded ensemble model
    """
    ensemble = joblib.load(filepath)
    print(f"Ensemble model loaded from {filepath}")
    return ensemble


def train_and_evaluate_ensemble_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    X_test: pd.DataFrame,
    use_lightgbm: bool = True,
    use_catboost: bool = True,
    save_model_path: str = None
) -> Tuple[SimpleEnsemble, Dict[str, float], np.ndarray]:
    """
    Complete ensemble training and evaluation pipeline - drop-in replacement for train_and_evaluate_pipeline().
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        use_lightgbm: Whether to include LightGBM
        use_catboost: Whether to include CatBoost
        save_model_path: Path to save the model (optional)
    
    Returns:
        Tuple of (trained_ensemble, evaluation_metrics, test_predictions)
    """
    # Train ensemble
    ensemble = train_ensemble_model(X_train, y_train, X_val, y_val, use_lightgbm, use_catboost)
    
    # Evaluate ensemble
    metrics = evaluate_ensemble_model(ensemble, X_val, y_val)
    
    # Get feature importance (average across models) - normalized scale
    feature_importance = ensemble.get_feature_importances(normalized=True)
    if feature_importance is not None:
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 20 Most Important Features (Ensemble Average - Normalized 0-1 Scale):")
        print(importance_df.head(20))
    
    # Make test predictions
    test_predictions = ensemble.predict_proba(X_test)[:, 1]
    print(f"Made ensemble predictions for {len(X_test)} test samples")
    
    # Save ensemble if path provided
    if save_model_path:
        save_ensemble_model(ensemble, save_model_path)
    
    return ensemble, metrics, test_predictions
