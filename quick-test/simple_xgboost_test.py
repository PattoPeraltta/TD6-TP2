#!/usr/bin/env python3
"""
Simple XGBoost Test Script

Quick test of XGBoost model without extensive hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the data."""
    print("Loading data...")
    
    # Load training data
    with tqdm(total=2, desc="Loading data") as pbar:
        train_data = pd.read_csv('train_data_cleaned.csv')
        pbar.update(1)
        print(f"Training data shape: {train_data.shape}")
        
        # Load test data
        test_data = pd.read_csv('test_data_cleaned.csv')
        pbar.update(1)
        print(f"Test data shape: {test_data.shape}")
    
    # Prepare features and target
    y_train = train_data['reason_end']
    X_train = train_data.drop(['reason_end', 'obs_id'], axis=1, errors='ignore')
    X_test = test_data.drop(['obs_id'], axis=1, errors='ignore')
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Preprocess categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    label_encoders = {}
    for col in tqdm(categorical_cols, desc="Encoding categorical columns"):
        le = LabelEncoder()
        # Combine train and test for consistent encoding
        all_values = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    return X_train, y_train, X_test, test_data['obs_id']

def train_and_evaluate_model(X_train, y_train):
    """Train XGBoost model and evaluate it."""
    print("Training XGBoost model...")
    
    # Split data for validation
    p80 = int(0.8 * len(X_train))
    X_train_split = X_train.iloc[:p80]
    X_val_split = X_train.iloc[p80:]
    y_train_split = y_train.iloc[:p80]
    y_val_split = y_train.iloc[p80:]

    # Create and train model (using classifier for binary classification)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting model...")
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False
    )
    
    # Make predictions on validation set
    y_val_pred = model.predict(X_val_split)
    y_val_pred_proba = model.predict_proba(X_val_split)[:, 1]  # Probability of class 1
    
    # Calculate metrics
    mse = mean_squared_error(y_val_split, y_val_pred)
    mae = mean_absolute_error(y_val_split, y_val_pred)
    r2 = r2_score(y_val_split, y_val_pred)
    auc_roc = roc_auc_score(y_val_split, y_val_pred_proba)
    
    print(f"\nValidation Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"AUC-ROC: {auc_roc:.6f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_val_split, y_val_pred))
    
    return model

def make_predictions(model, X_test, obs_ids):
    """Make predictions on test data."""
    print("Making predictions on test data...")
    
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    
    print(f"Probability statistics:")
    print(f"  Mean: {predictions_proba.mean():.6f}")
    print(f"  Std: {predictions_proba.std():.6f}")
    print(f"  Min: {predictions_proba.min():.6f}")
    print(f"  Max: {predictions_proba.max():.6f}")
    
    # Create submission dataframe (using probabilities for better performance)
    submission_df = pd.DataFrame({
        'obs_id': obs_ids,
        'target': predictions_proba
    })
    
    return submission_df

def main():
    """Main function."""
    print("=" * 50)
    print("Simple XGBoost Test")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        X_train, y_train, X_test, obs_ids = load_and_preprocess_data()
        
        # Train model
        model = train_and_evaluate_model(X_train, y_train)
        
        # Make predictions
        predictions_df = make_predictions(model, X_test, obs_ids)
        
        # Save predictions
        output_path = 'simple_xgboost_predictions.csv'
        predictions_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        # Show feature importance
        print(f"\nTop 10 Feature Importance:")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
