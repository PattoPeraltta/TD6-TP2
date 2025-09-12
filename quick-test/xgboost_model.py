"""
XGBoost Model Implementation and Testing Script

This script implements and tests an XGBoost model for predicting target values
using the cleaned training and test data.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

class XGBoostModel:
    """XGBoost model implementation with comprehensive evaluation."""
    
    def __init__(self, random_state=42):
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.scaler = StandardScaler()
        self.random_state = random_state
        
    def load_data(self, train_path, test_path):
        """Load and prepare training and test data."""
        print("Loading data...")
        
        # Load training data
        with tqdm(total=2, desc="Loading data") as pbar:
            self.train_data = pd.read_csv(train_path)
            pbar.update(1)
            print(f"Training data shape: {self.train_data.shape}")
            
            # Load test data
            self.test_data = pd.read_csv(test_path)
            pbar.update(1)
            print(f"Test data shape: {self.test_data.shape}")
        
        # Check if target column exists in training data
        if 'reason_end' in self.train_data.columns:
            self.y_train = self.train_data['reason_end']
            self.X_train = self.train_data.drop(['reason_end', 'obs_id'], axis=1, errors='ignore')
        else:
            raise ValueError("Target column 'reason_end' not found in training data")
        
        # Prepare test features (no target)
        self.X_test = self.test_data.drop(['obs_id'], axis=1, errors='ignore')
        
        print(f"Training features shape: {self.X_train.shape}")
        print(f"Test features shape: {self.X_test.shape}")
        
        return self.X_train, self.y_train, self.X_test
    
    def preprocess_data(self):
        """Preprocess the data for XGBoost."""
        print("Preprocessing data...")
        
        # Combine train and test for consistent preprocessing
        all_data = pd.concat([self.X_train, self.X_test], ignore_index=True)
        
        # Identify categorical and numerical columns
        categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = all_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        for col in tqdm(categorical_cols, desc="Processing categorical columns"):
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col].astype(str))
            self.label_encoders[col] = le
        
        # Split back to train and test
        train_size = len(self.X_train)
        self.X_train_processed = all_data[:train_size]
        self.X_test_processed = all_data[train_size:]
        
        # Store feature names
        self.feature_names = self.X_train_processed.columns.tolist()
        
        print(f"Processed training features shape: {self.X_train_processed.shape}")
        print(f"Processed test features shape: {self.X_test_processed.shape}")
        
        return self.X_train_processed, self.X_test_processed
    
    def train_model(self, use_grid_search=True):
        """Train the XGBoost model with optional hyperparameter tuning."""
        print("Training XGBoost model...")
        
        # Split data for validation
        p80 = int(0.8 * len(self.X_train_processed))
        X_train_split = self.X_train_processed.iloc[:p80]
        X_val_split = self.X_train_processed.iloc[p80:]
        y_train_split = self.y_train.iloc[:p80]
        y_val_split = self.y_train.iloc[p80:]
        
        if use_grid_search:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid (smaller for testing)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.9, 1.0],
                'colsample_bytree': [0.9, 1.0]
            }
            
            # Create base model (using classifier for binary classification)
            # Note: early_stopping_rounds not used with GridSearchCV as it doesn't provide validation set
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Grid search with cross-validation (using AUC-ROC for classification)
            grid_search = GridSearchCV(
                xgb_model, param_grid, 
                cv=3, scoring='roc_auc',
                n_jobs=-1, verbose=0  # Disable sklearn's verbose to avoid conflicts
            )
            
            print("Starting grid search...")
            start_time = time.time()
            
            # Calculate total number of fits for progress bar
            total_fits = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * \
                        len(param_grid['learning_rate']) * len(param_grid['subsample']) * \
                        len(param_grid['colsample_bytree']) * 3  # 3 CV folds
            
            print(f"Total combinations to test: {total_fits}")
            
            # Create a progress bar that shows elapsed time during grid search
            with tqdm(total=100, desc="Grid Search Progress", 
                     bar_format='{l_bar}{bar}| {elapsed} elapsed, {rate_fmt}') as pbar:
                
                # Start a background thread to update progress bar
                import threading
                import time as time_module
                
                def update_progress():
                    while not pbar.disable:
                        time_module.sleep(1)  # Update every second
                        if pbar.n < 100:
                            pbar.update(1)
                        else:
                            break
                
                # Start progress updater
                progress_thread = threading.Thread(target=update_progress)
                progress_thread.daemon = True
                progress_thread.start()
                
                # Run grid search
                grid_search.fit(X_train_split, y_train_split)
                
                # Complete progress bar
                pbar.n = 100
                pbar.refresh()
                pbar.disable = True
            
            end_time = time.time()
            
            print(f"Grid search completed in {end_time - start_time:.2f} seconds")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV AUC-ROC: {grid_search.best_score_:.6f}")
            
            self.model = grid_search.best_estimator_
            
        else:
            # Train with default parameters (using classifier for binary classification)
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=-1,
                early_stopping_rounds=10
            )
            
            print("Fitting model with early stopping...")
            self.model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
        
        # Make predictions on validation set
        y_val_pred = self.model.predict(X_val_split)
        y_val_pred_proba = self.model.predict_proba(X_val_split)[:, 1]  # Probability of class 1
        
        # Calculate validation metrics
        val_mse = mean_squared_error(y_val_split, y_val_pred)
        val_mae = mean_absolute_error(y_val_split, y_val_pred)
        val_r2 = r2_score(y_val_split, y_val_pred)
        val_auc_roc = roc_auc_score(y_val_split, y_val_pred_proba)
        
        print(f"\nValidation Metrics:")
        print(f"MSE: {val_mse:.6f}")
        print(f"MAE: {val_mae:.6f}")
        print(f"R²: {val_r2:.6f}")
        print(f"AUC-ROC: {val_auc_roc:.6f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_val_split, y_val_pred))
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model using cross-validation."""
        print("Evaluating model with cross-validation...")
        
        # Create a copy of the model without early stopping for cross-validation
        # (cross_val_score doesn't provide validation sets for early stopping)
        model_for_cv = xgb.XGBClassifier(
            n_estimators=self.model.n_estimators,
            max_depth=self.model.max_depth,
            learning_rate=self.model.learning_rate,
            subsample=self.model.subsample,
            colsample_bytree=self.model.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("Performing cross-validation...")
        # Cross-validation with AUC-ROC
        cv_auc_scores = cross_val_score(
            model_for_cv, self.X_train_processed, self.y_train,
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        # Cross-validation with MSE
        cv_scores = cross_val_score(
            model_for_cv, self.X_train_processed, self.y_train,
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        cv_mse = -cv_scores
        
        # Cross-validation with MAE
        cv_mae_scores = cross_val_score(
            model_for_cv, self.X_train_processed, self.y_train,
            cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        cv_mae = -cv_mae_scores
        
        print(f"\nCross-Validation Results:")
        print(f"AUC-ROC: {cv_auc_scores.mean():.6f} (+/- {cv_auc_scores.std() * 2:.6f})")
        print(f"MSE: {cv_mse.mean():.6f} (+/- {cv_mse.std() * 2:.6f})")
        print(f"MAE: {cv_mae.mean():.6f} (+/- {cv_mae.std() * 2:.6f})")
        
        return cv_auc_scores, cv_mse, cv_mae
    
    def make_predictions(self):
        """Make predictions on test data."""
        print("Making predictions on test data...")
        
        # Predict on test data
        self.test_predictions = self.model.predict(self.X_test_processed)
        self.test_predictions_proba = self.model.predict_proba(self.X_test_processed)[:, 1]  # Probability of class 1
        
        print(f"Predictions shape: {self.test_predictions.shape}")
        print(f"Prediction statistics:")
        print(f"  Mean: {self.test_predictions.mean():.6f}")
        print(f"  Std: {self.test_predictions.std():.6f}")
        print(f"  Min: {self.test_predictions.min():.6f}")
        print(f"  Max: {self.test_predictions.max():.6f}")
        
        print(f"Probability statistics:")
        print(f"  Mean: {self.test_predictions_proba.mean():.6f}")
        print(f"  Std: {self.test_predictions_proba.std():.6f}")
        print(f"  Min: {self.test_predictions_proba.min():.6f}")
        print(f"  Max: {self.test_predictions_proba.max():.6f}")
        
        return self.test_predictions_proba
    
    def get_feature_importance(self, top_n=20):
        """Get and display feature importance."""
        print(f"\nTop {top_n} Feature Importance:")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance."""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_distribution(self, save_path=None):
        """Plot distribution of predictions."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(self.test_predictions, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Test Predictions (Binary)')
        plt.xlabel('Predicted Class')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.hist(self.test_predictions_proba, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Test Probabilities')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        plt.boxplot(self.test_predictions_proba)
        plt.title('Box Plot of Test Probabilities')
        plt.ylabel('Predicted Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions distribution plot saved to {save_path}")
        
        plt.show()
    
    def save_predictions(self, output_path):
        """Save predictions to CSV file."""
        # Get obs_id from test data
        obs_ids = self.test_data['obs_id']
        
        # Create submission dataframe (using probabilities for better performance)
        submission_df = pd.DataFrame({
            'obs_id': obs_ids,
            'target': self.test_predictions_proba
        })
        
        submission_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return submission_df
    
    def run_full_pipeline(self, train_path, test_path, output_path=None, use_grid_search=True):
        """Run the complete pipeline."""
        print("=" * 60)
        print("XGBoost Model Pipeline")
        print("=" * 60)
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Preprocess data
        self.preprocess_data()
        
        # Train model
        self.train_model(use_grid_search=use_grid_search)
        
        # Evaluate model
        self.evaluate_model()
        
        # Make predictions
        self.make_predictions()
        
        # Feature importance
        self.get_feature_importance()
        
        # # Visualizations
        # self.plot_feature_importance(save_path='feature_importance.png')
        # self.plot_predictions_distribution(save_path='predictions_distribution.png')
        
        # Save predictions if output path provided
        if output_path:
            self.save_predictions(output_path)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
        return self.model, self.test_predictions


def main():
    """Main function to run the XGBoost model."""
    
    # File paths
    train_path = 'data/train_data_cleaned.csv'
    test_path = 'data/test_data_cleaned.csv'
    output_path = 'xgboost_predictions.csv'
    
    # Check if files exist
    if not Path(train_path).exists():
        print(f"Error: {train_path} not found!")
        return
    
    if not Path(test_path).exists():
        print(f"Error: {test_path} not found!")
        return
    
    # Create model instance
    model_pipeline = XGBoostModel(random_state=42)
    
    # Run full pipeline
    try:
        model, predictions = model_pipeline.run_full_pipeline(
            train_path=train_path,
            test_path=test_path,
            output_path=output_path,
            use_grid_search=True  # Set to False for faster execution
        )
        
        print(f"\nModel training completed!")
        print(f"Predictions saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
