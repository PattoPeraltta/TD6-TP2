import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import json
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
from itertools import product
import random

from config import TUNING_CONFIG, XGB_PARAM_GRID, XGB_FIXED_PARAMS, SEED

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning class for XGBoost models.
    Supports both grid search and random search with cross-validation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            config: Tuning configuration dictionary
        """
        self.config = config or TUNING_CONFIG
        self.best_params = None
        self.best_score = 0.0
        self.tuning_results = []
        self.cv_scores = []
        
        # Set random seed for reproducibility
        random.seed(SEED)
        np.random.seed(SEED)
    
    def _generate_random_params(self, n_trials: int) -> List[Dict[str, Any]]:
        """
        Generate random parameter combinations for random search.
        
        Args:
            n_trials: Number of random parameter combinations to generate
            
        Returns:
            List of parameter dictionaries
        """
        param_combinations = []
        
        for _ in range(n_trials):
            params = {}
            for param_name, param_values in XGB_PARAM_GRID.items():
                params[param_name] = random.choice(param_values)
            
            # Add fixed parameters
            params.update(XGB_FIXED_PARAMS)
            param_combinations.append(params)
        
        return param_combinations
    
    def _generate_grid_params(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search.
        
        Returns:
            List of parameter dictionaries
        """
        param_combinations = []
        
        # Generate all combinations
        for params in ParameterGrid(XGB_PARAM_GRID):
            # Add fixed parameters
            params.update(XGB_FIXED_PARAMS)
            param_combinations.append(params)
        
        return param_combinations
    
    def _cross_validate_params(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        params: Dict[str, Any]
    ) -> Tuple[float, List[float]]:
        """
        Perform cross-validation for a given parameter set.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: XGBoost parameters
            
        Returns:
            Tuple of (mean_cv_score, list_of_fold_scores)
        """
        cv_scores = []
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'], 
            shuffle=True, 
            random_state=SEED
        )
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and train model
            model = XGBClassifier(**params)
            
            # Train with early stopping
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            # Make predictions and calculate AUC
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(fold_auc)
            
            if self.config['verbose']:
                print(f"    Fold {fold + 1}: AUC = {fold_auc:.4f}")
        
        mean_score = np.mean(cv_scores)
        return mean_score, cv_scores
    
    def tune_hyperparameters(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using the specified search method.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for final evaluation)
            y_val: Validation labels (optional, for final evaluation)
            
        Returns:
            Dictionary with best parameters and tuning results
        """
        print("=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        print(f"Search type: {self.config['search_type']}")
        print(f"CV folds: {self.config['cv_folds']}")
        
        # Generate parameter combinations
        if self.config['search_type'] == 'random':
            param_combinations = self._generate_random_params(self.config['n_trials'])
            print(f"Random search trials: {len(param_combinations)}")
        else:
            param_combinations = self._generate_grid_params()
            print(f"Grid search combinations: {len(param_combinations)}")
        
        print(f"Total parameter combinations to test: {len(param_combinations)}")
        print("-" * 60)
        
        # Perform tuning
        start_time = time.time()
        
        for i, params in enumerate(param_combinations):
            print(f"\nTrial {i + 1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            # Cross-validate parameters
            mean_cv_score, fold_scores = self._cross_validate_params(X_train, y_train, params)
            
            # Store results
            result = {
                'trial': i + 1,
                'params': params.copy(),
                'mean_cv_score': mean_cv_score,
                'fold_scores': fold_scores,
                'std_cv_score': np.std(fold_scores)
            }
            self.tuning_results.append(result)
            
            print(f"Mean CV AUC: {mean_cv_score:.4f} (+/- {np.std(fold_scores):.4f})")
            
            # Update best parameters
            if mean_cv_score > self.best_score:
                self.best_score = mean_cv_score
                self.best_params = params.copy()
                print(f"🎉 New best score! AUC: {self.best_score:.4f}")
        
        tuning_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TUNING COMPLETED")
        print("=" * 60)
        print(f"Total tuning time: {tuning_time:.2f} seconds")
        print(f"Best CV AUC: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Final evaluation on validation set if provided
        if X_val is not None and y_val is not None:
            print("\n" + "-" * 40)
            print("FINAL VALIDATION EVALUATION")
            print("-" * 40)
            
            final_model = XGBClassifier(**self.best_params)
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_val_pred = final_model.predict_proba(X_val)[:, 1]
            final_auc = roc_auc_score(y_val, y_val_pred)
            print(f"Final validation AUC: {final_auc:.4f}")
        
        # Save results
        self._save_tuning_results()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tuning_results': self.tuning_results,
            'tuning_time': tuning_time
        }
    
    def _save_tuning_results(self):
        """Save tuning results to files."""
        # Create results directory
        results_dir = Path("tuning_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = results_dir / "tuning_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = []
            for result in self.tuning_results:
                json_result = result.copy()
                json_result['fold_scores'] = [float(score) for score in result['fold_scores']]
                json_result['mean_cv_score'] = float(result['mean_cv_score'])
                json_result['std_cv_score'] = float(result['std_cv_score'])
                json_results.append(json_result)
            
            json.dump({
                'best_params': self.best_params,
                'best_score': float(self.best_score),
                'tuning_results': json_results,
                'config': self.config
            }, f, indent=2)
        
        # Save best parameters summary
        summary_file = results_dir / "best_params_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("BEST HYPERPARAMETERS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best CV AUC: {self.best_score:.4f}\n\n")
            f.write("Best Parameters:\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
        
        print(f"\nTuning results saved to:")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")
    
    def get_top_parameters(self, n_top: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N parameter combinations by CV score.
        
        Args:
            n_top: Number of top combinations to return
            
        Returns:
            List of top parameter combinations
        """
        sorted_results = sorted(
            self.tuning_results, 
            key=lambda x: x['mean_cv_score'], 
            reverse=True
        )
        
        return sorted_results[:n_top]
    
    def plot_tuning_results(self, save_path: str = None):
        """
        Create plots of tuning results (requires matplotlib).
        
        Args:
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract scores and parameters for plotting
            scores = [result['mean_cv_score'] for result in self.tuning_results]
            trials = [result['trial'] for result in self.tuning_results]
            
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Score progression
            plt.subplot(2, 2, 1)
            plt.plot(trials, scores, 'b-', alpha=0.7)
            plt.axhline(y=self.best_score, color='r', linestyle='--', 
                       label=f'Best: {self.best_score:.4f}')
            plt.xlabel('Trial')
            plt.ylabel('CV AUC Score')
            plt.title('Hyperparameter Tuning Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Score distribution
            plt.subplot(2, 2, 2)
            plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=self.best_score, color='r', linestyle='--', 
                       label=f'Best: {self.best_score:.4f}')
            plt.xlabel('CV AUC Score')
            plt.ylabel('Frequency')
            plt.title('Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Parameter importance (variance in scores)
            param_importance = {}
            for param_name in XGB_PARAM_GRID.keys():
                param_scores = []
                for result in self.tuning_results:
                    param_value = result['params'][param_name]
                    param_scores.append((param_value, result['mean_cv_score']))
                
                # Calculate variance for this parameter
                param_df = pd.DataFrame(param_scores, columns=['value', 'score'])
                param_variance = param_df.groupby('value')['score'].var().mean()
                param_importance[param_name] = param_variance
            
            plt.subplot(2, 2, 3)
            params = list(param_importance.keys())
            importances = list(param_importance.values())
            plt.bar(params, importances, alpha=0.7, color='lightcoral')
            plt.xlabel('Parameter')
            plt.ylabel('Score Variance')
            plt.title('Parameter Importance (Variance)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Best parameters
            plt.subplot(2, 2, 4)
            best_param_names = list(self.best_params.keys())
            best_param_values = list(self.best_params.values())
            
            # Convert values to strings for better display
            best_param_values_str = [str(v) for v in best_param_values]
            
            plt.barh(best_param_names, best_param_values_str, alpha=0.7, color='lightgreen')
            plt.xlabel('Parameter Value')
            plt.title('Best Parameters')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping plotting.")
        except Exception as e:
            print(f"Error creating plots: {e}")


def create_submission_file(
    model: XGBClassifier, 
    X_test: pd.DataFrame, 
    test_ids: pd.Series = None,
    submission_path: str = "submission.csv"
) -> pd.DataFrame:
    """
    Create submission file with predictions.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        test_ids: Test sample IDs (optional)
        submission_path: Path to save submission file
        
    Returns:
        DataFrame with predictions
    """
    print(f"\nCreating submission file...")
    
    # Make predictions
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Create submission DataFrame
    if test_ids is not None:
        submission_df = pd.DataFrame({
            'id': test_ids,
            'prediction': predictions
        })
    else:
        submission_df = pd.DataFrame({
            'id': range(len(predictions)),
            'prediction': predictions
        })
    
    # Save submission file
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    return submission_df

def run_complete_tuning_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    config: Dict[str, Any] = None
) -> Tuple[XGBClassifier, Dict[str, Any], pd.DataFrame]:
    """
    Run the complete hyperparameter tuning and submission pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        config: Tuning configuration
        
    Returns:
        Tuple of (best_model, tuning_results, submission_df)
    """
    print("🚀 Starting Complete Hyperparameter Tuning Pipeline")
    print("=" * 60)
    
    # Initialize tuner
    tuner = HyperparameterTuner(config)
    
    # Perform hyperparameter tuning
    tuning_results = tuner.tune_hyperparameters(
        X_train, y_train, X_val, y_val
    )
    
    # Train final model with best parameters
    print("\n🏆 Training final model with best parameters...")
    best_model = XGBClassifier(**tuning_results['best_params'])
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Save best model if configured
    if tuner.config.get('save_best_model', False):
        model_path = tuner.config.get('best_model_path', 'models/best_xgboost_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        print(f"Best model saved to: {model_path}")
    
    # Create submission file
    submission_df = create_submission_file(best_model, X_test)
    
    # Create tuning plots
    tuner.plot_tuning_results("tuning_results/tuning_plots.png")
    
    print("\n✅ Complete tuning pipeline finished successfully!")
    
    return best_model, tuning_results, submission_df
