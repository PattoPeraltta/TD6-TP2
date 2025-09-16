import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import json
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from src.config import XGB_FIXED_PARAMS, SEED
from src.model import generate_submission_filename
from src.data import per_user_time_split


class HyperoptTuner:
    """
    Hyperparameter tuning class using Hyperopt's Tree-structured Parzen Estimator (TPE).
    More efficient than grid/random search for finding optimal hyperparameters.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Hyperopt tuner.
        
        Args:
            config: Tuning configuration dictionary
        """
        
        
        self.config = config or self._get_default_config()
        self.best_params = None
        self.best_score = 0.0
        self.tuning_results = []
        self.trials = None
        self.space = self._define_search_space()
        
        # Set random seed for reproducibility
        np.random.seed(SEED)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Hyperopt tuning."""
        return {
            'n_trials': 100,  # Number of optimization iterations
            'cv_folds': 1,  # Using temporal validation (single fold)
            'verbose': True,
            'save_best_model': True,
            'best_model_path': 'models/best_xgboost_model_hyperopt.pkl',
            'early_stopping_rounds': 50
        }
    
    def _define_search_space(self) -> Dict[str, Any]:
        """
        Define the search space for Hyperopt optimization.
        Uses more flexible distributions than discrete grids.
        """
        
        space = {
            # Continuous parameters
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(10.0)),
            
            # Discrete parameters
            'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 1000, 1500, 2000, 3500, 5000]),
            
            # Add fixed parameters
            **XGB_FIXED_PARAMS
        }

        
        return space
    
    def _objective(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Objective function for Hyperopt optimization.
        
        Args:
            params: Parameter dictionary from Hyperopt
            X_train: Feature matrix
            y_train: Target vector
            X_val: Feature matrix
            y_val: Target vector
            
        Returns:
            Dictionary with loss and status
        """
        try:
            # Convert float parameters to appropriate types
            params = self._convert_params(params)
            
            # Perform cross-validation
            mean_cv_score, fold_scores = self._cross_validate_params(X_train, y_train, X_val, y_val, params)
            
            # Store results
            result = {
                'params': params.copy(),
                'mean_cv_score': mean_cv_score,
                'fold_scores': fold_scores,
                'std_cv_score': np.std(fold_scores)
            }
            self.tuning_results.append(result)
            
            if self.config['verbose']:
                print(f"Trial {len(self.tuning_results)}: Temporal AUC = {mean_cv_score:.4f}")
                if self.config.get('verbose_params', False):
                    print(f"  Params: {params}")
            
            # Return negative AUC (Hyperopt minimizes)
            return {
                'loss': -mean_cv_score,
                'status': STATUS_OK,
                'auc': mean_cv_score,
                'params': params
            }
            
        except Exception as e:
            if self.config['verbose']:
                print(f"Error in objective function: {e}")
            return {
                'loss': 0.0,
                'status': STATUS_OK,
                'auc': 0.0,
                'params': params
            }
    
    def _convert_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Hyperopt parameter types to appropriate XGBoost types."""
        converted = params.copy()
        
        # Convert float parameters to int where needed
        if 'n_estimators' in converted:
            converted['n_estimators'] = int(converted['n_estimators'])
        if 'max_depth' in converted:
            converted['max_depth'] = int(converted['max_depth'])
        if 'min_child_weight' in converted:
            converted['min_child_weight'] = int(converted['min_child_weight'])
        
        return converted
    
    def _cross_validate_params(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        params: Dict[str, Any],
    ) -> Tuple[float, List[float]]:
        """
        Perform temporal validation for a given parameter set using pre-computed temporal split.
        
        Args:
            X_train: Feature matrix
            y_train: Target vector
            X_val: Feature matrix
            y_val: Target vector
            params: XGBoost parameters
            
        Returns:
            Tuple of (mean_cv_score, list_of_fold_scores)
        """
        
        # Create and train model
        model_params = params.copy()
        model_params['early_stopping_rounds'] = self.config.get('early_stopping_rounds', 50)
        model = XGBClassifier(**model_params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Make predictions and calculate AUC
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, y_pred_proba)
        
        # Return single score as a list for compatibility
        cv_scores = [fold_auc]
        mean_score = fold_auc
        
        return mean_score, cv_scores
    
    def tune_hyperparameters(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Hyperopt TPE.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for final evaluation)
            y_val: Validation labels (optional, for final evaluation)
            
        Returns:
            Dictionary with best parameters and tuning results
        """
        print("=" * 60)
        print("HYPEROPT HYPERPARAMETER TUNING")
        print("=" * 60)
        print(f"Max evaluations: {self.config['n_trials']}")
        print(f"Validation method: Temporal per-user split (80/20)")
        print(f"Search space: Tree-structured Parzen Estimator (TPE)")
        print("-" * 60)
        
        # Initialize trials
        self.trials = Trials()
        
        # Create objective function with data
        objective_with_data = lambda params: self._objective(params, X_train, y_train, X_val, y_val)
        
        # Perform optimization
        start_time = time.time()
        
        print("Starting Hyperopt optimization...")
        best = fmin(
            fn=objective_with_data,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.config['n_trials'],
            trials=self.trials,
            rstate=np.random.default_rng(SEED),
            verbose=self.config['verbose']
        )
        
        # Convert best parameters back to proper types
        self.best_params = self._convert_params(space_eval(self.space, best))
        
        # Find best score from trials
        best_trial = min(self.trials.trials, key=lambda x: x['result']['loss'])
        self.best_score = best_trial['result']['auc']
        
        tuning_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("HYPEROPT TUNING COMPLETED")
        print("=" * 60)
        print(f"Total tuning time: {tuning_time:.2f} seconds")
        print(f"Best Temporal AUC: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        print(f"Total trials: {len(self.trials.trials)}")
        
        # Final evaluation on validation set if provided
        if X_val is not None and y_val is not None:
            print("\n" + "-" * 40)
            print("FINAL VALIDATION EVALUATION")
            print("-" * 40)
            
            final_model_params = self.best_params.copy()
            final_model_params['early_stopping_rounds'] = self.config.get('early_stopping_rounds', 50)
            final_model = XGBClassifier(**final_model_params)
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
            'tuning_time': tuning_time,
            'trials': self.trials
        }
    
    def _save_tuning_results(self):
        """Save Hyperopt tuning results to files."""
        # Create results directory
        results_dir = Path("tuning_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = results_dir / "hyperopt_tuning_results.json"
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
                'config': self.config,
                'total_trials': len(self.trials.trials) if self.trials else 0
            }, f, indent=2)
        
        # Save best parameters summary
        summary_file = results_dir / "hyperopt_best_params_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("HYPEROPT BEST HYPERPARAMETERS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best Temporal AUC: {self.best_score:.4f}\n")
            f.write(f"Validation Method: Per-user temporal split (80/20)\n")
            f.write(f"Total Trials: {len(self.trials.trials) if self.trials else 0}\n\n")
            f.write("Best Parameters:\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
        
        print(f"\nHyperopt tuning results saved to:")
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
    

def run_complete_hyperopt_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any] = None
) -> Tuple[XGBClassifier, Dict[str, Any], pd.DataFrame]:
    """
    Run the complete Hyperopt hyperparameter tuning and submission pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Tuning configuration
        
    Returns:
        tuning_results
    """
    print("Starting Complete Hyperopt Hyperparameter Tuning Pipeline")
    print("=" * 60)
    
    # Initialize Hyperopt tuner
    tuner = HyperoptTuner(config)
    
    # Perform hyperparameter tuning
    tuning_results = tuner.tune_hyperparameters(
        X_train, y_train, X_val, y_val
    )
    
    print("\nComplete Hyperopt tuning pipeline finished successfully!")
    
    return tuning_results
