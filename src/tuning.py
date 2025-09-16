import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import json
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from src.config import XGB_FIXED_PARAMS, SEED
from src.model import generate_submission_filename, SimpleEnsemble
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

        best_params_manual = {
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
        best_auc_manual = 0.8063

        # Insert manual trial with proper structure
        import time
        current_time = time.time()
        
        manual_trial = {
            'state': 2,  # STATUS_OK = 2
            'tid': 0,
            'spec': None,  # Required field
            'result': {
                'loss': 1 - best_auc_manual, 
                'status': STATUS_OK,
                'auc': best_auc_manual,
                'params': best_params_manual
            },
            'misc': {
                'tid': 0,
                'cmd': ('domain_attachment', 'FMinIter_Domain'),
                'workdir': None,
                'vals': {
                    'n_estimators': [best_params_manual['n_estimators']],
                    'learning_rate': [best_params_manual['learning_rate']],
                    'max_depth': [best_params_manual['max_depth']],
                    'min_child_weight': [best_params_manual['min_child_weight']],
                    'subsample': [best_params_manual['subsample']],
                    'colsample_bytree': [best_params_manual['colsample_bytree']],
                    'reg_alpha': [best_params_manual['reg_alpha']],
                    'reg_lambda': [best_params_manual['reg_lambda']],
                },
                'idxs': {
                    'n_estimators': [0],
                    'learning_rate': [0],
                    'max_depth': [0],
                    'min_child_weight': [0],
                    'subsample': [0],
                    'colsample_bytree': [0],
                    'reg_alpha': [0],
                    'reg_lambda': [0],
                }
            },
            'exp_key': None,
            'owner': None,
            'version': 0,
            'book_time': current_time,
            'refresh_time': current_time
        }
        
        self.trials.insert_trial_docs([manual_trial])
        self.trials.refresh()

        
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


# =============================================================================
# ENSEMBLE HYPERPARAMETER TUNING
# =============================================================================

class EnsembleHyperoptTuner:
    """
    Hyperparameter tuning for ensemble models (XGBoost + LightGBM + CatBoost).
    Optimizes parameters for all three models simultaneously.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ensemble tuner."""
        self.config = config or self._get_default_config()
        self.best_params = None
        self.best_score = 0.0
        self.tuning_results = []
        self.trials = None
        self.space = self._define_ensemble_search_space()
        np.random.seed(SEED)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ensemble tuning."""
        return {
            'n_trials': 50,  # Fewer trials since we're tuning 3 models
            'verbose': True,
            'save_best_model': True,
            'best_model_path': 'models/best_ensemble_model_hyperopt.pkl',
            'early_stopping_rounds': 50
        }
    
    def _define_ensemble_search_space(self) -> Dict[str, Any]:
        """Define search space for all three models."""
        space = {
            # XGBoost parameters
            'xgb_n_estimators': hp.choice('xgb_n_estimators', [100, 200, 300, 500]),
            'xgb_learning_rate': hp.loguniform('xgb_learning_rate', np.log(0.01), np.log(0.2)),
            'xgb_max_depth': hp.quniform('xgb_max_depth', 3, 8, 1),
            'xgb_subsample': hp.uniform('xgb_subsample', 0.6, 1.0),
            'xgb_colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.6, 1.0),
            'xgb_reg_alpha': hp.loguniform('xgb_reg_alpha', np.log(0.01), np.log(10.0)),
            'xgb_reg_lambda': hp.loguniform('xgb_reg_lambda', np.log(0.1), np.log(10.0)),
            
            # LightGBM parameters
            'lgb_n_estimators': hp.choice('lgb_n_estimators', [100, 200, 300, 500]),
            'lgb_learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.01), np.log(0.2)),
            'lgb_max_depth': hp.quniform('lgb_max_depth', 3, 8, 1),
            'lgb_subsample': hp.uniform('lgb_subsample', 0.6, 1.0),
            'lgb_colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.6, 1.0),
            'lgb_reg_alpha': hp.loguniform('lgb_reg_alpha', np.log(0.01), np.log(10.0)),
            'lgb_reg_lambda': hp.loguniform('lgb_reg_lambda', np.log(0.1), np.log(10.0)),
            
            # CatBoost parameters
            'cat_iterations': hp.choice('cat_iterations', [100, 200, 300, 500]),
            'cat_learning_rate': hp.loguniform('cat_learning_rate', np.log(0.01), np.log(0.2)),
            'cat_depth': hp.quniform('cat_depth', 3, 8, 1),
            'cat_subsample': hp.uniform('cat_subsample', 0.6, 1.0),
            'cat_reg_lambda': hp.loguniform('cat_reg_lambda', np.log(0.1), np.log(10.0)),
        }
        
        return space
    
    def _convert_ensemble_params(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Convert Hyperopt parameters to model-specific parameters."""
        # XGBoost parameters
        xgb_params = {
            'n_estimators': int(params['xgb_n_estimators']),
            'learning_rate': params['xgb_learning_rate'],
            'max_depth': int(params['xgb_max_depth']),
            'subsample': params['xgb_subsample'],
            'colsample_bytree': params['xgb_colsample_bytree'],
            'reg_alpha': params['xgb_reg_alpha'],
            'reg_lambda': params['xgb_reg_lambda'],
            'tree_method': 'hist',
            'random_state': SEED,
            'eval_metric': 'auc'
        }
        
        # LightGBM parameters
        lgb_params = {
            'n_estimators': int(params['lgb_n_estimators']),
            'learning_rate': params['lgb_learning_rate'],
            'max_depth': int(params['lgb_max_depth']),
            'subsample': params['lgb_subsample'],
            'colsample_bytree': params['lgb_colsample_bytree'],
            'reg_alpha': params['lgb_reg_alpha'],
            'reg_lambda': params['lgb_reg_lambda'],
            'random_state': SEED,
            'verbose': -1
        }
        
        # CatBoost parameters
        cat_params = {
            'iterations': int(params['cat_iterations']),
            'learning_rate': params['cat_learning_rate'],
            'depth': int(params['cat_depth']),
            'subsample': params['cat_subsample'],
            'reg_lambda': params['cat_reg_lambda'],
            'random_seed': SEED,
            'verbose': False
        }
        
        return {
            'xgb': xgb_params,
            'lgb': lgb_params,
            'cat': cat_params
        }
    
    def _objective(self, params: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Objective function for ensemble hyperparameter optimization."""
        try:
            # Convert parameters
            model_params = self._convert_ensemble_params(params)
            
            # Create ensemble with tuned parameters
            ensemble = SimpleEnsemble(use_lightgbm=True, use_catboost=True, verbose=False)
            
            # Update model parameters
            for model_name, model_params_dict in model_params.items():
                if model_name in ensemble.models:
                    # Update the model with new parameters
                    model_class = type(ensemble.models[model_name])
                    ensemble.models[model_name] = model_class(**model_params_dict)
            
            # Train ensemble
            ensemble.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate ensemble
            y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            # Store results
            result = {
                'params': params.copy(),
                'auc': auc_score,
                'model_params': model_params
            }
            self.tuning_results.append(result)
            
            if self.config['verbose']:
                print(f"Trial {len(self.tuning_results)}: Ensemble AUC = {auc_score:.4f}")
            
            return {
                'loss': -auc_score,
                'status': STATUS_OK,
                'auc': auc_score,
                'params': params
            }
            
        except Exception as e:
            if self.config['verbose']:
                print(f"Error in ensemble objective function: {e}")
            return {
                'loss': 0.0,
                'status': STATUS_OK,
                'auc': 0.0,
                'params': params
            }
    
    def tune_ensemble_hyperparameters(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> Dict[str, Any]:
        """Perform ensemble hyperparameter tuning."""
        print("=" * 60)
        print("ENSEMBLE HYPERPARAMETER TUNING")
        print("=" * 60)
        print(f"Max evaluations: {self.config['n_trials']}")
        print(f"Models: XGBoost + LightGBM + CatBoost")
        print(f"Validation method: Temporal per-user split")
        print("-" * 60)
        
        # Initialize trials
        self.trials = Trials()
        
        # Create objective function with data
        objective_with_data = lambda params: self._objective(params, X_train, y_train, X_val, y_val)
        
        # Perform optimization
        start_time = time.time()
        
        print("Starting ensemble hyperparameter optimization...")
        best = fmin(
            fn=objective_with_data,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.config['n_trials'],
            trials=self.trials,
            rstate=np.random.default_rng(SEED),
            verbose=self.config['verbose']
        )
        
        # Get best parameters
        self.best_params = space_eval(self.space, best)
        
        # Find best score from trials
        best_trial = min(self.trials.trials, key=lambda x: x['result']['loss'])
        self.best_score = best_trial['result']['auc']
        
        tuning_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TUNING COMPLETED")
        print("=" * 60)
        print(f"Total tuning time: {tuning_time:.2f} seconds")
        print(f"Best Ensemble AUC: {self.best_score:.4f}")
        print(f"Total trials: {len(self.trials.trials)}")
        
        # Convert best parameters to model format
        best_model_params = self._convert_ensemble_params(self.best_params)
        print(f"\nBest Parameters:")
        for model_name, params in best_model_params.items():
            print(f"  {model_name.upper()}:")
            for param, value in params.items():
                print(f"    {param}: {value}")
        
        # Save results
        self._save_ensemble_tuning_results()
        
        return {
            'best_params': self.best_params,
            'best_model_params': best_model_params,
            'best_score': self.best_score,
            'tuning_results': self.tuning_results,
            'tuning_time': tuning_time,
            'trials': self.trials
        }
    
    def _save_ensemble_tuning_results(self):
        """Save ensemble tuning results to files."""
        results_dir = Path("tuning_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = results_dir / "ensemble_hyperopt_tuning_results.json"
        with open(results_file, 'w') as f:
            json_results = []
            for result in self.tuning_results:
                json_result = result.copy()
                json_result['auc'] = float(result['auc'])
                json_results.append(json_result)
            
            json.dump({
                'best_params': self.best_params,
                'best_score': float(self.best_score),
                'tuning_results': json_results,
                'config': self.config,
                'total_trials': len(self.trials.trials) if self.trials else 0
            }, f, indent=2)
        
        # Save best parameters summary
        summary_file = results_dir / "ensemble_hyperopt_best_params_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("ENSEMBLE HYPERPARAMETER TUNING RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best Ensemble AUC: {self.best_score:.4f}\n")
            f.write(f"Total Trials: {len(self.trials.trials) if self.trials else 0}\n\n")
            
            best_model_params = self._convert_ensemble_params(self.best_params)
            for model_name, params in best_model_params.items():
                f.write(f"{model_name.upper()} Parameters:\n")
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
        
        print(f"\nEnsemble tuning results saved to:")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")


def run_ensemble_hyperopt_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any] = None
) -> Tuple[SimpleEnsemble, Dict[str, Any]]:
    """
    Run the complete ensemble hyperparameter tuning pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Tuning configuration
        
    Returns:
        Tuple of (best_ensemble_model, tuning_results)
    """
    print("Starting Complete Ensemble Hyperparameter Tuning Pipeline")
    print("=" * 60)
    
    # Initialize ensemble tuner
    tuner = EnsembleHyperoptTuner(config)
    
    # Perform hyperparameter tuning
    tuning_results = tuner.tune_ensemble_hyperparameters(
        X_train, y_train, X_val, y_val
    )
    
    # Create final ensemble with best parameters and train it
    best_model_params = tuning_results['best_model_params']
    final_ensemble = SimpleEnsemble(use_lightgbm=True, use_catboost=True, verbose=False)
    
    # Update with best parameters
    for model_name, model_params_dict in best_model_params.items():
        if model_name in final_ensemble.models:
            model_class = type(final_ensemble.models[model_name])
            final_ensemble.models[model_name] = model_class(**model_params_dict)
    
    # Train the final ensemble with best parameters
    print("Training final ensemble with best parameters...")
    final_ensemble.fit(X_train, y_train, X_val, y_val)
    
    print("\nComplete ensemble hyperparameter tuning pipeline finished successfully!")
    
    return final_ensemble, tuning_results
