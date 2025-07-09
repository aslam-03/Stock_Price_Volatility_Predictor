import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import mlflow
from mlflow_init import initialize_mlflow
from datetime import datetime
from mlflow import log_artifact, log_params, log_metrics
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation for volatility prediction models with enhanced MLflow (DagsHub) support
    """

    def __init__(self, experiment_name="Volatility_Evaluation"):
        """
        Initialize ModelEvaluator with MLflow tracking
        
        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        self.metrics_history = []
        self.current_run_id = None
        self.experiment_name = experiment_name
        
        # Initialize MLflow with DagsHub authentication
        try:
            initialize_mlflow()
            # Set experiment if not default
            if experiment_name != "Stock_Volatility_Prediction":
                mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: MLflow initialization failed: {str(e)}")
            print("Model evaluation will continue without MLflow tracking")

    def _start_mlflow_run(self, model_name=None):
        """Start a new MLflow run if one isn't already active"""
        if not mlflow.active_run():
            run_name = f"{model_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if model_name else None
            mlflow.start_run(run_name=run_name)
            self.current_run_id = mlflow.active_run().info.run_id
        return self.current_run_id

    def evaluate_model(self, y_true, y_pred, model_name=None):
        """
        Evaluate model predictions with comprehensive metrics and MLflow logging
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            model_name: Optional name of model being evaluated
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            min_len = min(len(y_true), len(y_pred))
            y_true, y_pred = y_true[:min_len], y_pred[:min_len]
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true, y_pred = y_true[mask], y_pred[mask]

            if len(y_true) == 0:
                raise ValueError("No valid data points for evaluation")

            # Calculate base metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
                'r2': r2_score(y_true, y_pred),
                'directional_accuracy': (
                    np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)) * 100
                    if len(y_true) > 1 else 0
                ),
                'hit_ratio': (
                    np.mean(np.abs(y_pred - y_true) / (y_true + 1e-8) <= 0.1) * 100
                ),
                'mean_true_vol': np.mean(y_true),
                'mean_pred_vol': np.mean(y_pred),
                'vol_bias': np.mean(y_pred - y_true),
                'vol_correlation': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0,
                'theil_u': self.calculate_theil_u(y_true, y_pred),
                'information_coefficient': self.calculate_ic(y_true, y_pred)
            }

            # Add risk metrics
            risk_metrics = self.calculate_risk_metrics(y_true, y_pred)
            metrics.update(risk_metrics)

            if model_name:
                metrics['model_name'] = model_name
                metrics['evaluation_date'] = pd.Timestamp.now().isoformat()
                self.metrics_history.append(metrics.copy())
                
                # Log to MLflow
                self._log_evaluation_metrics(metrics, model_name)

            return metrics

        except Exception as e:
            error_metrics = {
                'rmse': np.inf,
                'mae': np.inf,
                'mape': np.inf,
                'r2': -np.inf,
                'directional_accuracy': 0,
                'hit_ratio': 0,
                'mean_true_vol': 0,
                'mean_pred_vol': 0,
                'vol_bias': 0,
                'vol_correlation': 0,
                'theil_u': np.inf,
                'information_coefficient': 0,
                'error': str(e)
            }
            if model_name:
                self._log_evaluation_metrics(error_metrics, model_name)
            return error_metrics

    def _log_evaluation_metrics(self, metrics, model_name):
        """Log evaluation metrics to MLflow"""
        try:
            self._start_mlflow_run(model_name)
            
            # Log model name as parameter
            mlflow.log_params({'evaluated_model': model_name})
            
            # Prepare metrics for logging (only numeric values)
            metrics_to_log = {k: v for k, v in metrics.items() 
                             if isinstance(v, (int, float)) and not np.isnan(v)}
            
            # Log all metrics at once
            mlflow.log_metrics(metrics_to_log)
            
            # Save full metrics as JSON artifact
            artifact_path = f"{model_name}_metrics.json"
            with open(artifact_path, "w") as f:
                json.dump(metrics, f, indent=4)
            mlflow.log_artifact(artifact_path)
            os.remove(artifact_path)  # Clean up
            
            # Additional diagnostic logging
            mlflow.set_tag("evaluation_type", "volatility_prediction")
            mlflow.set_tag("model_family", model_name.split('_')[0] if model_name else "unknown")
            
        except Exception as e:
            warnings.warn(f"Failed to log metrics to MLflow: {str(e)}")

    def calculate_theil_u(self, y_true, y_pred):
        try:
            if len(y_true) <= 1: return np.inf
            mse_pred = np.mean((y_true - y_pred) ** 2)
            naive_pred = np.roll(y_true, 1)[1:]
            y_true_subset = y_true[1:]
            mse_naive = np.mean((y_true_subset - naive_pred) ** 2)
            return 0 if mse_naive == 0 and mse_pred == 0 else (
                np.inf if mse_naive == 0 else np.sqrt(mse_pred) / np.sqrt(mse_naive)
            )
        except:
            return np.inf

    def calculate_ic(self, y_true, y_pred):
        try:
            if len(y_true) <= 1: return 0
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_true, y_pred)
            return ic if not np.isnan(ic) else 0
        except:
            try:
                return np.corrcoef(y_true, y_pred)[0, 1]
            except:
                return 0

    def calculate_risk_metrics(self, y_true, y_pred):
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if len(y_true) == 0:
                return {}

            errors = y_pred - y_true
            var_95 = np.percentile(np.abs(errors), 95)
            tail_errors = np.abs(errors)[np.abs(errors) >= var_95]

            return {
                'error_var_95': var_95,
                'error_es_95': np.mean(tail_errors) if len(tail_errors) > 0 else 0,
                'max_error': np.max(np.abs(errors)),
                'error_volatility': np.std(errors),
                'prediction_sharpe': -np.mean(errors) / (np.std(errors) + 1e-8)
            }
        except:
            return {}

    def get_evaluation_summary(self, y_true, y_pred, model_name="Model"):
        """
        Comprehensive evaluation with automatic MLflow logging
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of model being evaluated
            
        Returns:
            dict: Combined evaluation metrics
        """
        summary = self.evaluate_model(y_true, y_pred, model_name)
        return summary

    def log_to_mlflow(self, metrics_dict, model_name):
        """
        Explicit method to log metrics to MLflow (alternative to automatic logging)
        """
        self._log_evaluation_metrics(metrics_dict, model_name)

    def compare_models(self, results_dict):
        """
        Compare multiple models' results with MLflow logging
        
        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        if not results_dict:
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(results_dict).T
            df = df.sort_values('rmse', ascending=True)
            
            # Calculate ranks
            for metric in ['rmse', 'mae', 'mape']:
                df[f'{metric}_rank'] = df[metric].rank(ascending=True)
            for metric in ['r2', 'hit_ratio', 'directional_accuracy', 'vol_correlation']:
                if metric in df.columns:
                    df[f'{metric}_rank'] = df[metric].rank(ascending=False)
            
            # Log comparison to MLflow
            self._log_model_comparison(df)
            
            return df
        except Exception as e:
            warnings.warn(f"Model comparison failed: {str(e)}")
            return pd.DataFrame()

    def _log_model_comparison(self, comparison_df):
        """Log model comparison results to MLflow"""
        try:
            self._start_mlflow_run("model_comparison")
            
            # Log top model for each metric
            for metric in ['rmse', 'mae', 'r2', 'hit_ratio']:
                if metric in comparison_df.columns:
                    top_model = comparison_df.sort_values(metric, ascending=metric not in ['r2', 'hit_ratio']).index[0]
                    mlflow.log_metric(f"top_{metric}_model", 1)  # Dummy value
                    mlflow.set_tag(f"top_{metric}_model", top_model)
            
            # Save full comparison as artifact
            artifact_path = "model_comparison.csv"
            comparison_df.to_csv(artifact_path)
            log_artifact(artifact_path)
            os.remove(artifact_path)
            
        except Exception as e:
            warnings.warn(f"Failed to log model comparison: {str(e)}")

    def export_results(self, filepath=None):
        """
        Export evaluation history with optional MLflow artifact logging
        
        Args:
            filepath: Optional path to save results
            
        Returns:
            pd.DataFrame: DataFrame of all historical metrics
        """
        if not self.metrics_history:
            return pd.DataFrame()
            
        try:
            df = pd.DataFrame(self.metrics_history)
            
            if filepath:
                df.to_csv(filepath, index=False)
                # Log to MLflow if active
                if mlflow.active_run():
                    log_artifact(filepath)
            
            return df
        except Exception as e:
            warnings.warn(f"Failed to export results: {str(e)}")
            return pd.DataFrame()