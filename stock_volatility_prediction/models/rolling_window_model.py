import numpy as np
import pandas as pd
import mlflow
from mlflow_init import setup_mlflow, initialize_mlflow
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class RollingWindowModel:
    """
    Rolling Window model for volatility prediction with enhanced MLflow tracking.
    Predicts volatility using a rolling window mean approach.
    """
    
    def __init__(self, window_size=30):
        """
        Initialize Rolling Window model with MLflow support
        
        Args:
            window_size (int): Number of periods to include in rolling window
        """
        self.window_size = window_size
        self.mlflow_run_id = None
        self.ticker = None  # Will be set during prediction
    
    def _log_model_parameters(self):
        """Log model parameters to MLflow"""
        if self.mlflow_run_id and mlflow.active_run():
            mlflow.log_params({
                'model_type': 'RollingWindow',
                'window_size': self.window_size,
                'ticker': self.ticker if self.ticker else 'unknown'
            })
    
    def _log_prediction_metrics(self, predictions):
        """Log prediction metrics to MLflow"""
        if self.mlflow_run_id and mlflow.active_run() and len(predictions) > 0:
            try:
                mlflow.log_metrics({
                    'prediction_mean': np.mean(predictions),
                    'prediction_std': np.std(predictions),
                    'prediction_max': np.max(predictions),
                    'prediction_min': np.min(predictions)
                })
            except Exception as e:
                warnings.warn(f"Failed to log prediction metrics: {str(e)}")
    
    def _detect_ticker(self, data):
        """Attempt to detect ticker symbol from data"""
        if 'ticker' in data.columns:
            self.ticker = data['ticker'].iloc[0] if len(data['ticker']) > 0 else 'unknown'
        elif 'Close' in data.columns:
            self.ticker = 'Close'
        elif 'close' in data.columns:
            self.ticker = 'close'
        else:
            self.ticker = 'unknown'
    
    def fit_predict(self, train_data, test_data):
        """
        Generate rolling window predictions with comprehensive MLflow tracking.

        Args:
            train_data: DataFrame containing training data with volatility information.
            test_data: DataFrame containing test data with volatility information.
            
        Returns:
            Array of predictions based on rolling window mean.
        """
        # Initialize MLflow tracking first
        initialize_mlflow()
        
        # Initialize MLflow tracking if not already done
        if not self.mlflow_run_id:
            self._detect_ticker(train_data)
            self.mlflow_run_id = setup_mlflow(
                model_type="RollingWindow",
                ticker=self.ticker if self.ticker else "unknown",
                params_path="params.yaml"
            )
        
        rolling_pred = []
        train_size = len(train_data)
        
        # Start MLflow run
        with mlflow.start_run(run_id=self.mlflow_run_id, nested=True):
            # Log parameters
            self._log_model_parameters()
            
            # Track prediction steps
            prediction_steps = []
            
            for i in range(len(test_data)):
                if i < self.window_size:
                    continue

                # Extract the window data
                if i == 0:
                    window_data = train_data['volatility'].iloc[-self.window_size:]
                else:
                    window_data = test_data['volatility'].iloc[max(0, i - self.window_size):i]

                # Calculate the mean of the window data and append to predictions
                current_pred = window_data.mean()
                rolling_pred.append(current_pred)
                
                # Track each prediction step
                prediction_steps.append({
                    'step': i,
                    'prediction': current_pred,
                    'window_size': len(window_data)
                })
            
            # Convert predictions to numpy array
            predictions = np.array(rolling_pred)
            
            # Log prediction metrics
            self._log_prediction_metrics(predictions)
            
            # Log additional information about the predictions
            if len(predictions) > 0:
                mlflow.log_metric("num_predictions", len(predictions))
                mlflow.log_metric("first_prediction", predictions[0])
                mlflow.log_metric("last_prediction", predictions[-1])
                
                # Create artifacts directory if it doesn't exist
                artifacts_dir = "Artifacts"
                os.makedirs(artifacts_dir, exist_ok=True)
                
                # Log prediction statistics as artifacts
                pred_stats = pd.DataFrame({
                    'step': range(len(predictions)),
                    'prediction': predictions,
                    'model_type': ['RollingWindow'] * len(predictions),
                    'ticker': [self.ticker] * len(predictions),
                    'timestamp': [datetime.now().isoformat()] * len(predictions)
                })
                
                # Save to Artifacts folder
                artifact_path = os.path.join(artifacts_dir, f"rolling_window_predictions_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                pred_stats.to_csv(artifact_path, index=False)
                mlflow.log_artifact(artifact_path)
                
                # Log model summary
                summary_data = {
                    'model_type': 'RollingWindow',
                    'ticker': self.ticker,
                    'window_size': self.window_size,
                    'num_predictions': len(predictions),
                    'mean_prediction': np.mean(predictions),
                    'std_prediction': np.std(predictions),
                    'min_prediction': np.min(predictions),
                    'max_prediction': np.max(predictions),
                    'timestamp': datetime.now().isoformat()
                }
                summary_df = pd.DataFrame([summary_data])
                summary_path = os.path.join(artifacts_dir, f"rolling_window_summary_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                summary_df.to_csv(summary_path, index=False)
                mlflow.log_artifact(summary_path)
            
            return predictions
    
    def forecast_future(self, horizon):
        """
        Placeholder for future forecast method with MLflow tracking.
        Note: Rolling Window is typically not used for future forecasting.

        Args:
            horizon: The number of future time steps to forecast.
            
        Returns:
            Array of future predictions (placeholders).
        """
        # Start MLflow run if not already active
        if not mlflow.active_run() and self.mlflow_run_id:
            mlflow.start_run(run_id=self.mlflow_run_id, nested=True)
        
        # Log that this is a placeholder forecast
        if mlflow.active_run():
            mlflow.log_param("forecast_warning", "RollingWindow not designed for forecasting")
            mlflow.log_metric("forecast_horizon", horizon)
        
        # Rolling Window models are generally not used for future forecasting
        # This method is a placeholder to maintain consistency with other models
        return np.zeros(horizon)