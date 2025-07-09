import numpy as np
import pandas as pd
from arch import arch_model
import warnings
import mlflow
from mlflow_init import setup_mlflow, initialize_mlflow
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class GARCHModel:
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model for volatility prediction
    with MLflow experiment tracking.
    """
    
    def __init__(self, p=1, q=1, mean='constant', vol='GARCH', dist='normal'):
        """
        Initialize GARCH model
        
        Args:
            p (int): Number of lag terms for volatility
            q (int): Number of lag terms for squared residuals
            mean (str): Mean model specification
            vol (str): Volatility model specification
            dist (str): Distribution assumption
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model = None
        self.fitted_model = None
        self.returns_data = None
        self.mlflow_run_id = None
        self.ticker = None  # Will be set when data is prepared
        
    def prepare_data(self, data):
        """
        Prepare returns data for GARCH modeling
        
        Args:
            data (pd.DataFrame): Input data with returns or price data
            
        Returns:
            pd.Series: Returns data scaled by 100
        """
        returns = None
        
        # Try multiple column names for returns
        if 'Returns' in data.columns:
            returns = data['Returns'].dropna()
        elif 'Log_Returns' in data.columns:
            returns = data['Log_Returns'].dropna()
        elif 'returns' in data.columns:
            returns = data['returns'].dropna()
        elif 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna()
            self.ticker = 'Close'  # Set ticker for MLflow
        elif 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            self.ticker = 'close'  # Set ticker for MLflow
        elif 'volatility' in data.columns:
            # If only volatility data is available, generate synthetic returns
            vol_data = data['volatility'].dropna()
            if len(vol_data) > 1:
                # Create synthetic returns from volatility (simplified approach)
                returns = vol_data.diff().dropna()
            else:
                returns = vol_data
        else:
            # Last resort: use the first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                first_col = numeric_cols[0]
                returns = data[first_col].pct_change().dropna()
                self.ticker = first_col  # Set ticker for MLflow
            else:
                raise ValueError("No suitable returns data found in input")
        
        if returns is None or len(returns) == 0:
            raise ValueError("No valid returns data could be prepared")
        
        # Scale returns by 100 for better numerical stability
        returns = returns * 100
        
        # Remove extreme outliers (beyond 5 standard deviations)
        if len(returns) > 5:
            std_threshold = 5
            mean_val = returns.mean()
            std_val = returns.std()
            if std_val > 0:
                returns = returns[np.abs(returns - mean_val) <= std_threshold * std_val]
        
        # Ensure we have enough data
        if len(returns) < 10:
            raise ValueError(f"Insufficient returns data after cleaning: {len(returns)} observations")
        
        return returns
    
    def _log_model_parameters(self):
        """Log model parameters to MLflow"""
        if self.mlflow_run_id and mlflow.active_run():
            mlflow.log_params({
                'model_type': 'GARCH',
                'p': self.p,
                'q': self.q,
                'mean_model': self.mean,
                'vol_model': self.vol,
                'distribution': self.dist,
                'ticker': self.ticker if self.ticker else 'unknown'
            })
    
    def _log_model_metrics(self, metrics):
        """Log model metrics to MLflow"""
        if self.mlflow_run_id and mlflow.active_run() and metrics:
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                warnings.warn(f"Failed to log metrics to MLflow: {str(e)}")
    
    def _log_artifacts(self, artifacts):
        """Log artifacts to MLflow"""
        if self.mlflow_run_id and mlflow.active_run() and artifacts:
            try:
                for artifact in artifacts:
                    mlflow.log_artifact(artifact)
            except Exception as e:
                warnings.warn(f"Failed to log artifacts to MLflow: {str(e)}")
    
    def fit(self, data):
        """
        Fit GARCH model to the data with MLflow tracking
        
        Args:
            data (pd.DataFrame): Training data
            
        Returns:
            self: Fitted model
        """
        try:
            # Initialize MLflow tracking first
            initialize_mlflow()
            
            # Initialize MLflow tracking
            if not self.mlflow_run_id:
                self.mlflow_run_id = setup_mlflow(
                    model_type="GARCH",
                    ticker=self.ticker if self.ticker else "unknown",
                    params_path="params.yaml"
                )
            
            # Prepare returns data
            self.returns_data = self.prepare_data(data)
            
            if len(self.returns_data) < 50:
                raise ValueError("Insufficient data for GARCH modeling (need at least 50 observations)")
            
            # Create GARCH model
            self.model = arch_model(
                self.returns_data,
                mean=self.mean,
                vol=self.vol,
                p=self.p,
                q=self.q,
                dist=self.dist
            )
            
            # Fit the model
            if not mlflow.active_run():
                with mlflow.start_run(run_id=self.mlflow_run_id):
                    self.fitted_model = self.model.fit(disp='off', show_warning=False)
                    
                    # Log model parameters
                    self._log_model_parameters()
                    
                    # Log model summary metrics
                    summary = self.get_model_summary()
                    if summary:
                        metrics_to_log = {
                            'AIC': summary.get('AIC', 0),
                            'BIC': summary.get('BIC', 0),
                            'Log_Likelihood': summary.get('Log_Likelihood', 0)
                        }
                        self._log_model_metrics(metrics_to_log)
                    
                    # Log diagnostics
                    diagnostics = self.diagnose_model()
                    if diagnostics:
                        self._log_model_metrics({
                            'mean_residual': diagnostics.get('mean_residual', 0),
                            'std_residual': diagnostics.get('std_residual', 0),
                            'skewness': diagnostics.get('skewness', 0),
                            'kurtosis': diagnostics.get('kurtosis', 0)
                        })
            else:
                self.fitted_model = self.model.fit(disp='off', show_warning=False)
                
                # Log model parameters
                self._log_model_parameters()
                
                # Log model summary metrics
                summary = self.get_model_summary()
                if summary:
                    metrics_to_log = {
                        'AIC': summary.get('AIC', 0),
                        'BIC': summary.get('BIC', 0),
                        'Log_Likelihood': summary.get('Log_Likelihood', 0)
                    }
                    self._log_model_metrics(metrics_to_log)
                
                # Log diagnostics
                diagnostics = self.diagnose_model()
                if diagnostics:
                    self._log_model_metrics({
                        'mean_residual': diagnostics.get('mean_residual', 0),
                        'std_residual': diagnostics.get('std_residual', 0),
                        'skewness': diagnostics.get('skewness', 0),
                        'kurtosis': diagnostics.get('kurtosis', 0)
                    })
            
            return self
            
        except Exception as e:
            # Fallback to simpler model if fitting fails
            try:
                if not mlflow.active_run():
                    with mlflow.start_run(run_id=self.mlflow_run_id):
                        self.model = arch_model(
                            self.returns_data,
                            mean='zero',
                            vol='ARCH',
                            p=1,
                            dist='normal'
                        )
                        self.fitted_model = self.model.fit(disp='off', show_warning=False)
                        
                        # Update parameters in MLflow
                        self.p = 1
                        self.q = 0
                        self.mean = 'zero'
                        self.vol = 'ARCH'
                        self.dist = 'normal'
                        self._log_model_parameters()
                        
                        # Log fallback warning
                        mlflow.log_param('fallback_model', True)
                else:
                    self.model = arch_model(
                        self.returns_data,
                        mean='zero',
                        vol='ARCH',
                        p=1,
                        dist='normal'
                    )
                    self.fitted_model = self.model.fit(disp='off', show_warning=False)
                    
                    # Update parameters in MLflow
                    self.p = 1
                    self.q = 0
                    self.mean = 'zero'
                    self.vol = 'ARCH'
                    self.dist = 'normal'
                    self._log_model_parameters()
                    
                    # Log fallback warning
                    mlflow.log_param('fallback_model', True)
                        
                return self
            except:
                raise Exception(f"GARCH model fitting failed: {str(e)}")
    
    def predict(self, horizon=1):
        """
        Predict volatility using fitted GARCH model
        
        Args:
            horizon (int): Prediction horizon (number of periods ahead)
            
        Returns:
            np.ndarray: Predicted volatility values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate forecasts
            forecasts = self.fitted_model.forecast(horizon=horizon, reindex=False)
            
            # Extract conditional volatility
            predicted_variance = forecasts.variance.values[-1, :]
            predicted_volatility = np.sqrt(predicted_variance) / 100  # Scale back from percentage
            
            return predicted_volatility
            
        except Exception as e:
            # Fallback prediction using last observed volatility
            last_volatility = np.sqrt(self.fitted_model.conditional_volatility.iloc[-1]) / 100
            return np.array([last_volatility] * horizon)
    
    def fit_predict(self, train_data, test_data, horizon=1):
        """
        Fit model and make predictions with MLflow tracking
        
        Args:
            train_data (pd.DataFrame): Training data
            test_data (pd.DataFrame): Test data
            horizon (int): Prediction horizon
            
        Returns:
            np.ndarray: Predictions for test period
        """
        # Initialize MLflow tracking first
        initialize_mlflow()
        
        # Initialize MLflow tracking if not already done
        if not self.mlflow_run_id:
            self.mlflow_run_id = setup_mlflow(
                model_type="GARCH",
                ticker=self.ticker if self.ticker else "unknown",
                params_path="params.yaml"
            )
        
        # Start a new MLflow run for fit_predict
        with mlflow.start_run(run_id=self.mlflow_run_id, nested=True):
            # Fit model on training data
            self.fit(train_data)
            
            # Make rolling predictions for test period
            predictions = []
            current_data = train_data.copy()
            
            for i in range(len(test_data)):
                try:
                    # Predict next volatility
                    pred = self.predict(horizon=1)
                    predictions.append(pred[0])
                    
                    # Log prediction if MLflow is active
                    if mlflow.active_run():
                        mlflow.log_metric('test_prediction', pred[0], step=i)
                    
                    # Update data with new observation for next prediction
                    if i < len(test_data) - 1:
                        new_row = test_data.iloc[i:i+1]
                        current_data = pd.concat([current_data, new_row])
                        
                        # Refit model with updated data (every 10 steps to balance accuracy and speed)
                        if i % 10 == 0:
                            self.fit(current_data)
                            
                except Exception as e:
                    # Use last valid prediction or mean volatility as fallback
                    if predictions:
                        predictions.append(predictions[-1])
                    else:
                        predictions.append(train_data['volatility'].mean() if 'volatility' in train_data.columns else 0.02)
                    
                    # Log prediction error
                    if mlflow.active_run():
                        mlflow.log_metric('prediction_error', 1, step=i)
            
            # Log predictions as artifacts (inside MLflow run context)
            if len(predictions) > 0:
                print(f"🎯 GARCH: About to log artifacts - MLflow run ID: {self.mlflow_run_id}, Active run: {mlflow.active_run() is not None}")
                
                # Create artifacts directory if it doesn't exist
                artifacts_dir = "Artifacts"
                os.makedirs(artifacts_dir, exist_ok=True)
                
                # Log prediction statistics as artifacts
                pred_stats = pd.DataFrame({
                    'step': range(len(predictions)),
                    'prediction': predictions,
                    'model_type': ['GARCH'] * len(predictions),
                    'ticker': [self.ticker] * len(predictions),
                    'timestamp': [datetime.now().isoformat()] * len(predictions),
                    'p_param': [self.p] * len(predictions),
                    'q_param': [self.q] * len(predictions),
                    'mean_model': [self.mean] * len(predictions),
                    'vol_model': [self.vol] * len(predictions),
                    'distribution': [self.dist] * len(predictions)
                })
                
                # Save to Artifacts folder
                artifact_path = os.path.join(artifacts_dir, f"garch_predictions_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                pred_stats.to_csv(artifact_path, index=False)
                print(f"💾 GARCH: Saved predictions to {artifact_path}")
                
                # Try to log to MLflow if run is active
                if mlflow.active_run():
                    try:
                        mlflow.log_artifact(artifact_path)
                        print(f"✅ GARCH: Logged artifact to MLflow")
                    except Exception as e:
                        print(f"❌ GARCH: Failed to log artifact to MLflow: {e}")
                else:
                    print(f"⚠️ GARCH: No active MLflow run - skipping MLflow artifact logging")
                
                # Log model summary
                summary_data = {
                    'model_type': 'GARCH',
                    'ticker': self.ticker,
                    'p_param': self.p,
                    'q_param': self.q,
                    'mean_model': self.mean,
                    'vol_model': self.vol,
                    'distribution': self.dist,
                    'num_predictions': len(predictions),
                    'mean_prediction': np.mean(predictions),
                    'std_prediction': np.std(predictions),
                    'min_prediction': np.min(predictions),
                    'max_prediction': np.max(predictions),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add model fit statistics if available
                if self.fitted_model is not None:
                    try:
                        summary_data.update({
                            'aic': self.fitted_model.aic,
                            'bic': self.fitted_model.bic,
                            'log_likelihood': self.fitted_model.loglikelihood
                        })
                    except:
                        pass
                
                summary_df = pd.DataFrame([summary_data])
                summary_path = os.path.join(artifacts_dir, f"garch_summary_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                summary_df.to_csv(summary_path, index=False)
                print(f"💾 GARCH: Saved summary to {summary_path}")
                
                # Try to log summary to MLflow if run is active
                if mlflow.active_run():
                    try:
                        mlflow.log_artifact(summary_path)
                        print(f"✅ GARCH: Logged summary to MLflow")
                    except Exception as e:
                        print(f"❌ GARCH: Failed to log summary to MLflow: {e}")
                else:
                    print(f"⚠️ GARCH: No active MLflow run - skipping MLflow summary logging")
        
        return np.array(predictions)
    
    def forecast_future(self, horizon):
        """
        Forecast volatility for future periods with MLflow tracking
        
        Args:
            horizon (int): Number of periods to forecast
            
        Returns:
            np.ndarray: Future volatility forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            forecasts = self.fitted_model.forecast(horizon=horizon, reindex=False)
            predicted_variance = forecasts.variance.values[-1, :]
            predicted_volatility = np.sqrt(predicted_variance) / 100
            
            # Log forecast summary
            if self.mlflow_run_id and mlflow.active_run():
                mlflow.log_metrics({
                    'forecast_mean': np.mean(predicted_volatility),
                    'forecast_std': np.std(predicted_volatility),
                    'forecast_max': np.max(predicted_volatility),
                    'forecast_min': np.min(predicted_volatility)
                })
            
            return predicted_volatility
            
        except Exception as e:
            # Fallback to last observed volatility
            last_volatility = np.sqrt(self.fitted_model.conditional_volatility.iloc[-1]) / 100
            return np.array([last_volatility] * horizon)
    
    def get_model_summary(self):
        """
        Get model summary statistics
        
        Returns:
            dict: Model summary information
        """
        if self.fitted_model is None:
            return None
        
        try:
            summary_info = {
                'AIC': self.fitted_model.aic,
                'BIC': self.fitted_model.bic,
                'Log_Likelihood': self.fitted_model.loglikelihood,
                'Parameters': dict(self.fitted_model.params),
                'Model_Specification': f"GARCH({self.p},{self.q})"
            }
            return summary_info
        except:
            return None
    
    def get_conditional_volatility(self):
        """
        Get conditional volatility from fitted model
        
        Returns:
            pd.Series: Conditional volatility series
        """
        if self.fitted_model is None:
            return None
        
        return self.fitted_model.conditional_volatility / 100
    
    def diagnose_model(self):
        """
        Perform model diagnostics
        
        Returns:
            dict: Diagnostic test results
        """
        if self.fitted_model is None:
            return None
        
        try:
            # Get standardized residuals
            std_resid = self.fitted_model.std_resid
            
            # Basic diagnostics
            diagnostics = {
                'mean_residual': np.mean(std_resid),
                'std_residual': np.std(std_resid),
                'skewness': pd.Series(std_resid).skew(),
                'kurtosis': pd.Series(std_resid).kurtosis(),
                'jarque_bera_pvalue': None  # Would need scipy.stats for proper test
            }
            
            return diagnostics
            
        except Exception as e:
            return None
        




