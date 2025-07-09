"""
MLflow initialization and configuration for DagsHub tracking.
Fixed version with robust environment variable handling.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import yaml

def initialize_mlflow():
    """
    Initialize MLflow with DagsHub authentication - robust version
    
    Returns:
        MlflowClient: Initialized MLflow client
    """
    try:
        print("🔧 Starting MLflow initialization...")
        
        # Method 1: Try loading from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=".env", override=True)
            print("✅ Loaded .env file")
        except Exception as e:
            print(f"⚠️ Could not load .env file: {e}")
        
        # Method 2: Get environment variables (with fallback values)
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI') or 'https://dagshub.com/aslam-03/stock_volatility_prediction.mlflow'
        mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME') or 'aslam-03'
        mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD') or 'f026026aaafaa0a5b4e9c6ca5719a73cad1cd323'
        
        # Set MLflow configuration
        mlflow.set_tracking_uri(mlflow_uri)
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
        
        print(f"✅ MLflow URI: {mlflow_uri}")
        print(f"✅ Username: {mlflow_username}")
        print(f"✅ Password: {'*' * len(mlflow_password)}")
        
        # Initialize client
        client = MlflowClient()
        
        # Test connection
        try:
            experiments = client.list_experiments()
            print(f"✅ Connection successful - {len(experiments)} experiments found")
        except Exception as e:
            print(f"⚠️ Connection test failed: {e}")
            # Continue anyway - might still work for logging
        
        # Set up experiment
        experiment_name = "Stock_Volatility_Prediction"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
            else:
                print(f"✅ Using experiment: {experiment_name} (ID: {experiment.experiment_id})")
            
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"⚠️ Experiment setup failed: {e}")
            print("📝 Will use default experiment")
        
        print("🎉 MLflow initialization complete!")
        return client
        
    except Exception as e:
        print(f"❌ MLflow initialization failed: {e}")
        # Return a basic client for local use
        mlflow.set_tracking_uri("file:./mlruns")
        return MlflowClient()
        
    except Exception as e:
        print(f"❌ MLflow initialization failed: {str(e)}")
        print("Falling back to local MLflow tracking")
        
        # Fallback to local tracking
        mlflow.set_tracking_uri("file:./mlruns")
        client = MlflowClient()
        
        try:
            mlflow.set_experiment("Default")
        except:
            pass
            
        return client

def get_run_name(model_type, ticker):
    """
    Generate a standardized run name for MLflow tracking.
    
    Args:
        model_type (str): Type of model (e.g., 'LSTM', 'GARCH', 'RollingWindow')
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        str: Formatted run name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{ticker}_{timestamp}"

def log_params_from_yaml(params_path):
    """
    Log parameters from a YAML file to MLflow.
    
    Args:
        params_path (str): Path to the YAML parameters file
    """
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
        mlflow.log_params(params)

def setup_mlflow(model_type, ticker, params_path=None):
    """
    Complete MLflow setup including initialization and run creation.
    
    Args:
        model_type (str): Type of model being trained
        ticker (str): Stock ticker symbol
        params_path (str, optional): Path to parameters YAML file
    
    Returns:
        str: Run ID of the created MLflow run
    """
    try:
        # Initialize MLflow (this handles DagsHub authentication)
        client = initialize_mlflow()
        
        # Start a new run
        run_name = get_run_name(model_type, ticker)
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters if provided
            if params_path and os.path.exists(params_path):
                try:
                    log_params_from_yaml(params_path)
                except Exception as e:
                    print(f"Warning: Could not log params from {params_path}: {str(e)}")
            
            # Add some useful tags
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("ticker", ticker)
            mlflow.set_tag("framework", "custom_volatility")
            
            print(f"✅ Started MLflow run: {run_name}")
            return run.info.run_id
            
    except Exception as e:
        print(f"❌ MLflow setup failed: {str(e)}")
        # Return None if setup fails
        return None