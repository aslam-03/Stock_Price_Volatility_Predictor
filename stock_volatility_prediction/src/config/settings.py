"""
Application settings and configuration constants.
Uses environment variables for sensitive credentials.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== DAGsHub MLflow Configuration ====================
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://dagshub.com/aslam-03/stock_volatility_prediction.mlflow"
)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Stock_Volatility_Prediction")

# Authentication (must be set in .env)
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ==================== API Settings ====================
DEFAULT_EXCHANGE_RATE = float(os.getenv("DEFAULT_EXCHANGE_RATE", 83.0))  # Fallback USD to INR rate
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 1))  # Cache time-to-live for data

# ==================== Model Parameters ====================
DEFAULT_WINDOW_SIZE = int(os.getenv("DEFAULT_WINDOW_SIZE", 20))
DEFAULT_PREDICTION_HORIZON = int(os.getenv("DEFAULT_PREDICTION_HORIZON", 5))
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", 0.8))

# GARCH Model Settings
GARCH_P = int(os.getenv("GARCH_P", 1))  # Lag terms for volatility
GARCH_Q = int(os.getenv("GARCH_Q", 1))  # Lag terms for squared residuals

# ML Model Settings
LSTM_WINDOW_SIZE = int(os.getenv("LSTM_WINDOW_SIZE", 30))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 10))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# ==================== Technical Indicators Settings ====================
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))
BB_PERIOD = int(os.getenv("BB_PERIOD", 20))
BB_STD = int(os.getenv("BB_STD", 2))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))

# ==================== UI Settings ====================
CHART_HEIGHT = int(os.getenv("CHART_HEIGHT", 600))
COLOR_SCHEME = {
    'primary': os.getenv("COLOR_PRIMARY", '#1f77b4'),
    'secondary': os.getenv("COLOR_SECONDARY", '#ff7f0e'),
    'success': os.getenv("COLOR_SUCCESS", '#2ca02c'),
    'danger': os.getenv("COLOR_DANGER", '#d62728'),
    'warning': os.getenv("COLOR_WARNING", '#ff9800'),
    'info': os.getenv("COLOR_INFO", '#17a2b8')
}

# ==================== Currency Settings ====================
BASE_CURRENCY = os.getenv("BASE_CURRENCY", 'USD')
TARGET_CURRENCY = os.getenv("TARGET_CURRENCY", 'INR')
CURRENCY_SYMBOL = os.getenv("CURRENCY_SYMBOL", '₹')

# ==================== Validation ====================
def validate_settings():
    """Validate critical settings at startup"""
    if not MLFLOW_TRACKING_USERNAME or not MLFLOW_TRACKING_PASSWORD:
        raise ValueError("DAGsHub credentials not configured. Please set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD in .env")

    if not MLFLOW_TRACKING_URI.startswith("https://"):
        raise ValueError("Invalid MLflow tracking URI")

validate_settings()