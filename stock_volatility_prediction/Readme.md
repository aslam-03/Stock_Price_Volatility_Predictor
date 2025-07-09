# Stock Volatility Prediction System 📈

An advanced machine learning system for predicting stock volatility using multiple models (GARCH, LSTM, and Rolling Window) with comprehensive technical analysis and MLOps integration. Built with Streamlit for interactive visualization and DagsHub for experiment tracking.

## 🚀 Key Features

### 📊 Multi-Model Prediction System
- **GARCH Model**: Econometric approach for volatility clustering and heteroskedasticity modeling
- **LSTM Model**: Deep learning approach using RandomForest with sequence features for time series patterns
- **Rolling Window Model**: Statistical baseline model for benchmark comparisons

### 🔧 Technical Analysis Suite
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillator, Williams %R, CCI, ATR, OBV
- **Price Pattern Recognition**: Doji patterns, Higher Highs/Lower Lows detection
- **Volume Analysis**: Volume ratios, On-Balance Volume (OBV)
- **Support/Resistance Levels**: Dynamic support and resistance identification
- **Momentum Indicators**: Rate of Change (ROC), Momentum oscillators

### 💱 Currency Conversion & Localization
- **Real-time USD to INR conversion** using live exchange rates
- **Localized price display** with Indian Rupee formatting (₹)
- **Multi-currency support** with automatic conversion

### 📈 Advanced Visualization
- **Interactive Plotly charts** with zoom, pan, and hover capabilities
- **Multi-panel dashboards** showing price, volume, and technical indicators
- **Volatility regime visualization** with color-coded periods
- **Model comparison charts** with performance metrics
- **Candlestick charts** with technical overlays

### 🎯 Model Performance Analytics
- **Comprehensive evaluation metrics**: RMSE, MAE, R², Hit Ratio, Directional Accuracy
- **Model caching system** for faster re-training
- **Backtesting capabilities** with walk-forward validation
- **Risk metrics calculation**: VaR, CVaR, Sharpe Ratio, Maximum Drawdown

### 🔄 MLOps Integration
- **MLflow experiment tracking** with automatic logging
- **DVC data versioning** for reproducible pipelines
- **Model artifact management** with versioning
- **Automated parameter logging** and metric tracking

### 🐳 Docker Support
- **Multi-stage Docker builds** for optimized images
- **Secure credential management** with runtime environment variables
- **Production-ready deployment** with health checks

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://dagshub.com/aslam-03/stock_volatility_prediction.git
cd stock_volatility_prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. DagsHub Configuration

#### Create `.env` file with your DagsHub credentials:
```bash
# MLflow Tracking Configuration
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/stock_volatility_prediction.mlflow
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token

# Optional: Currency and Display Settings
DEFAULT_EXCHANGE_RATE=83.0
BASE_CURRENCY=USD
TARGET_CURRENCY=INR
CURRENCY_SYMBOL=₹
```

#### Get Your DagsHub Token:
1. Go to [DagsHub](https://dagshub.com)
2. Sign up/Login to your account
3. Go to Settings → Access Tokens
4. Generate a new token
5. Copy the token to your `.env` file

### 4. Initialize DVC (Data Version Control)
```bash
# Initialize DVC
dvc init

# Add DagsHub remote
dvc remote add origin https://dagshub.com/your-username/stock_volatility_prediction.dvc

# Configure DVC with DagsHub
dvc remote modify origin --local auth basic
dvc remote modify origin --local user your-dagshub-username
dvc remote modify origin --local password your-dagshub-token
```

### 5. Run DVC Pipeline (Optional)
```bash
# Run the complete data pipeline
dvc repro

# Pull pre-processed data
dvc pull
```

## 🚀 Running the Application

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment (Secure)
```bash
# Build and run with secure environment variables
docker-secure-setup.bat

# Or manual Docker commands
docker build -t stock-volatility-predictor .
docker run -d -p 8501:8501 --env-file .env stock-volatility-predictor
```

### Access the Application
- **Local**: http://localhost:8501
- **Docker**: http://localhost:8501

## 🧠 Models Description

### 1. GARCH Model (Generalized Autoregressive Conditional Heteroskedasticity)
```python
# Configuration
p=1, q=1, mean='constant', vol='GARCH', dist='normal'
```
- **Purpose**: Models volatility clustering and time-varying volatility
- **Strengths**: Captures volatility clustering, well-established in finance
- **Use Cases**: Risk management, option pricing, portfolio optimization
- **Features**: Supports multiple GARCH variants, automatic parameter estimation

### 2. LSTM Model (Long Short-Term Memory)
```python
# Configuration
window_size=30, n_estimators=100, max_depth=10
```
- **Purpose**: Captures long-term dependencies and non-linear patterns
- **Implementation**: RandomForest with sequence features for robust performance
- **Strengths**: Handles complex patterns, incorporates technical indicators
- **Use Cases**: Multi-step ahead forecasting, regime detection

### 3. Rolling Window Model
```python
# Configuration
window_size=30, train_test_split=0.8
```
- **Purpose**: Simple baseline model using rolling statistics
- **Strengths**: Interpretable, fast computation, robust baseline
- **Use Cases**: Benchmark comparisons, real-time applications
- **Features**: Adaptive window sizing, online learning capability

## 📊 Technical Indicators Implemented

### Momentum Indicators
- **RSI (Relative Strength Index)**: 14-period momentum oscillator
- **Stochastic Oscillator**: %K and %D lines for overbought/oversold signals
- **Williams %R**: Momentum indicator similar to stochastic
- **ROC (Rate of Change)**: Price momentum over 10 periods

### Trend Indicators
- **Moving Averages**: SMA (10, 20, 50), EMA (12, 26)
- **MACD**: Moving Average Convergence Divergence with signal line
- **Bollinger Bands**: 20-period bands with 2 standard deviations

### Volume Indicators
- **OBV (On-Balance Volume)**: Cumulative volume indicator
- **Volume Ratio**: Current volume vs. 20-period average
- **Volume SMA**: 20-period volume moving average

### Volatility Indicators
- **ATR (Average True Range)**: 14-period volatility measure
- **Bollinger Band Width**: Band expansion/contraction indicator
- **Rolling Volatility**: 20-period standard deviation

## 📁 Project Structure

```
stock_volatility_prediction/
├── 📄 app.py                          # Main Streamlit application
├── 📄 requirements.txt                # Python dependencies
├── 📄 params.yaml                     # Model parameters configuration
├── 📄 dvc.yaml                        # DVC pipeline configuration
├── 📄 dvc.lock                        # DVC lock file
├── 📄 mlflow_init.py                  # MLflow initialization
├── 📄 Dockerfile                      # Docker configuration
├── 📄 docker-secure-setup.bat         # Secure Docker deployment script
├── 📄 security-check.bat              # Security verification script
├── 📄 .env                            # Environment variables (create this)
├── 📄 .dockerignore                   # Docker ignore patterns
├── 📄 .gitignore                      # Git ignore patterns
├── 📄 .dvcignore                      # DVC ignore patterns
│
├── 📁 models/                          # Machine Learning Models
│   ├── 📄 garch_model.py              # GARCH volatility model
│   ├── 📄 lstm_model.py               # LSTM/RandomForest model
│   ├── 📄 rolling_window_model.py     # Rolling window baseline model
│   ├── 📄 model_evaluator.py          # Model evaluation utilities
│   └── 📁 saved/                      # Saved model artifacts
│       ├── 📄 garch_model_AAPL_*.pkl
│       ├── 📄 lstm_model_AAPL_*.pkl
│       └── 📄 rolling_window_model_AAPL_*.pkl
│
├── 📁 Research/                        # Data Processing & Feature Engineering
│   ├── 📄 data_processor.py           # Stock data fetching and preprocessing
│   ├── 📄 feature_engineering.py      # Technical indicators and features
│   ├── 📄 test_data_processor.py      # Unit tests for data processing
│   ├── 📁 processed/                  # Processed feature data
│   └── 📁 processed2/                 # Raw stock data
│       └── 📄 AAPL_raw.csv
│
├── 📁 src/                            # Source Code Modules
│   ├── 📁 config/                     # Configuration Management
│   │   └── 📄 settings.py             # Application settings and parameters
│   ├── 📁 ui/                         # User Interface Components
│   │   └── 📄 visualization.py        # Plotly charts and visualizations
│   └── 📁 utils/                      # Utility Functions
│       └── 📄 utils.py                # Helper functions and utilities
│
├── 📁 mlruns/                         # MLflow Experiment Tracking
│   └── 📁 0/                          # Default experiment
│       └── 📁 experiment_runs/        # Individual run artifacts
│
├── 📁 Artifacts/                      # Model Artifacts and Outputs
│   └── 📄 prediction_results.csv      # Model predictions export
│
└── 📁 .dvc/                           # DVC Configuration
    ├── 📄 config                      # DVC configuration
    └── 📁 cache/                      # DVC cache directory
```

## 🎛️ Configuration Parameters

### Model Parameters (`params.yaml`)
```yaml
# GARCH Model
garch:
  p: 1                    # Volatility lag terms
  q: 1                    # Squared residuals lag terms
  distribution: 'normal'  # Error distribution

# LSTM Model
lstm:
  input_window: 60        # Sequence length
  epochs: 100            # Training epochs
  batch_size: 32         # Batch size
  hidden_units: 50       # Hidden layer units

# Rolling Window Model
rolling_window:
  train_test_split: 0.8  # Training data proportion
  refit_frequency: 5     # Retraining frequency
```

### Technical Indicators Configuration
```yaml
features:
  window_size: 20        # Default window for indicators
  lag_periods: [1,2,3,5] # Lag features
  volatility_window: 10  # Volatility calculation window
```

## 📈 Usage Examples

### 1. Basic Stock Analysis
```python
# In the Streamlit interface:
# 1. Enter stock symbol (e.g., AAPL, GOOGL, TSLA)
# 2. Select date range
# 3. Choose models (GARCH, LSTM, Rolling Window)
# 4. Click "Run Analysis"
```

### 2. Model Comparison
```python
# Select multiple models for comparison
selected_models = ["GARCH", "LSTM", "Rolling Window"]
# View side-by-side performance metrics
# Compare prediction accuracy and visualizations
```

### 3. Export Results
```python
# Download prediction results as CSV
# Export model performance metrics
# Save charts as PNG/HTML
```

## 🔧 Advanced Features

### Model Caching
- **Session-based caching**: Trained models are cached for faster re-runs
- **Parameter tracking**: Cache invalidation based on parameter changes
- **Memory optimization**: Efficient model storage and retrieval

### Backtesting Framework
- **Walk-forward validation**: Time-series aware cross-validation
- **Out-of-sample testing**: Separate test sets for unbiased evaluation
- **Performance attribution**: Detailed breakdown of model performance

### Risk Management
- **Value at Risk (VaR)**: 95% confidence interval risk metrics
- **Conditional VaR**: Expected shortfall calculations
- **Maximum Drawdown**: Worst-case loss scenarios
- **Sharpe Ratio**: Risk-adjusted return metrics

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Production
```bash
docker-secure-setup.bat
```

### Cloud Deployment
- **AWS ECS**: Container-based deployment
- **Azure Container Instances**: Serverless containers
- **Google Cloud Run**: Managed containerized applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 📞 Contact

[LinkedIn: Mohamed Aslam I](https://www.linkedin.com/in/mohamed-aslam-i)



---

**Built with ❤️ using Python, Streamlit, MLflow, and DagsHub**