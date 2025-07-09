import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import warnings
import requests
warnings.filterwarnings('ignore')

def calculate_volatility(data, window=20, method='returns'):
    """
    Calculate rolling volatility from stock price data
    
    Args:
        data (pd.DataFrame): Stock price data
        window (int): Rolling window size
        method (str): Method to calculate volatility ('returns', 'log_returns', 'parkinson')
        
    Returns:
        pd.Series: Volatility time series
    """
    try:
        if method == 'returns':
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        elif method == 'log_returns':
            log_returns = np.log(data['Close'] / data['Close'].shift(1))
            volatility = log_returns.rolling(window=window).std() * np.sqrt(252)
        
        elif method == 'parkinson':
            # Parkinson volatility estimator (using high-low prices)
            hl_ratio = np.log(data['High'] / data['Low'])
            volatility = hl_ratio.rolling(window=window).apply(
                lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2)))
            ) * np.sqrt(252)
        
        else:
            raise ValueError(f"Unknown volatility method: {method}")
        
        return volatility.fillna(method='bfill')
        
    except Exception as e:
        st.error(f"Error calculating volatility: {str(e)}")
        return pd.Series(index=data.index, data=0.02)  # Default 2% volatility

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    """
    Get basic stock information
    
    Args:
        symbol (str): Stock ticker symbol
        
    Returns:
        dict: Stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        stock_info = {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': format_market_cap(info.get('marketCap', 0)),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'N/A')
        }
        
        return stock_info
        
    except Exception as e:
        return {
            'name': symbol,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 'N/A',
            'currency': 'USD',
            'exchange': 'N/A'
        }

def format_market_cap(market_cap):
    """
    Format market cap value for display
    
    Args:
        market_cap (int): Market cap value
        
    Returns:
        str: Formatted market cap
    """
    if market_cap == 0:
        return 'N/A'
    
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"${market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.2f}M"
    else:
        return f"${market_cap:,.0f}"

def calculate_technical_signals(data):
    """
    Calculate basic technical trading signals
    
    Args:
        data (pd.DataFrame): Stock data with technical indicators
        
    Returns:
        pd.DataFrame: Data with signal columns added
    """
    try:
        signals = data.copy()
        
        # RSI signals
        if 'RSI' in signals.columns:
            signals['RSI_Signal'] = 0
            signals.loc[signals['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold - Buy
            signals.loc[signals['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought - Sell
        
        # MACD signals
        if 'MACD' in signals.columns and 'MACD_Signal' in signals.columns:
            signals['MACD_Signal_Action'] = 0
            signals.loc[signals['MACD'] > signals['MACD_Signal'], 'MACD_Signal_Action'] = 1
            signals.loc[signals['MACD'] < signals['MACD_Signal'], 'MACD_Signal_Action'] = -1
        
        # Bollinger Bands signals
        if all(col in signals.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
            signals['BB_Signal'] = 0
            signals.loc[signals['Close'] <= signals['BB_Lower'], 'BB_Signal'] = 1  # Buy at lower band
            signals.loc[signals['Close'] >= signals['BB_Upper'], 'BB_Signal'] = -1  # Sell at upper band
        
        # Volume signals
        if 'Volume_Ratio' in signals.columns:
            signals['Volume_Signal'] = 0
            signals.loc[signals['Volume_Ratio'] > 1.5, 'Volume_Signal'] = 1  # High volume
            signals.loc[signals['Volume_Ratio'] < 0.5, 'Volume_Signal'] = -1  # Low volume
        
        return signals
        
    except Exception as e:
        st.error(f"Error calculating technical signals: {str(e)}")
        return data

def detect_volatility_regime(volatility_series, threshold_factor=1.5):
    """
    Detect volatility regimes (low, normal, high)
    
    Args:
        volatility_series (pd.Series): Volatility time series
        threshold_factor (float): Factor to determine regime thresholds
        
    Returns:
        pd.Series: Regime classifications
    """
    try:
        mean_vol = volatility_series.mean()
        std_vol = volatility_series.std()
        
        low_threshold = mean_vol - threshold_factor * std_vol
        high_threshold = mean_vol + threshold_factor * std_vol
        
        regimes = pd.Series(index=volatility_series.index, data='Normal')
        regimes[volatility_series < low_threshold] = 'Low'
        regimes[volatility_series > high_threshold] = 'High'
        
        return regimes
        
    except Exception as e:
        return pd.Series(index=volatility_series.index, data='Normal')

def calculate_risk_metrics(returns_series):
    """
    Calculate various risk metrics from returns
    
    Args:
        returns_series (pd.Series): Returns time series
        
    Returns:
        dict: Risk metrics
    """
    try:
        returns = returns_series.dropna()
        
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = returns.mean()
        metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        metrics['sharpe_ratio'] = metrics['mean_return'] / (returns.std() + 1e-8) * np.sqrt(252)
        
        # Value at Risk (95% confidence)
        metrics['var_95'] = returns.quantile(0.05)
        
        # Expected Shortfall (Conditional VaR)
        var_95 = metrics['var_95']
        tail_returns = returns[returns <= var_95]
        metrics['cvar_95'] = tail_returns.mean() if len(tail_returns) > 0 else var_95
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Calmar ratio
        metrics['calmar_ratio'] = metrics['mean_return'] * 252 / abs(metrics['max_drawdown'])
        
        return metrics
        
    except Exception as e:
        return {}

def validate_input_data(data, required_columns):
    """
    Validate input data for required columns and data quality
    
    Args:
        data (pd.DataFrame): Input data
        required_columns (list): List of required column names
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if data.empty:
            return False, "Data is empty"
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for sufficient data
        if len(data) < 30:
            return False, "Insufficient data (need at least 30 observations)"
        
        # Check for valid prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                if (data[col] <= 0).any():
                    return False, f"Invalid {col} prices (must be positive)"
        
        # Check for valid volume
        if 'Volume' in data.columns:
            if (data['Volume'] < 0).any():
                return False, "Invalid volume data (must be non-negative)"
        
        # Check for excessive NaN values
        nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if nan_ratio > 0.1:  # More than 10% NaN values
            return False, f"Too many missing values ({nan_ratio:.1%})"
        
        return True, "Data validation passed"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def export_predictions_to_csv(predictions_dict, test_data, filename="volatility_predictions.csv"):
    """
    Export predictions to CSV format
    
    Args:
        predictions_dict (dict): Dictionary of model predictions
        test_data (pd.DataFrame): Test data with dates
        filename (str): Output filename
        
    Returns:
        str: CSV content as string
    """
    try:
        # Create DataFrame with predictions
        export_df = pd.DataFrame(index=test_data.index)
        export_df['Actual_Volatility'] = test_data['volatility']
        
        for model_name, predictions in predictions_dict.items():
            if len(predictions) > 0:
                pred_series = pd.Series(predictions, index=test_data.index[:len(predictions)])
                export_df[f'{model_name}_Prediction'] = pred_series
        
        # Calculate prediction errors
        for model_name in predictions_dict.keys():
            if f'{model_name}_Prediction' in export_df.columns:
                export_df[f'{model_name}_Error'] = (
                    export_df[f'{model_name}_Prediction'] - export_df['Actual_Volatility']
                )
                export_df[f'{model_name}_Abs_Error'] = export_df[f'{model_name}_Error'].abs()
        
        # Add metadata
        export_df.insert(0, 'Date', export_df.index.strftime('%Y-%m-%d'))
        
        return export_df.to_csv(index=False)
        
    except Exception as e:
        st.error(f"Error exporting predictions: {str(e)}")
        return ""

def create_model_summary_report(evaluation_results, model_configs=None):
    """
    Create a comprehensive model summary report
    
    Args:
        evaluation_results (dict): Model evaluation results
        model_configs (dict, optional): Model configuration details
        
    Returns:
        str: Formatted summary report
    """
    try:
        report = []
        report.append("=" * 60)
        report.append("VOLATILITY PREDICTION MODEL SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall comparison
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        
        if evaluation_results:
            best_rmse = min(evaluation_results.values(), key=lambda x: x.get('rmse', float('inf')))
            best_model = [name for name, metrics in evaluation_results.items() 
                         if metrics.get('rmse') == best_rmse['rmse']][0]
            
            report.append(f"Best performing model: {best_model}")
            report.append(f"Best RMSE: {best_rmse['rmse']:.6f}")
            report.append("")
            
            # Detailed metrics for each model
            for model_name, metrics in evaluation_results.items():
                report.append(f"{model_name.upper()} MODEL")
                report.append("-" * 20)
                report.append(f"RMSE: {metrics.get('rmse', 'N/A'):.6f}")
                report.append(f"MAE: {metrics.get('mae', 'N/A'):.6f}")
                report.append(f"R²: {metrics.get('r2', 'N/A'):.4f}")
                report.append(f"Hit Ratio: {metrics.get('hit_ratio', 'N/A'):.2f}%")
                report.append(f"Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2f}%")
                report.append("")
        
        # Model configurations (if provided)
        if model_configs:
            report.append("MODEL CONFIGURATIONS")
            report.append("-" * 30)
            for model_name, config in model_configs.items():
                report.append(f"{model_name}: {config}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

def get_market_hours():
    """
    Get current market status and hours
    
    Returns:
        dict: Market status information
    """
    try:
        now = datetime.now()
        
        # Simple US market hours check (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if it's a weekday
        is_weekday = now.weekday() < 5
        
        # Check if market is open (simplified - doesn't account for holidays)
        is_market_open = is_weekday and market_open <= now <= market_close
        
        return {
            'is_open': is_market_open,
            'next_open': market_open + timedelta(days=1) if not is_market_open else None,
            'next_close': market_close if is_market_open else None,
            'timezone': 'ET',
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {
            'is_open': False,
            'error': str(e)
        }

@st.cache_data(ttl=3600)
def get_usd_to_inr_rate():
    """
    Get USD to INR exchange rate
    
    Returns:
        float: USD to INR conversion rate
    """
    try:
        # Using Yahoo Finance to get USD/INR rate
        ticker = yf.Ticker("USDINR=X")
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        else:
            # Fallback rate if API fails
            return 83.0  # Approximate rate
    except:
        # Fallback rate
        return 83.0

def convert_usd_to_inr(usd_amount, exchange_rate=None):
    """
    Convert USD amount to INR
    
    Args:
        usd_amount (float): Amount in USD
        exchange_rate (float, optional): USD to INR rate
        
    Returns:
        float: Amount in INR
    """
    if exchange_rate is None:
        exchange_rate = get_usd_to_inr_rate()
    return usd_amount * exchange_rate

def format_inr_price(inr_amount):
    """
    Format INR amount for display with proper formatting
    
    Args:
        inr_amount (float): Amount in INR
        
    Returns:
        str: Formatted INR amount
    """
    if inr_amount >= 1e7:  # 1 crore
        return f"₹{inr_amount/1e7:.2f} Cr"
    elif inr_amount >= 1e5:  # 1 lakh
        return f"₹{inr_amount/1e5:.2f} L"
    elif inr_amount >= 1000:
        return f"₹{inr_amount/1000:.2f}K"
    else:
        return f"₹{inr_amount:.2f}"
