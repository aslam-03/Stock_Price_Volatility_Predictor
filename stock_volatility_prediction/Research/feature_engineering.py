import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for creating technical indicators and preparing data for modeling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.features_created = False
    
    def create_features(self, data):
        """
        Create comprehensive technical indicators and features
        
        Args:
            data (pd.DataFrame): Stock price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with technical indicators added
        """
        df = data.copy()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        macd_line = df['EMA_12'] - df['EMA_26']
        signal_line = macd_line.ewm(span=9).mean()
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = macd_line - signal_line
        
        # Bollinger Bands
        bb_sma = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = bb_sma + (bb_std * 2)
        df['BB_Mid'] = bb_sma
        df['BB_Lower'] = bb_sma - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Average True Range (ATR)
        df['ATR'] = self._calculate_atr(df, 14)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df, 14)
        
        # Commodity Channel Index (CCI)
        df['CCI'] = self._calculate_cci(df, 20)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # On Balance Volume
        df['OBV'] = self._calculate_obv(df)
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['Doji'] = (abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1).astype(int)
        
        # Volatility measures
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Momentum indicators
        df['ROC'] = (df['Close'] / df['Close'].shift(10) - 1) * 100  # Rate of Change
        df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Support and Resistance levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Price_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Returns_Mean_{window}'] = df['Returns'].rolling(window=window).mean()
            df[f'Returns_Std_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'Price_Range_{window}'] = df['High'].rolling(window=window).max() - df['Low'].rolling(window=window).min()
        
        # Calculate realized volatility (target variable)
        df['volatility'] = df['Returns'].rolling(window=20).std()
        
        # Fill missing values
        df = df.ffill().bfill()
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        self.features_created = True
        return df
    
    def select_features(self, data):
        """
        Select the most relevant features for modeling
        
        Args:
            data (pd.DataFrame): Data with all features
            
        Returns:
            list: List of selected feature column names
        """
        # Core technical indicators
        core_features = [
            'RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'BB_Position',
            'ATR', 'Stoch_K', 'Williams_R', 'CCI', 'Volume_Ratio',
            'High_Low_Pct', 'ROC', 'Momentum', 'Price_Position'
        ]
        
        # Rolling statistics features
        rolling_features = [
            'Returns_Mean_5', 'Returns_Std_5', 'Returns_Mean_10', 'Returns_Std_10',
            'Volume_Mean_5', 'Price_Range_5', 'Price_Range_10'
        ]
        
        # Lag features
        lag_features = [
            'Returns_Lag_1', 'Returns_Lag_2', 'Volume_Lag_1'
        ]
        
        # Combine all features
        selected_features = core_features + rolling_features + lag_features
        
        # Filter features that actually exist in the data
        available_features = [f for f in selected_features if f in data.columns]
        
        return available_features
    
    def prepare_model_data(self, data, volatility_series, window_size):
        """
        Prepare data for machine learning models
        
        Args:
            data (pd.DataFrame): Feature data
            volatility_series (pd.Series): Target volatility values
            window_size (int): Size of the rolling window
            
        Returns:
            pd.DataFrame: Prepared model data
        """
        # Select features
        feature_columns = self.select_features(data)
        
        # Create model dataset
        model_data = data[feature_columns].copy()
        model_data['volatility'] = volatility_series
        
        # Remove NaN values
        model_data = model_data.dropna()
        
        # Ensure minimum data size
        if len(model_data) < window_size:
            raise ValueError(f"Insufficient data: {len(model_data)} rows, need at least {window_size}")
        
        return model_data
    
    def create_sequences(self, data, window_size, target_col='volatility'):
        """
        Create sequences for LSTM model
        
        Args:
            data (pd.DataFrame): Input data
            window_size (int): Size of input sequence
            target_col (str): Name of target column
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        feature_columns = [col for col in data.columns if col != target_col]
        
        X, y = [], []
        
        for i in range(window_size, len(data)):
            # Input sequence
            X.append(data[feature_columns].iloc[i-window_size:i].values)
            # Target value
            y.append(data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def scale_features(self, train_data, test_data=None):
        """
        Scale features using StandardScaler
        
        Args:
            train_data (pd.DataFrame): Training data
            test_data (pd.DataFrame, optional): Test data
            
        Returns:
            tuple: (scaled_train_data, scaled_test_data)
        """
        feature_columns = [col for col in train_data.columns if col != 'volatility']
        
        # Fit scaler on training data
        train_scaled = train_data.copy()
        train_scaled[feature_columns] = self.scaler.fit_transform(train_data[feature_columns])
        
        if test_data is not None:
            test_scaled = test_data.copy()
            test_scaled[feature_columns] = self.scaler.transform(test_data[feature_columns])
            return train_scaled, test_scaled
        
        return train_scaled
    
    def get_feature_importance(self, data):
        """
        Calculate feature importance using correlation with target
        
        Args:
            data (pd.DataFrame): Data with features and target
            
        Returns:
            pd.Series: Feature importance scores
        """
        if 'volatility' not in data.columns:
            return None
        
        feature_columns = [col for col in data.columns if col != 'volatility']
        correlations = data[feature_columns].corrwith(data['volatility']).abs()
        
        return correlations.sort_values(ascending=False)
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data, window=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window).mean()
    
    def _calculate_stochastic(self, data, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, data, window=14):
        """Calculate Williams %R"""
        high_max = data['High'].rolling(window=window).max()
        low_min = data['Low'].rolling(window=window).min()
        return -100 * ((high_max - data['Close']) / (high_max - low_min))
    
    def _calculate_cci(self, data, window=20):
        """Calculate Commodity Channel Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        return (typical_price - sma) / (0.015 * mean_deviation)
    
    def _calculate_obv(self, data):
        """Calculate On Balance Volume"""
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=data.index)
