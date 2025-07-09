import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import os


class DataProcessor:
    """
    Handles data fetching and preprocessing for stock analysis
    """

    def __init__(self):
        self.cache_timeout = 3600  # 1 hour cache

    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance

        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for data
            end_date (datetime): End date for data

        Returns:
            pd.DataFrame: Stock price data with OHLCV columns
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            data = data.ffill()  # Updated per FutureWarning
            data = data.dropna()

            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def validate_data(self, data):
        """
        Validate the fetched stock data

        Args:
            data (pd.DataFrame): Stock data to validate

        Returns:
            bool: True if data is valid, False otherwise
        """
        if data.empty:
            return False

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False

        if len(data) < 30:
            return False

        if (data['Close'] <= 0).any() or (data['Volume'] < 0).any():
            return False

        return True

    def preprocess_data(self, data):
        """
        Preprocess stock data for analysis

        Args:
            data (pd.DataFrame): Raw stock data

        Returns:
            pd.DataFrame: Preprocessed stock data
        """
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Price_Change'] = data['Close'] - data['Close'].shift(1)
        data['Price_Change_Pct'] = data['Price_Change'] / data['Close'].shift(1) * 100
        data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
        data['OC_Ratio'] = (data['Open'] - data['Close']) / data['Close']
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

        data = data.dropna()
        return data

    def get_multiple_stocks(self, symbols, start_date, end_date):
        """
        Fetch data for multiple stocks and save each to a CSV in 'Research/processed2'

        Args:
            symbols (list): List of stock ticker symbols
            start_date (datetime): Start date for data
            end_date (datetime): End date for data

        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        output_dir = "Research/processed2"
        os.makedirs(output_dir, exist_ok=True)

        stock_data = {}

        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    stock_data[symbol] = data

                    # Save raw data to CSV (DVC will track this)
                    output_path = os.path.join(output_dir, f"{symbol}_raw.csv")
                    data.to_csv(output_path, index=True)

            except Exception as e:
                print(f"Could not fetch data for {symbol}: {str(e)}")
                continue

        return stock_data

    def resample_data(self, data, frequency='D'):
        """
        Resample data to different frequencies

        Args:
            data (pd.DataFrame): Stock data
            frequency (str): Resampling frequency ('D', 'W', 'M')

        Returns:
            pd.DataFrame: Resampled data
        """
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        for col in data.columns:
            if col not in agg_dict and data[col].dtype in ['float64', 'int64']:
                agg_dict[col] = 'last'

        resampled = data.resample(frequency).agg(agg_dict)
        resampled = resampled.dropna()

        return resampled
