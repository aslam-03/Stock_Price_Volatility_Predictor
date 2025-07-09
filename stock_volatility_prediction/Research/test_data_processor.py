"""
Unit tests for data processing module.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from Research.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor()
        self.test_symbol = "AAPL"
        self.start_date = datetime.now() - timedelta(days=30)
        self.end_date = datetime.now()
    
    def test_fetch_stock_data(self):
        """Test basic stock data fetching"""
        data = self.processor.fetch_stock_data(
            self.test_symbol, 
            self.start_date, 
            self.end_date
        )
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('Close', data.columns)
    
    def test_validate_data(self):
        """Test data validation"""
        # Create sample data
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        
        is_valid = self.processor.validate_data(sample_data)
        self.assertTrue(is_valid)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        
        processed_data = self.processor.preprocess_data(sample_data)
        self.assertIsInstance(processed_data, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()