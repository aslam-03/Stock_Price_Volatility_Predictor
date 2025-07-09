import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from Research.data_processor import DataProcessor
from Research.feature_engineering import FeatureEngineer
from models.garch_model import GARCHModel
from models.lstm_model import LSTMModel
from models.rolling_window_model import RollingWindowModel
from models.model_evaluator import ModelEvaluator
from src.ui.visualization import Visualizer
from src.utils.utils import calculate_volatility, get_stock_info, get_usd_to_inr_rate, convert_usd_to_inr, format_inr_price
from src.config.settings import *

st.set_page_config(
    page_title="Stock Volatility Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📈 Stock Volatility Prediction System")
    st.markdown("Advanced volatility forecasting using GARCH and LSTM models with technical indicators")
    
    # Initialize session state for model caching
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'last_training_params' not in st.session_state:
        st.session_state.last_training_params = {}
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Stock selection
    stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, GOOGL, TSLA)")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 5)
    window_size = st.sidebar.slider("Rolling Window Size", 10, 100, 30)
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models",
        ["GARCH", "LSTM", "Rolling Window"],
        default=["GARCH", "LSTM"]
    )
    
    # Clear cache button
    if st.sidebar.button("🗑️ Clear Model Cache", help="Clear all cached models to force retraining"):
        st.session_state.trained_models = {}
        st.session_state.last_training_params = {}
        st.sidebar.success("✅ Cache cleared!")
    
    # Show cache status
    if st.session_state.trained_models:
        st.sidebar.info(f"📋 Cached models: {list(st.session_state.trained_models.keys())}")
    else:
        st.sidebar.info("📋 No cached models")
    
    if st.sidebar.button("Run Analysis", type="primary"):
        if not stock_symbol:
            st.error("Please enter a stock symbol")
            return
            
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
            
        try:
            # Initialize components
            data_processor = DataProcessor()
            feature_engineer = FeatureEngineer()
            visualizer = Visualizer()
            
            # Data loading progress
            with st.spinner(f"Loading data for {stock_symbol}..."):
                stock_data = data_processor.fetch_stock_data(stock_symbol, start_date, end_date)
                
            if stock_data.empty:
                st.error(f"No data found for symbol {stock_symbol}")
                return
                
            # Get USD to INR exchange rate
            with st.spinner("Fetching exchange rate..."):
                exchange_rate = get_usd_to_inr_rate()
                st.sidebar.info(f"USD to INR: ₹{exchange_rate:.2f}")
            
            # Display stock info
            stock_info = get_stock_info(stock_symbol)
            if stock_info:
                st.info(f"**{stock_info['name']}** ({stock_symbol}) - {stock_info['sector']} | Market Cap: {stock_info['market_cap']}")
            
            # Feature engineering
            with st.spinner("Calculating technical indicators..."):
                features_data = feature_engineer.create_features(stock_data)
                
            # Calculate volatility
            volatility_data = calculate_volatility(stock_data, window=window_size)
            
            # Display main metrics in INR
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_price_usd = stock_data['Close'].iloc[-1]
                current_price_inr = convert_usd_to_inr(current_price_usd, exchange_rate)
                st.metric("Current Price", format_inr_price(current_price_inr))
            with col2:
                price_change = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2] * 100
                st.metric("Daily Change", f"{price_change:.2f}%")
            with col3:
                current_vol = volatility_data.iloc[-1] * 100
                st.metric("Current Volatility", f"{current_vol:.2f}%")
            with col4:
                avg_vol = volatility_data.mean() * 100
                st.metric("Average Volatility", f"{avg_vol:.2f}%")
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔮 Predictions", "📈 Technical Analysis", "🎯 Model Evaluation"])
            
            with tab1:
                st.subheader("Stock Price and Volatility Overview")
                
                # Price chart
                fig_price = visualizer.plot_stock_price(stock_data)
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Volatility chart
                fig_vol = visualizer.plot_volatility(volatility_data)
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Data summary
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Price Statistics")
                    price_stats = stock_data['Close'].describe()
                    st.dataframe(price_stats)
                    
                with col2:
                    st.subheader("Volatility Statistics")
                    vol_stats = volatility_data.describe()
                    st.dataframe(vol_stats)
            
            with tab2:
                st.subheader("Volatility Predictions")
                
                if not selected_models:
                    st.warning("Please select at least one model for prediction")
                else:
                    predictions = {}
                    evaluation_results = {}
                    
                    # Prepare data for modeling
                    model_data = feature_engineer.prepare_model_data(features_data, volatility_data, window_size)
                    
                    if len(model_data) < window_size * 2:
                        st.error("Insufficient data for modeling. Please select a longer date range.")
                        return
                    
                    # Split data
                    train_size = int(len(model_data) * 0.8)
                    train_data = model_data[:train_size]
                    test_data = model_data[train_size:]
                    
                    # Model evaluation
                    evaluator = ModelEvaluator()
                    
                    # Check if retraining is needed
                    need_retraining = check_need_retraining(
                        stock_symbol, start_date, end_date, selected_models, window_size, prediction_horizon
                    )
                    
                    # Display current cache status
                    st.info(f"📋 Cache Status: {list(st.session_state.trained_models.keys())}")
                    
                    if need_retraining:
                        st.info("🔄 Training models with new parameters...")
                    else:
                        st.info("♻️ Using cached models (same parameters)")
                    
                    # GARCH Model
                    garch_model = None
                    if "GARCH" in selected_models:
                        if need_retraining or "GARCH" not in st.session_state.trained_models:
                            with st.spinner("Training GARCH model..."):
                                print(f"🎯 Starting GARCH model training...")
                                try:
                                    garch_model = GARCHModel()
                                    garch_model.ticker = stock_symbol  # Set ticker for MLflow
                                    # Pass the original stock data with price information for GARCH
                                    train_stock_data = stock_data.iloc[:train_size]
                                    test_stock_data = stock_data.iloc[train_size:train_size+len(test_data)]
                                    garch_pred = garch_model.fit_predict(train_stock_data, test_stock_data, prediction_horizon)
                                    
                                    # Cache the model and predictions
                                    st.session_state.trained_models["GARCH"] = {
                                        'model': garch_model,
                                        'predictions': garch_pred
                                    }
                                    print(f"✅ GARCH model cached successfully")
                                    st.success("✅ GARCH model trained and artifacts saved!")
                                    
                                except Exception as e:
                                    print(f"❌ GARCH model failed: {str(e)}")
                                    st.warning(f"GARCH model failed: {str(e)}. Skipping GARCH predictions.")
                                    if "GARCH" in selected_models:
                                        selected_models.remove("GARCH")
                        else:
                            print(f"♻️ Using cached GARCH model")
                            # Get the cached model object
                            garch_model = st.session_state.trained_models["GARCH"]['model']
                        
                        # Use cached predictions
                        if "GARCH" in st.session_state.trained_models:
                            garch_pred = st.session_state.trained_models["GARCH"]['predictions']
                            predictions["GARCH"] = garch_pred
                            
                            # Evaluate GARCH
                            garch_metrics = evaluator.evaluate_model(
                                test_data['volatility'].values[:len(garch_pred)], 
                                garch_pred,
                                model_name="GARCH"
                            )
                            evaluation_results["GARCH"] = garch_metrics
                    
                    # LSTM Model
                    lstm_model = None
                    if "LSTM" in selected_models:
                        if need_retraining or "LSTM" not in st.session_state.trained_models:
                            with st.spinner("Training LSTM model..."):
                                print(f"🎯 Starting LSTM model training...")
                                try:
                                    lstm_model = LSTMModel(window_size=window_size)
                                    lstm_model.ticker = stock_symbol  # Set ticker for MLflow
                                    lstm_pred = lstm_model.fit_predict(train_data, test_data, prediction_horizon)
                                    
                                    # Cache the model and predictions
                                    st.session_state.trained_models["LSTM"] = {
                                        'model': lstm_model,
                                        'predictions': lstm_pred
                                    }
                                    print(f"✅ LSTM model cached successfully")
                                    st.success("✅ LSTM model trained and artifacts saved!")
                                    
                                except Exception as e:
                                    print(f"❌ LSTM model failed: {str(e)}")
                                    st.warning(f"LSTM model failed: {str(e)}. Skipping LSTM predictions.")
                                    if "LSTM" in selected_models:
                                        selected_models.remove("LSTM")
                        else:
                            print(f"♻️ Using cached LSTM model")
                            # Get the cached model object
                            lstm_model = st.session_state.trained_models["LSTM"]['model']
                        
                        # Use cached predictions
                        if "LSTM" in st.session_state.trained_models:
                            lstm_pred = st.session_state.trained_models["LSTM"]['predictions']
                            predictions["LSTM"] = lstm_pred
                            
                            # Evaluate LSTM
                            lstm_metrics = evaluator.evaluate_model(
                                test_data['volatility'].values[:len(lstm_pred)], 
                                lstm_pred,
                                model_name="LSTM"
                            )
                            evaluation_results["LSTM"] = lstm_metrics
                    
                    # Rolling Window Model
                    if "Rolling Window" in selected_models:
                        if need_retraining or "Rolling Window" not in st.session_state.trained_models:
                            with st.spinner("Calculating rolling window predictions..."):
                                print(f"🎯 Starting Rolling Window model training...")
                                try:
                                    rolling_window_model = RollingWindowModel(window_size=window_size)
                                    rolling_window_model.ticker = stock_symbol  # Set ticker for MLflow
                                    rolling_pred = rolling_window_model.fit_predict(train_data, test_data)
                                    
                                    # Cache the model and predictions
                                    st.session_state.trained_models["Rolling Window"] = {
                                        'model': rolling_window_model,
                                        'predictions': rolling_pred
                                    }
                                    print(f"✅ Rolling Window model cached successfully")
                                    st.success("✅ Rolling Window model trained and artifacts saved!")
                                    
                                except Exception as e:
                                    print(f"❌ Rolling Window model failed: {str(e)}")
                                    st.warning(f"Rolling Window model failed: {str(e)}. Skipping Rolling Window predictions.")
                                    if "Rolling Window" in selected_models:
                                        selected_models.remove("Rolling Window")
                        else:
                            print(f"♻️ Using cached Rolling Window model")
                        
                        # Use cached predictions
                        if "Rolling Window" in st.session_state.trained_models:
                            rolling_pred = st.session_state.trained_models["Rolling Window"]['predictions']
                            predictions["Rolling Window"] = rolling_pred                            # Evaluate Rolling Window
                            if len(rolling_pred) > 0:
                                rolling_metrics = evaluator.evaluate_model(
                                    test_data['volatility'].values[:len(rolling_pred)],
                                    np.array(rolling_pred),
                                    model_name="Rolling Window"
                                )
                                evaluation_results["Rolling Window"] = rolling_metrics

                    
                    # Display predictions
                    if predictions:
                        fig_pred = visualizer.plot_predictions(test_data, predictions)
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Future predictions
                        st.subheader("Future Volatility Forecast")
                        future_predictions = {}
                        
                        for model_name in predictions.keys():
                            if model_name == "GARCH" and "GARCH" in selected_models and garch_model is not None:
                                future_pred = garch_model.forecast_future(prediction_horizon)
                                future_predictions[model_name] = future_pred
                            elif model_name == "LSTM" and "LSTM" in selected_models and lstm_model is not None:
                                future_pred = lstm_model.forecast_future(prediction_horizon)
                                future_predictions[model_name] = future_pred
                        
                        if future_predictions:
                            fig_future = visualizer.plot_future_predictions(future_predictions, prediction_horizon)
                            st.plotly_chart(fig_future, use_container_width=True)
                        
                        # Download predictions
                        if st.button("Download Predictions"):
                            pred_df = pd.DataFrame(predictions)
                            pred_df.index = test_data.index[:len(pred_df)]
                            csv = pred_df.to_csv()
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                                file_name=f"{stock_symbol}_volatility_predictions.csv",
                                mime="text/csv"
                            )
            
            with tab3:
                st.subheader("Technical Analysis")
                
                # Technical indicators
                fig_indicators = visualizer.plot_technical_indicators(features_data)
                st.plotly_chart(fig_indicators, use_container_width=True)
                
                # Correlation analysis
                st.subheader("Feature Correlation Analysis")
                
                # Debug: Show available columns
                with st.expander("Available Columns (Debug)"):
                    st.write("All columns:", features_data.columns.tolist())
                
                # Use available columns with proper capitalization
                correlation_columns = []
                if 'volatility' in features_data.columns:
                    correlation_columns.append('volatility')
                elif 'Volatility' in features_data.columns:
                    correlation_columns.append('Volatility')
                
                # Find technical indicator columns
                potential_cols = ['RSI', 'ATR', 'BB_Width', 'Volume_Ratio', 'MACD', 'Stoch_K', 'Williams_R', 'CCI', 'OBV']
                for col in features_data.columns:
                    if any(indicator.lower() in col.lower() for indicator in potential_cols):
                        if col not in correlation_columns:
                            correlation_columns.append(col)
                
                # Limit to first 8 columns for readability
                correlation_columns = correlation_columns[:8]
                
                if len(correlation_columns) > 1:
                    correlation_data = features_data[correlation_columns].corr()
                    fig_corr = px.imshow(
                        correlation_data,
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough correlation features available in the dataset")
                
                # Feature importance (if available)
                if 'LSTM' in selected_models and lstm_model is not None:
                    st.subheader("Feature Analysis")
                    try:
                        feature_importance = lstm_model.get_feature_importance()
                        if feature_importance is not None:
                            fig_importance = px.bar(
                                x=list(feature_importance.keys()),
                                y=list(feature_importance.values()),
                                title="Feature Importance (LSTM Model)"
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                    except Exception as e:
                        st.info(f"Feature importance not available: {str(e)}")
            
            with tab4:
                st.subheader("Model Evaluation")
                
                if evaluation_results:
                    # Metrics comparison
                    metrics_df = pd.DataFrame(evaluation_results).T
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Metrics visualization
                    fig_metrics = visualizer.plot_model_comparison(evaluation_results)
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Best model recommendation
                    best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['rmse'])
                    st.success(f"**Best Performing Model:** {best_model} (Lowest RMSE: {evaluation_results[best_model]['rmse']:.4f})")
                    
                    # Residual analysis
                    if predictions:
                        st.subheader("Residual Analysis")
                        fig_residuals = visualizer.plot_residuals(test_data, predictions)
                        st.plotly_chart(fig_residuals, use_container_width=True)
                else:
                    st.info("Run predictions to see model evaluation results")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

def check_need_retraining(stock_symbol, start_date, end_date, selected_models, window_size, prediction_horizon):
    """
    Check if models need retraining based on parameters
    
    Returns:
        bool: True if retraining is needed
    """
    current_params = {
        'stock_symbol': stock_symbol,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'selected_models': sorted(selected_models),
        'window_size': window_size,
        'prediction_horizon': prediction_horizon
    }
    
    # Check if parameters have changed
    if st.session_state.last_training_params != current_params:
        print(f"🔄 Parameters changed - need retraining")
        print(f"Old params: {st.session_state.last_training_params}")
        print(f"New params: {current_params}")
        st.session_state.last_training_params = current_params
        st.session_state.trained_models = {}  # Clear cached models
        return True
    
    # Check if required models are already trained
    missing_models = []
    for model_name in selected_models:
        if model_name not in st.session_state.trained_models:
            missing_models.append(model_name)
    
    if missing_models:
        print(f"🔄 Missing models {missing_models} - need retraining")
        return True
    
    print(f"♻️ All models already trained with same parameters - using cache")
    return False

if __name__ == "__main__":
    main()
