import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.utils.utils import convert_usd_to_inr, format_inr_price, get_usd_to_inr_rate
import streamlit as st

class Visualizer:
    """
    Comprehensive visualization class for stock volatility prediction system
    """
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'purple': '#9467bd',
            'brown': '#8c564b',
            'pink': '#e377c2',
            'gray': '#7f7f7f'
        }
    
    def plot_stock_price(self, data, title="Stock Price Analysis"):
        """
        Plot stock price with volume in INR
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Price chart
        """
        try:
            # Get exchange rate and convert prices to INR
            exchange_rate = get_usd_to_inr_rate()
            data_inr = data.copy()
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in data_inr.columns:
                    data_inr[col] = convert_usd_to_inr(data_inr[col], exchange_rate)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price (INR)', 'Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick chart with INR prices
            fig.add_trace(
                go.Candlestick(
                    x=data_inr.index,
                    open=data_inr['Open'],
                    high=data_inr['High'],
                    low=data_inr['Low'],
                    close=data_inr['Close'],
                    name='Price (INR)',
                    increasing_line_color=self.colors['success'],
                    decreasing_line_color=self.colors['danger']
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors = ['red' if row['Close'] < row['Open'] else 'green' 
                     for index, row in data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"{title} (Prices in INR @ ₹{exchange_rate:.2f}/USD)",
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")
            return go.Figure()
    
    def plot_volatility(self, volatility_data, title="Volatility Analysis"):
        """
        Plot volatility over time
        
        Args:
            volatility_data (pd.Series): Volatility time series
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Volatility chart
        """
        try:
            fig = go.Figure()
            
            # Volatility line
            fig.add_trace(
                go.Scatter(
                    x=volatility_data.index,
                    y=volatility_data.values * 100,
                    mode='lines',
                    name='Volatility',
                    line=dict(color=self.colors['primary'], width=2)
                )
            )
            
            # Add volatility bands
            mean_vol = volatility_data.mean() * 100
            std_vol = volatility_data.std() * 100
            
            fig.add_hline(
                y=mean_vol,
                line_dash="dash",
                line_color=self.colors['secondary'],
                annotation_text="Mean"
            )
            
            fig.add_hline(
                y=mean_vol + 2*std_vol,
                line_dash="dot",
                line_color=self.colors['danger'],
                annotation_text="High Volatility"
            )
            
            fig.add_hline(
                y=mean_vol - 2*std_vol,
                line_dash="dot",
                line_color=self.colors['success'],
                annotation_text="Low Volatility"
            )
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility chart: {str(e)}")
            return go.Figure()
    
    def plot_technical_indicators(self, data, title="Technical Indicators"):
        """
        Plot technical indicators
        
        Args:
            data (pd.DataFrame): Data with technical indicators
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Technical indicators chart
        """
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'ATR', 'Stochastic', 'Volume Ratio'),
                vertical_spacing=0.1
            )
            
            # RSI
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color=self.colors['primary'])
                    ),
                    row=1, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            
            # MACD
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color=self.colors['primary'])
                    ),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD_Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color=self.colors['secondary'])
                    ),
                    row=1, col=2
                )
            
            # Bollinger Bands
            if all(col in data.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color=self.colors['danger'])
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color=self.colors['primary'])
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color=self.colors['success'])
                    ),
                    row=2, col=1
                )
            
            # ATR
            if 'ATR' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ATR'],
                        mode='lines',
                        name='ATR',
                        line=dict(color=self.colors['warning'])
                    ),
                    row=2, col=2
                )
            
            # Stochastic
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Stoch_K'],
                        mode='lines',
                        name='%K',
                        line=dict(color=self.colors['primary'])
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Stoch_D'],
                        mode='lines',
                        name='%D',
                        line=dict(color=self.colors['secondary'])
                    ),
                    row=3, col=1
                )
                fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
            
            # Volume Ratio
            if 'Volume_Ratio' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Volume_Ratio'],
                        mode='lines',
                        name='Volume Ratio',
                        line=dict(color=self.colors['purple'])
                    ),
                    row=3, col=2
                )
                fig.add_hline(y=1, line_dash="dash", line_color="gray", row=3, col=2)
            
            fig.update_layout(
                title=title,
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating technical indicators chart: {str(e)}")
            return go.Figure()
    
    def plot_predictions(self, test_data, predictions, title="Volatility Predictions"):
        """
        Plot actual vs predicted volatility
        
        Args:
            test_data (pd.DataFrame): Test data with actual volatility
            predictions (dict): Dictionary of model predictions
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Predictions chart
        """
        try:
            fig = go.Figure()
            
            # Actual volatility
            fig.add_trace(
                go.Scatter(
                    x=test_data.index[:len(test_data)],
                    y=test_data['volatility'].values * 100,
                    mode='lines',
                    name='Actual',
                    line=dict(color=self.colors['primary'], width=3)
                )
            )
            
            # Model predictions
            color_map = {
                'GARCH': self.colors['secondary'],
                'LSTM': self.colors['success'],
                'Rolling Window': self.colors['purple']
            }
            
            for model_name, pred_values in predictions.items():
                if len(pred_values) > 0:
                    pred_index = test_data.index[:len(pred_values)]
                    fig.add_trace(
                        go.Scatter(
                            x=pred_index,
                            y=pred_values * 100,
                            mode='lines',
                            name=f'{model_name} Prediction',
                            line=dict(
                                color=color_map.get(model_name, self.colors['gray']),
                                width=2,
                                dash='dash'
                            )
                        )
                    )
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating predictions chart: {str(e)}")
            return go.Figure()
    
    def plot_future_predictions(self, future_predictions, horizon, title="Future Volatility Forecast"):
        """
        Plot future volatility predictions
        
        Args:
            future_predictions (dict): Dictionary of future predictions
            horizon (int): Prediction horizon
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Future predictions chart
        """
        try:
            fig = go.Figure()
            
            # Create future dates
            future_dates = pd.date_range(
                start=pd.Timestamp.now().date(),
                periods=horizon,
                freq='D'
            )
            
            color_map = {
                'GARCH': self.colors['secondary'],
                'LSTM': self.colors['success'],
                'Rolling Window': self.colors['purple']
            }
            
            for model_name, pred_values in future_predictions.items():
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=pred_values * 100,
                        mode='lines+markers',
                        name=f'{model_name}',
                        line=dict(
                            color=color_map.get(model_name, self.colors['gray']),
                            width=3
                        ),
                        marker=dict(size=8)
                    )
                )
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Predicted Volatility (%)",
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating future predictions chart: {str(e)}")
            return go.Figure()
    
    def plot_model_comparison(self, evaluation_results, title="Model Performance Comparison"):
        """
        Plot model comparison metrics
        
        Args:
            evaluation_results (dict): Dictionary of evaluation metrics
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Model comparison chart
        """
        try:
            metrics = ['rmse', 'mae', 'mape', 'r2', 'hit_ratio', 'directional_accuracy']
            model_names = list(evaluation_results.keys())
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=metrics,
                vertical_spacing=0.15
            )
            
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
            
            for i, metric in enumerate(metrics):
                if i < len(positions):
                    row, col = positions[i]
                    
                    values = [evaluation_results[model].get(metric, 0) for model in model_names]
                    
                    fig.add_trace(
                        go.Bar(
                            x=model_names,
                            y=values,
                            name=metric.upper(),
                            marker_color=self.colors['primary']
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=title,
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating model comparison chart: {str(e)}")
            return go.Figure()
    
    def plot_residuals(self, test_data, predictions, title="Residual Analysis"):
        """
        Plot residual analysis for model predictions
        
        Args:
            test_data (pd.DataFrame): Test data
            predictions (dict): Model predictions
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Residuals chart
        """
        try:
            fig = make_subplots(
                rows=len(predictions), cols=2,
                subplot_titles=[f'{model} Residuals' for model in predictions.keys()] + 
                              [f'{model} Q-Q Plot' for model in predictions.keys()],
                vertical_spacing=0.1
            )
            
            actual = test_data['volatility'].values
            
            for i, (model_name, pred_values) in enumerate(predictions.items()):
                # Calculate residuals
                min_len = min(len(actual), len(pred_values))
                residuals = actual[:min_len] - pred_values[:min_len]
                
                # Residuals time series
                fig.add_trace(
                    go.Scatter(
                        x=test_data.index[:min_len],
                        y=residuals,
                        mode='lines+markers',
                        name=f'{model_name} Residuals',
                        line=dict(color=self.colors['primary'])
                    ),
                    row=i+1, col=1
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=i+1, col=1)
                
                # Residuals histogram (as Q-Q plot substitute)
                fig.add_trace(
                    go.Histogram(
                        x=residuals,
                        name=f'{model_name} Distribution',
                        marker_color=self.colors['secondary'],
                        opacity=0.7
                    ),
                    row=i+1, col=2
                )
            
            fig.update_layout(
                title=title,
                height=300 * len(predictions),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating residuals chart: {str(e)}")
            return go.Figure()
    
    def create_dashboard_summary(self, stock_data, predictions, metrics):
        """
        Create a summary dashboard figure
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            predictions (dict): Model predictions
            metrics (dict): Evaluation metrics
            
        Returns:
            plotly.graph_objects.Figure: Dashboard summary
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Movement', 'Volatility Predictions', 'Model Performance', 'Risk Metrics'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Price movement
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color=self.colors['primary'])
                ),
                row=1, col=1
            )
            
            # Add more subplots as needed...
            
            fig.update_layout(
                title="Portfolio Dashboard",
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating dashboard: {str(e)}")
            return go.Figure()
