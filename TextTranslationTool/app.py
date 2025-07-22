import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import numpy as np
from typing import Dict, List, Any
import logging

# Import custom modules
from config import Config
from data.live_data_loader import DataLoader
from pattern_analysis.ml_candlestick_patterns import MLCandlestickDiscovery
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from models.deep_learning_ensemble import DeepLearningEnsemble
from backtesting.strategy_tester import StrategyTester
from portfolio.optimizer import PortfolioOptimizer
from risk_management.position_sizing import RiskManager
from utils.logger import setup_logging
from utils.helpers import format_currency, calculate_returns

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class MarketAIEngine:
    """Main application class for the Ultimate Market AI Engine"""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.pattern_discovery = MLCandlestickDiscovery()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_ensemble = DeepLearningEnsemble()
        self.strategy_tester = StrategyTester()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_manager = RiskManager()
        
    async def load_market_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Load market data for all timeframes"""
        try:
            data = await self.data_loader.fetch_multi_timeframe_data(
                symbol=symbol,
                timeframes=self.config.TIMEFRAMES,
                lookback_days=self.config.HISTORICAL_DAYS
            )
            return data
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            st.error(f"Failed to load market data: {e}")
            return {}
    
    def analyze_patterns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Discover ML-powered candlestick patterns"""
        try:
            patterns = {}
            for timeframe, df in data.items():
                if len(df) > 50:  # Minimum data requirement
                    pattern_results = self.pattern_discovery.discover_patterns(df)
                    patterns[timeframe] = pattern_results
            return patterns
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {}
    
    def calculate_technical_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate 40+ technical indicators with pattern analysis"""
        try:
            indicators = {}
            for timeframe, df in data.items():
                if len(df) > 50:
                    indicator_results = self.technical_analyzer.calculate_all_indicators(df)
                    indicators[timeframe] = indicator_results
            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def generate_predictions(self, data: Dict[str, pd.DataFrame], 
                           indicators: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate ML predictions using ensemble methods"""
        try:
            # Prepare feature matrix
            features = self.ml_ensemble.prepare_features(data, indicators)
            
            # Generate predictions
            predictions = self.ml_ensemble.predict(features)
            
            return predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {}

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Ultimate Market AI Engine",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the engine
    if 'engine' not in st.session_state:
        st.session_state.engine = MarketAIEngine()
    
    engine = st.session_state.engine
    
    # Header
    st.title("🚀 Ultimate Market AI Engine")
    st.markdown("*Proprietary AI-powered market intelligence for Indian stocks*")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Only editable parameter - Stock Symbol
    symbol = st.sidebar.text_input(
        "Stock Symbol (NSE)",
        value=engine.config.DEFAULT_SYMBOL,
        help="Enter Indian stock symbol (e.g., RELIANCE.NS, TCS.NS)"
    ).upper()
    
    if not symbol.endswith('.NS'):
        symbol += '.NS'
    
    # Display current configuration
    st.sidebar.markdown("### Current Settings")
    st.sidebar.info(f"**Symbol:** {symbol}")
    st.sidebar.info(f"**Timeframes:** {', '.join(engine.config.TIMEFRAMES)}")
    st.sidebar.info(f"**Historical Data:** {engine.config.HISTORICAL_DAYS} days")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Live Analysis", 
        "🔍 Pattern Discovery", 
        "📈 Technical Analysis",
        "🤖 AI Predictions",
        "📋 Backtesting",
        "💼 Portfolio & Risk"
    ])
    
    # Load data button
    if st.sidebar.button("🔄 Analyze Stock", type="primary"):
        with st.spinner("Loading market data and running AI analysis..."):
            # Load market data
            data = asyncio.run(engine.load_market_data(symbol))
            
            if data:
                st.session_state.market_data = data
                st.session_state.current_symbol = symbol
                
                # Run all analyses
                st.session_state.patterns = engine.analyze_patterns(data)
                st.session_state.indicators = engine.calculate_technical_indicators(data)
                st.session_state.predictions = engine.generate_predictions(data, st.session_state.indicators)
                
                st.sidebar.success("✅ Analysis Complete!")
            else:
                st.sidebar.error("❌ Failed to load data")
    
    # Tab 1: Live Analysis
    with tab1:
        st.header("📊 Live Market Analysis")
        
        if 'market_data' in st.session_state:
            data = st.session_state.market_data
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price chart
                st.subheader("Price Chart")
                timeframe = st.selectbox("Select Timeframe", engine.config.TIMEFRAMES)
                
                if timeframe in data:
                    df = data[timeframe].tail(100)  # Last 100 periods
                    
                    fig = go.Figure(data=go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name=f"{symbol} - {timeframe}"
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} - {timeframe} Chart",
                        xaxis_title="Time",
                        yaxis_title="Price (₹)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Current stats
                st.subheader("Current Stats")
                latest_data = data['1d'].iloc[-1]
                
                st.metric("Current Price", f"₹{latest_data['Close']:.2f}")
                
                change = latest_data['Close'] - latest_data['Open']
                change_pct = (change / latest_data['Open']) * 100
                
                st.metric(
                    "Day Change",
                    f"₹{change:.2f}",
                    f"{change_pct:.2f}%"
                )
                
                st.metric("Volume", f"{latest_data['Volume']:,.0f}")
                st.metric("High", f"₹{latest_data['High']:.2f}")
                st.metric("Low", f"₹{latest_data['Low']:.2f}")
        else:
            st.info("👆 Select a stock symbol and click 'Analyze Stock' to begin")
    
    # Tab 2: Pattern Discovery
    with tab2:
        st.header("🔍 ML Pattern Discovery")
        
        if 'patterns' in st.session_state:
            patterns = st.session_state.patterns
            
            st.subheader("Discovered Patterns")
            
            for timeframe, pattern_data in patterns.items():
                with st.expander(f"📊 {timeframe} Patterns"):
                    if 'discovered_patterns' in pattern_data:
                        discovered = pattern_data['discovered_patterns']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Pattern Summary:**")
                            st.write(f"- Total Patterns: {len(discovered)}")
                            
                            if discovered:
                                avg_success = np.mean([p.get('success_rate', 0) for p in discovered])
                                st.write(f"- Average Success Rate: {avg_success:.1f}%")
                        
                        with col2:
                            if discovered:
                                st.write("**Top Patterns:**")
                                sorted_patterns = sorted(discovered, 
                                                       key=lambda x: x.get('success_rate', 0), 
                                                       reverse=True)[:3]
                                
                                for i, pattern in enumerate(sorted_patterns, 1):
                                    st.write(f"{i}. Pattern {pattern.get('id', 'Unknown')} - "
                                           f"{pattern.get('success_rate', 0):.1f}% success")
        else:
            st.info("Run analysis to discover ML patterns")
    
    # Tab 3: Technical Analysis
    with tab3:
        st.header("📈 Advanced Technical Analysis")
        
        if 'indicators' in st.session_state:
            indicators = st.session_state.indicators
            
            timeframe = st.selectbox("Analysis Timeframe", engine.config.TIMEFRAMES, key="tech_tf")
            
            if timeframe in indicators:
                indicator_data = indicators[timeframe]
                
                # Create tabs for different indicator categories
                ind_tab1, ind_tab2, ind_tab3, ind_tab4 = st.tabs([
                    "Trend", "Momentum", "Volatility", "Volume"
                ])
                
                with ind_tab1:
                    st.subheader("Trend Indicators")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'SMA_20' in indicator_data.columns:
                            st.metric("SMA 20", f"₹{indicator_data['SMA_20'].iloc[-1]:.2f}")
                        if 'EMA_20' in indicator_data.columns:
                            st.metric("EMA 20", f"₹{indicator_data['EMA_20'].iloc[-1]:.2f}")
                    
                    with col2:
                        if 'ADX' in indicator_data.columns:
                            adx_val = indicator_data['ADX'].iloc[-1]
                            st.metric("ADX (Trend Strength)", f"{adx_val:.1f}")
                            
                            if adx_val > 25:
                                st.success("Strong Trend")
                            elif adx_val > 20:
                                st.warning("Moderate Trend")
                            else:
                                st.info("Weak Trend")
                
                with ind_tab2:
                    st.subheader("Momentum Indicators")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'RSI' in indicator_data.columns:
                            rsi_val = indicator_data['RSI'].iloc[-1]
                            st.metric("RSI", f"{rsi_val:.1f}")
                            
                            if rsi_val > 70:
                                st.error("Overbought")
                            elif rsi_val < 30:
                                st.success("Oversold")
                            else:
                                st.info("Neutral")
                    
                    with col2:
                        if 'MACD' in indicator_data.columns:
                            macd_val = indicator_data['MACD'].iloc[-1]
                            st.metric("MACD", f"{macd_val:.3f}")
        else:
            st.info("Run analysis to view technical indicators")
    
    # Tab 4: AI Predictions
    with tab4:
        st.header("🤖 AI Predictions")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions
            
            if predictions:
                # Main prediction display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    direction = predictions.get('direction', 'Unknown')
                    confidence = predictions.get('direction_confidence', 0)
                    
                    st.metric(
                        "Direction Prediction",
                        direction,
                        f"{confidence:.1f}% confidence"
                    )
                
                with col2:
                    target_price = predictions.get('target_price', 0)
                    current_price = predictions.get('current_price', 0)
                    
                    if target_price and current_price:
                        change_pct = ((target_price - current_price) / current_price) * 100
                        st.metric(
                            "Target Price",
                            f"₹{target_price:.2f}",
                            f"{change_pct:+.1f}%"
                        )
                
                with col3:
                    risk_level = predictions.get('risk_level', 'Unknown')
                    st.metric("Risk Level", risk_level)
                
                # Detailed predictions
                st.subheader("Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Time Horizons:**")
                    horizons = predictions.get('time_horizons', {})
                    for horizon, data in horizons.items():
                        st.write(f"- {horizon}: {data.get('probability', 0):.1f}% probability")
                
                with col2:
                    st.write("**Feature Importance:**")
                    features = predictions.get('feature_importance', {})
                    for feature, importance in list(features.items())[:5]:
                        st.write(f"- {feature}: {importance:.3f}")
                
                # Risk metrics
                st.subheader("Risk Assessment")
                
                risk_metrics = predictions.get('risk_metrics', {})
                if risk_metrics:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        stop_loss = risk_metrics.get('stop_loss', 0)
                        if stop_loss:
                            st.metric("Recommended Stop Loss", f"₹{stop_loss:.2f}")
                    
                    with col2:
                        position_size = risk_metrics.get('position_size', 0)
                        if position_size:
                            st.metric("Position Size", f"{position_size:.1f}%")
                    
                    with col3:
                        max_risk = risk_metrics.get('max_risk', 0)
                        if max_risk:
                            st.metric("Max Risk", f"{max_risk:.1f}%")
        else:
            st.info("Run analysis to view AI predictions")
    
    # Tab 5: Backtesting
    with tab5:
        st.header("📋 Strategy Backtesting")
        
        st.subheader("Backtest Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capital = st.number_input("Initial Capital (₹)", value=100000, min_value=1000)
            commission = st.number_input("Commission (%)", value=0.1, min_value=0.0, max_value=2.0, step=0.01)
        
        with col2:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", value=datetime.now())
        
        if st.button("🚀 Run Backtest"):
            if 'market_data' in st.session_state:
                with st.spinner("Running backtest..."):
                    # Run backtest
                    backtest_results = engine.strategy_tester.run_backtest(
                        symbol=st.session_state.current_symbol,
                        data=st.session_state.market_data,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        commission=commission
                    )
                    
                    if backtest_results:
                        # Display results
                        st.subheader("Backtest Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_return = backtest_results.get('total_return', 0)
                            st.metric("Total Return", f"{total_return:.2f}%")
                        
                        with col2:
                            sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        
                        with col3:
                            max_drawdown = backtest_results.get('max_drawdown', 0)
                            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                        
                        with col4:
                            win_rate = backtest_results.get('win_rate', 0)
                            st.metric("Win Rate", f"{win_rate:.1f}%")
                        
                        # Equity curve
                        if 'equity_curve' in backtest_results:
                            st.subheader("Equity Curve")
                            equity_df = pd.DataFrame(backtest_results['equity_curve'])
                            
                            fig = px.line(equity_df, x='date', y='equity', 
                                        title="Portfolio Equity Over Time")
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please run analysis first")
    
    # Tab 6: Portfolio & Risk
    with tab6:
        st.header("💼 Portfolio Optimization & Risk Management")
        
        st.subheader("Portfolio Composition")
        
        # Multi-stock portfolio input
        st.write("**Add Stocks to Portfolio:**")
        
        if 'portfolio_stocks' not in st.session_state:
            st.session_state.portfolio_stocks = []
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            new_stock = st.text_input("Stock Symbol", placeholder="RELIANCE.NS")
        
        with col2:
            allocation = st.number_input("Allocation (%)", min_value=0.0, max_value=100.0, value=10.0)
        
        with col3:
            if st.button("Add Stock"):
                if new_stock and allocation > 0:
                    st.session_state.portfolio_stocks.append({
                        'symbol': new_stock.upper(),
                        'allocation': allocation
                    })
                    st.success(f"Added {new_stock}")
        
        # Display current portfolio
        if st.session_state.portfolio_stocks:
            st.write("**Current Portfolio:**")
            portfolio_df = pd.DataFrame(st.session_state.portfolio_stocks)
            st.dataframe(portfolio_df, use_container_width=True)
            
            total_allocation = portfolio_df['allocation'].sum()
            if total_allocation != 100:
                st.warning(f"Total allocation: {total_allocation}% (should be 100%)")
            
            # Optimize portfolio
            if st.button("🎯 Optimize Portfolio") and total_allocation == 100:
                with st.spinner("Optimizing portfolio..."):
                    optimization_results = engine.portfolio_optimizer.optimize(
                        stocks=st.session_state.portfolio_stocks
                    )
                    
                    if optimization_results:
                        st.subheader("Optimization Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            expected_return = optimization_results.get('expected_return', 0)
                            st.metric("Expected Annual Return", f"{expected_return:.2f}%")
                            
                            volatility = optimization_results.get('volatility', 0)
                            st.metric("Portfolio Volatility", f"{volatility:.2f}%")
                        
                        with col2:
                            sharpe = optimization_results.get('sharpe_ratio', 0)
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            
                            var = optimization_results.get('var_95', 0)
                            st.metric("VaR (95%)", f"{var:.2f}%")
        
        # Risk management section
        st.subheader("Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_value = st.number_input("Portfolio Value (₹)", value=1000000, min_value=1000)
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
        
        with col2:
            max_position_size = st.slider("Max Position Size (%)", 1, 20, 5)
            stop_loss_pct = st.slider("Stop Loss (%)", 1, 10, 3)
        
        if st.button("📊 Calculate Risk Metrics"):
            risk_metrics = engine.risk_manager.calculate_risk_metrics(
                portfolio_value=portfolio_value,
                risk_tolerance=risk_tolerance,
                max_position_size=max_position_size,
                stop_loss_pct=stop_loss_pct
            )
            
            if risk_metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_risk = risk_metrics.get('max_risk_per_trade', 0)
                    st.metric("Max Risk per Trade", f"₹{max_risk:,.0f}")
                
                with col2:
                    position_size = risk_metrics.get('recommended_position_size', 0)
                    st.metric("Recommended Position Size", f"₹{position_size:,.0f}")
                
                with col3:
                    kelly_fraction = risk_metrics.get('kelly_fraction', 0)
                    st.metric("Kelly Fraction", f"{kelly_fraction:.2f}")

if __name__ == "__main__":
    main()
