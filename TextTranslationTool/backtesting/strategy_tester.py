"""
Comprehensive Backtesting Framework
Tests strategies with performance metrics and risk analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade execution record"""
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    direction: str = "LONG"  # LONG or SHORT
    entry_signal: str = ""
    exit_signal: str = ""
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0

@dataclass 
class Position:
    """Current position tracking"""
    symbol: str
    quantity: int = 0
    entry_price: float = 0.0
    entry_date: Optional[datetime] = None
    unrealized_pnl: float = 0.0
    direction: str = "FLAT"

class StrategyTester:
    """Advanced backtesting system with comprehensive metrics"""
    
    def __init__(self):
        self.config = Config()
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    def _calculate_commission(self, price: float, quantity: int, commission_pct: float) -> float:
        """Calculate commission costs"""
        return price * quantity * (commission_pct / 100)
    
    def _calculate_slippage(self, price: float, quantity: int, slippage_pct: float) -> float:
        """Calculate slippage costs"""
        return price * quantity * (slippage_pct / 100)
    
    def _generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on ML predictions and technical analysis"""
        
        signals_df = data.copy()
        signals_df['signal'] = 0  # 0=HOLD, 1=BUY, -1=SELL
        signals_df['signal_strength'] = 0.0
        signals_df['stop_loss'] = 0.0
        signals_df['take_profit'] = 0.0
        
        if indicators.empty:
            logger.warning("No indicators available for signal generation")
            return signals_df
        
        # Combine data with indicators
        combined_df = signals_df.join(indicators, how='inner')
        
        try:
            # Multi-factor signal generation
            
            # 1. Trend Following Signals
            trend_signals = pd.Series(0, index=combined_df.index)
            
            if 'Golden_Cross' in combined_df.columns and 'Death_Cross' in combined_df.columns:
                trend_signals += combined_df['Golden_Cross'] * 2
                trend_signals -= combined_df['Death_Cross'] * 2
            
            if 'ADX' in combined_df.columns and 'Bullish_Trend' in combined_df.columns:
                strong_trend = combined_df['ADX'] > 25
                trend_signals += (strong_trend & combined_df['Bullish_Trend']).astype(int)
                trend_signals -= (strong_trend & ~combined_df['Bullish_Trend']).astype(int)
            
            # 2. Momentum Signals
            momentum_signals = pd.Series(0, index=combined_df.index)
            
            if 'RSI' in combined_df.columns:
                rsi_oversold = combined_df['RSI'] < 30
                rsi_overbought = combined_df['RSI'] > 70
                momentum_signals += rsi_oversold.astype(int)
                momentum_signals -= rsi_overbought.astype(int)
            
            if 'MACD_Bull_Cross' in combined_df.columns and 'MACD_Bear_Cross' in combined_df.columns:
                momentum_signals += combined_df['MACD_Bull_Cross']
                momentum_signals -= combined_df['MACD_Bear_Cross']
            
            # 3. Volume Confirmation
            volume_signals = pd.Series(0, index=combined_df.index)
            
            if 'Price_Up_Vol_Up' in combined_df.columns and 'Price_Down_Vol_Up' in combined_df.columns:
                volume_signals += combined_df['Price_Up_Vol_Up']
                volume_signals -= combined_df['Price_Down_Vol_Up']
            
            # 4. Volatility Signals
            volatility_signals = pd.Series(0, index=combined_df.index)
            
            if 'BB_Squeeze' in combined_df.columns and 'BB_Expansion' in combined_df.columns:
                # Breakout after squeeze
                squeeze_breakout = (combined_df['BB_Squeeze'].shift(1) == 1) & (combined_df['BB_Expansion'] == 1)
                price_direction = combined_df['Close'] > combined_df['Close'].shift(1)
                volatility_signals += (squeeze_breakout & price_direction).astype(int)
                volatility_signals -= (squeeze_breakout & ~price_direction).astype(int)
            
            # 5. Confluence Score
            if 'Confluence_Score' in combined_df.columns:
                confluence_signals = pd.Series(0, index=combined_df.index)
                confluence_signals += (combined_df['Confluence_Score'] > 3).astype(int)
                confluence_signals -= (combined_df['Confluence_Score'] < -3).astype(int)
            else:
                confluence_signals = pd.Series(0, index=combined_df.index)
            
            # Combine all signals with weights
            final_signals = (
                trend_signals * 0.3 +
                momentum_signals * 0.25 +
                volume_signals * 0.15 +
                volatility_signals * 0.15 +
                confluence_signals * 0.15
            )
            
            # Generate discrete signals
            signals_df['signal'] = np.where(
                final_signals > 1.5, 1,  # Strong BUY
                np.where(final_signals < -1.5, -1, 0)  # Strong SELL
            )
            
            signals_df['signal_strength'] = np.abs(final_signals)
            
            # Dynamic stop loss and take profit based on volatility
            if 'ATR' in combined_df.columns:
                atr = combined_df['ATR']
                signals_df['stop_loss'] = atr * 2  # 2x ATR stop loss
                signals_df['take_profit'] = atr * 3  # 3x ATR take profit
            else:
                # Default values
                signals_df['stop_loss'] = combined_df['Close'] * 0.02  # 2% stop loss
                signals_df['take_profit'] = combined_df['Close'] * 0.06  # 6% take profit
            
            logger.info(f"Generated signals: {(signals_df['signal'] == 1).sum()} BUY, {(signals_df['signal'] == -1).sum()} SELL")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals_df
    
    def _execute_trade(self, symbol: str, signal: int, price: float, date: datetime,
                      quantity: int, commission_pct: float, slippage_pct: float,
                      signal_strength: float, stop_loss: float, take_profit: float) -> Optional[Trade]:
        """Execute a trade based on signal"""
        
        if signal == 0:
            return None
        
        position = self.positions.get(symbol, Position(symbol=symbol))
        
        # Calculate costs
        commission = self._calculate_commission(price, quantity, commission_pct)
        slippage = self._calculate_slippage(price, quantity, slippage_pct)
        
        # Adjust price for slippage
        if signal == 1:  # BUY
            execution_price = price * (1 + slippage_pct / 100)
            direction = "LONG"
        else:  # SELL
            execution_price = price * (1 - slippage_pct / 100)
            direction = "SHORT"
        
        trade = Trade(
            entry_date=date,
            entry_price=execution_price,
            quantity=quantity,
            direction=direction,
            entry_signal=f"Signal_{signal}_Strength_{signal_strength:.2f}",
            commission=commission,
            slippage=slippage
        )
        
        # Update position
        if position.quantity == 0:  # No existing position
            position.quantity = quantity if signal == 1 else -quantity
            position.entry_price = execution_price
            position.entry_date = date
            position.direction = direction
        else:
            # Close existing position if opposite signal
            if (position.quantity > 0 and signal == -1) or (position.quantity < 0 and signal == 1):
                # Close position
                exit_price = execution_price
                pnl = (exit_price - position.entry_price) * position.quantity
                
                # Find the corresponding opening trade and update it
                for open_trade in reversed(self.trades):
                    if (open_trade.exit_date is None and 
                        open_trade.direction == position.direction):
                        open_trade.exit_date = date
                        open_trade.exit_price = exit_price
                        open_trade.pnl = pnl - open_trade.commission - commission - open_trade.slippage - slippage
                        open_trade.return_pct = open_trade.pnl / (open_trade.entry_price * open_trade.quantity) * 100
                        open_trade.exit_signal = f"Signal_{signal}_Strength_{signal_strength:.2f}"
                        break
                
                # Reset position
                position.quantity = 0
                position.entry_price = 0.0
                position.entry_date = None
                position.direction = "FLAT"
        
        self.positions[symbol] = position
        self.trades.append(trade)
        
        return trade
    
    def run_backtest(self, symbol: str, data: Dict[str, pd.DataFrame],
                    start_date: datetime, end_date: datetime,
                    initial_capital: float = 100000,
                    commission: float = 0.1, slippage: float = 0.05) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Reset state
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        
        # Use daily data for backtesting
        primary_timeframe = '1d'
        if primary_timeframe not in data:
            primary_timeframe = list(data.keys())[0]
        
        price_data = data[primary_timeframe]
        
        # Filter data by date range
        price_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
        
        if len(price_data) < 10:
            logger.warning("Insufficient data for backtesting")
            return {'status': 'failed', 'reason': 'insufficient_data'}
        
        # Generate mock indicators for backtesting (in real implementation, use actual indicators)
        indicators_data = self._generate_mock_indicators(price_data)
        
        # Generate signals
        signals_df = self._generate_signals(price_data, indicators_data)
        
        # Initialize portfolio
        current_capital = initial_capital
        position_size_pct = 10  # 10% per trade
        
        equity_history = []
        
        try:
            for i, (date, row) in enumerate(signals_df.iterrows()):
                current_price = row['Close']
                signal = row['signal']
                signal_strength = row.get('signal_strength', 1.0)
                stop_loss = row.get('stop_loss', current_price * 0.02)
                take_profit = row.get('take_profit', current_price * 0.06)
                
                # Calculate position size
                position_value = current_capital * (position_size_pct / 100)
                quantity = int(position_value / current_price)
                
                if quantity > 0:
                    # Execute trade
                    trade = self._execute_trade(
                        symbol=symbol,
                        signal=signal,
                        price=current_price,
                        date=date,
                        quantity=quantity,
                        commission_pct=commission,
                        slippage_pct=slippage,
                        signal_strength=signal_strength,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                
                # Update unrealized P&L
                position = self.positions.get(symbol, Position(symbol=symbol))
                if position.quantity != 0:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                
                # Calculate current equity
                realized_pnl = sum([t.pnl for t in self.trades if t.exit_date is not None])
                unrealized_pnl = position.unrealized_pnl
                current_equity = initial_capital + realized_pnl + unrealized_pnl
                
                equity_history.append({
                    'date': date,
                    'equity': current_equity,
                    'realized_pnl': realized_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'price': current_price,
                    'signal': signal
                })
                
                current_capital = current_equity - unrealized_pnl  # Available capital for new trades
            
            # Close any remaining positions
            if symbol in self.positions and self.positions[symbol].quantity != 0:
                final_price = signals_df.iloc[-1]['Close']
                final_date = signals_df.index[-1]
                
                for trade in reversed(self.trades):
                    if trade.exit_date is None:
                        trade.exit_date = final_date
                        trade.exit_price = final_price
                        position = self.positions[symbol]
                        trade.pnl = (final_price - trade.entry_price) * trade.quantity - trade.commission - trade.slippage
                        trade.return_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100
                        trade.exit_signal = "End_of_Period"
                        break
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(equity_history, initial_capital)
            
            self.equity_curve = equity_history
            self.performance_metrics = performance_metrics
            
            logger.info("Backtest completed successfully")
            
            return {
                'status': 'success',
                'equity_curve': equity_history,
                'trades': [self._trade_to_dict(t) for t in self.trades],
                'performance_metrics': performance_metrics,
                'initial_capital': initial_capital,
                'final_equity': equity_history[-1]['equity'] if equity_history else initial_capital,
                'total_trades': len(self.trades),
                'period': f"{start_date.date()} to {end_date.date()}"
            }
            
        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def _generate_mock_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic technical indicators for backtesting"""
        
        indicators = pd.DataFrame(index=price_data.index)
        
        try:
            close = price_data['Close']
            high = price_data['High']
            low = price_data['Low']
            volume = price_data['Volume']
            
            # Moving averages
            indicators['SMA_20'] = close.rolling(20).mean()
            indicators['SMA_50'] = close.rolling(50).mean()
            indicators['EMA_12'] = close.ewm(span=12).mean()
            
            # Golden/Death Cross
            indicators['Golden_Cross'] = ((indicators['SMA_20'] > indicators['SMA_50']) & 
                                        (indicators['SMA_20'].shift(1) <= indicators['SMA_50'].shift(1))).astype(int)
            indicators['Death_Cross'] = ((indicators['SMA_20'] < indicators['SMA_50']) & 
                                       (indicators['SMA_20'].shift(1) >= indicators['SMA_50'].shift(1))).astype(int)
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            indicators['MACD'] = exp1 - exp2
            indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
            indicators['MACD_Bull_Cross'] = ((indicators['MACD'] > indicators['MACD_Signal']) & 
                                           (indicators['MACD'].shift(1) <= indicators['MACD_Signal'].shift(1))).astype(int)
            indicators['MACD_Bear_Cross'] = ((indicators['MACD'] < indicators['MACD_Signal']) & 
                                           (indicators['MACD'].shift(1) >= indicators['MACD_Signal'].shift(1))).astype(int)
            
            # Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            indicators['BB_Upper'] = bb_middle + (bb_std * 2)
            indicators['BB_Lower'] = bb_middle - (bb_std * 2)
            bb_width = (indicators['BB_Upper'] - indicators['BB_Lower']) / bb_middle
            indicators['BB_Squeeze'] = (bb_width < bb_width.rolling(20).quantile(0.2)).astype(int)
            indicators['BB_Expansion'] = (bb_width > bb_width.shift(1)).astype(int)
            
            # ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['ATR'] = tr.rolling(14).mean()
            
            # ADX (simplified)
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm = plus_dm.where(plus_dm > minus_dm.abs(), 0).abs()
            minus_dm = minus_dm.where(minus_dm.abs() > plus_dm, 0).abs()
            
            plus_di = 100 * (plus_dm.rolling(14).mean() / indicators['ATR'])
            minus_di = 100 * (minus_dm.rolling(14).mean() / indicators['ATR'])
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            indicators['ADX'] = dx.rolling(14).mean()
            indicators['Bullish_Trend'] = (plus_di > minus_di).astype(int)
            
            # Volume indicators
            volume_ma = volume.rolling(20).mean()
            indicators['Price_Up_Vol_Up'] = ((close > close.shift(1)) & (volume > volume_ma)).astype(int)
            indicators['Price_Down_Vol_Up'] = ((close < close.shift(1)) & (volume > volume_ma)).astype(int)
            
            # Confluence score
            bullish_signals = [
                (indicators['RSI'] < 30).astype(int),
                indicators['MACD_Bull_Cross'],
                indicators['Golden_Cross'],
                indicators['Price_Up_Vol_Up']
            ]
            
            bearish_signals = [
                (indicators['RSI'] > 70).astype(int),
                indicators['MACD_Bear_Cross'],
                indicators['Death_Cross'],
                indicators['Price_Down_Vol_Up']
            ]
            
            indicators['Confluence_Score'] = (sum(bullish_signals) - sum(bearish_signals))
            
        except Exception as e:
            logger.error(f"Error generating mock indicators: {e}")
        
        return indicators.fillna(0)
    
    def _calculate_performance_metrics(self, equity_history: List[Dict], 
                                     initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not equity_history:
            return {}
        
        # Convert to series for easier calculation
        equity_series = pd.Series([eq['equity'] for eq in equity_history])
        dates = [eq['date'] for eq in equity_history]
        
        final_equity = equity_series.iloc[-1]
        
        # Basic returns
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        # Annualized metrics
        trading_days = len(equity_series)
        years = trading_days / 252  # Approximate trading days per year
        
        if years > 0:
            annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0
        
        # Volatility
        if len(daily_returns) > 1:
            volatility = daily_returns.std() * np.sqrt(252) * 100
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Trade statistics
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        
        if completed_trades:
            win_trades = [t for t in completed_trades if t.pnl > 0]
            lose_trades = [t for t in completed_trades if t.pnl <= 0]
            
            win_rate = len(win_trades) / len(completed_trades) * 100
            avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t.pnl for t in lose_trades]) if lose_trades else 0
            profit_factor = abs(sum([t.pnl for t in win_trades]) / sum([t.pnl for t in lose_trades])) if lose_trades and sum([t.pnl for t in lose_trades]) != 0 else 0
            
            avg_trade_duration = np.mean([(t.exit_date - t.entry_date).days for t in completed_trades if t.exit_date])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        # Risk metrics
        if len(daily_returns) > 0:
            var_95 = np.percentile(daily_returns, 5) * 100  # 95% Value at Risk
            cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100  # Conditional VaR
        else:
            var_95 = 0
            cvar_95 = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 1:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            sortino_ratio = (daily_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0
        
        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_trades': len(completed_trades),
            'avg_trade_duration': round(avg_trade_duration, 1),
            'var_95': round(var_95, 2),
            'cvar_95': round(cvar_95, 2),
            'final_equity': round(final_equity, 2),
            'initial_capital': initial_capital,
            'trading_period_days': trading_days,
            'years_traded': round(years, 2)
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert Trade object to dictionary"""
        return {
            'entry_date': trade.entry_date.isoformat() if trade.entry_date else None,
            'entry_price': round(trade.entry_price, 2),
            'exit_date': trade.exit_date.isoformat() if trade.exit_date else None,
            'exit_price': round(trade.exit_price, 2) if trade.exit_price else None,
            'quantity': trade.quantity,
            'direction': trade.direction,
            'entry_signal': trade.entry_signal,
            'exit_signal': trade.exit_signal,
            'commission': round(trade.commission, 2),
            'slippage': round(trade.slippage, 2),
            'pnl': round(trade.pnl, 2),
            'return_pct': round(trade.return_pct, 2)
        }
    
    def get_trade_analysis(self) -> Dict[str, Any]:
        """Get detailed trade analysis"""
        
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        
        if not completed_trades:
            return {'status': 'no_trades'}
        
        # Monthly performance
        monthly_returns = {}
        for trade in completed_trades:
            if trade.exit_date:
                month_key = trade.exit_date.strftime('%Y-%m')
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                monthly_returns[month_key].append(trade.pnl)
        
        monthly_pnl = {month: sum(pnl_list) for month, pnl_list in monthly_returns.items()}
        
        # Trade distribution
        pnl_values = [t.pnl for t in completed_trades]
        
        return {
            'monthly_pnl': monthly_pnl,
            'pnl_distribution': {
                'mean': np.mean(pnl_values),
                'std': np.std(pnl_values),
                'min': min(pnl_values),
                'max': max(pnl_values),
                'median': np.median(pnl_values)
            },
            'trade_duration_stats': {
                'avg_days': np.mean([(t.exit_date - t.entry_date).days for t in completed_trades if t.exit_date]),
                'min_days': min([(t.exit_date - t.entry_date).days for t in completed_trades if t.exit_date]),
                'max_days': max([(t.exit_date - t.entry_date).days for t in completed_trades if t.exit_date])
            }
        }
