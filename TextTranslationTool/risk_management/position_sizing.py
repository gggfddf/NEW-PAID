"""
Risk Management and Position Sizing
Implements various position sizing methods and risk controls
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class RiskManager:
    """Advanced risk management and position sizing system"""
    
    def __init__(self):
        self.config = Config()
        
    def calculate_position_size_fixed_percentage(self, portfolio_value: float, 
                                               risk_per_trade: float,
                                               entry_price: float,
                                               stop_loss: float) -> Dict[str, Any]:
        """Calculate position size using fixed percentage risk"""
        
        if stop_loss <= 0 or entry_price <= 0:
            return {'error': 'Invalid price parameters'}
        
        # Risk amount in currency
        risk_amount = portfolio_value * (risk_per_trade / 100)
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return {'error': 'No risk per share (entry price equals stop loss)'}
        
        # Position size in shares
        position_size_shares = int(risk_amount / risk_per_share)
        
        # Position value
        position_value = position_size_shares * entry_price
        
        # Position size as percentage of portfolio
        position_size_pct = (position_value / portfolio_value) * 100
        
        return {
            'position_size_shares': position_size_shares,
            'position_value': position_value,
            'position_size_pct': position_size_pct,
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'method': 'Fixed Percentage Risk'
        }
    
    def calculate_position_size_kelly_criterion(self, win_probability: float,
                                              avg_win: float, avg_loss: float,
                                              portfolio_value: float,
                                              max_kelly: float = 0.25) -> Dict[str, Any]:
        """Calculate position size using Kelly Criterion"""
        
        if win_probability <= 0 or win_probability >= 1:
            return {'error': 'Win probability must be between 0 and 1'}
        
        if avg_loss >= 0:
            return {'error': 'Average loss must be negative'}
        
        # Kelly formula: f = (bp - q) / b
        # where:
        # f = fraction of capital to wager
        # b = odds of winning (avg_win / |avg_loss|)
        # p = probability of winning
        # q = probability of losing (1 - p)
        
        b = avg_win / abs(avg_loss)  # Odds ratio
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap the Kelly fraction to prevent over-leveraging
        kelly_fraction = max(0, min(kelly_fraction, max_kelly))
        
        # Position value
        position_value = portfolio_value * kelly_fraction
        
        return {
            'kelly_fraction': kelly_fraction,
            'position_value': position_value,
            'position_size_pct': kelly_fraction * 100,
            'odds_ratio': b,
            'win_probability': p,
            'method': 'Kelly Criterion'
        }
    
    def calculate_position_size_volatility_adjusted(self, portfolio_value: float,
                                                  target_volatility: float,
                                                  asset_volatility: float,
                                                  correlation_adjustment: float = 1.0) -> Dict[str, Any]:
        """Calculate position size adjusted for volatility"""
        
        if asset_volatility <= 0:
            return {'error': 'Asset volatility must be positive'}
        
        # Volatility scaling factor
        volatility_scalar = target_volatility / asset_volatility
        
        # Adjust for correlation with existing positions
        correlation_scalar = 1.0 / correlation_adjustment
        
        # Base allocation (could be from other methods)
        base_allocation = 0.1  # 10% default
        
        # Adjusted position size
        adjusted_allocation = base_allocation * volatility_scalar * correlation_scalar
        
        # Cap the allocation
        max_allocation = 0.3  # 30% maximum
        adjusted_allocation = min(adjusted_allocation, max_allocation)
        
        position_value = portfolio_value * adjusted_allocation
        
        return {
            'position_value': position_value,
            'position_size_pct': adjusted_allocation * 100,
            'volatility_scalar': volatility_scalar,
            'correlation_scalar': correlation_scalar,
            'target_volatility': target_volatility,
            'asset_volatility': asset_volatility,
            'method': 'Volatility Adjusted'
        }
    
    def calculate_var_position_sizing(self, portfolio_value: float,
                                    var_limit: float,
                                    asset_var: float,
                                    confidence_level: float = 0.05) -> Dict[str, Any]:
        """Calculate position size based on Value at Risk limits"""
        
        if asset_var >= 0:
            return {'error': 'Asset VaR should be negative (loss)'}
        
        # Maximum position size that keeps portfolio VaR within limit
        max_var_amount = portfolio_value * (var_limit / 100)
        
        # Position size to achieve target VaR
        position_value = max_var_amount / abs(asset_var)
        
        # Cap position size
        max_position_value = portfolio_value * 0.5  # 50% maximum
        position_value = min(position_value, max_position_value)
        
        position_size_pct = (position_value / portfolio_value) * 100
        
        return {
            'position_value': position_value,
            'position_size_pct': position_size_pct,
            'var_limit': var_limit,
            'asset_var': asset_var,
            'confidence_level': confidence_level,
            'method': 'VaR-based Sizing'
        }
    
    def calculate_optimal_stop_loss(self, entry_price: float,
                                  direction: str,
                                  atr: float,
                                  volatility: float,
                                  support_resistance_levels: List[float] = None) -> Dict[str, Any]:
        """Calculate optimal stop loss levels using multiple methods"""
        
        stop_loss_methods = {}
        
        # 1. ATR-based stop loss
        atr_multiplier = 2.0  # 2x ATR
        if direction.upper() == 'LONG':
            atr_stop = entry_price - (atr * atr_multiplier)
        else:
            atr_stop = entry_price + (atr * atr_multiplier)
        
        stop_loss_methods['atr_stop'] = {
            'price': atr_stop,
            'distance_pct': abs(atr_stop - entry_price) / entry_price * 100
        }
        
        # 2. Volatility-based stop loss
        vol_multiplier = 2.5
        vol_stop_distance = entry_price * volatility * vol_multiplier
        
        if direction.upper() == 'LONG':
            vol_stop = entry_price - vol_stop_distance
        else:
            vol_stop = entry_price + vol_stop_distance
        
        stop_loss_methods['volatility_stop'] = {
            'price': vol_stop,
            'distance_pct': abs(vol_stop - entry_price) / entry_price * 100
        }
        
        # 3. Percentage-based stop loss
        pct_stop_distance = self.config.DEFAULT_STOP_LOSS_PCT / 100
        
        if direction.upper() == 'LONG':
            pct_stop = entry_price * (1 - pct_stop_distance)
        else:
            pct_stop = entry_price * (1 + pct_stop_distance)
        
        stop_loss_methods['percentage_stop'] = {
            'price': pct_stop,
            'distance_pct': pct_stop_distance * 100
        }
        
        # 4. Support/Resistance based stop loss
        if support_resistance_levels:
            if direction.upper() == 'LONG':
                # Find nearest support below entry price
                supports_below = [level for level in support_resistance_levels if level < entry_price]
                if supports_below:
                    sr_stop = max(supports_below) * 0.99  # Slightly below support
                    stop_loss_methods['support_resistance_stop'] = {
                        'price': sr_stop,
                        'distance_pct': abs(sr_stop - entry_price) / entry_price * 100
                    }
            else:
                # Find nearest resistance above entry price
                resistances_above = [level for level in support_resistance_levels if level > entry_price]
                if resistances_above:
                    sr_stop = min(resistances_above) * 1.01  # Slightly above resistance
                    stop_loss_methods['support_resistance_stop'] = {
                        'price': sr_stop,
                        'distance_pct': abs(sr_stop - entry_price) / entry_price * 100
                    }
        
        # Choose optimal stop loss (balance between too tight and too loose)
        optimal_method = 'atr_stop'  # Default
        optimal_distance = stop_loss_methods['atr_stop']['distance_pct']
        
        # Prefer stops between 1% and 5%
        for method, data in stop_loss_methods.items():
            distance = data['distance_pct']
            if 1.0 <= distance <= 5.0:
                if abs(distance - 2.5) < abs(optimal_distance - 2.5):  # Closer to 2.5%
                    optimal_method = method
                    optimal_distance = distance
        
        return {
            'all_methods': stop_loss_methods,
            'recommended': {
                'method': optimal_method,
                'price': stop_loss_methods[optimal_method]['price'],
                'distance_pct': stop_loss_methods[optimal_method]['distance_pct']
            }
        }
    
    def calculate_take_profit_levels(self, entry_price: float,
                                   stop_loss: float,
                                   direction: str,
                                   risk_reward_ratios: List[float] = [1.5, 2.0, 3.0]) -> Dict[str, Any]:
        """Calculate multiple take profit levels"""
        
        risk_amount = abs(entry_price - stop_loss)
        take_profit_levels = {}
        
        for ratio in risk_reward_ratios:
            profit_amount = risk_amount * ratio
            
            if direction.upper() == 'LONG':
                tp_price = entry_price + profit_amount
            else:
                tp_price = entry_price - profit_amount
            
            take_profit_levels[f'tp_{ratio}x'] = {
                'price': tp_price,
                'risk_reward_ratio': ratio,
                'profit_pct': (profit_amount / entry_price) * 100
            }
        
        return take_profit_levels
    
    def assess_portfolio_risk(self, positions: List[Dict[str, Any]], 
                            portfolio_value: float) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        
        if not positions:
            return {'status': 'no_positions'}
        
        total_position_value = sum([pos.get('position_value', 0) for pos in positions])
        total_risk_amount = sum([pos.get('risk_amount', 0) for pos in positions])
        
        # Portfolio metrics
        portfolio_utilization = (total_position_value / portfolio_value) * 100
        portfolio_risk_pct = (total_risk_amount / portfolio_value) * 100
        
        # Concentration risk
        largest_position = max([pos.get('position_value', 0) for pos in positions])
        concentration_risk = (largest_position / portfolio_value) * 100
        
        # Sector/correlation risk (simplified)
        num_positions = len(positions)
        diversification_score = min(100, (num_positions / 10) * 100)  # Simple scoring
        
        # Risk levels
        risk_level = 'LOW'
        risk_warnings = []
        
        if portfolio_risk_pct > 15:
            risk_level = 'HIGH'
            risk_warnings.append('Portfolio risk exceeds 15%')
        elif portfolio_risk_pct > 8:
            risk_level = 'MEDIUM'
            risk_warnings.append('Portfolio risk is elevated')
        
        if concentration_risk > 30:
            risk_warnings.append('High concentration risk in single position')
        
        if portfolio_utilization > 90:
            risk_warnings.append('High portfolio utilization - limited diversification capacity')
        
        if diversification_score < 50:
            risk_warnings.append('Low diversification - consider more positions')
        
        return {
            'portfolio_metrics': {
                'total_value': portfolio_value,
                'total_position_value': total_position_value,
                'portfolio_utilization_pct': portfolio_utilization,
                'total_risk_amount': total_risk_amount,
                'portfolio_risk_pct': portfolio_risk_pct
            },
            'risk_analysis': {
                'risk_level': risk_level,
                'concentration_risk_pct': concentration_risk,
                'diversification_score': diversification_score,
                'num_positions': num_positions,
                'largest_position_pct': concentration_risk
            },
            'warnings': risk_warnings,
            'recommendations': self._generate_risk_recommendations(
                portfolio_risk_pct, concentration_risk, diversification_score, portfolio_utilization
            )
        }
    
    def _generate_risk_recommendations(self, portfolio_risk_pct: float,
                                     concentration_risk: float,
                                     diversification_score: float,
                                     portfolio_utilization: float) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        if portfolio_risk_pct > 10:
            recommendations.append("Consider reducing position sizes to lower overall portfolio risk")
        
        if concentration_risk > 25:
            recommendations.append("Reduce concentration in largest position")
        
        if diversification_score < 60:
            recommendations.append("Increase diversification by adding more positions")
        
        if portfolio_utilization > 80:
            recommendations.append("Consider keeping some cash for new opportunities")
        
        if portfolio_utilization < 40:
            recommendations.append("Portfolio is under-utilized - consider increasing position sizes")
        
        if not recommendations:
            recommendations.append("Portfolio risk profile appears well-balanced")
        
        return recommendations
    
    def calculate_risk_metrics(self, portfolio_value: float,
                             risk_tolerance: str,
                             max_position_size: float,
                             stop_loss_pct: float) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for the portfolio"""
        
        # Risk tolerance mapping
        risk_tolerance_map = {
            'Conservative': {'max_risk_per_trade': 1.0, 'max_portfolio_risk': 5.0, 'max_positions': 15},
            'Moderate': {'max_risk_per_trade': 2.0, 'max_portfolio_risk': 8.0, 'max_positions': 12},
            'Aggressive': {'max_risk_per_trade': 3.0, 'max_portfolio_risk': 12.0, 'max_positions': 10}
        }
        
        risk_params = risk_tolerance_map.get(risk_tolerance, risk_tolerance_map['Moderate'])
        
        # Calculate maximum risk per trade
        max_risk_per_trade = portfolio_value * (risk_params['max_risk_per_trade'] / 100)
        
        # Calculate recommended position size
        # Assuming average stop loss distance
        avg_stop_distance = stop_loss_pct / 100
        recommended_position_size = max_risk_per_trade / avg_stop_distance
        
        # Cap position size by maximum percentage
        max_position_by_pct = portfolio_value * (max_position_size / 100)
        recommended_position_size = min(recommended_position_size, max_position_by_pct)
        
        # Kelly fraction estimation (simplified)
        # Assuming 55% win rate and 1.5:1 reward/risk ratio
        win_prob = 0.55
        avg_win = stop_loss_pct * 1.5
        avg_loss = -stop_loss_pct
        
        if avg_loss < 0:
            kelly_fraction = ((avg_win / abs(avg_loss)) * win_prob - (1 - win_prob)) / (avg_win / abs(avg_loss))
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.05  # Default 5%
        
        kelly_position_size = portfolio_value * kelly_fraction
        
        # Risk-adjusted metrics
        sharpe_assumption = 1.2  # Assumed Sharpe ratio
        sortino_assumption = 1.5  # Assumed Sortino ratio
        
        return {
            'max_risk_per_trade': max_risk_per_trade,
            'recommended_position_size': recommended_position_size,
            'max_position_by_percentage': max_position_by_pct,
            'kelly_fraction': kelly_fraction,
            'kelly_position_size': kelly_position_size,
            'risk_tolerance_level': risk_tolerance,
            'max_portfolio_risk_pct': risk_params['max_portfolio_risk'],
            'max_simultaneous_positions': risk_params['max_positions'],
            'estimated_annual_volatility': stop_loss_pct * 0.5 * 16,  # Rough estimation
            'risk_adjusted_metrics': {
                'assumed_sharpe_ratio': sharpe_assumption,
                'assumed_sortino_ratio': sortino_assumption,
                'risk_tolerance_score': {'Conservative': 3, 'Moderate': 6, 'Aggressive': 9}.get(risk_tolerance, 6)
            },
            'recommendations': {
                'position_sizing_method': 'Kelly Criterion' if kelly_fraction > 0.02 else 'Fixed Percentage',
                'suggested_stop_loss_range': f"{max(1, stop_loss_pct-1)}% - {min(5, stop_loss_pct+1)}%",
                'portfolio_heat': 'Low' if risk_params['max_portfolio_risk'] < 6 else 'Medium' if risk_params['max_portfolio_risk'] < 10 else 'High'
            }
        }
    
    def dynamic_position_sizing(self, market_conditions: Dict[str, Any],
                              volatility_regime: str,
                              portfolio_performance: Dict[str, Any]) -> Dict[str, float]:
        """Dynamically adjust position sizing based on market conditions"""
        
        base_size = 1.0  # Base multiplier
        
        # Market condition adjustments
        vix_level = market_conditions.get('vix', 20)  # Default VIX
        if vix_level > 30:
            volatility_adjustment = 0.7  # Reduce size in high volatility
        elif vix_level > 25:
            volatility_adjustment = 0.85
        elif vix_level < 15:
            volatility_adjustment = 1.2  # Increase size in low volatility
        else:
            volatility_adjustment = 1.0
        
        # Volatility regime adjustment
        regime_adjustment = {
            'LOW': 1.1,
            'NORMAL': 1.0,
            'HIGH': 0.8,
            'EXTREME': 0.5
        }.get(volatility_regime, 1.0)
        
        # Performance-based adjustment
        recent_performance = portfolio_performance.get('recent_return_pct', 0)
        if recent_performance > 10:
            performance_adjustment = 1.1  # Increase after good performance
        elif recent_performance < -10:
            performance_adjustment = 0.8  # Reduce after poor performance
        else:
            performance_adjustment = 1.0
        
        # Drawdown adjustment
        current_drawdown = portfolio_performance.get('current_drawdown_pct', 0)
        if current_drawdown > 15:
            drawdown_adjustment = 0.6
        elif current_drawdown > 10:
            drawdown_adjustment = 0.8
        else:
            drawdown_adjustment = 1.0
        
        # Combined adjustment
        final_multiplier = (base_size * volatility_adjustment * regime_adjustment * 
                           performance_adjustment * drawdown_adjustment)
        
        # Cap the multiplier
        final_multiplier = max(0.2, min(2.0, final_multiplier))
        
        return {
            'position_size_multiplier': final_multiplier,
            'components': {
                'volatility_adjustment': volatility_adjustment,
                'regime_adjustment': regime_adjustment,
                'performance_adjustment': performance_adjustment,
                'drawdown_adjustment': drawdown_adjustment
            },
            'market_conditions': {
                'vix_level': vix_level,
                'volatility_regime': volatility_regime,
                'recent_performance': recent_performance,
                'current_drawdown': current_drawdown
            }
        }
