"""
Portfolio Optimization using Modern Portfolio Theory
Implements efficient frontier, risk parity, and other optimization strategies
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import asyncio

from config import Config
from data.live_data_loader import DataLoader

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Advanced portfolio optimization using modern portfolio theory"""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        
    async def _fetch_portfolio_data(self, stocks: List[Dict[str, Any]], 
                                  lookback_days: int = 252) -> pd.DataFrame:
        """Fetch historical data for portfolio stocks"""
        
        all_data = {}
        
        for stock_info in stocks:
            symbol = stock_info['symbol']
            try:
                # Fetch data for the stock
                stock_data = await self.data_loader.fetch_multi_timeframe_data(
                    symbol=symbol,
                    timeframes=['1d'],
                    lookback_days=lookback_days
                )
                
                if '1d' in stock_data:
                    all_data[symbol] = stock_data['1d']['Close']
                    logger.info(f"Fetched data for {symbol}: {len(stock_data['1d'])} days")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid stock data fetched")
        
        # Combine into single DataFrame
        price_df = pd.DataFrame(all_data)
        
        # Handle missing values
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        
        return price_df
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns"""
        returns = price_data.pct_change().dropna()
        return returns
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> Tuple[float, float]:
        """Calculate portfolio expected return and volatility"""
        
        # Expected returns (annualized)
        expected_returns = returns.mean() * 252
        
        # Covariance matrix (annualized)
        cov_matrix = returns.cov() * 252
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return portfolio_return, portfolio_volatility
    
    def _sharpe_objective(self, weights: np.ndarray, returns: pd.DataFrame, 
                         risk_free_rate: float = 0.06) -> float:
        """Objective function for Sharpe ratio maximization (negative for minimization)"""
        
        portfolio_return, portfolio_volatility = self._calculate_portfolio_metrics(weights, returns)
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return -sharpe_ratio  # Negative for minimization
    
    def _volatility_objective(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Objective function for minimum volatility"""
        
        _, portfolio_volatility = self._calculate_portfolio_metrics(weights, returns)
        
        return portfolio_volatility
    
    def _return_objective(self, weights: np.ndarray, returns: pd.DataFrame, 
                         target_return: float) -> float:
        """Objective function for targeting specific return"""
        
        portfolio_return, portfolio_volatility = self._calculate_portfolio_metrics(weights, returns)
        
        # Penalty for not meeting target return
        return_penalty = 1000 * abs(portfolio_return - target_return)
        
        return portfolio_volatility + return_penalty
    
    def _risk_parity_objective(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Objective function for risk parity"""
        
        cov_matrix = returns.cov() * 252
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Marginal risk contributions
        marginal_contribs = np.dot(cov_matrix, weights)
        
        # Risk contributions
        risk_contribs = weights * marginal_contribs / portfolio_var
        
        # Target equal risk contribution
        n_assets = len(weights)
        target_contrib = 1.0 / n_assets
        
        # Sum of squared deviations from equal risk
        risk_parity_error = np.sum((risk_contribs - target_contrib) ** 2)
        
        return risk_parity_error
    
    def _create_constraints_and_bounds(self, n_assets: int, 
                                     min_weight: float = 0.01,
                                     max_weight: float = 0.30) -> Tuple[List[Dict], List[Tuple]]:
        """Create optimization constraints and bounds"""
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds for each weight
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        return constraints, bounds
    
    def optimize_sharpe(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio for maximum Sharpe ratio"""
        
        n_assets = len(returns.columns)
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints and bounds
        constraints, bounds = self._create_constraints_and_bounds(n_assets)
        
        # Optimization
        result = minimize(
            fun=self._sharpe_objective,
            x0=initial_weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility = self._calculate_portfolio_metrics(optimal_weights, returns)
            sharpe_ratio = (portfolio_return - self.config.RISK_FREE_RATE) / portfolio_volatility
            
            return {
                'success': True,
                'weights': dict(zip(returns.columns, optimal_weights)),
                'expected_return': portfolio_return * 100,
                'volatility': portfolio_volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'optimization_method': 'Maximum Sharpe Ratio'
            }
        else:
            return {
                'success': False,
                'error': result.message
            }
    
    def optimize_min_volatility(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio for minimum volatility"""
        
        n_assets = len(returns.columns)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        constraints, bounds = self._create_constraints_and_bounds(n_assets)
        
        result = minimize(
            fun=self._volatility_objective,
            x0=initial_weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility = self._calculate_portfolio_metrics(optimal_weights, returns)
            sharpe_ratio = (portfolio_return - self.config.RISK_FREE_RATE) / portfolio_volatility
            
            return {
                'success': True,
                'weights': dict(zip(returns.columns, optimal_weights)),
                'expected_return': portfolio_return * 100,
                'volatility': portfolio_volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'optimization_method': 'Minimum Volatility'
            }
        else:
            return {
                'success': False,
                'error': result.message
            }
    
    def optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio for risk parity"""
        
        n_assets = len(returns.columns)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        constraints, bounds = self._create_constraints_and_bounds(n_assets)
        
        result = minimize(
            fun=self._risk_parity_objective,
            x0=initial_weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility = self._calculate_portfolio_metrics(optimal_weights, returns)
            sharpe_ratio = (portfolio_return - self.config.RISK_FREE_RATE) / portfolio_volatility
            
            return {
                'success': True,
                'weights': dict(zip(returns.columns, optimal_weights)),
                'expected_return': portfolio_return * 100,
                'volatility': portfolio_volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'optimization_method': 'Risk Parity'
            }
        else:
            return {
                'success': False,
                'error': result.message
            }
    
    def generate_efficient_frontier(self, returns: pd.DataFrame, 
                                  n_portfolios: int = 100) -> Dict[str, Any]:
        """Generate efficient frontier"""
        
        # Calculate return range
        min_return = returns.mean().min() * 252
        max_return = returns.mean().max() * 252
        
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            n_assets = len(returns.columns)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            constraints, bounds = self._create_constraints_and_bounds(n_assets)
            
            result = minimize(
                fun=self._return_objective,
                x0=initial_weights,
                args=(returns, target_return),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return, portfolio_volatility = self._calculate_portfolio_metrics(optimal_weights, returns)
                sharpe_ratio = (portfolio_return - self.config.RISK_FREE_RATE) / portfolio_volatility
                
                efficient_portfolios.append({
                    'target_return': target_return * 100,
                    'expected_return': portfolio_return * 100,
                    'volatility': portfolio_volatility * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'weights': dict(zip(returns.columns, optimal_weights))
                })
        
        return {
            'efficient_portfolios': efficient_portfolios,
            'n_portfolios': len(efficient_portfolios)
        }
    
    def calculate_var_cvar(self, weights: Dict[str, float], returns: pd.DataFrame,
                          confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        
        weights_array = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns.dot(weights_array)
        
        # Value at Risk
        var = np.percentile(portfolio_returns, confidence_level * 100)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return {
            'var_95': var * 100,  # Convert to percentage
            'cvar_95': cvar * 100,
            'var_99': np.percentile(portfolio_returns, 1) * 100,
            'cvar_99': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean() * 100
        }
    
    def optimize(self, stocks: List[Dict[str, Any]], 
                method: str = 'sharpe') -> Dict[str, Any]:
        """Main optimization method"""
        
        logger.info(f"Starting portfolio optimization with method: {method}")
        
        try:
            # Fetch historical data
            price_data = asyncio.run(self._fetch_portfolio_data(stocks))
            
            if price_data.empty:
                return {
                    'success': False,
                    'error': 'No price data available'
                }
            
            # Calculate returns
            returns = self._calculate_returns(price_data)
            
            if len(returns) < 30:  # Minimum data requirement
                return {
                    'success': False,
                    'error': 'Insufficient historical data for optimization'
                }
            
            # Choose optimization method
            if method == 'sharpe':
                result = self.optimize_sharpe(returns)
            elif method == 'min_vol':
                result = self.optimize_min_volatility(returns)
            elif method == 'risk_parity':
                result = self.optimize_risk_parity(returns)
            else:
                # Default to Sharpe ratio
                result = self.optimize_sharpe(returns)
            
            if result.get('success'):
                # Add additional metrics
                weights = result['weights']
                
                # Risk metrics
                risk_metrics = self.calculate_var_cvar(weights, returns)
                result.update(risk_metrics)
                
                # Correlation analysis
                correlation_matrix = returns.corr()
                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                result['avg_correlation'] = avg_correlation
                
                # Diversification ratio
                individual_vols = returns.std() * np.sqrt(252)
                weights_array = np.array([weights.get(col, 0) for col in returns.columns])
                weighted_avg_vol = np.dot(weights_array, individual_vols)
                diversification_ratio = weighted_avg_vol / result['volatility'] * 100
                result['diversification_ratio'] = diversification_ratio
                
                # Performance attribution
                contribution_to_return = {}
                for stock in returns.columns:
                    stock_weight = weights.get(stock, 0)
                    stock_return = returns[stock].mean() * 252 * 100
                    contribution_to_return[stock] = stock_weight * stock_return
                
                result['return_contribution'] = contribution_to_return
                
                logger.info(f"Portfolio optimization completed successfully")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def rebalance_portfolio(self, current_weights: Dict[str, float],
                           target_weights: Dict[str, float],
                           transaction_cost: float = 0.1) -> Dict[str, Any]:
        """Calculate rebalancing trades with transaction costs"""
        
        rebalancing_trades = {}
        total_transaction_cost = 0
        
        all_stocks = set(current_weights.keys()) | set(target_weights.keys())
        
        for stock in all_stocks:
            current_weight = current_weights.get(stock, 0)
            target_weight = target_weights.get(stock, 0)
            
            weight_change = target_weight - current_weight
            
            if abs(weight_change) > 0.01:  # Only rebalance if change > 1%
                rebalancing_trades[stock] = {
                    'current_weight': current_weight * 100,
                    'target_weight': target_weight * 100,
                    'weight_change': weight_change * 100,
                    'action': 'BUY' if weight_change > 0 else 'SELL',
                    'transaction_cost_pct': transaction_cost
                }
                
                total_transaction_cost += abs(weight_change) * transaction_cost
        
        # Net benefit calculation
        # This is simplified - in practice, you'd need expected returns to calculate properly
        estimated_benefit = 0  # Would need more sophisticated calculation
        
        return {
            'rebalancing_trades': rebalancing_trades,
            'total_transaction_cost': total_transaction_cost * 100,
            'number_of_trades': len(rebalancing_trades),
            'recommend_rebalance': total_transaction_cost < 0.5,  # Simple threshold
            'estimated_net_benefit': estimated_benefit
        }
    
    def monte_carlo_simulation(self, weights: Dict[str, float], 
                             returns: pd.DataFrame,
                             initial_portfolio_value: float = 100000,
                             time_horizon_days: int = 252,
                             n_simulations: int = 1000) -> Dict[str, Any]:
        """Monte Carlo simulation for portfolio performance"""
        
        weights_array = np.array([weights.get(col, 0) for col in returns.columns])
        
        # Calculate portfolio statistics
        portfolio_mean = np.dot(weights_array, returns.mean())
        portfolio_cov = np.dot(weights_array.T, np.dot(returns.cov(), weights_array))
        portfolio_std = np.sqrt(portfolio_cov)
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.normal(
            portfolio_mean, 
            portfolio_std, 
            (n_simulations, time_horizon_days)
        )
        
        # Calculate cumulative portfolio values
        portfolio_values = np.zeros((n_simulations, time_horizon_days + 1))
        portfolio_values[:, 0] = initial_portfolio_value
        
        for i in range(time_horizon_days):
            portfolio_values[:, i + 1] = portfolio_values[:, i] * (1 + random_returns[:, i])
        
        final_values = portfolio_values[:, -1]
        
        # Calculate statistics
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        value_percentiles = {
            f'percentile_{p}': np.percentile(final_values, p)
            for p in percentiles
        }
        
        # Probability of loss
        prob_loss = (final_values < initial_portfolio_value).mean() * 100
        
        # Average return
        avg_return = (final_values.mean() / initial_portfolio_value - 1) * 100
        
        return {
            'simulation_results': {
                'initial_value': initial_portfolio_value,
                'final_values_stats': {
                    'mean': final_values.mean(),
                    'std': final_values.std(),
                    'min': final_values.min(),
                    'max': final_values.max()
                },
                'value_percentiles': value_percentiles,
                'probability_of_loss': prob_loss,
                'average_return': avg_return,
                'time_horizon_days': time_horizon_days,
                'n_simulations': n_simulations
            }
        }
