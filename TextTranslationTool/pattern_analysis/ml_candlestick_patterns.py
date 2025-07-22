"""
ML-Powered Candlestick Pattern Discovery Engine
Discovers NEW proprietary patterns using unsupervised machine learning
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class MLCandlestickDiscovery:
    """Advanced ML system for discovering proprietary candlestick patterns"""
    
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.patterns_database = {}
        self.pattern_counter = 0
        
    def _extract_candlestick_features(self, df: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Extract comprehensive features from candlestick sequences"""
        
        features_list = []
        
        for i in range(sequence_length, len(df)):
            sequence = df.iloc[i-sequence_length:i]
            feature_vector = []
            
            # Basic OHLC features
            opens = sequence['Open'].values
            highs = sequence['High'].values
            lows = sequence['Low'].values
            closes = sequence['Close'].values
            volumes = sequence['Volume'].values
            
            # 1. Body and shadow ratios
            bodies = np.abs(closes - opens)
            total_ranges = highs - lows
            body_ratios = np.divide(bodies, total_ranges, out=np.zeros_like(bodies), where=total_ranges!=0)
            
            upper_shadows = np.maximum(highs - np.maximum(opens, closes), 0)
            lower_shadows = np.maximum(np.minimum(opens, closes) - lows, 0)
            
            upper_shadow_ratios = np.divide(upper_shadows, total_ranges, out=np.zeros_like(upper_shadows), where=total_ranges!=0)
            lower_shadow_ratios = np.divide(lower_shadows, total_ranges, out=np.zeros_like(lower_shadows), where=total_ranges!=0)
            
            # 2. Price momentum features
            price_changes = np.diff(closes) / closes[:-1]
            price_accelerations = np.diff(price_changes)
            
            # 3. Volume-price relationships
            volume_price_ratios = volumes / closes
            volume_changes = np.diff(volumes) / volumes[:-1]
            
            # 4. Relative positioning
            close_positions = (closes - lows) / (highs - lows)  # Where close is in the range
            open_positions = (opens - lows) / (highs - lows)   # Where open is in the range
            
            # 5. Statistical features
            volatility = np.std(closes) / np.mean(closes)
            skewness = stats.skew(closes)
            kurtosis = stats.kurtosis(closes)
            
            # 6. Pattern-specific features
            doji_score = 1 - np.mean(body_ratios)  # Higher for doji-like patterns
            hammer_score = np.mean(lower_shadow_ratios) - np.mean(upper_shadow_ratios)
            spinning_top_score = np.mean(upper_shadow_ratios + lower_shadow_ratios) - np.mean(body_ratios)
            
            # 7. Multi-candle relationships
            if sequence_length > 1:
                # Engulfing patterns
                engulfing_scores = []
                for j in range(1, len(bodies)):
                    if bodies[j] > bodies[j-1]:
                        engulfing_scores.append(bodies[j] / bodies[j-1])
                    else:
                        engulfing_scores.append(0)
                
                # Gap analysis
                gaps = []
                for j in range(1, len(sequence)):
                    gap = (opens[j] - closes[j-1]) / closes[j-1]
                    gaps.append(gap)
                
                # Trend consistency
                trend_consistency = np.corrcoef(range(len(closes)), closes)[0, 1]
            
            # Compile feature vector
            feature_vector.extend([
                # Basic statistics
                np.mean(body_ratios), np.std(body_ratios),
                np.mean(upper_shadow_ratios), np.std(upper_shadow_ratios),
                np.mean(lower_shadow_ratios), np.std(lower_shadow_ratios),
                
                # Price dynamics
                np.mean(price_changes), np.std(price_changes),
                np.mean(price_accelerations) if len(price_accelerations) > 0 else 0,
                
                # Volume features
                np.mean(volume_price_ratios), np.std(volume_price_ratios),
                np.mean(volume_changes) if len(volume_changes) > 0 else 0,
                
                # Position features
                np.mean(close_positions), np.std(close_positions),
                np.mean(open_positions), np.std(open_positions),
                
                # Statistical measures
                volatility, skewness, kurtosis,
                
                # Pattern scores
                doji_score, hammer_score, spinning_top_score,
            ])
            
            # Multi-candle features
            if sequence_length > 1:
                feature_vector.extend([
                    np.mean(engulfing_scores) if engulfing_scores else 0,
                    np.mean(gaps) if gaps else 0,
                    trend_consistency if not np.isnan(trend_consistency) else 0,
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # Sequence-specific features
            feature_vector.extend([
                closes[-1] / opens[0] - 1,  # Total return over sequence
                (highs.max() - lows.min()) / opens[0],  # Total range relative to start
                np.sum(volumes),  # Total volume
                sequence_length,  # Sequence length as feature
            ])
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def _find_optimal_clusters(self, features: np.ndarray, max_clusters: int = 20) -> int:
        """Find optimal number of clusters using multiple methods"""
        
        if len(features) < 10:
            return min(3, len(features))
        
        inertias = []
        silhouette_scores = []
        
        k_range = range(2, min(max_clusters + 1, len(features)))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                inertias.append(kmeans.inertia_)
                
                if len(set(cluster_labels)) > 1:
                    sil_score = silhouette_score(features, cluster_labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
                    
            except Exception as e:
                logger.warning(f"Error calculating clusters for k={k}: {e}")
                continue
        
        if not silhouette_scores:
            return 5  # Default fallback
        
        # Use elbow method for inertias and best silhouette score
        best_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        # Simple elbow detection
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_k = k_range[np.argmax(second_diffs) + 1] if len(second_diffs) > 0 else best_k_silhouette
        else:
            elbow_k = best_k_silhouette
        
        # Combine both methods
        optimal_k = int((best_k_silhouette + elbow_k) / 2)
        return max(2, min(optimal_k, max_clusters))
    
    def _cluster_patterns_kmeans(self, features: np.ndarray) -> Dict[str, Any]:
        """Discover patterns using K-Means clustering"""
        
        optimal_k = self._find_optimal_clusters(features)
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        patterns = {}
        
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) < 3:  # Skip very small clusters
                continue
            
            pattern_id = f"KMEANS_{self.pattern_counter}"
            self.pattern_counter += 1
            
            # Calculate pattern characteristics
            centroid = kmeans.cluster_centers_[cluster_id]
            intra_cluster_distance = np.mean(pdist(cluster_features))
            
            patterns[pattern_id] = {
                'type': 'kmeans',
                'cluster_id': cluster_id,
                'size': len(cluster_features),
                'centroid': centroid,
                'intra_distance': intra_cluster_distance,
                'indices': np.where(cluster_mask)[0].tolist(),
                'confidence': 1.0 / (1.0 + intra_cluster_distance)  # Lower distance = higher confidence
            }
        
        return patterns
    
    def _cluster_patterns_dbscan(self, features: np.ndarray) -> Dict[str, Any]:
        """Discover patterns using DBSCAN clustering"""
        
        # Determine optimal eps using k-distance graph
        distances = pdist(features)
        eps = np.percentile(distances, 10)  # Start with 10th percentile
        
        dbscan = DBSCAN(eps=eps, min_samples=max(3, len(features) // 20))
        cluster_labels = dbscan.fit_predict(features)
        
        patterns = {}
        
        unique_labels = set(cluster_labels)
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            
            pattern_id = f"DBSCAN_{self.pattern_counter}"
            self.pattern_counter += 1
            
            # Calculate pattern characteristics
            centroid = np.mean(cluster_features, axis=0)
            density = len(cluster_features) / (np.max(pdist(cluster_features)) + 1e-6)
            
            patterns[pattern_id] = {
                'type': 'dbscan',
                'cluster_id': cluster_id,
                'size': len(cluster_features),
                'centroid': centroid,
                'density': density,
                'indices': np.where(cluster_mask)[0].tolist(),
                'confidence': min(1.0, density / 10.0)  # Normalize density to confidence
            }
        
        return patterns
    
    def _cluster_patterns_gmm(self, features: np.ndarray) -> Dict[str, Any]:
        """Discover patterns using Gaussian Mixture Models"""
        
        optimal_k = self._find_optimal_clusters(features)
        
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        gmm.fit(features)
        cluster_labels = gmm.predict(features)
        probabilities = gmm.predict_proba(features)
        
        patterns = {}
        
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_probs = probabilities[cluster_mask, cluster_id]
            
            if len(cluster_features) < 3:
                continue
            
            pattern_id = f"GMM_{self.pattern_counter}"
            self.pattern_counter += 1
            
            # Calculate pattern characteristics
            mean_prob = np.mean(cluster_probs)
            covariance = gmm.covariances_[cluster_id]
            
            patterns[pattern_id] = {
                'type': 'gmm',
                'cluster_id': cluster_id,
                'size': len(cluster_features),
                'mean': gmm.means_[cluster_id],
                'covariance': covariance.tolist() if hasattr(covariance, 'tolist') else float(covariance),
                'weight': gmm.weights_[cluster_id],
                'indices': np.where(cluster_mask)[0].tolist(),
                'confidence': mean_prob
            }
        
        return patterns
    
    def _calculate_pattern_success_rates(self, patterns: Dict[str, Any], 
                                       df: pd.DataFrame, 
                                       sequence_length: int,
                                       forward_periods: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """Calculate success rates for discovered patterns"""
        
        enhanced_patterns = {}
        
        for pattern_id, pattern_data in patterns.items():
            indices = pattern_data['indices']
            
            success_rates = {}
            profitability = {}
            
            for periods in forward_periods:
                successful_predictions = 0
                total_predictions = 0
                total_returns = []
                
                for idx in indices:
                    actual_idx = idx + sequence_length
                    
                    # Ensure we have future data
                    if actual_idx + periods < len(df):
                        current_price = df.iloc[actual_idx]['Close']
                        future_price = df.iloc[actual_idx + periods]['Close']
                        
                        future_return = (future_price - current_price) / current_price
                        total_returns.append(future_return)
                        
                        # Consider it successful if return > 1%
                        if future_return > 0.01:
                            successful_predictions += 1
                        
                        total_predictions += 1
                
                if total_predictions > 0:
                    success_rates[f'{periods}d'] = (successful_predictions / total_predictions) * 100
                    profitability[f'{periods}d'] = {
                        'mean_return': np.mean(total_returns) * 100,
                        'std_return': np.std(total_returns) * 100,
                        'sharpe_ratio': np.mean(total_returns) / (np.std(total_returns) + 1e-6),
                        'win_rate': (successful_predictions / total_predictions) * 100
                    }
                else:
                    success_rates[f'{periods}d'] = 0
                    profitability[f'{periods}d'] = {
                        'mean_return': 0, 'std_return': 0, 'sharpe_ratio': 0, 'win_rate': 0
                    }
            
            # Calculate overall pattern score
            avg_success_rate = np.mean(list(success_rates.values()))
            avg_return = np.mean([p['mean_return'] for p in profitability.values()])
            avg_sharpe = np.mean([p['sharpe_ratio'] for p in profitability.values()])
            
            pattern_score = (avg_success_rate * 0.4 + 
                           (avg_return + 100) * 0.3 +  # Shift to make positive
                           (avg_sharpe + 2) * 50 * 0.3)  # Normalize sharpe
            
            enhanced_patterns[pattern_id] = {
                **pattern_data,
                'success_rates': success_rates,
                'profitability': profitability,
                'pattern_score': pattern_score,
                'statistical_significance': self._calculate_statistical_significance(
                    [p['mean_return'] for p in profitability.values()]
                )
            }
        
        return enhanced_patterns
    
    def _calculate_statistical_significance(self, returns: List[float]) -> Dict[str, float]:
        """Calculate statistical significance of pattern returns"""
        
        if len(returns) < 3:
            return {'p_value': 1.0, 't_statistic': 0.0, 'significant': False}
        
        # Test if returns are significantly different from zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        return {
            'p_value': p_value,
            't_statistic': t_stat,
            'significant': p_value < 0.05
        }
    
    def _generate_pattern_rules(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading rules for discovered patterns"""
        
        rules = {}
        
        for pattern_id, pattern_data in patterns.items():
            success_rates = pattern_data.get('success_rates', {})
            profitability = pattern_data.get('profitability', {})
            
            # Determine best timeframe
            best_period = max(success_rates.keys(), key=lambda k: success_rates[k])
            best_success_rate = success_rates[best_period]
            
            # Generate entry rule
            if best_success_rate > 60:
                entry_signal = "STRONG_BUY"
            elif best_success_rate > 50:
                entry_signal = "BUY"
            else:
                entry_signal = "HOLD"
            
            # Generate risk management rules
            pattern_profitability = profitability.get(best_period, {})
            
            stop_loss = abs(pattern_profitability.get('mean_return', 2)) + \
                       pattern_profitability.get('std_return', 2)
            
            take_profit = pattern_profitability.get('mean_return', 3) * 2
            
            rules[pattern_id] = {
                'entry_signal': entry_signal,
                'confidence': pattern_data.get('confidence', 0),
                'best_holding_period': best_period,
                'expected_return': pattern_profitability.get('mean_return', 0),
                'risk_reward_ratio': abs(take_profit / stop_loss) if stop_loss > 0 else 0,
                'stop_loss_pct': min(stop_loss, 5),  # Cap at 5%
                'take_profit_pct': min(take_profit, 15),  # Cap at 15%
                'min_occurrences': pattern_data.get('size', 0)
            }
        
        return rules
    
    def discover_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main method to discover patterns using multiple ML algorithms"""
        
        logger.info("Starting ML pattern discovery...")
        
        if len(df) < self.config.MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient data for pattern discovery: {len(df)} rows")
            return {'discovered_patterns': [], 'summary': 'Insufficient data'}
        
        all_patterns = {}
        
        # Try different sequence lengths
        for seq_length in self.config.PATTERN_SEQUENCE_LENGTHS:
            if len(df) < seq_length + 10:  # Need extra data for analysis
                continue
            
            logger.info(f"Analyzing sequences of length {seq_length}")
            
            try:
                # Extract features
                features = self._extract_candlestick_features(df, seq_length)
                
                if len(features) < 10:
                    continue
                
                # Scale features
                features_scaled = self.scaler.fit_transform(features)
                
                # Apply different clustering algorithms
                kmeans_patterns = self._cluster_patterns_kmeans(features_scaled)
                dbscan_patterns = self._cluster_patterns_dbscan(features_scaled)
                gmm_patterns = self._cluster_patterns_gmm(features_scaled)
                
                # Combine patterns
                sequence_patterns = {**kmeans_patterns, **dbscan_patterns, **gmm_patterns}
                
                # Calculate success rates
                sequence_patterns = self._calculate_pattern_success_rates(
                    sequence_patterns, df, seq_length
                )
                
                # Filter patterns by significance
                significant_patterns = {
                    k: v for k, v in sequence_patterns.items()
                    if (v.get('statistical_significance', {}).get('significant', False) or
                        v.get('pattern_score', 0) > 50)
                }
                
                all_patterns.update(significant_patterns)
                
                logger.info(f"Found {len(significant_patterns)} significant patterns for sequence length {seq_length}")
                
            except Exception as e:
                logger.error(f"Error processing sequence length {seq_length}: {e}")
                continue
        
        # Generate trading rules
        trading_rules = self._generate_pattern_rules(all_patterns)
        
        # Sort patterns by quality
        sorted_patterns = sorted(
            all_patterns.items(),
            key=lambda x: x[1].get('pattern_score', 0),
            reverse=True
        )
        
        # Create summary
        summary = {
            'total_patterns_discovered': len(all_patterns),
            'statistically_significant': len([p for p in all_patterns.values() 
                                            if p.get('statistical_significance', {}).get('significant', False)]),
            'high_quality_patterns': len([p for p in all_patterns.values() 
                                        if p.get('pattern_score', 0) > 70]),
            'avg_success_rate': np.mean([
                np.mean(list(p.get('success_rates', {}).values())) 
                for p in all_patterns.values()
            ]) if all_patterns else 0
        }
        
        logger.info(f"Pattern discovery complete. Found {len(all_patterns)} patterns")
        
        return {
            'discovered_patterns': [
                {
                    'id': pattern_id,
                    **pattern_data,
                    'trading_rules': trading_rules.get(pattern_id, {})
                }
                for pattern_id, pattern_data in sorted_patterns
            ],
            'summary': summary,
            'total_sequences_analyzed': sum(
                len(df) - seq_len for seq_len in self.config.PATTERN_SEQUENCE_LENGTHS
                if len(df) > seq_len
            )
        }
    
    def predict_pattern_breakout(self, df: pd.DataFrame, 
                               recent_periods: int = 20) -> Dict[str, Any]:
        """Predict potential breakouts based on discovered patterns"""
        
        if len(self.patterns_database) == 0:
            logger.warning("No patterns in database for breakout prediction")
            return {'breakout_probability': 0, 'confidence': 0}
        
        recent_data = df.tail(recent_periods)
        breakout_signals = []
        
        for seq_length in self.config.PATTERN_SEQUENCE_LENGTHS:
            if len(recent_data) < seq_length:
                continue
            
            # Extract features from recent data
            features = self._extract_candlestick_features(recent_data, seq_length)
            
            if len(features) == 0:
                continue
            
            latest_features = features[-1].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Compare with known patterns
            for pattern_id, pattern_data in self.patterns_database.items():
                if 'centroid' in pattern_data:
                    centroid = pattern_data['centroid'].reshape(1, -1)
                    distance = np.linalg.norm(latest_features_scaled - centroid)
                    
                    similarity = 1 / (1 + distance)
                    
                    if similarity > 0.7:  # High similarity threshold
                        breakout_prob = pattern_data.get('success_rates', {}).get('1d', 0) / 100
                        breakout_signals.append({
                            'pattern_id': pattern_id,
                            'similarity': similarity,
                            'breakout_probability': breakout_prob,
                            'expected_return': pattern_data.get('profitability', {}).get('1d', {}).get('mean_return', 0)
                        })
        
        if not breakout_signals:
            return {'breakout_probability': 0.5, 'confidence': 0.1, 'signals': []}
        
        # Aggregate signals
        avg_breakout_prob = np.mean([s['breakout_probability'] for s in breakout_signals])
        avg_similarity = np.mean([s['similarity'] for s in breakout_signals])
        avg_expected_return = np.mean([s['expected_return'] for s in breakout_signals])
        
        return {
            'breakout_probability': avg_breakout_prob,
            'confidence': avg_similarity,
            'expected_return': avg_expected_return,
            'num_matching_patterns': len(breakout_signals),
            'signals': breakout_signals
        }
