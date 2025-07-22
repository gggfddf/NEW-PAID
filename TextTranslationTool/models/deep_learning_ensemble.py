"""
Multi-Architecture Deep Learning Prediction System
Ensemble of neural networks specifically designed for market prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
from datetime import datetime

from config import Config

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM Network for sequential pattern recognition"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # Use the last time step
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc_layers(last_output)
        
        return output

class CNNModel(nn.Module):
    """CNN for visual chart pattern recognition"""
    
    def __init__(self, input_channels: int = 1, sequence_length: int = 60, 
                 num_features: int = 5, output_size: int = 1):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # Reshape for CNN: (batch, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        conv_out = self.conv_layers(x)
        output = self.fc_layers(conv_out)
        
        return output

class TransformerModel(nn.Module):
    """Transformer model for long-range dependencies"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, output_size: int = 1, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Output layers
        output = self.output_layers(pooled)
        
        return output

class AutoEncoder(nn.Module):
    """Autoencoder for anomaly detection and feature learning"""
    
    def __init__(self, input_size: int, encoding_dim: int = 32):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size),
            nn.Sigmoid()
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        anomaly_score = self.anomaly_head(encoded)
        
        return decoded, anomaly_score, encoded

class DeepLearningEnsemble:
    """Advanced ensemble of neural networks for market prediction"""
    
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = {}
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.3,
            'cnn': 0.2,
            'transformer': 0.25,
            'autoencoder': 0.25
        }
        
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
            targets.append(data[i, 0])  # Predict close price (first feature)
        
        return np.array(sequences), np.array(targets)
    
    def _prepare_cnn_data(self, sequences: np.ndarray) -> np.ndarray:
        """Prepare data for CNN input (2D representation)"""
        
        # Convert sequences to 2D images
        cnn_data = []
        
        for seq in sequences:
            # Reshape sequence to create a "chart" image
            # Using OHLCV data to create candlestick-like representation
            if seq.shape[1] >= 5:  # Ensure we have at least OHLCV
                # Create a visual representation
                height = min(seq.shape[1], 20)  # Limit height
                width = seq.shape[0]  # Time dimension
                
                # Normalize to [0, 1] for better CNN performance
                seq_normalized = (seq - seq.min()) / (seq.max() - seq.min() + 1e-8)
                
                # Resize if necessary
                if height < 20:
                    # Pad with zeros
                    padded = np.zeros((width, 20))
                    padded[:, :height] = seq_normalized.T
                    cnn_data.append(padded)
                else:
                    cnn_data.append(seq_normalized.T[:20])
            else:
                # Fallback for insufficient features
                cnn_data.append(np.zeros((seq.shape[0], 20)))
        
        return np.array(cnn_data)
    
    def prepare_features(self, data: Dict[str, pd.DataFrame], 
                        indicators: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Prepare feature matrix from all components"""
        
        logger.info("Preparing features for ML models...")
        
        # Use daily data as primary timeframe
        primary_timeframe = '1d'
        if primary_timeframe not in data:
            primary_timeframe = list(data.keys())[0]
        
        df_price = data[primary_timeframe]
        df_indicators = indicators.get(primary_timeframe, pd.DataFrame())
        
        # Align dataframes by index
        if not df_indicators.empty:
            df_combined = df_price.join(df_indicators, how='inner')
        else:
            df_combined = df_price.copy()
        
        # Select relevant features
        feature_columns = []
        
        # Price features
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns.extend([col for col in price_features if col in df_combined.columns])
        
        # Technical indicator features
        if not df_indicators.empty:
            indicator_features = [
                'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'SMA_20', 'EMA_12',
                'ADX', 'ATR', 'OBV', 'Stoch_K', 'CCI', 'Williams_R', 'ROC',
                'Confluence_Score', 'Vol_Adj_Momentum', 'Breakout_Probability'
            ]
            feature_columns.extend([col for col in indicator_features if col in df_combined.columns])
        
        # Extract feature matrix
        if not feature_columns:
            logger.warning("No valid features found, using basic OHLCV")
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        feature_matrix = df_combined[feature_columns].values
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        logger.info(f"Prepared feature matrix: {feature_matrix.shape}")
        return feature_matrix
    
    def _calculate_feature_importance(self, model_name: str, features: np.ndarray, 
                                    targets: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using permutation importance"""
        
        try:
            model = self.models[model_name]
            model.eval()
            
            # Baseline performance
            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)
                if model_name == 'lstm' or model_name == 'transformer':
                    # Use sequences for LSTM/Transformer
                    sequences, seq_targets = self._create_sequences(
                        np.column_stack([targets.reshape(-1, 1), features]), 
                        self.config.LSTM_LOOKBACK
                    )
                    if len(sequences) > 0:
                        X_tensor = torch.FloatTensor(sequences[:, :, 1:]).to(self.device)
                        baseline_pred = model(X_tensor).cpu().numpy()
                        baseline_mse = mean_squared_error(seq_targets, baseline_pred.flatten())
                    else:
                        return {}
                elif model_name == 'cnn':
                    sequences, seq_targets = self._create_sequences(
                        np.column_stack([targets.reshape(-1, 1), features]), 
                        self.config.LSTM_LOOKBACK
                    )
                    if len(sequences) > 0:
                        cnn_data = self._prepare_cnn_data(sequences[:, :, 1:])
                        X_tensor = torch.FloatTensor(cnn_data).to(self.device)
                        baseline_pred = model(X_tensor).cpu().numpy()
                        baseline_mse = mean_squared_error(seq_targets, baseline_pred.flatten())
                    else:
                        return {}
                else:  # autoencoder
                    baseline_pred = model(X_tensor)[1].cpu().numpy()  # anomaly scores
                    baseline_mse = mean_squared_error(targets, baseline_pred.flatten())
            
            # Permutation importance
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            importance_dict = {}
            
            for i in range(min(features.shape[1], 10)):  # Limit to top 10 features
                # Permute feature i
                features_permuted = features.copy()
                np.random.shuffle(features_permuted[:, i])
                
                with torch.no_grad():
                    if model_name == 'lstm' or model_name == 'transformer':
                        sequences_perm, _ = self._create_sequences(
                            np.column_stack([targets.reshape(-1, 1), features_permuted]), 
                            self.config.LSTM_LOOKBACK
                        )
                        if len(sequences_perm) > 0:
                            X_perm = torch.FloatTensor(sequences_perm[:, :, 1:]).to(self.device)
                            perm_pred = model(X_perm).cpu().numpy()
                            perm_mse = mean_squared_error(seq_targets, perm_pred.flatten())
                        else:
                            continue
                    elif model_name == 'cnn':
                        sequences_perm, _ = self._create_sequences(
                            np.column_stack([targets.reshape(-1, 1), features_permuted]), 
                            self.config.LSTM_LOOKBACK
                        )
                        if len(sequences_perm) > 0:
                            cnn_data_perm = self._prepare_cnn_data(sequences_perm[:, :, 1:])
                            X_perm = torch.FloatTensor(cnn_data_perm).to(self.device)
                            perm_pred = model(X_perm).cpu().numpy()
                            perm_mse = mean_squared_error(seq_targets, perm_pred.flatten())
                        else:
                            continue
                    else:
                        X_perm = torch.FloatTensor(features_permuted).to(self.device)
                        perm_pred = model(X_perm)[1].cpu().numpy()
                        perm_mse = mean_squared_error(targets, perm_pred.flatten())
                
                importance = perm_mse - baseline_mse
                importance_dict[feature_names[i]] = importance
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating feature importance for {model_name}: {e}")
            return {}
    
    def train_models(self, features: np.ndarray, targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        
        logger.info("Training deep learning ensemble...")
        
        if targets is None:
            # Create targets from price changes
            close_prices = features[:, 3]  # Assume Close is 4th column
            targets = np.diff(close_prices) / close_prices[:-1]  # Price returns
            features = features[:-1]  # Align features
        
        if len(features) < self.config.MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient data for training: {len(features)} samples")
            return {'status': 'failed', 'reason': 'insufficient_data'}
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        features_scaled = self.scalers['features'].fit_transform(features)
        
        self.scalers['targets'] = StandardScaler()
        targets_scaled = self.scalers['targets'].fit_transform(targets.reshape(-1, 1)).flatten()
        
        training_results = {}
        
        try:
            # 1. Train LSTM Model
            logger.info("Training LSTM model...")
            lstm_sequences, lstm_targets = self._create_sequences(
                np.column_stack([targets_scaled.reshape(-1, 1), features_scaled]), 
                self.config.LSTM_LOOKBACK
            )
            
            if len(lstm_sequences) > 0:
                self.models['lstm'] = LSTMModel(
                    input_size=features.shape[1],
                    hidden_size=128,
                    num_layers=2
                ).to(self.device)
                
                X_lstm = torch.FloatTensor(lstm_sequences[:, :, 1:]).to(self.device)
                y_lstm = torch.FloatTensor(lstm_targets).to(self.device)
                
                optimizer = optim.Adam(self.models['lstm'].parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                self.models['lstm'].train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    outputs = self.models['lstm'](X_lstm)
                    loss = criterion(outputs.flatten(), y_lstm)
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 10 == 0:
                        logger.info(f"LSTM Epoch {epoch}, Loss: {loss.item():.6f}")
                
                training_results['lstm'] = {'final_loss': loss.item()}
            
            # 2. Train CNN Model
            logger.info("Training CNN model...")
            if len(lstm_sequences) > 0:  # Reuse sequences
                self.models['cnn'] = CNNModel(
                    sequence_length=self.config.LSTM_LOOKBACK,
                    num_features=features.shape[1]
                ).to(self.device)
                
                cnn_data = self._prepare_cnn_data(lstm_sequences[:, :, 1:])
                X_cnn = torch.FloatTensor(cnn_data).to(self.device)
                
                optimizer = optim.Adam(self.models['cnn'].parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                self.models['cnn'].train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    outputs = self.models['cnn'](X_cnn)
                    loss = criterion(outputs.flatten(), y_lstm)
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 10 == 0:
                        logger.info(f"CNN Epoch {epoch}, Loss: {loss.item():.6f}")
                
                training_results['cnn'] = {'final_loss': loss.item()}
            
            # 3. Train Transformer Model
            logger.info("Training Transformer model...")
            if len(lstm_sequences) > 0:
                self.models['transformer'] = TransformerModel(
                    input_size=features.shape[1],
                    d_model=128,
                    nhead=8,
                    num_layers=4
                ).to(self.device)
                
                optimizer = optim.Adam(self.models['transformer'].parameters(), lr=0.0005)
                criterion = nn.MSELoss()
                
                self.models['transformer'].train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    outputs = self.models['transformer'](X_lstm)
                    loss = criterion(outputs.flatten(), y_lstm)
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 10 == 0:
                        logger.info(f"Transformer Epoch {epoch}, Loss: {loss.item():.6f}")
                
                training_results['transformer'] = {'final_loss': loss.item()}
            
            # 4. Train AutoEncoder
            logger.info("Training AutoEncoder model...")
            self.models['autoencoder'] = AutoEncoder(
                input_size=features.shape[1],
                encoding_dim=32
            ).to(self.device)
            
            X_ae = torch.FloatTensor(features_scaled).to(self.device)
            
            optimizer = optim.Adam(self.models['autoencoder'].parameters(), lr=0.001)
            
            self.models['autoencoder'].train()
            for epoch in range(50):
                optimizer.zero_grad()
                decoded, anomaly_scores, encoded = self.models['autoencoder'](X_ae)
                
                # Reconstruction loss
                recon_loss = nn.MSELoss()(decoded, X_ae)
                
                # Anomaly detection loss (predict if next price change is significant)
                anomaly_targets = (np.abs(targets_scaled) > np.std(targets_scaled)).astype(float)
                anomaly_targets_tensor = torch.FloatTensor(anomaly_targets).to(self.device)
                anomaly_loss = nn.BCELoss()(anomaly_scores.flatten(), anomaly_targets_tensor)
                
                total_loss = recon_loss + 0.5 * anomaly_loss
                total_loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logger.info(f"AutoEncoder Epoch {epoch}, Total Loss: {total_loss.item():.6f}")
            
            training_results['autoencoder'] = {'final_loss': total_loss.item()}
            
            # Calculate feature importance for each model
            for model_name in self.models.keys():
                self.feature_importance[model_name] = self._calculate_feature_importance(
                    model_name, features_scaled, targets_scaled
                )
            
            logger.info("Deep learning ensemble training completed successfully")
            
            return {
                'status': 'success',
                'models_trained': list(self.models.keys()),
                'training_results': training_results,
                'data_shape': features.shape
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate predictions using ensemble methods"""
        
        logger.info("Generating ensemble predictions...")
        
        if not self.models:
            logger.warning("No trained models available for prediction")
            return self._generate_fallback_prediction(features)
        
        # Scale features
        if 'features' in self.scalers:
            features_scaled = self.scalers['features'].transform(features)
        else:
            logger.warning("No feature scaler available, using raw features")
            features_scaled = features
        
        predictions = {}
        confidences = {}
        
        try:
            current_price = features[-1, 3] if features.shape[1] > 3 else 100  # Default fallback
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                model.eval()
                
                try:
                    with torch.no_grad():
                        if model_name == 'lstm' or model_name == 'transformer':
                            if len(features_scaled) >= self.config.LSTM_LOOKBACK:
                                sequence = features_scaled[-self.config.LSTM_LOOKBACK:]
                                X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                                pred = model(X_tensor).cpu().item()
                            else:
                                continue
                                
                        elif model_name == 'cnn':
                            if len(features_scaled) >= self.config.LSTM_LOOKBACK:
                                sequence = features_scaled[-self.config.LSTM_LOOKBACK:]
                                cnn_data = self._prepare_cnn_data(sequence.reshape(1, *sequence.shape))
                                X_tensor = torch.FloatTensor(cnn_data).to(self.device)
                                pred = model(X_tensor).cpu().item()
                            else:
                                continue
                                
                        elif model_name == 'autoencoder':
                            latest_features = features_scaled[-1:] 
                            X_tensor = torch.FloatTensor(latest_features).to(self.device)
                            _, anomaly_score, _ = model(X_tensor)
                            pred = anomaly_score.cpu().item()
                        
                        # Transform prediction back to original scale if scaler exists
                        if 'targets' in self.scalers and model_name != 'autoencoder':
                            pred_scaled = np.array([[pred]])
                            pred = self.scalers['targets'].inverse_transform(pred_scaled)[0, 0]
                        
                        predictions[model_name] = pred
                        confidences[model_name] = min(1.0, 1.0 / (1.0 + abs(pred)))
                        
                except Exception as e:
                    logger.error(f"Error in {model_name} prediction: {e}")
                    continue
            
            if not predictions:
                return self._generate_fallback_prediction(features)
            
            # Ensemble prediction
            weighted_pred = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                if model_name != 'autoencoder':  # Exclude anomaly scores from price prediction
                    weight = self.model_weights.get(model_name, 0.25) * confidences.get(model_name, 0.5)
                    weighted_pred += pred * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction = weighted_pred / total_weight
            else:
                ensemble_prediction = np.mean(list(predictions.values()))
            
            # Calculate target price
            target_price = current_price * (1 + ensemble_prediction)
            
            # Determine direction and confidence
            direction = "UP" if ensemble_prediction > 0 else "DOWN"
            direction_confidence = abs(ensemble_prediction) * 100
            direction_confidence = min(95, max(5, direction_confidence))  # Clamp between 5-95%
            
            # Risk assessment
            volatility = np.std(list(predictions.values())) if len(predictions) > 1 else 0.02
            anomaly_score = predictions.get('autoencoder', 0.5)
            
            if volatility > 0.05 or anomaly_score > 0.7:
                risk_level = "HIGH"
            elif volatility > 0.02 or anomaly_score > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Time horizons
            time_horizons = {
                '1_day': {'probability': min(95, direction_confidence)},
                '3_days': {'probability': max(5, direction_confidence * 0.8)},
                '1_week': {'probability': max(5, direction_confidence * 0.6)}
            }
            
            # Feature importance (aggregated)
            agg_feature_importance = {}
            for model_importance in self.feature_importance.values():
                for feature, importance in model_importance.items():
                    if feature in agg_feature_importance:
                        agg_feature_importance[feature] += importance
                    else:
                        agg_feature_importance[feature] = importance
            
            # Risk metrics
            stop_loss_pct = min(5.0, max(1.0, volatility * 100))
            stop_loss = current_price * (1 - stop_loss_pct / 100) if direction == "UP" else current_price * (1 + stop_loss_pct / 100)
            
            risk_metrics = {
                'stop_loss': stop_loss,
                'position_size': min(20, max(1, 100 / (volatility * 100 + 1))),
                'max_risk': stop_loss_pct
            }
            
            result = {
                'direction': direction,
                'direction_confidence': direction_confidence,
                'target_price': target_price,
                'current_price': current_price,
                'expected_return': ensemble_prediction * 100,
                'risk_level': risk_level,
                'time_horizons': time_horizons,
                'feature_importance': dict(sorted(agg_feature_importance.items(), 
                                                key=lambda x: abs(x[1]), reverse=True)[:10]),
                'risk_metrics': risk_metrics,
                'model_predictions': predictions,
                'model_confidences': confidences,
                'ensemble_confidence': np.mean(list(confidences.values())),
                'volatility_estimate': volatility,
                'anomaly_score': anomaly_score
            }
            
            logger.info(f"Ensemble prediction completed: {direction} with {direction_confidence:.1f}% confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error during ensemble prediction: {e}")
            return self._generate_fallback_prediction(features)
    
    def _generate_fallback_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate basic prediction when models are not available"""
        
        current_price = features[-1, 3] if features.shape[1] > 3 else 100
        
        # Simple technical analysis fallback
        if len(features) >= 5:
            recent_prices = features[-5:, 3]  # Last 5 close prices
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            direction = "UP" if price_trend > 0 else "DOWN"
            confidence = min(75, abs(price_trend) * 1000)
            
            target_price = current_price * (1 + price_trend * 0.5)
        else:
            direction = "HOLD"
            confidence = 10
            target_price = current_price
            price_trend = 0
        
        return {
            'direction': direction,
            'direction_confidence': confidence,
            'target_price': target_price,
            'current_price': current_price,
            'expected_return': price_trend * 100,
            'risk_level': "MEDIUM",
            'time_horizons': {
                '1_day': {'probability': confidence},
                '3_days': {'probability': confidence * 0.8},
                '1_week': {'probability': confidence * 0.6}
            },
            'feature_importance': {'price_trend': 1.0},
            'risk_metrics': {
                'stop_loss': current_price * 0.98,
                'position_size': 5,
                'max_risk': 2
            },
            'model_predictions': {'fallback': price_trend},
            'model_confidences': {'fallback': confidence / 100},
            'ensemble_confidence': confidence / 100,
            'volatility_estimate': 0.02,
            'anomaly_score': 0.5,
            'note': 'Fallback prediction - models not trained'
        }
    
    def save_models(self, filepath: str = 'models/ensemble_models.pkl'):
        """Save trained models and scalers"""
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'models_state': {name: model.state_dict() for name, model in self.models.items()},
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'model_weights': self.model_weights,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str = 'models/ensemble_models.pkl'):
        """Load trained models and scalers"""
        
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore scalers and other data
            self.scalers = model_data.get('scalers', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.model_weights = model_data.get('model_weights', self.model_weights)
            self.training_history = model_data.get('training_history', {})
            
            # Recreate and load models
            models_state = model_data.get('models_state', {})
            
            for model_name, state_dict in models_state.items():
                try:
                    if model_name == 'lstm':
                        self.models[model_name] = LSTMModel(input_size=len(self.scalers.get('features', {}).get('mean_', [10])))
                    elif model_name == 'cnn':
                        self.models[model_name] = CNNModel()
                    elif model_name == 'transformer':
                        self.models[model_name] = TransformerModel(input_size=len(self.scalers.get('features', {}).get('mean_', [10])))
                    elif model_name == 'autoencoder':
                        self.models[model_name] = AutoEncoder(input_size=len(self.scalers.get('features', {}).get('mean_', [10])))
                    
                    self.models[model_name].load_state_dict(state_dict)
                    self.models[model_name].to(self.device)
                    self.models[model_name].eval()
                    
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
                    continue
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
