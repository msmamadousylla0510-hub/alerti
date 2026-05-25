"""
LSTM Model for Flood Prediction using Time Series Data
Predicts floods based on meteorological time series (precipitation, etc.)
"""
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
import keras
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import sys
from typing import Dict, List

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config import LSTM_SEQUENCE_LENGTH, LSTM_FORECAST_DAYS, MODEL_DIR

# Entraînement synthétique / lstm_model.h5 : (30 jours × 5 features météo)
MODEL_FEATURE_NAMES = [
    'precipitation', 'temperature', 'humidity', 'pressure', 'soil_moisture',
]

class LSTMPredictor:
    """LSTM model for time series flood prediction"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
        
        self.model_path = model_path
        self.scaler_path = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = LSTM_SEQUENCE_LENGTH
        self.forecast_days = LSTM_FORECAST_DAYS
        self.feature_names = list(MODEL_FEATURE_NAMES)
        self.n_features = len(MODEL_FEATURE_NAMES)
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self._sync_features_from_model()
                self._compile_model()
                print(
                    f"LSTM model loaded successfully "
                    f"({self.sequence_length}×{self.n_features})"
                )
            except Exception as e:
                print(f"Error loading LSTM model: {e}. Creating new model...")
                self._create_model()
        else:
            print("Creating new LSTM model...")
            self._create_model()
    
    def _sync_features_from_model(self):
        """Aligne n_features sur le modèle .h5 chargé (ex. 5, pas 30)."""
        if self.model is None or not getattr(self.model, 'input_shape', None):
            return
        shape = self.model.input_shape
        if shape and len(shape) >= 3 and shape[-1]:
            self.n_features = int(shape[-1])
            self.feature_names = MODEL_FEATURE_NAMES[: self.n_features]

    def _create_model(self):
        """Create bidirectional LSTM model architecture"""
        input_shape = (self.sequence_length, self.n_features)
        
        inputs = keras.Input(shape=input_shape)
        
        # Bidirectional LSTM layers
        lstm1 = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(inputs)
        
        lstm2 = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False, dropout=0.2)
        )(lstm1)
        
        # Attention mechanism (simple implementation)
        attention = layers.Dense(32, activation='tanh')(lstm2)
        attention = layers.Dense(1, activation='softmax')(attention)
        
        # Dense layers for prediction
        dense1 = layers.Dense(64, activation='relu')(lstm2)
        dropout = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(32, activation='relu')(dropout)
        
        # Output: single flood probability value
        outputs = layers.Dense(1, activation='sigmoid')(dense2)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self._compile_model()
        
        print("LSTM model created")
    
    def _compile_model(self):
        """Compile model with default optimizer/loss/metrics."""
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae']
        )
    
    def _antecedent_precip(self, history: List[float], days: int) -> float:
        if not history:
            return 0.0
        return float(sum(history[-days:]))

    def _estimate_soil_moisture(self, precip_history: List[float]) -> float:
        """Proxy humidité sol (0-1) à partir des précipitations récentes."""
        total_14 = self._antecedent_precip(precip_history, 14)
        return float(min(1.0, 0.2 + total_14 / 80.0))

    def _build_meteo_row(
        self,
        precip: float,
        temp: float,
        humidity: float,
        pressure: float,
        precip_history: List[float],
    ) -> List[float]:
        """Vecteur (n_features) aligné sur le modèle chargé (5 par défaut)."""
        row = [
            float(precip),
            float(temp),
            float(humidity),
            float(pressure),
            self._estimate_soil_moisture(precip_history),
        ]
        return row[: self.n_features]

    def prepare_time_series(self, meteo_data, lat: float, lon: float, location_name: str):
        """Série (sequence_length, n_features) pour le modèle LSTM."""
        daily_data = meteo_data.get('chirps', {}).get('daily_data', [])

        if not daily_data:
            return self._generate_synthetic_sequence(lat, lon, location_name)

        df = pd.DataFrame(daily_data)
        if 'precipitation' not in df.columns:
            return self._generate_synthetic_sequence(lat, lon, location_name)

        precip_series = df['precipitation'].astype(float).tolist()
        if len(precip_series) < self.sequence_length:
            pad = [0.0] * (self.sequence_length - len(precip_series))
            precip_series = pad + precip_series

        rows = []
        window = precip_series[-self.sequence_length:]
        history = precip_series[:-self.sequence_length] if len(precip_series) > self.sequence_length else []

        for i, precip in enumerate(window):
            hist = history + window[: i + 1]
            rows.append(
                self._build_meteo_row(
                    precip=precip,
                    temp=28.0,
                    humidity=65.0,
                    pressure=1013.0,
                    precip_history=hist,
                )
            )

        return np.array(rows, dtype=np.float32)

    def _generate_synthetic_sequence(self, lat: float, lon: float, location_name: str):
        """Série synthétique (30×5) quand les données CHIRPS manquent."""
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        sequence = []
        precip_history: List[float] = []

        for _ in range(self.sequence_length):
            precip = max(0.0, float(np.random.exponential(2.0)))
            precip_history.append(precip)
            sequence.append(
                self._build_meteo_row(
                    precip=precip,
                    temp=float(np.random.normal(25, 3)),
                    humidity=float(np.random.uniform(50, 80)),
                    pressure=float(np.random.normal(1013, 5)),
                    precip_history=precip_history,
                )
            )

        return np.array(sequence, dtype=np.float32)
    
    def predict(self, lat, lon, location_name, forecast_data=None):
        """
        Predict flood probability for location
        Args:
            lat: Latitude
            lon: Longitude
            location_name: Name of location
            forecast_data: Optional forecast data from WeatherForecastService
        """
        try:
            # Import here to avoid circular dependency
            from services.africa_data_service import AfricaDataService
            from services.weather_forecast_service import WeatherForecastService
            
            data_service = AfricaDataService()
            
            # Get historical meteorological data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.sequence_length)
            meteo_data = data_service.get_comprehensive_meteo_data(lat, lon, days_back=self.sequence_length)
            
            # Get forecast data if not provided
            if forecast_data is None:
                forecast_service = WeatherForecastService()
                forecast_data = forecast_service.get_forecast_for_lstm(lat, lon, days=self.forecast_days)
            
            # Prepare time series with historical data
            sequence = self.prepare_time_series(meteo_data, lat, lon, location_name)
            precip_history = [float(row[0]) for row in sequence]

            # Enhance sequence with forecast data if available
            if forecast_data and len(forecast_data) > 0:
                for forecast_day in forecast_data[: min(7, len(forecast_data))]:
                    precip = float(forecast_day.get('precipitation', 0))
                    precip_history.append(precip)
                    forecast_row = self._build_meteo_row(
                        precip=precip,
                        temp=float(forecast_day.get('temperature', 25)),
                        humidity=float(forecast_day.get('humidity', 60)),
                        pressure=float(forecast_day.get('pressure', 1013)),
                        precip_history=precip_history,
                    )
                    sequence = np.append(sequence[1:], [forecast_row], axis=0)

            nf = self.n_features
            flat = sequence.reshape(-1, nf)
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == nf:
                scaled_flat = self.scaler.transform(flat)
            else:
                scaled_flat = self.scaler.fit_transform(flat)
            sequence_scaled = scaled_flat.reshape(1, self.sequence_length, nf)
            
            # Predict
            if self.model:
                prediction = self.model.predict(sequence_scaled, verbose=0)[0][0]
            else:
                # Fallback if model not loaded
                total_precip = sum(seq[0] for seq in sequence)
                prediction = min(1.0, total_precip / 100)  # Simple heuristic
            
            # Determine risk level
            risk_level = self._get_risk_level(prediction)
            
            return {
                'flood_probability': float(prediction),
                'risk_level': risk_level,
                'forecast_days': self.forecast_days,
                'model_type': 'LSTM',
                'location': location_name,
                'timestamp': datetime.now().isoformat(),
                'features_used': self.feature_names
            }
            
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            # Return default prediction
            return {
                'flood_probability': 0.3,
                'risk_level': 'low',
                'forecast_days': self.forecast_days,
                'model_type': 'LSTM',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_risk_level(self, probability):
        """Get risk level from probability"""
        if probability >= 0.8:
            return 'critical'
        elif probability >= 0.6:
            return 'high'
        elif probability >= 0.4:
            return 'medium'
        elif probability >= 0.2:
            return 'low'
        else:
            return 'none'
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the LSTM model"""
        if self.model is None:
            self._create_model()
        
        # Fit scaler on training data
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        # Scale data
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                save_best_only=True,
                monitor='val_loss' if X_val is not None else 'loss'
            )
        ]
        
        validation_data = None
        if X_val is not None:
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_reshaped).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        
        history = self.model.fit(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        return history

