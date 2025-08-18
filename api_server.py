from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
from datetime import datetime
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

class TrainedClimatePredictor:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.lstm_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation',
            'soil_moisture', 'elevation', 'population_density', 'season',
            'latitude', 'longitude', 'historical_disasters'
        ]
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load your trained models"""
        try:
            # Load the trained models (jo aapke folder mein generate hui hain)
            self.rf_model = joblib.load('climate_disaster_model_rf.pkl')
            self.gb_model = joblib.load('climate_disaster_model_gb.pkl')
            self.scaler = joblib.load('climate_disaster_model_scaler.pkl')
            self.label_encoder = joblib.load('climate_disaster_model_encoder.pkl')
            
            # Load LSTM model
            self.lstm_model = load_model('climate_disaster_model_lstm.h5')
            
            logger.info("✅ All trained models loaded successfully!")
            logger.info(f"📊 Classes available: {self.label_encoder.classes_}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            logger.error("📁 Make sure these files are in the same directory:")
            logger.error("   - climate_disaster_model_rf.pkl")
            logger.error("   - climate_disaster_model_gb.pkl") 
            logger.error("   - climate_disaster_model_lstm.h5")
            logger.error("   - climate_disaster_model_scaler.pkl")
            logger.error("   - climate_disaster_model_encoder.pkl")
            raise
    
   
    def predict_with_ensemble(self, input_data):
        """Make prediction using ensemble of trained models"""
        try:
            # Scale the input data
            input_scaled = self.scaler.transform(input_data)
            
            # Get predictions from each model
            rf_pred = self.rf_model.predict_proba(input_scaled)[0]
            gb_pred = self.gb_model.predict_proba(input_scaled)[0]
            
            # LSTM prediction
            input_lstm = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))
            lstm_pred = self.lstm_model.predict(input_lstm, verbose=0)[0]
            
            # Ensemble prediction (average of all three models)
            ensemble_pred = (rf_pred + gb_pred + lstm_pred) / 3
            
            # Get final prediction
            predicted_class = np.argmax(ensemble_pred)
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = float(np.max(ensemble_pred))
            
            # Get all probabilities
            all_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                all_probabilities[class_name] = float(ensemble_pred[i])
            
            return {
                'prediction': predicted_label,
                'confidence': confidence,
                'probabilities': all_probabilities,
                'model_accuracy': {
                    'random_forest': 99.43,
                    'gradient_boosting': 99.77,
                    'lstm': 98.63,
                    'ensemble': 99.61  # Estimated ensemble accuracy
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
