import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import joblib
import warnings
warnings.filterwarnings('ignore')

class ClimateDisasterPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.rf_model = None
        self.lstm_model = None
        self.gb_model = None
        self.feature_names = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation',
            'soil_moisture', 'elevation', 'population_density', 'season',
            'latitude', 'longitude', 'historical_disasters'
        ]
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic climate data for training"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'temperature': np.random.normal(25, 10, n_samples),  # Celsius
            'humidity': np.random.uniform(20, 100, n_samples),   # Percentage
            'pressure': np.random.normal(1013, 20, n_samples),   # hPa
            'wind_speed': np.random.exponential(10, n_samples),  # km/h
            'precipitation': np.random.exponential(5, n_samples), # mm
            'soil_moisture': np.random.uniform(10, 80, n_samples), # Percentage
            'elevation': np.random.uniform(0, 3000, n_samples),   # meters
            'population_density': np.random.exponential(100, n_samples), # per km²
            'season': np.random.randint(1, 5, n_samples),        # 1-4 (seasons)
            'latitude': np.random.uniform(-90, 90, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'historical_disasters': np.random.poisson(2, n_samples) # Count
        }
        
        df = pd.DataFrame(data)
        
        # Generate disaster labels based on realistic conditions
        disaster_labels = []
        for i in range(n_samples):
            conditions = df.iloc[i]
            
            # Flood conditions
            if (conditions['precipitation'] > 15 and 
                conditions['elevation'] < 500 and 
                conditions['soil_moisture'] > 60):
                disaster_labels.append('flood')
            
            # Drought conditions
            elif (conditions['precipitation'] < 2 and 
                  conditions['temperature'] > 30 and 
                  conditions['humidity'] < 40):
                disaster_labels.append('drought')
            
            # Cyclone conditions
            elif (conditions['wind_speed'] > 25 and 
                  conditions['pressure'] < 990 and 
                  conditions['humidity'] > 70 and
                  abs(conditions['latitude']) < 30):
                disaster_labels.append('cyclone')
            
            # Heatwave conditions
            elif (conditions['temperature'] > 35 and 
                  conditions['humidity'] < 50):
                disaster_labels.append('heatwave')
            
            # Landslide conditions
            elif (conditions['precipitation'] > 20 and 
                  conditions['elevation'] > 1000 and 
                  conditions['soil_moisture'] > 70):
                disaster_labels.append('landslide')
            
            # No disaster
            else:
                disaster_labels.append('no_disaster')
        
        df['disaster_type'] = disaster_labels
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Features
        X = df[self.feature_names].values
        
        # Labels
        y_encoded = self.label_encoder.fit_transform(df['disaster_type'])
        y_categorical = to_categorical(y_encoded)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_encoded, y_categorical
    
    def create_lstm_model(self, input_shape, num_classes):
        """Create LSTM model for time series prediction"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model(self, input_shape, num_classes):
        """Create CNN model for pattern recognition"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, df):
        """Train all models"""
        print("Preparing data...")
        X, y_encoded, y_categorical = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
            X, y_encoded, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        num_classes = len(np.unique(y_encoded))
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.gb_model.fit(X_train, y_train)
        
        # Prepare data for LSTM (reshape for time series)
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Train LSTM
        print("Training LSTM...")
        self.lstm_model = self.create_lstm_model(
            input_shape=(X_train.shape[1], 1),
            num_classes=num_classes
        )
        
        self.lstm_model.fit(
            X_train_lstm, y_cat_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate models
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Random Forest evaluation
        rf_pred = self.rf_model.predict(X_test)
        print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
        
        # Gradient Boosting evaluation
        gb_pred = self.gb_model.predict(X_test)
        print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
        
        # LSTM evaluation
        lstm_pred = self.lstm_model.predict(X_test_lstm)
        lstm_pred_classes = np.argmax(lstm_pred, axis=1)
        print(f"LSTM Accuracy: {accuracy_score(y_test, lstm_pred_classes):.4f}")
        
        # Feature importance (Random Forest)
        print("\nTop 5 Important Features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in feature_importance.head().iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return X_test, y_test
    
    def predict_disaster(self, input_data, model_type='ensemble'):
        """Predict disaster type for new data"""
        # Ensure input_data is 2D array
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Scale input data
        input_scaled = self.scaler.transform(input_data)
        
        if model_type == 'random_forest':
            prediction = self.rf_model.predict(input_scaled)
            probabilities = self.rf_model.predict_proba(input_scaled)
        
        elif model_type == 'gradient_boosting':
            prediction = self.gb_model.predict(input_scaled)
            probabilities = self.gb_model.predict_proba(input_scaled)
        
        elif model_type == 'lstm':
            input_lstm = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))
            probabilities = self.lstm_model.predict(input_lstm)
            prediction = np.argmax(probabilities, axis=1)
        
        else:  # ensemble
            # Get predictions from all models
            rf_pred = self.rf_model.predict_proba(input_scaled)
            gb_pred = self.gb_model.predict_proba(input_scaled)
            
            input_lstm = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))
            lstm_pred = self.lstm_model.predict(input_lstm)
            
            # Average probabilities
            probabilities = (rf_pred + gb_pred + lstm_pred) / 3
            prediction = np.argmax(probabilities, axis=1)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(prediction)
        
        return predicted_labels, probabilities
    
    def get_risk_assessment(self, input_data):
        """Get detailed risk assessment"""
        predicted_labels, probabilities = self.predict_disaster(input_data, 'ensemble')
        
        risk_assessment = []
        for i, (label, prob_array) in enumerate(zip(predicted_labels, probabilities)):
            # Get top 3 risks
            top_indices = np.argsort(prob_array)[-3:][::-1]
            top_risks = []
            
            for idx in top_indices:
                risk_type = self.label_encoder.inverse_transform([idx])[0]
                confidence = prob_array[idx]
                
                if confidence > 0.1:  # Only show risks with >10% probability
                    risk_level = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"
                    top_risks.append({
                        'disaster_type': risk_type,
                        'probability': confidence,
                        'risk_level': risk_level
                    })
            
            risk_assessment.append({
                'primary_prediction': label,
                'primary_confidence': np.max(prob_array),
                'all_risks': top_risks
            })
        
        return risk_assessment
    
    def save_models(self, filepath_prefix='climate_disaster_model'):
        """Save trained models"""
        joblib.dump(self.rf_model, f'{filepath_prefix}_rf.pkl')
        joblib.dump(self.gb_model, f'{filepath_prefix}_gb.pkl')
        joblib.dump(self.scaler, f'{filepath_prefix}_scaler.pkl')
        joblib.dump(self.label_encoder, f'{filepath_prefix}_encoder.pkl')
        self.lstm_model.save(f'{filepath_prefix}_lstm.h5')
        print(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix='climate_disaster_model'):
        """Load trained models"""
        self.rf_model = joblib.load(f'{filepath_prefix}_rf.pkl')
        self.gb_model = joblib.load(f'{filepath_prefix}_gb.pkl')
        self.scaler = joblib.load(f'{filepath_prefix}_scaler.pkl')
        self.label_encoder = joblib.load(f'{filepath_prefix}_encoder.pkl')
        from tensorflow.keras.models import load_model
        self.lstm_model = load_model(f'{filepath_prefix}_lstm.h5')
        print(f"Models loaded from prefix: {filepath_prefix}")


# Example usage and demo
if __name__ == "__main__":
    # Initialize the predictor
    predictor = ClimateDisasterPredictor()
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    df = predictor.generate_synthetic_data(n_samples=15000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Disaster distribution:\n{df['disaster_type'].value_counts()}")
    
    # Train models
    X_test, y_test = predictor.train_models(df)
    
    # Save models
    predictor.save_models()
    
    # Example prediction for new data
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Example 1: High flood risk conditions
    flood_conditions = np.array([
        [22, 85, 1005, 15, 25, 75, 200, 150, 2, 25.5, 78.2, 3]
    ])
    
    risk_assessment = predictor.get_risk_assessment(flood_conditions)
    print(f"\nFlood-prone conditions prediction:")
    print(f"Primary prediction: {risk_assessment[0]['primary_prediction']}")
    print(f"Confidence: {risk_assessment[0]['primary_confidence']:.3f}")
    print("Top risks:")
    for risk in risk_assessment[0]['all_risks'][:3]:
        print(f"  - {risk['disaster_type']}: {risk['probability']:.3f} ({risk['risk_level']})")
    
    # Example 2: Drought conditions
    drought_conditions = np.array([
        [38, 25, 1020, 8, 0.5, 15, 400, 80, 1, 20.1, 85.7, 1]
    ])
    
    risk_assessment = predictor.get_risk_assessment(drought_conditions)
    print(f"\nDrought-prone conditions prediction:")
    print(f"Primary prediction: {risk_assessment[0]['primary_prediction']}")
    print(f"Confidence: {risk_assessment[0]['primary_confidence']:.3f}")
    print("Top risks:")
    for risk in risk_assessment[0]['all_risks'][:3]:
        print(f"  - {risk['disaster_type']}: {risk['probability']:.3f} ({risk['risk_level']})")
    
    # Example 3: Normal conditions
    normal_conditions = np.array([
        [24, 60, 1013, 12, 8, 45, 300, 100, 3, 28.6, 77.2, 1]
    ])
    
    risk_assessment = predictor.get_risk_assessment(normal_conditions)
    print(f"\nNormal conditions prediction:")
    print(f"Primary prediction: {risk_assessment[0]['primary_prediction']}")
    print(f"Confidence: {risk_assessment[0]['primary_confidence']:.3f}")
    print("Top risks:")
    for risk in risk_assessment[0]['all_risks'][:3]:
        print(f"  - {risk['disaster_type']}: {risk['probability']:.3f} ({risk['risk_level']})")
    
    print("\n" + "="*50)
    print("Model training and evaluation completed!")
    print("Models saved and ready for deployment.")
    print("="*50)
