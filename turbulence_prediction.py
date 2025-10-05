"""
Advanced Turbulence Prediction Module
Professional machine learning system for turbulence forecasting
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('turbulence_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTurbulencePredictor:
    """
    Simplified turbulence prediction system
    """
    
    def __init__(self, model_save_path="models/turbulence"):
        """
        Initialize the turbulence predictor
        
        Args:
            model_save_path (str): Path to save trained models
        """
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.model_metrics = {}
        
        logger.info("Advanced Turbulence Predictor initialized")
    
    def engineer_features(self, df):
        """
        Simplified feature engineering for turbulence prediction
        
        Args:
            df (pd.DataFrame): Input data with atmospheric features
            
        Returns:
            pd.DataFrame: Engineered feature set
        """
        logger.info("Engineering features for turbulence prediction")
        features = pd.DataFrame()
        
        # Basic features
        if 'combined_wind_speed' in df.columns:
            features['wind_speed'] = df['combined_wind_speed']
            features['wind_speed_squared'] = df['combined_wind_speed'] ** 2
        
        if 'combined_turbulence_index' in df.columns:
            features['turbulence'] = df['combined_turbulence_index']
            features['turbulence_log'] = np.log1p(df['combined_turbulence_index'])
        
        if 'combined_ozone_concentration' in df.columns:
            features['ozone'] = df['combined_ozone_concentration']
            features['ozone_normalized'] = (df['combined_ozone_concentration'] - df['combined_ozone_concentration'].mean()) / (df['combined_ozone_concentration'].std() + 1e-8)
        
        if 'geos5_T' in df.columns:
            features['temperature'] = df['geos5_T']
        
        if 'geos5_RH' in df.columns:
            features['humidity'] = df['geos5_RH']
        
        # Time-based features
        if 'timestamp' in df.columns:
            features['hour'] = df['timestamp'].dt.hour
            features['month'] = df['timestamp'].dt.month
        
        # Fill missing values
        features = features.fillna(0)
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Feature engineering completed. Total features: {len(features.columns)}")
        return features
    
    def create_target_variable(self, df):
        """
        Create target variable for turbulence prediction
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.Series: Binary target variable
        """
        logger.info("Creating target variable for turbulence prediction")
        
        # Use the combined turbulence index as primary indicator
        if 'combined_turbulence_index' in df.columns:
            turbulence = df['combined_turbulence_index']
        else:
            # Create proxy from other variables
            turbulence = pd.Series(0, index=df.index)
            if 'combined_wind_speed' in df.columns:
                turbulence += df['combined_wind_speed'] / 50
        
        # Create binary target (0: No turbulence, 1: Turbulence)
        threshold = 0.5
        target = (turbulence > threshold).astype(int)
        
        # Check class distribution
        class_dist = target.value_counts(normalize=True)
        logger.info(f"Class distribution: {class_dist.to_dict()}")
        
        return target
    
    def train(self, df, perform_pca=False, feature_selection_method='simple', balance_method='none'):
        """
        Train the turbulence prediction model
        
        Args:
            df (pd.DataFrame): Training data
            perform_pca (bool): Whether to perform PCA dimensionality reduction
            feature_selection_method (str): Feature selection method
            balance_method (str): Class balancing method
            
        Returns:
            dict: Model performance metrics
        """
        logger.info("Training turbulence prediction model")
        
        # Feature engineering
        X = self.engineer_features(df)
        y = self.create_target_variable(df)
        
        # Feature selection
        if feature_selection_method == 'simple' and X.shape[1] > 10:
            selector = SelectKBest(f_classif, k=10)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            X = pd.DataFrame(X_selected, columns=selected_features)
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        self.model_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
        }
        
        logger.info(f"Model trained with accuracy: {self.model_metrics['accuracy']:.4f}")
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return self.model_metrics
    
    def predict(self, df, return_probabilities=True):
        """
        Predict turbulence probability using the trained model
        
        Args:
            df (pd.DataFrame): Input data
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            np.array or tuple: Predictions and optionally probabilities
        """
        if not self.is_trained:
            logger.error("Model must be trained before making predictions")
            raise ValueError("Model must be trained before making predictions")
        
        # Feature engineering
        X = self.engineer_features(df)
        
        # Select features
        X_selected = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        # Predictions
        predictions = self.model.predict(X_scaled)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        else:
            return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from the model
        
        Returns:
            pd.DataFrame: Feature importance sorted by importance
        """
        if not self.is_trained:
            logger.error("Model must be trained first")
            raise ValueError("Model must be trained first")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance
    
    def get_model_metrics(self):
        """
        Get model performance metrics
        
        Returns:
            dict: Model metrics
        """
        return self.model_metrics
    
    def save_model(self):
        """
        Save trained model to disk
        """
        logger.info("Saving model...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics
        }
        
        model_file = "turbulence_model.joblib"
        joblib.dump(model_data, model_file)
        logger.info(f"Model saved to {model_file}")
        print(f"✅ Model saved to: {model_file}")
    
    def load_model(self):
        """
        Load trained model from disk
        """
        model_file = "turbulence_model.joblib"
        
        if not Path(model_file).exists():
            logger.error(f"No saved model found at {model_file}")
            raise FileNotFoundError(f"No saved model found at {model_file}")
        
        logger.info(f"Loading model from {model_file}")
        model_data = joblib.load(model_file)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data['model_metrics']
        
        self.is_trained = True
        logger.info("Model loaded successfully")
    
    def generate_model_report(self):
        """
        Generate model performance report
        
        Returns:
            dict: Model performance report
        """
        if not self.is_trained:
            logger.error("Model must be trained first")
            raise ValueError("Model must be trained first")
        
        report = {
            'model_summary': {
                'is_trained': self.is_trained,
                'feature_count': len(self.feature_columns)
            },
            'performance_metrics': self.model_metrics,
            'feature_importance': self.get_feature_importance().to_dict('records')
        }
        
        return report