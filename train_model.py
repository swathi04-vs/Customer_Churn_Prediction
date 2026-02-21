"""
Model Training Script for Customer Churn Prediction
Trains and evaluates the RandomForestClassifier model
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_VARIABLE,
    MODEL_PATH, PREPROCESSOR_PATH
)
from utils.preprocess import DataPreprocessor, handle_missing_values, encode_target_variable
from utils.helper import log_info, log_error, log_warning


class ChurnModelTrainer:
    """
    Trainer class for Customer Churn Prediction model
    """
    
    def __init__(self, random_state=42):
        """Initialize trainer"""
        self.model = None
        self.preprocessor = None
        self.accuracy = None
        self.conf_matrix = None
        self.class_report = None
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.random_state = random_state
    
    def load_data(self, filepath):
        """
        Load data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(filepath)
            log_info(f"Data loaded from {filepath}. Shape: {data.shape}")
            return data
        except Exception as e:
            log_error(f"Error loading data: {str(e)}")
            raise Exception(f"Failed to load data: {str(e)}")
    
    def prepare_data(self, data):
        """
        Prepare data for modeling
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            tuple: (X, y) prepared features and target
        """
        try:
            # Remove duplicate rows if any
            data = data.drop_duplicates()
            log_info(f"Duplicates removed. Shape: {data.shape}")
            
            # Handle missing values
            data = handle_missing_values(data, strategy='mean')
            
            # Convert TotalCharges to numeric if needed
            if 'TotalCharges' in data.columns:
                data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
                data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
            
            # Separate features and target
            X = data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
            y = data[TARGET_VARIABLE]
            
            # Encode target variable
            y = encode_target_variable(y)
            
            log_info(f"Data prepared. Features shape: {X.shape}, Target shape: {y.shape}")
            return X, y
            
        except Exception as e:
            log_error(f"Error preparing data: {str(e)}")
            raise Exception(f"Failed to prepare data: {str(e)}")
    
    def preprocess_features(self, X, fit=True):
        """
        Preprocess features
        
        Args:
            X (pd.DataFrame): Features dataframe
            fit (bool): Whether to fit the preprocessor
            
        Returns:
            np.ndarray: Preprocessed features
        """
        try:
            if self.preprocessor is None:
                self.preprocessor = DataPreprocessor()
            
            if fit:
                X_transformed = self.preprocessor.fit_transform(X)
            else:
                X_transformed = self.preprocessor.transform(X)
            
            log_info(f"Features preprocessed. Shape: {X_transformed.shape}")
            return X_transformed
            
        except Exception as e:
            log_error(f"Error preprocessing features: {str(e)}")
            raise Exception(f"Failed to preprocess features: {str(e)}")
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into train and test sets
        
        Args:
            X (pd.DataFrame): Features
            y (np.ndarray): Target
            test_size (float): Test set proportion
        """
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            log_info(f"Data split. Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        except Exception as e:
            log_error(f"Error splitting data: {str(e)}")
            raise Exception(f"Failed to split data: {str(e)}")
    
    def train_model(self, n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2):
        """
        Train RandomForestClassifier model
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples to split
            min_samples_leaf (int): Minimum samples in leaf
        """
        try:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.model.fit(self.X_train, self.y_train)
            log_info("Model training completed successfully")
            
        except Exception as e:
            log_error(f"Error training model: {str(e)}")
            raise Exception(f"Failed to train model: {str(e)}")
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        try:
            # Predictions
            y_pred = self.model.predict(self.X_test)
            
            # Accuracy
            self.accuracy = accuracy_score(self.y_test, y_pred)
            
            # Confusion Matrix
            self.conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            # Classification Report
            self.class_report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Feature Importance
            self.feature_importance = self.model.feature_importances_
            
            log_info(f"Model Accuracy: {self.accuracy:.4f}")
            log_info(f"\nConfusion Matrix:\n{self.conf_matrix}")
            log_info(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
            
        except Exception as e:
            log_error(f"Error evaluating model: {str(e)}")
            raise Exception(f"Failed to evaluate model: {str(e)}")
    
    def save_model(self, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH):
        """
        Save model and preprocessor
        
        Args:
            model_path (str): Path to save model
            preprocessor_path (str): Path to save preprocessor
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            joblib.dump(self.model, model_path)
            log_info(f"Model saved to {model_path}")
            
            self.preprocessor.save(preprocessor_path)
            log_info(f"Preprocessor saved to {preprocessor_path}")
            
        except Exception as e:
            log_error(f"Error saving model: {str(e)}")
            raise Exception(f"Failed to save model: {str(e)}")
    
    def get_feature_importance_df(self, top_n=10):
        """
        Get top N most important features
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_importance is None:
            return None
        
        feature_imp_df = pd.DataFrame({
            'feature': self.preprocessor.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp_df.head(top_n)
    
    def train_full_pipeline(self, filepath, test_size=0.2):
        """
        Execute full training pipeline
        
        Args:
            filepath (str): Path to CSV file
            test_size (float): Test set proportion
        """
        try:
            log_info("Starting full training pipeline...")
            
            # Load and prepare data
            data = self.load_data(filepath)
            X, y = self.prepare_data(data)
            
            # Preprocess features
            X_transformed = self.preprocess_features(X, fit=True)
            
            # Split data
            self.split_data(X_transformed, y, test_size=test_size)
            
            # Train model
            self.train_model()
            
            # Evaluate model
            self.evaluate_model()
            
            # Save model
            self.save_model()
            
            log_info("Training pipeline completed successfully!")
            
            return {
                'accuracy': self.accuracy,
                'confusion_matrix': self.conf_matrix.tolist(),
                'classification_report': self.class_report
            }
            
        except Exception as e:
            log_error(f"Error in training pipeline: {str(e)}")
            raise Exception(f"Training pipeline failed: {str(e)}")


def create_sample_data():
    """
    Create sample dataset for demonstration
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18, 120, n_samples),
        'TotalCharges': np.random.uniform(18, 8684, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    }
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    log_info("Customer Churn Prediction Model Training Script")
    log_info("=" * 50)
    
    try:
        # Check if sample data file exists, if not create it
        sample_data_path = 'sample_data.csv'
        
        if not os.path.exists(sample_data_path):
            log_info("Creating sample dataset...")
            sample_data = create_sample_data()
            sample_data.to_csv(sample_data_path, index=False)
            log_info(f"Sample data created at {sample_data_path}")
        
        # Initialize trainer
        trainer = ChurnModelTrainer()
        
        # Run training pipeline
        results = trainer.train_full_pipeline(sample_data_path)
        
        log_info("\n" + "=" * 50)
        log_info("TRAINING COMPLETED SUCCESSFULLY!")
        log_info(f"Model Accuracy: {results['accuracy']:.4f}")
        log_info("=" * 50)
        
    except Exception as e:
        log_error(f"Training failed: {str(e)}")
        sys.exit(1)
