"""Model training module."""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from joblib import dump
import logging

logger = logging.getLogger(__name__)

def train_xgboost_model(data_path: str, model_path: str) -> None:
    """
    Train XGBoost model for loan default prediction.
    
    Args:
        data_path: Path to processed data CSV
        model_path: Path to save trained model
    """
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Split features and target
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
        
        # Train model
        logger.info("Training XGBoost model...")
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X, y)
        
        # Save model
        logger.info(f"Saving model to {model_path}")
        dump(model, model_path)
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise 