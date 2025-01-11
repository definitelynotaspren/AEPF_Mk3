"""Prediction module."""
import pandas as pd
try:
    from joblib import load
except ImportError:
    from sklearn.externals.joblib import load
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_predictions(model_path: str, data_path: str, output_path: str) -> None:
    """
    Generate predictions using trained model.
    
    Args:
        model_path: Path to trained model file
        data_path: Path to input data CSV
        output_path: Path to save predictions
    """
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = load(model_path)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Generate predictions
        logger.info("Generating predictions...")
        probabilities = model.predict_proba(df)
        predictions = model.predict(df)
        
        # Add predictions to dataframe
        df['default_probability'] = probabilities[:, 1]
        df['predicted_status'] = predictions
        
        # Save results
        logger.info(f"Saving predictions to {output_path}")
        df.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise 