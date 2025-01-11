import pandas as pd
import numpy as np
from joblib import load
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('loan_prediction')

class LoanDefaultPredictor:
    def __init__(self, config_path: str = '../config/model_config.json'):
        """Initialize predictor with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_columns = self.config['feature_columns']
        
    def _load_config(self, config_path: str) -> dict:
        """Load model configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def load_model(self, model_path: str):
        """Load trained model."""
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def predict(self, input_path: str, output_path: str):
        """
        Make predictions on input data and save results.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions
        """
        try:
            # Load input data
            logger.info(f"Loading input data from {input_path}")
            df = pd.read_csv(input_path)
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Extract features for prediction
            X = df[self.feature_columns]
            
            # Make predictions
            logger.info("Making predictions...")
            probabilities = self.model.predict_proba(X)
            predictions = self.model.predict(X)
            
            # Add predictions to dataframe
            df['default_probability'] = probabilities[:, 1]  # Probability of default
            df['predicted_status'] = predictions
            
            # Add prediction metadata
            df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df['model_version'] = self.config.get('model_version', 'default')
            
            # Save results
            logger.info(f"Saving predictions to {output_path}")
            df.to_csv(output_path, index=False)
            
            # Log prediction summary
            logger.info(f"Total predictions made: {len(df)}")
            logger.info(f"Predicted defaults: {sum(predictions == 1)}")
            logger.info(f"Predicted non-defaults: {sum(predictions == 0)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

def main():
    """Main execution function."""
    try:
        # Setup paths
        base_path = Path(__file__).parent.parent
        input_path = base_path / 'data' / 'processed' / 'loan_data_preprocessed.csv'
        output_path = base_path / 'outputs' / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        model_path = base_path / 'models' / 'xgboost_default_model.joblib'
        config_path = base_path / 'config' / 'model_config.json'
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize predictor
        predictor = LoanDefaultPredictor(str(config_path))
        
        # Load model
        predictor.load_model(str(model_path))
        
        # Make predictions
        predictions_df = predictor.predict(str(input_path), str(output_path))
        
        logger.info("Prediction pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    main() 