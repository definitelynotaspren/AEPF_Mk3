import logging
import os
from pathlib import Path
from datetime import datetime
import sys

# Add the scripts directory to Python path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger('loan_pipeline')

class LoanDefaultPipeline:
    def __init__(self):
        """Initialize pipeline with paths and configuration."""
        self.base_path = Path(__file__).parent.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directory paths
        self.data_dir = self.base_path / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.models_dir = self.base_path / 'models'
        self.outputs_dir = self.base_path / 'outputs'
        
        # Setup file paths
        self.raw_data = self.raw_dir / 'loan_data.csv'
        self.processed_data = self.processed_dir / f'loan_data_preprocessed_{self.timestamp}.csv'
        self.model_path = self.models_dir / f'xgboost_model_{self.timestamp}.joblib'
        self.predictions_path = self.outputs_dir / f'predictions_{self.timestamp}.csv'
        
        # Create necessary directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.outputs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")
    
    def run_preprocessing(self):
        """Run data preprocessing step."""
        try:
            logger.info("Starting preprocessing...")
            
            if not self.raw_data.exists():
                raise FileNotFoundError(f"Raw data file not found: {self.raw_data}")
            
            from preprocessing import preprocess_data
            preprocess_data(
                input_path=str(self.raw_data),
                output_path=str(self.processed_data)
            )
            
            logger.info(f"Preprocessing completed. Output saved to {self.processed_data}")
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return False
    
    def train_model(self):
        """Run model training step."""
        try:
            logger.info("Starting model training...")
            
            if not self.processed_data.exists():
                raise FileNotFoundError(f"Processed data not found: {self.processed_data}")
            
            from train_model import train_xgboost_model
            train_xgboost_model(
                data_path=str(self.processed_data),
                model_path=str(self.model_path)
            )
            
            logger.info(f"Model training completed. Model saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False
    
    def generate_predictions(self):
        """Run prediction generation step."""
        try:
            logger.info("Starting prediction generation...")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            from predict import generate_predictions
            generate_predictions(
                model_path=str(self.model_path),
                data_path=str(self.processed_data),
                output_path=str(self.predictions_path)
            )
            
            logger.info(f"Predictions generated and saved to {self.predictions_path}")
            return True
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {str(e)}")
            return False
    
    def run_pipeline(self):
        """Execute the complete pipeline."""
        logger.info("Starting loan default prediction pipeline...")
        
        # Run preprocessing
        if not self.run_preprocessing():
            logger.error("Pipeline failed at preprocessing step")
            return False
        
        # Train model
        if not self.train_model():
            logger.error("Pipeline failed at model training step")
            return False
        
        # Generate predictions
        if not self.generate_predictions():
            logger.error("Pipeline failed at prediction generation step")
            return False
        
        logger.info("Pipeline completed successfully!")
        return True

def main():
    """Main execution function."""
    try:
        # Initialize and run pipeline
        pipeline = LoanDefaultPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            logger.info("Pipeline execution completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline execution failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 