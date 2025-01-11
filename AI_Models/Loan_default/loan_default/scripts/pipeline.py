"""Pipeline module."""
import logging
from pathlib import Path
from datetime import datetime
from .preprocessing import preprocess_data
from .train_model import train_xgboost_model
from .predict import generate_predictions

logger = logging.getLogger(__name__)

class LoanDefaultPipeline:
    def __init__(self):
        """Initialize pipeline with paths."""
        self.base_path = Path(__file__).parent.parent.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup paths
        self._setup_paths()
        self._setup_directories()
    
    def _setup_paths(self):
        """Setup directory paths based on base path."""
        self.data_dir = self.base_path / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.models_dir = self.base_path / 'models'
        self.outputs_dir = self.base_path / 'outputs'
    
    @property
    def base_path(self):
        """Get base path."""
        return self._base_path
    
    @base_path.setter
    def base_path(self, path):
        """Set base path and update all dependent paths."""
        self._base_path = Path(path)
        self._setup_paths()
    
    def _setup_directories(self):
        """Create required directories."""
        for directory in [self.raw_dir, self.processed_dir, self.models_dir, self.outputs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")
    
    def run_preprocessing(self):
        """Run preprocessing step."""
        try:
            input_path = self.raw_dir / 'loan_data.csv'
            output_path = self.processed_dir / f'processed_{self.timestamp}.csv'
            preprocess_data(str(input_path), str(output_path))
            return True
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return False
    
    def train_model(self):
        """Run model training step."""
        try:
            data_path = next(self.processed_dir.glob('processed_*.csv'))
            model_path = self.models_dir / f'model_{self.timestamp}.joblib'
            train_xgboost_model(str(data_path), str(model_path))
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def generate_predictions(self):
        """Run prediction generation step."""
        try:
            model_path = next(self.models_dir.glob('model_*.joblib'))
            data_path = next(self.processed_dir.glob('processed_*.csv'))
            output_path = self.outputs_dir / f'predictions_{self.timestamp}.csv'
            generate_predictions(str(model_path), str(data_path), str(output_path))
            return True
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return False 