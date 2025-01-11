"""Generate sample predictions for testing."""
try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError(
        "Required packages not found. Please run: pip install pandas numpy"
    )

from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_predictions(n_samples: int = 1000):
    """Generate sample predictions data."""
    try:
        np.random.seed(42)
        
        # Generate predictions
        data = {
            'id': range(1, n_samples + 1),
            'actual': np.random.randint(0, 2, n_samples),
            'predicted': np.random.randint(0, 2, n_samples),
            'probability': np.random.random(n_samples),
            'loan_amount': np.random.uniform(1000, 50000, n_samples),
            'interest_rate': np.random.uniform(5, 15, n_samples),
            'term': np.random.choice([12, 24, 36, 48, 60], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save predictions
        base_path = Path(__file__).parent.parent
        output_dir = base_path / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'predictions.csv'
        df.to_csv(output_path, index=False)
        
        logger.info(f"Generated {n_samples} sample predictions")
        logger.info(f"Saved to {output_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error generating sample predictions: {e}")
        raise

if __name__ == "__main__":
    try:
        generate_sample_predictions()
        logger.info("Sample predictions generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate sample predictions: {e}") 