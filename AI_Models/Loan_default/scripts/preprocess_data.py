"""Data preprocessing module for loan default prediction."""
import os
import sys
from pathlib import Path
from typing import Union, Optional

# Get module base path
BASE_PATH = Path(__file__).parent.parent

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Scientific imports
try:
    import pandas as pd
    import numpy as np
    from numpy import random, exp
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    logger.error("Please run: pip install pandas numpy scikit-learn")
    raise

def run(input_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Main execution function.
    
    Args:
        input_path: Path to input data file. If None, uses default path.
        
    Returns:
        pd.DataFrame: Processed data
    """
    try:
        # Setup paths
        base_path = Path(__file__).parent.parent
        if input_path is None:
            input_path = base_path / 'data' / 'raw' / 'loan_data.csv'
        else:
            input_path = Path(input_path) if isinstance(input_path, str) else input_path
            
        output_path = base_path / 'data' / 'processed' / 'loan_data_preprocessed.csv'
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading data from {input_path}")
        
        # Generate sample data if input file doesn't exist
        if not input_path.exists():
            logger.info("Input file not found. Generating sample data...")
            raw_df = generate_sample_data()
            input_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(input_path, index=False)
        else:
            raw_df = pd.read_csv(input_path)
        
        logger.info("Starting preprocessing...")
        
        # Preprocess data
        processed_df = preprocess_data(raw_df)
        
        # Save processed data
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")
        
        # Log summary statistics using raw data for readability
        logger.info("\nData Summary:")
        logger.info(f"Total samples: {len(raw_df)}")
        if 'loan_status' in raw_df.columns:
            logger.info(f"Default rate: {raw_df['loan_status'].mean():.2%}")
        logger.info(f"Average loan amount: ${raw_df['loan_amount'].mean():,.2f}")
        logger.info(f"Average interest rate: {raw_df['interest_rate'].mean():.1f}%")
        logger.info(f"Average term: {raw_df['term'].mean():.1f} months")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic loan data for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base features
    data = {
        'loan_amount': np.random.uniform(5000, 100000, n_samples),
        'term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'interest_rate': np.random.uniform(5, 15, n_samples),
        'employment_length': np.random.randint(0, 30, n_samples),
        'annual_income': np.random.uniform(30000, 200000, n_samples),
        'debt_to_income': np.random.uniform(5, 40, n_samples),
        'credit_score': np.random.uniform(580, 850, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate risk score
    df['risk_score'] = (
        0.3 * (df['loan_amount'] / 100000) +
        0.2 * (df['interest_rate'] / 15) +
        0.2 * (df['debt_to_income'] / 40) +
        0.3 * ((850 - df['credit_score']) / 270)
    )
    
    # Generate loan status based on risk score
    probabilities = 1 / (1 + np.exp(-10 * (df['risk_score'] - 0.5)))  # Logistic function
    df['loan_status'] = np.random.binomial(1, probabilities)
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess loan data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Fill missing values
    df = df.fillna({
        'employment_length': 0,
        'debt_to_income': df['debt_to_income'].median(),
        'credit_score': df['credit_score'].median()
    })
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['loan_amount', 'annual_income', 'debt_to_income', 'credit_score']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Add engineered features
    df['payment_to_income'] = (
        df['loan_amount'] * (df['interest_rate']/1200) * 
        (1 + df['interest_rate']/1200)**df['term']
    ) / ((1 + df['interest_rate']/1200)**df['term'] - 1) / (df['annual_income']/12)
    
    df['credit_utilization'] = df['debt_to_income'] / 100
    
    return df

if __name__ == "__main__":
    run() 