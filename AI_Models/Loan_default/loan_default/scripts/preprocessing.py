"""Data preprocessing module."""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def preprocess_data(input_path: str, output_path: str) -> None:
    """
    Preprocess loan data.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
    """
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Fill missing values
        df = df.fillna(0)
        
        # Encode categorical variables
        categorical_cols = ['home_ownership', 'verification_status', 'purpose']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # Save processed data
        logger.info(f"Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise 