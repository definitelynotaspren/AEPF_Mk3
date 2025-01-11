"""Technical metric calculation functions."""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Configure logging
logger = logging.getLogger(__name__)

def analyze_feature_importance(feature_importance: dict) -> dict:
    """Analyze and normalize feature importance scores."""
    try:
        total = sum(feature_importance.values())
        return {k: v/total for k, v in feature_importance.items()}
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")
        return {}

def generate_feature_statistics(df: pd.DataFrame) -> dict:
    """Generate statistical analysis of features."""
    try:
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isnull().sum()
            }
        return stats
    except Exception as e:
        logger.error(f"Error generating feature statistics: {e}")
        return {}

def generate_correlation_analysis(df: pd.DataFrame) -> dict:
    """Generate correlation analysis between features."""
    try:
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        return corr_matrix.to_dict()
    except Exception as e:
        logger.error(f"Error generating correlation analysis: {e}")
        return {} 