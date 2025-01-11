"""Fairness metric calculation functions."""
import logging
from typing import List
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def safe_mean(values: List[float]) -> float:
    """Safely calculate mean of a list of values."""
    if not values:
        return 0.5
    return sum(values) / len(values)

def calculate_demographic_parity(df: pd.DataFrame, predictions_df: pd.DataFrame) -> float:
    """Calculate demographic parity score."""
    try:
        protected_attrs = ['gender', 'race', 'age_group']
        scores = []
        
        for attr in protected_attrs:
            if attr in df.columns:
                groups = df[attr].unique()
                pred_rates = []
                for group in groups:
                    group_mask = df[attr] == group
                    pred_series = predictions_df.loc[group_mask, 'predicted']
                    group_pred_rate = pred_series.mean() if not pred_series.empty else 0.0
                    pred_rates.append(float(group_pred_rate))
                max_diff = max(pred_rates) - min(pred_rates)
                scores.append(1 - max_diff)
        
        return safe_mean(scores)
    except Exception as e:
        logger.error(f"Error calculating demographic parity: {e}")
        return 0.5

def calculate_equal_opportunity(df: pd.DataFrame, predictions_df: pd.DataFrame) -> float:
    """Calculate equal opportunity score."""
    try:
        protected_attrs = ['gender', 'race', 'age_group']
        scores = []
        
        for attr in protected_attrs:
            if attr in df.columns:
                groups = df[attr].unique()
                tpr_rates = []
                for group in groups:
                    group_mask = (df[attr] == group) & (df['actual'] == 1)
                    pred_series = predictions_df.loc[group_mask, 'predicted']
                    tpr = pred_series.mean() if not pred_series.empty else 0.0
                    tpr_rates.append(float(tpr))
                max_diff = max(tpr_rates) - min(tpr_rates)
                scores.append(1 - max_diff)
        
        return safe_mean(scores)
    except Exception as e:
        logger.error(f"Error calculating equal opportunity: {e}")
        return 0.5

def calculate_disparate_impact(df: pd.DataFrame, predictions_df: pd.DataFrame) -> float:
    """Calculate disparate impact score."""
    try:
        protected_attrs = ['gender', 'race', 'age_group']
        scores = []
        
        for attr in protected_attrs:
            if attr in df.columns:
                groups = df[attr].unique()
                pred_ratios = []
                for group in groups:
                    group_mask = df[attr] == group
                    pred_series = predictions_df.loc[group_mask, 'predicted']
                    group_pred_rate = pred_series.mean() if not pred_series.empty else 0.0
                    pred_ratios.append(float(group_pred_rate))
                min_ratio = min(pred_ratios) / max(pred_ratios) if max(pred_ratios) > 0 else 0.0
                scores.append(min_ratio)
        
        return safe_mean(scores)
    except Exception as e:
        logger.error(f"Error calculating disparate impact: {e}")
        return 0.5 