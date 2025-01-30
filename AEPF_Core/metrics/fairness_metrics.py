"""Fairness metrics for evaluating model bias."""
from typing import Dict, List, Union, Optional
from typing_extensions import TypeAlias
import numpy as np
import pandas as pd

# Type aliases
NumericArray: TypeAlias = Union[np.ndarray, pd.Series, List[float]]
GroupData: TypeAlias = Dict[str, NumericArray]

class FairnessMetrics:
    """Calculate various fairness metrics for model evaluation."""
    
    def __init__(self, predictions: NumericArray, sensitive_features: pd.DataFrame):
        """Initialize with predictions and sensitive feature data."""
        self.predictions = np.asarray(predictions, dtype=float)
        self.sensitive_features = sensitive_features
    
    def demographic_parity(self, group_col: str) -> Dict[str, float]:
        """Calculate demographic parity across groups."""
        groups = self.sensitive_features[group_col].unique()
        group_predictions: Dict[str, float] = {}
        
        for group in groups:
            mask = self.sensitive_features[group_col] == group
            group_preds = self.predictions[mask]
            # Convert to float explicitly
            group_predictions[str(group)] = float(np.mean(group_preds))
            
        return group_predictions
    
    def equal_opportunity(self, group_col: str, actual: NumericArray) -> Dict[str, float]:
        """Calculate equal opportunity difference across groups."""
        groups = self.sensitive_features[group_col].unique()
        actual = np.asarray(actual, dtype=float)
        opportunity_rates: Dict[str, float] = {}
        
        for group in groups:
            mask = self.sensitive_features[group_col] == group
            group_preds = self.predictions[mask]
            group_actual = actual[mask]
            # Calculate true positive rate
            pos_mask = group_actual == 1
            if pos_mask.any():
                tpr = float(np.mean(group_preds[pos_mask]))
                opportunity_rates[str(group)] = tpr
            
        return opportunity_rates
    
    def disparate_impact(self, group_col: str) -> Dict[str, float]:
        """Calculate disparate impact ratios across groups."""
        groups = self.sensitive_features[group_col].unique()
        impact_ratios: Dict[str, float] = {}
        
        # Calculate acceptance rates for each group
        for group in groups:
            mask = self.sensitive_features[group_col] == group
            group_preds = self.predictions[mask]
            # Convert to float explicitly
            impact_ratios[str(group)] = float(np.mean(group_preds))
            
        # Calculate ratios relative to highest acceptance rate
        max_rate = max(impact_ratios.values())
        if max_rate > 0:
            impact_ratios = {
                k: float(v / max_rate) 
                for k, v in impact_ratios.items()
            }
            
        return impact_ratios 