"""Risk assessment metric calculation functions."""
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_bias_risk_severity(df: pd.DataFrame, predictions_df: pd.DataFrame) -> float:
    """Calculate severity of bias risk."""
    try:
        protected_attrs = ['gender', 'race', 'age_group']
        max_disparity = 0
        
        for attr in protected_attrs:
            if attr in df.columns:
                groups = df[attr].unique()
                pred_rates = []
                for group in groups:
                    group_mask = df[attr] == group
                    pred_series = predictions_df.loc[group_mask, 'predicted']
                    pred_rate = pred_series.mean() if not pred_series.empty else 0.0
                    pred_rates.append(float(pred_rate))
                disparity = max(pred_rates) - min(pred_rates)
                max_disparity = max(max_disparity, disparity)
        
        return min(max_disparity, 1.0)
    except Exception as e:
        logger.error(f"Error calculating bias risk severity: {e}")
        return 0.5

def calculate_privacy_risk_severity(model_results: Dict[str, Any]) -> float:
    """Calculate severity of privacy risk."""
    try:
        feature_importance = model_results.get('feature_importance', {})
        sensitive_features = ['ssn', 'address', 'phone', 'email']
        
        privacy_risk = sum(
            feature_importance.get(feature, 0) 
            for feature in sensitive_features 
            if feature in feature_importance
        )
        
        return min(privacy_risk * 2, 1.0)  # Scale up for severity
    except Exception as e:
        logger.error(f"Error calculating privacy risk severity: {e}")
        return 0.5

def calculate_overall_risk_score(model_results: Dict[str, Any]) -> float:
    """Calculate overall risk score considering multiple factors."""
    try:
        risk_factors = {
            'data_quality': model_results.get('data_quality_score', 0.5),
            'model_complexity': model_results.get('model_complexity_score', 0.5),
            'deployment_environment': model_results.get('deployment_risk_score', 0.5),
            'monitoring_capability': model_results.get('monitoring_score', 0.5)
        }
        
        # Weighted average of risk factors
        weights = {
            'data_quality': 0.3,
            'model_complexity': 0.2,
            'deployment_environment': 0.3,
            'monitoring_capability': 0.2
        }
        
        weighted_score = sum(
            score * weights[factor]
            for factor, score in risk_factors.items()
        )
        
        return min(max(weighted_score, 0), 1)
    except Exception as e:
        logger.error(f"Error calculating overall risk score: {e}")
        return 0.5

def calculate_bias_risk_likelihood(df: pd.DataFrame) -> float:
    """Calculate likelihood of bias risk materializing."""
    try:
        # Check data imbalance in protected attributes
        protected_attrs = ['gender', 'race', 'age_group']
        imbalance_scores = []
        
        for attr in protected_attrs:
            if attr in df.columns:
                value_counts = df[attr].value_counts()
                if not value_counts.empty:
                    max_count = value_counts.max()
                    min_count = value_counts.min()
                    imbalance = 1 - (min_count / max_count if max_count > 0 else 0)
                    imbalance_scores.append(imbalance)
        
        return sum(imbalance_scores) / len(imbalance_scores) if imbalance_scores else 0.5
    except Exception as e:
        logger.error(f"Error calculating bias risk likelihood: {e}")
        return 0.5

def calculate_privacy_risk_likelihood(model_results: Dict[str, Any]) -> float:
    """Calculate likelihood of privacy risk materializing."""
    try:
        # Assess factors that influence privacy risk likelihood
        factors = {
            'data_exposure': model_results.get('data_exposure_score', 0.5),
            'access_controls': model_results.get('access_control_score', 0.5),
            'data_retention': model_results.get('data_retention_score', 0.5),
            'third_party_sharing': model_results.get('third_party_sharing_score', 0.5)
        }
        
        # Weight the factors
        weights = {
            'data_exposure': 0.3,
            'access_controls': 0.3,
            'data_retention': 0.2,
            'third_party_sharing': 0.2
        }
        
        weighted_score = sum(
            score * weights[factor]
            for factor, score in factors.items()
        )
        
        return min(max(weighted_score, 0), 1)
    except Exception as e:
        logger.error(f"Error calculating privacy risk likelihood: {e}")
        return 0.5