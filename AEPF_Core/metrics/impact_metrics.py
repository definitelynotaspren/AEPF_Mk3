"""Impact assessment metric calculation functions."""
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_transparency_score(model_results: dict) -> float:
    """Calculate model transparency score."""
    try:
        feature_importance = model_results.get('feature_importance', {})
        documentation = model_results.get('documentation', {})
        
        # Score based on feature importance clarity
        feature_score = min(len(feature_importance) / 10, 1.0)
        
        # Score based on documentation completeness
        doc_score = min(len(documentation) / 5, 1.0)
        
        return (feature_score + doc_score) / 2
    except Exception as e:
        logger.error(f"Error calculating transparency score: {e}")
        return 0.5

def calculate_accountability_score(model_results: Dict[str, Any]) -> float:
    """Calculate model accountability score."""
    try:
        # Check for key accountability factors
        factors = {
            'model_card': model_results.get('model_card', {}),
            'audit_trail': model_results.get('audit_trail', {}),
            'version_control': model_results.get('version_control', {}),
            'validation_results': model_results.get('validation_results', {})
        }
        
        # Score each factor
        scores = []
        for factor, data in factors.items():
            if isinstance(data, dict):
                completeness = min(len(data) / 5, 1.0)
                scores.append(completeness)
        
        return sum(scores) / len(scores) if scores else 0.5
    except Exception as e:
        logger.error(f"Error calculating accountability score: {e}")
        return 0.5

def calculate_social_impact_score(df: pd.DataFrame, predictions_df: pd.DataFrame) -> float:
    """Calculate social impact score."""
    try:
        # Analyze impact on different demographic groups
        protected_attrs = ['gender', 'race', 'age_group', 'income_level']
        impact_scores = []
        
        for attr in protected_attrs:
            if attr in df.columns:
                groups = df[attr].unique()
                group_scores = []
                
                for group in groups:
                    group_mask = df[attr] == group
                    pred_series = predictions_df.loc[group_mask, 'predicted']
                    approval_rate = pred_series.mean() if not pred_series.empty else 0.0
                    group_scores.append(approval_rate)
                
                # Calculate disparity in impact
                if group_scores:
                    disparity = max(group_scores) - min(group_scores)
                    impact_scores.append(1 - disparity)
        
        return sum(impact_scores) / len(impact_scores) if impact_scores else 0.5
    except Exception as e:
        logger.error(f"Error calculating social impact score: {e}")
        return 0.5

def calculate_individual_rights_score(model_results: Dict[str, Any]) -> float:
    """Calculate individual rights protection score."""
    try:
        # Check for privacy and rights protection measures
        measures = {
            'data_privacy': model_results.get('privacy_measures', {}),
            'consent_management': model_results.get('consent_management', {}),
            'data_access': model_results.get('data_access_controls', {}),
            'right_to_explanation': model_results.get('explanation_capability', {})
        }
        
        # Score each measure
        scores = []
        for measure, data in measures.items():
            if isinstance(data, dict):
                implementation = min(len(data) / 3, 1.0)  # At least 3 components per measure
                scores.append(implementation)
        
        return sum(scores) / len(scores) if scores else 0.5
    except Exception as e:
        logger.error(f"Error calculating individual rights score: {e}")
        return 0.5