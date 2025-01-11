"""Model robustness assessment metrics."""
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

def calculate_stability_score(predictions_df: pd.DataFrame) -> float:
    """Calculate model prediction stability score."""
    try:
        # Check for prediction consistency across similar inputs
        if 'confidence' in predictions_df.columns:
            confidence_std = predictions_df['confidence'].std()
            stability_score = 1 - min(confidence_std, 1.0)
            return float(stability_score)
        return 0.5
    except Exception as e:
        logger.error(f"Error calculating stability score: {e}")
        return 0.5

def perform_sensitivity_analysis(df: pd.DataFrame, model_results: Dict[str, Any]) -> Dict[str, float]:
    """Analyze model sensitivity to input variations."""
    try:
        feature_importance = model_results.get('feature_importance', {})
        sensitivity_scores = {}
        
        for feature, importance in feature_importance.items():
            if feature in df.columns:
                # Calculate feature variability
                feature_std = df[feature].std()
                # Combine with feature importance
                sensitivity = float(feature_std * importance)
                sensitivity_scores[feature] = min(sensitivity, 1.0)
        
        return sensitivity_scores
    except Exception as e:
        logger.error(f"Error performing sensitivity analysis: {e}")
        return {}

def calculate_bias_metrics(df: pd.DataFrame, predictions_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive bias metrics."""
    try:
        protected_attrs = ['gender', 'race', 'age_group']
        metrics = {}
        
        for attr in protected_attrs:
            if attr in df.columns:
                groups = df[attr].unique()
                group_metrics = []
                
                for group in groups:
                    mask = df[attr] == group
                    pred_series = predictions_df.loc[mask, 'predicted']
                    actual_series = df.loc[mask, 'actual']
                    
                    if not pred_series.empty and not actual_series.empty:
                        # Calculate false positive rate
                        fp_mask = (actual_series == 0) & (pred_series == 1)
                        fp_rate = fp_mask.mean()
                        
                        # Calculate false negative rate
                        fn_mask = (actual_series == 1) & (pred_series == 0)
                        fn_rate = fn_mask.mean()
                        
                        group_metrics.append({
                            'fp_rate': float(fp_rate),
                            'fn_rate': float(fn_rate)
                        })
                
                if group_metrics:
                    metrics[attr] = {
                        'max_fp_disparity': max(m['fp_rate'] for m in group_metrics) - 
                                          min(m['fp_rate'] for m in group_metrics),
                        'max_fn_disparity': max(m['fn_rate'] for m in group_metrics) - 
                                          min(m['fn_rate'] for m in group_metrics)
                    }
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating bias metrics: {e}")
        return {}

def analyze_predictions(predictions_df: pd.DataFrame) -> Dict[str, float]:
    """Analyze prediction distribution and patterns."""
    try:
        metrics = {
            'prediction_mean': float(predictions_df['predicted'].mean()),
            'prediction_std': float(predictions_df['predicted'].std()),
            'positive_rate': float((predictions_df['predicted'] == 1).mean())
        }
        
        if 'confidence' in predictions_df.columns:
            metrics.update({
                'confidence_mean': float(predictions_df['confidence'].mean()),
                'confidence_std': float(predictions_df['confidence'].std())
            })
        
        return metrics
    except Exception as e:
        logger.error(f"Error analyzing predictions: {e}")
        return {}

def analyze_prediction_confidence(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze model prediction confidence patterns."""
    try:
        if 'confidence' not in predictions_df.columns:
            return {}
            
        confidence_metrics = {
            'mean_confidence': float(predictions_df['confidence'].mean()),
            'min_confidence': float(predictions_df['confidence'].min()),
            'max_confidence': float(predictions_df['confidence'].max()),
            'low_confidence_rate': float((predictions_df['confidence'] < 0.6).mean())
        }
        
        # Analyze confidence distribution
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(predictions_df['confidence'], bins=bins)
        confidence_metrics['distribution'] = {
            f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(count)
            for i, count in enumerate(hist)
        }
        
        return confidence_metrics
    except Exception as e:
        logger.error(f"Error analyzing prediction confidence: {e}")
        return {}

def identify_edge_cases(predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify potential edge cases in predictions."""
    try:
        edge_cases = []
        
        if 'confidence' in predictions_df.columns:
            # Find very low confidence predictions
            low_conf_mask = predictions_df['confidence'] < 0.55
            low_conf_cases = predictions_df[low_conf_mask]
            
            for idx, row in low_conf_cases.iterrows():
                edge_cases.append({
                    'index': int(idx),
                    'prediction': int(row['predicted']),
                    'confidence': float(row['confidence']),
                    'type': 'low_confidence'
                })
        
        # Find prediction outliers
        if len(edge_cases) > 0:
            edge_cases = sorted(edge_cases, key=lambda x: x['confidence'])[:10]
        
        return edge_cases
    except Exception as e:
        logger.error(f"Error identifying edge cases: {e}")
        return [] 