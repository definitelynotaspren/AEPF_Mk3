"""Stakeholder impact analysis metrics."""
import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_applicant_impact(df: pd.DataFrame, predictions_df: pd.DataFrame) -> float:
    """Calculate impact score for loan applicants."""
    try:
        # Analyze approval rates and fairness
        approval_rate = float(predictions_df['predicted'].mean())
        
        # Check for demographic disparities
        protected_attrs = ['gender', 'race', 'age_group']
        disparity_scores = []
        
        for attr in protected_attrs:
            if attr in df.columns:
                groups = df[attr].unique()
                group_rates = []
                
                for group in groups:
                    mask = df[attr] == group
                    pred_series = predictions_df.loc[mask, 'predicted']
                    if not pred_series.empty:
                        group_rate = pred_series.mean()
                        group_rates.append(float(group_rate))
                
                if group_rates:
                    disparity = max(group_rates) - min(group_rates)
                    disparity_scores.append(disparity)
        
        fairness_score = 1 - (sum(disparity_scores) / len(disparity_scores) if disparity_scores else 0)
        
        # Combine metrics
        impact_score = (0.6 * approval_rate + 0.4 * fairness_score)
        return min(max(impact_score, 0), 1)
    except Exception as e:
        logger.error(f"Error calculating applicant impact: {e}")
        return 0.5

def analyze_applicant_concerns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze potential concerns for loan applicants."""
    try:
        concerns = []
        
        # Check for demographic representation
        protected_attrs = ['gender', 'race', 'age_group']
        for attr in protected_attrs:
            if attr in df.columns:
                value_counts = df[attr].value_counts()
                if not value_counts.empty:
                    min_rep = value_counts.min() / len(df)
                    if min_rep < 0.1:  # Less than 10% representation
                        concerns.append({
                            'type': 'representation',
                            'attribute': attr,
                            'severity': float(1 - min_rep),
                            'description': f"Low representation in {attr} category"
                        })
        
        # Add other common concerns
        concerns.extend([
            {
                'type': 'transparency',
                'severity': 0.7,
                'description': "Need for clear explanation of decision factors"
            },
            {
                'type': 'privacy',
                'severity': 0.8,
                'description': "Protection of sensitive personal information"
            }
        ])
        
        return concerns
    except Exception as e:
        logger.error(f"Error analyzing applicant concerns: {e}")
        return []

def generate_applicant_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate recommendations for loan applicants."""
    try:
        recommendations = [
            {
                'priority': 'high',
                'category': 'transparency',
                'description': "Provide clear explanation of decision factors",
                'implementation': "Create detailed applicant guidelines"
            },
            {
                'priority': 'high',
                'category': 'rights',
                'description': "Ensure appeal process availability",
                'implementation': "Establish formal appeal procedure"
            },
            {
                'priority': 'medium',
                'category': 'support',
                'description': "Offer application assistance",
                'implementation': "Develop support resources"
            }
        ]
        
        # Add data-driven recommendations
        if 'credit_score' in df.columns:
            credit_threshold = df['credit_score'].quantile(0.7)
            recommendations.append({
                'priority': 'high',
                'category': 'qualification',
                'description': f"Target credit score above {credit_threshold:.0f}",
                'implementation': "Credit improvement guidance"
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating applicant recommendations: {e}")
        return []

def calculate_institution_impact(model_results: Dict[str, Any]) -> float:
    """Calculate impact score for the financial institution."""
    try:
        factors = {
            'model_performance': model_results.get('performance_score', 0.7),
            'operational_efficiency': model_results.get('efficiency_score', 0.6),
            'risk_management': model_results.get('risk_score', 0.8),
            'regulatory_compliance': model_results.get('compliance_score', 0.9)
        }
        
        weights = {
            'model_performance': 0.3,
            'operational_efficiency': 0.2,
            'risk_management': 0.25,
            'regulatory_compliance': 0.25
        }
        
        impact_score = sum(
            score * weights[factor]
            for factor, score in factors.items()
        )
        
        return min(max(impact_score, 0), 1)
    except Exception as e:
        logger.error(f"Error calculating institution impact: {e}")
        return 0.5

def analyze_institution_concerns(model_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze potential concerns for the financial institution."""
    try:
        concerns = []
        
        # Risk-based concerns
        risk_factors = model_results.get('risk_factors', {})
        for factor, score in risk_factors.items():
            if score > 0.7:  # High risk threshold
                concerns.append({
                    'type': 'risk',
                    'factor': factor,
                    'severity': float(score),
                    'description': f"High risk level in {factor}"
                })
        
        # Add standard concerns
        concerns.extend([
            {
                'type': 'compliance',
                'severity': 0.8,
                'description': "Regulatory compliance requirements"
            },
            {
                'type': 'reputation',
                'severity': 0.7,
                'description': "Public perception and trust"
            }
        ])
        
        return concerns
    except Exception as e:
        logger.error(f"Error analyzing institution concerns: {e}")
        return []

def generate_institution_recommendations(model_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate recommendations for the financial institution."""
    try:
        recommendations = [
            {
                'priority': 'high',
                'category': 'compliance',
                'description': "Enhance regulatory compliance monitoring",
                'implementation': "Implement automated compliance checks"
            },
            {
                'priority': 'high',
                'category': 'risk',
                'description': "Strengthen risk management framework",
                'implementation': "Develop comprehensive risk assessment"
            }
        ]
        
        # Add performance-based recommendations
        if model_results.get('performance_score', 1.0) < 0.8:
            recommendations.append({
                'priority': 'high',
                'category': 'performance',
                'description': "Improve model accuracy and reliability",
                'implementation': "Regular model retraining and validation"
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating institution recommendations: {e}")
        return [] 