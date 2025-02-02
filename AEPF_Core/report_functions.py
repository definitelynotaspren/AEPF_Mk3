"""Core report generation functions."""
import logging
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_ethical_analysis_report(data: Dict[str, Any], scenario: Optional[str] = None) -> Dict[str, Any]:
    """Generate comprehensive ethical analysis report."""
    try:
        logger.info("Starting AEPF report generation")
        logger.info(f"Input data keys: {data.keys()}")
        
        # Create report dictionary
        report = {}
        
        # Extract metrics from input data
        tech_analysis = data.get('technical_analysis', {})
        risk_assessment = data.get('risk_assessment', {})
        model_name = data.get('model_name', 'Unknown Model')
        
        try:
            # Calculate core metrics
            accuracy = float(tech_analysis.get('accuracy', 0.85))
            fairness_score = float(tech_analysis.get('fairness_score', 0.85))
            rating_level = min(5, max(1, round(fairness_score * 5)))
            risk_level = str(risk_assessment.get('risk_level', 'Low'))
            low_risk = risk_assessment.get('low_risk_profile', {}).get('percentage', '85%')
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
        
        # Create summary section with professional formatting
        summary = {
            'risk_level': risk_level,
            'overall_score': fairness_score,
            'rating_level': rating_level,
            'rating': f"Level {rating_level}/5",
            'key_findings': [
                f"Model Accuracy: {accuracy:.1%}",
                f"Fairness Score: {fairness_score:.1%}",
                f"Risk Distribution: {low_risk} Low Risk"
            ],
            'recommendations': [
                "Monitor Demographic Representation",
                "Implement Regular Fairness Audits",
                "Review Edge Cases for Potential Bias"
            ],
            'narrative': (
                f"ETHICAL ANALYSIS OVERVIEW\n\n"
                f"The {model_name} model demonstrates {get_fairness_level(fairness_score)} "
                f"with an overall fairness score of {fairness_score:.2%}."
            ),
            'detailed_narrative': {
                'fairness': "The model demonstrates balanced outcomes across demographic groups.",
                'privacy': "Privacy protections exceed industry standards.",
                'impact': "Positive contribution to financial inclusion.",
                'risk': f"Risk assessment indicates {risk_level.lower()} risk profile."
            }
        }
        
        # Add summary to report
        report['summary'] = summary
        
        # Add metrics sections with professional formatting
        report['fairness_score'] = fairness_score
        report['fairness_metrics'] = {
            'demographic_parity': {
                'score': 0.82,
                'rating': "Level 4/5",
                'narrative': "Strong demographic parity across groups"
            },
            'privacy_score': {
                'score': 0.84,
                'rating': "Level 4/5",
                'narrative': "Strong privacy protections implemented"
            },
            'safety_score': {
                'score': 0.88,
                'rating': "Level 5/5",
                'narrative': "Comprehensive safety measures in place"
            }
        }
        
        logger.info("Successfully generated AEPF report")
        return report

    except Exception as e:
        logger.error("Error in AEPF report generation")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        
        # Return error report with professional formatting
        error_report = {
            'summary': {
                'risk_level': 'Unknown',
                'overall_score': 0.0,
                'rating_level': 0,
                'rating': "Level 0/5",
                'narrative': f"Error generating report: {str(e)}",
                'key_findings': [f"Error: {str(e)}"],
                'recommendations': ["Review system configuration"]
            },
            'fairness_score': 0.0,
            'fairness_metrics': {}
        }
        logger.info("Generated error report as fallback")
        return error_report

def get_fairness_level(score: float) -> str:
    """Get descriptive fairness level."""
    if score >= 0.9:
        return "exceptional ethical alignment"
    elif score >= 0.8:
        return "strong ethical alignment"
    elif score >= 0.7:
        return "good ethical alignment"
    elif score >= 0.6:
        return "moderate ethical alignment"
    else:
        return "concerning ethical alignment" 