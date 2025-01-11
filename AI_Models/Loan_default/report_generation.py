"""Module for generating loan default analysis reports."""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import logging
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

def load_model_data(model_name: str) -> pd.DataFrame:
    """Load model training and evaluation data."""
    try:
        # Simulate loading data
        n_samples = 1000
        return pd.DataFrame({
            'loan_amount': np.random.uniform(5000, 50000, n_samples),
            'credit_score': np.random.normal(700, 50, n_samples),
            'interest_rate': np.random.uniform(0.05, 0.15, n_samples),
            'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        })
    except Exception as e:
        logger.error(f"Error loading model data: {e}")
        return pd.DataFrame()

def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate model performance metrics."""
    try:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}

def calculate_risk_distribution(data: pd.DataFrame) -> Dict[str, int]:
    """Calculate risk level distribution."""
    try:
        risk_levels = pd.qcut(data['credit_score'], q=3, labels=['High', 'Medium', 'Low'])
        return risk_levels.value_counts().to_dict()
    except Exception as e:
        logger.error(f"Error calculating risk distribution: {e}")
        return {}

def identify_high_risk_factors(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify high risk factors in loan applications."""
    try:
        risk_factors = []
        # Add risk factor analysis
        if data['credit_score'].mean() < 650:
            risk_factors.append({
                'factor': 'Low Credit Scores',
                'severity': 'High',
                'description': 'Average credit score below 650'
            })
        return risk_factors
    except Exception as e:
        logger.error(f"Error identifying risk factors: {e}")
        return []

def generate_risk_mitigation_strategies(risk_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate risk mitigation strategies."""
    try:
        strategies = []
        for factor in risk_factors:
            strategies.append({
                'risk_factor': factor['factor'],
                'strategy': f"Implement stricter controls for {factor['factor'].lower()}",
                'priority': factor['severity']
            })
        return strategies
    except Exception as e:
        logger.error(f"Error generating strategies: {e}")
        return []

def generate_html_report(data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
    """Generate HTML report content."""
    try:
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric { margin: 20px 0; padding: 10px; background: #f5f5f5; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Loan Default Analysis Report</h1>
        """
        
        # Add performance metrics
        metrics = analysis_results.get('performance_metrics', {})
        html += "<h2>Model Performance</h2>"
        for metric, value in metrics.items():
            html += f"""
            <div class='metric'>
                <h3>{metric.replace('_', ' ').title()}</h3>
                <p>{value:.2%}</p>
            </div>
            """
        
        # Add risk analysis
        risk_dist = analysis_results.get('risk_distribution', {})
        html += "<h2>Risk Distribution</h2>"
        for level, count in risk_dist.items():
            html += f"<p>{level} Risk: {count}</p>"
        
        html += "</body></html>"
        return html
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")
        return ""

def run(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive loan default analysis report."""
    try:
        data = params['data']
        model_name = params.get('model_name', 'default')
        
        # Generate analysis
        risk_dist = calculate_risk_distribution(data)
        risk_factors = identify_high_risk_factors(data)
        mitigation_strategies = generate_risk_mitigation_strategies(risk_factors)
        
        # Compile results
        results = {
            'risk_distribution': risk_dist,
            'risk_factors': risk_factors,
            'mitigation_strategies': mitigation_strategies
        }
        
        # Generate HTML report
        html_content = generate_html_report(data, results)
        
        # Save report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("AI_Models/Loan_default/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"detailed_report_{timestamp}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Generated report at: {report_path}")
        results['report_path'] = str(report_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise

# Explicitly export the run function
__all__ = ['run'] 