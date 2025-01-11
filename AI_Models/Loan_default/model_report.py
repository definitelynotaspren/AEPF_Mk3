"""Module for generating AI model analysis reports."""
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

from .scripts import preprocess_data, generate_report

logger = logging.getLogger(__name__)

def generate_model_report(model_name: str, scenario: str) -> Dict[str, Any]:
    """Generate comprehensive AI model performance report."""
    try:
        # Get base paths
        base_path = Path(__file__).parent
        data_path = base_path / 'data' / f'{scenario.lower()}_data.csv'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run preprocessing
        processed_data = preprocess_data(str(data_path))
        
        # Generate report
        report = generate_report({
            'data': processed_data,
            'scenario': scenario,
            'model_name': model_name,
            'timestamp': timestamp
        })
        
        if 'report_path' not in report:
            raise ValueError("Report generation failed - no report path returned")
            
        logger.info(f"Generated report at: {report['report_path']}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating model report: {e}")
        raise RuntimeError(f"Error generating model report: {e}")

__all__ = ['generate_model_report']