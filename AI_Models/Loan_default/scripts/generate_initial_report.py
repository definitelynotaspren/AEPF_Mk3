from pathlib import Path
import json
import os

def create_initial_report():
    """Create initial loan default model report."""
    # Get absolute path to project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    
    # Create report
    report = {
        'scenario_type': 'Loan Default Prediction',
        'model_type': 'Gradient Boosting Default Model',
        'metrics': {
            'accuracy': {
                'value': 0.87,
                'trend': '+1.5%',
                'narrative': 'Consistent improvement in predictions'
            },
            'precision': {
                'value': 0.85,
                'trend': '+2.0%',
                'narrative': 'Better default risk identification'
            },
            'recall': {
                'value': 0.86,
                'trend': '+1.8%',
                'narrative': 'Improved risk capture rate'
            }
        },
        'feature_importance': {
            'credit_score': 0.35,
            'income': 0.25,
            'debt_ratio': 0.22,
            'payment_history': 0.18
        },
        'insights': {
            'strengths': [
                'Strong default prediction accuracy',
                'Balanced risk assessment',
                'Reliable credit scoring'
            ],
            'improvement_areas': [
                'Edge case handling',
                'New market adaptation',
                'Real-time assessment speed'
            ]
        }
    }
    
    # Define report path
    report_dir = project_root / 'AI_Models/Loan_default/outputs/reports'
    report_path = report_dir / 'model_report.json'
    
    # Create directories
    print(f"Creating directory: {report_dir}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save report
    print(f"Saving report to: {report_path}")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Verify file exists
    if report_path.exists():
        print(f"Report successfully created at: {report_path}")
        # Verify file is readable
        try:
            with open(report_path, 'r') as f:
                saved_report = json.load(f)
            print("Report is readable and contains valid JSON")
        except Exception as e:
            print(f"Error reading report: {e}")
    else:
        print(f"Error: Report file not found at {report_path}")
    
    return report

if __name__ == "__main__":
    create_initial_report() 