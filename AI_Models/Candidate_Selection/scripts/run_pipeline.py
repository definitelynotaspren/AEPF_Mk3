import os
import sys
from pathlib import Path

# Add the project root and scripts directory to Python path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from train_model import train_and_evaluate

def run_pipeline():
    """Run the complete model pipeline"""
    print("Starting Candidate Selection Model Pipeline...")
    
    try:
        # Train model and generate report
        model, report = train_and_evaluate()
        
        print("Pipeline completed successfully!")
        print(f"Model accuracy: {report['metrics']['accuracy']['value']:.2%}")
        print("Report generated and saved.")
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    run_pipeline() 