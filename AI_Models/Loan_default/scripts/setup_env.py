"""Environment setup script."""
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Install required packages and verify setup."""
    try:
        # Get requirements file path
        base_path = Path(__file__).parent.parent
        requirements_path = base_path / 'requirements.txt'
        
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            str(requirements_path)
        ])
        
        print("\nVerifying installations...")
        # Import test for each required package
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import accuracy_score
        import xgboost
        import joblib
        
        print("\nAll required packages installed successfully!")
        
        # Create required directories
        directories = [
            base_path / 'data' / 'raw',
            base_path / 'data' / 'processed',
            base_path / 'models',
            base_path / 'outputs',
            base_path / 'reports'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"Error importing required package: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_environment() 