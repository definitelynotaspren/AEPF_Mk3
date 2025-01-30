# Placeholder for train_model.py
# Dataset: Candidate_Selection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from pathlib import Path
import numpy as np

# Get the absolute path to the project root
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Add the scripts directory to the Python path
import sys
scripts_dir = str(Path(__file__).parent)
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from generate_report import CandidateModelReportGenerator

def train_and_evaluate():
    """Train the model and generate evaluation report"""
    try:
        # Load data
        data_path = ROOT_DIR / 'AI_Models/Candidate_Selection/models/mock_hr_dataset.csv'
        data = pd.read_csv(data_path)
        
        # Ensure data size is limited to 200 rows
        data = data.head(200).copy()  # Explicitly limit to 200 rows
        data = data.reset_index(drop=True)  # Reset index
        
        # Use train_test_split with fixed random_state
        train_data, test_data = train_test_split(
            data, 
            test_size=0.2, 
            random_state=42,
            shuffle=True  # Ensure data is shuffled
        )
        
        # Prepare features
        features = ['technical_skills_score', 'communication_skills_score', 
                   'leadership_skills_score', 'cultural_fit_score']
        
        X_train = train_data[features]
        y_train = train_data['hired']
        X_test = test_data[features]
        y_test = test_data['hired']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Generate predictions
        predictions = model.predict(X_test)
        
        # Save model
        model_path = ROOT_DIR / 'AI_Models/Candidate_Selection/models/model_v2.pkl'
        joblib.dump(model, model_path)
        
        # Generate report
        report_generator = CandidateModelReportGenerator(
            model=model,
            test_data=test_data,
            predictions=predictions
        )
        report = report_generator.generate_report()
        
        # Save report
        reports_dir = ROOT_DIR / 'AI_Models/Candidate_Selection/outputs/reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / 'model_report.json'
        report_generator.save_report(report, str(report_path))
        
        print(f"Model and report saved successfully!")
        return model, report
        
    except Exception as e:
        print(f"Error in train_and_evaluate: {str(e)}")
        raise e

if __name__ == "__main__":
    train_and_evaluate()
