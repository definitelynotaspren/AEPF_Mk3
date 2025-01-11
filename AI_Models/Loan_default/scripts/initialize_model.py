"""Script to initialize the loan default model and generate sample data."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib

def create_model_and_data():
    # Create directories if they don't exist
    base_path = Path(__file__).parent.parent
    model_dir = base_path / "models"
    data_dir = base_path / "data"
    pred_dir = base_path / "predictions"
    
    for dir_path in [model_dir, data_dir, pred_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Generate sample training data
    n_samples = 1000
    np.random.seed(42)
    
    data = {
        'income': np.random.normal(60000, 20000, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'debt_ratio': np.random.normal(0.3, 0.1, n_samples),
        'employment_length': np.random.normal(5, 2, n_samples)
    }
    
    # Create target variable based on simple rules
    df = pd.DataFrame(data)
    df['default'] = ((df['credit_score'] < 650) | 
                    (df['debt_ratio'] > 0.4) | 
                    (df['income'] < 40000)).astype(int)
    
    # Train the model
    feature_names = ['income', 'credit_score', 'debt_ratio', 'employment_length']
    X = df[feature_names]
    y = df['default']
    model.fit(X, y)
    
    # Add feature names to model
    model.feature_names_ = feature_names
    
    # Save model
    model_path = model_dir / "loan_default_model.pkl"
    joblib.dump(model, model_path)
    
    # Save scenario data
    scenarios = ['default', 'high_risk', 'low_risk']
    
    for scenario in scenarios:
        # Generate scenario-specific data
        if scenario == 'high_risk':
            data['credit_score'] = np.random.normal(650, 50, n_samples)
            data['debt_ratio'] = np.random.normal(0.4, 0.1, n_samples)
        elif scenario == 'low_risk':
            data['credit_score'] = np.random.normal(750, 30, n_samples)
            data['debt_ratio'] = np.random.normal(0.25, 0.05, n_samples)
            
        scenario_df = pd.DataFrame(data)
        scenario_df['default'] = ((scenario_df['credit_score'] < 650) | 
                                (scenario_df['debt_ratio'] > 0.4) | 
                                (scenario_df['income'] < 40000)).astype(int)
        
        # Save scenario data
        data_path = data_dir / f"{scenario}_data.csv"
        scenario_df.to_csv(data_path, index=False)
        
        # Generate predictions
        X_scenario = scenario_df[feature_names]
        y_pred = model.predict(X_scenario)
        y_prob = model.predict_proba(X_scenario)[:, 1]
        
        # Save predictions
        pred_df = pd.DataFrame({
            'predicted_default': y_pred,
            'default_probability': y_prob
        })
        pred_path = pred_dir / f"{scenario}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
    
    print("Model and data initialization complete!")

if __name__ == "__main__":
    create_model_and_data() 