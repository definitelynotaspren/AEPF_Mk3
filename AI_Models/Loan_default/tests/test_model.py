import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from loan_default.scripts.train_model import train_xgboost_model

@pytest.fixture
def sample_processed_data():
    """Create sample processed data for model training."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'term': np.random.choice([36, 60], n_samples),
        'interest_rate': np.random.uniform(5, 15, n_samples),
        'employment_length': np.random.randint(0, 20, n_samples),
        'home_ownership': np.random.randint(0, 3, n_samples),
        'annual_income': np.random.uniform(30000, 150000, n_samples),
        'verification_status': np.random.randint(0, 2, n_samples),
        'purpose': np.random.randint(0, 5, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'delinq_2yrs': np.random.randint(0, 5, n_samples),
        'credit_score': np.random.uniform(300, 850, n_samples),
        'loan_status': np.random.randint(0, 2, n_samples)  # Target variable
    })

def test_model_training(sample_processed_data, tmp_path):
    """Test if model training completes successfully."""
    data_path = tmp_path / 'processed_data.csv'
    model_path = tmp_path / 'model.joblib'
    
    sample_processed_data.to_csv(data_path, index=False)
    
    # Train model
    train_xgboost_model(str(data_path), str(model_path))
    
    # Check if model file exists
    assert model_path.exists()

def test_model_training_invalid_data(tmp_path):
    """Test model training with invalid data."""
    data_path = tmp_path / 'invalid_data.csv'
    model_path = tmp_path / 'model.joblib'
    
    # Create invalid data
    invalid_data = pd.DataFrame({
        'invalid_column': ['a', 'b', 'c']
    })
    invalid_data.to_csv(data_path, index=False)
    
    with pytest.raises(Exception):
        train_xgboost_model(str(data_path), str(model_path)) 