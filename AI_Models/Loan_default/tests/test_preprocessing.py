import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from loan_default.scripts.preprocessing import preprocess_data
from typing import Any

@pytest.fixture
def sample_data():
    """Create sample loan data for testing."""
    return pd.DataFrame({
        'loan_amount': [10000, 20000, 15000],
        'term': [36, 60, 36],
        'interest_rate': [5.5, 6.0, 5.8],
        'employment_length': [5, 10, 3],
        'home_ownership': ['RENT', 'OWN', 'MORTGAGE'],
        'annual_income': [50000, 75000, 60000],
        'verification_status': ['Verified', 'Not Verified', 'Verified'],
        'purpose': ['debt_consolidation', 'home_improvement', 'credit_card'],
        'dti': [15.5, 20.0, 18.5],
        'delinq_2yrs': [0, 1, 0],
        'credit_score': [680, 720, 700]
    })

def test_preprocessing_output_shape(sample_data, tmp_path):
    """Test if preprocessing maintains expected data shape."""
    input_path = tmp_path / 'input.csv'
    output_path = tmp_path / 'output.csv'
    
    # Save sample data
    sample_data.to_csv(input_path, index=False)
    
    # Run preprocessing
    preprocess_data(str(input_path), str(output_path))
    
    # Load processed data
    processed_data = pd.read_csv(output_path)
    
    # Check shape
    assert processed_data.shape[0] == sample_data.shape[0]
    assert all(col in processed_data.columns for col in sample_data.columns)

def test_preprocessing_handles_missing_values(sample_data, tmp_path):
    """Test handling of missing values."""
    # Add some missing values
    sample_data.loc[0, 'loan_amount'] = float('nan')
    
    input_path = tmp_path / 'input.csv'
    output_path = tmp_path / 'output.csv'
    
    sample_data.to_csv(input_path, index=False)
    preprocess_data(str(input_path), str(output_path))
    
    processed_data = pd.read_csv(output_path)
    assert not processed_data['loan_amount'].isnull().any()

def test_preprocessing_categorical_encoding(sample_data, tmp_path):
    """Test categorical variable encoding."""
    input_path = tmp_path / 'input.csv'
    output_path = tmp_path / 'output.csv'
    
    sample_data.to_csv(input_path, index=False)
    preprocess_data(str(input_path), str(output_path))
    
    processed_data = pd.read_csv(output_path)
    categorical_cols = ['home_ownership', 'verification_status', 'purpose']
    
    for col in categorical_cols:
        assert processed_data[col].dtype in ['int64', 'float64'] 