import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
import xgboost as xgb
from loan_default.scripts.predict import generate_predictions

@pytest.fixture
def sample_model(tmp_path):
    """Create a sample XGBoost model for testing."""
    model = xgb.XGBClassifier(random_state=42)
    X = np.random.rand(100, 11)  # 11 features
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    model_path = tmp_path / 'test_model.joblib'
    dump(model, model_path)
    return model_path

@pytest.fixture
def sample_test_data():
    """Create sample test data for predictions."""
    return pd.DataFrame({
        'loan_amount': np.random.uniform(1000, 50000, 10),
        'term': np.random.choice([36, 60], 10),
        'interest_rate': np.random.uniform(5, 15, 10),
        'employment_length': np.random.randint(0, 20, 10),
        'home_ownership': np.random.randint(0, 3, 10),
        'annual_income': np.random.uniform(30000, 150000, 10),
        'verification_status': np.random.randint(0, 2, 10),
        'purpose': np.random.randint(0, 5, 10),
        'dti': np.random.uniform(0, 40, 10),
        'delinq_2yrs': np.random.randint(0, 5, 10),
        'credit_score': np.random.uniform(300, 850, 10)
    })

def test_prediction_output(sample_model, sample_test_data, tmp_path):
    """Test if predictions are generated correctly."""
    data_path = tmp_path / 'test_data.csv'
    output_path = tmp_path / 'predictions.csv'
    
    sample_test_data.to_csv(data_path, index=False)
    
    generate_predictions(
        str(sample_model),
        str(data_path),
        str(output_path)
    )
    
    # Load predictions
    predictions = pd.read_csv(output_path)
    
    # Check predictions format
    assert 'default_probability' in predictions.columns
    assert 'predicted_status' in predictions.columns
    assert len(predictions) == len(sample_test_data)

def test_prediction_invalid_model_path(sample_test_data, tmp_path):
    """Test handling of invalid model path."""
    data_path = tmp_path / 'test_data.csv'
    output_path = tmp_path / 'predictions.csv'
    invalid_model_path = tmp_path / 'nonexistent_model.joblib'
    
    sample_test_data.to_csv(data_path, index=False)
    
    with pytest.raises(FileNotFoundError):
        generate_predictions(
            str(invalid_model_path),
            str(data_path),
            str(output_path)
        ) 