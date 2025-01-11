import pytest
from pathlib import Path
import pandas as pd
from loan_default.scripts.pipeline import LoanDefaultPipeline

@pytest.fixture
def pipeline():
    """Create a pipeline instance for testing."""
    return LoanDefaultPipeline()

def test_directory_setup(pipeline, tmp_path):
    """Test directory creation and structure."""
    # Override base path for testing
    pipeline.base_path = tmp_path
    
    # Update all directory paths based on new base path
    pipeline.data_dir = tmp_path / 'data'
    pipeline.raw_dir = pipeline.data_dir / 'raw'
    pipeline.processed_dir = pipeline.data_dir / 'processed'
    pipeline.models_dir = tmp_path / 'models'
    pipeline.outputs_dir = tmp_path / 'outputs'
    
    # Create directories
    pipeline._setup_directories()
    
    # Verify all required directories exist
    assert (tmp_path / 'data' / 'raw').exists()
    assert (tmp_path / 'data' / 'processed').exists()
    assert (tmp_path / 'models').exists()
    assert (tmp_path / 'outputs').exists()

def test_preprocessing_missing_data(pipeline):
    """Test preprocessing with missing input data."""
    result = pipeline.run_preprocessing()
    assert not result  # Should fail when input file doesn't exist

def test_model_training_missing_data(pipeline):
    """Test model training with missing processed data."""
    result = pipeline.train_model()
    assert not result  # Should fail when processed data doesn't exist

def test_prediction_missing_model(pipeline):
    """Test prediction with missing model file."""
    result = pipeline.generate_predictions()
    assert not result  # Should fail when model doesn't exist 