import pytest
from AEPF_Core.ethical_prisms.sentient_first import SentientFirstPrism

@pytest.fixture
def sentient_first_prism():
    """Fixture for initializing SentientFirstPrism."""
    return SentientFirstPrism()

def test_valid_input(sentient_first_prism):
    """Test SentientFirstPrism with valid input data."""
    valid_input = {
        'sentient_welfare': 0.85,
        'empathy_score': 0.9,
        'autonomy_respect': 0.75,
        'sentient_safety': 0.8,
        'organisational_welfare': 0.7
    }

    results = sentient_first_prism.evaluate(valid_input)
    assert len(results['metrics']) == 5, "Expected 5 output metrics"
    assert results['prism'] == "Sentient-First"
    for metric, details in results['metrics'].items():
        assert 'value' in details
        assert 'narrative' in details

def test_missing_input(sentient_first_prism):
    """Test SentientFirstPrism with missing input data."""
    invalid_input = {
        'sentient_welfare': 0.85,
        'empathy_score': 0.9,
        'autonomy_respect': 0.75,
        # 'sentient_safety' is missing
        'organisational_welfare': 0.7
    }

    with pytest.raises(ValueError, match="Missing required input data: sentient_safety"):
        sentient_first_prism.evaluate(invalid_input)

def test_invalid_input_type(sentient_first_prism):
    """Test SentientFirstPrism with invalid input data type."""
    invalid_input = {
        'sentient_welfare': 0.85,
        'empathy_score': 0.9,
        'autonomy_respect': 0.75,
        'sentient_safety': "invalid",  # Invalid type
        'organisational_welfare': 0.7
    }

    with pytest.raises(ValueError, match="sentient_safety must be a numeric value between 0 and 1."):
        sentient_first_prism.evaluate(invalid_input)

def test_empty_input(sentient_first_prism):
    """Test SentientFirstPrism with empty input."""
    empty_input = {}

    with pytest.raises(ValueError, match="Missing required input data: sentient_welfare"):
        sentient_first_prism.evaluate(empty_input)

def test_out_of_bounds_input(sentient_first_prism):
    """Test SentientFirstPrism with input values out of bounds."""
    invalid_input = {
        'sentient_welfare': 1.2,  # Out of bounds
        'empathy_score': 0.9,
        'autonomy_respect': 0.75,
        'sentient_safety': 0.8,
        'organisational_welfare': 0.7
    }

    with pytest.raises(ValueError, match="sentient_welfare must be a numeric value between 0 and 1."):
        sentient_first_prism.evaluate(invalid_input)
