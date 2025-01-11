import pytest
from AEPF_Core.ethical_prisms.human_centric import HumanCentricPrism

@pytest.fixture
def human_centric_prism():
    """Fixture to initialize the Human-Centric Prism."""
    return HumanCentricPrism()

def test_valid_input(human_centric_prism):
    """Test Human-Centric Prism with valid input data."""
    valid_input = {
        'wellbeing_data': [0.8, 0.9, 0.85],
        'autonomy_score': 0.7,
        'fairness_score': 0.9,
        'privacy_score': 0.75,
        'safety_score': 0.95
    }

    results = human_centric_prism.evaluate(valid_input)

    # Check for five output metrics
    assert len(results['metrics']) == 5, "Expected 5 output metrics"

    # Validate the structure of each metric
    for metric, data in results['metrics'].items():
        assert 'value' in data, f"Metric {metric} is missing a 'value'"
        assert 'narrative' in data, f"Metric {metric} is missing a 'narrative'"
        assert isinstance(data['value'], float), f"Metric {metric} value must be a float"
        assert isinstance(data['narrative'], str), f"Metric {metric} narrative must be a string"

def test_missing_input(human_centric_prism):
    """Test Human-Centric Prism with missing input data."""
    invalid_input = {
        'wellbeing_data': [0.8, 0.9, 0.85],
        'autonomy_score': 0.7,
        # 'fairness_score' is missing
        'privacy_score': 0.75,
        'safety_score': 0.95
    }

    with pytest.raises(ValueError, match="Missing required input data: fairness_score"):
        human_centric_prism.evaluate(invalid_input)

def test_invalid_input_type(human_centric_prism):
    """Test Human-Centric Prism with invalid input data type."""
    invalid_input = {
        'wellbeing_data': [0.8, 'invalid', 0.85],  # Invalid type in list
        'autonomy_score': 0.7,
        'fairness_score': 0.9,
        'privacy_score': 0.75,
        'safety_score': 0.95
    }

    with pytest.raises(ValueError, match="All wellbeing data must be numeric."):
        human_centric_prism.evaluate(invalid_input)

def test_empty_input(human_centric_prism):
    """Test Human-Centric Prism with empty input."""
    empty_input = {}

    with pytest.raises(ValueError, match="Missing required input data: wellbeing_data, autonomy_score, fairness_score, privacy_score, safety_score"):
        human_centric_prism.evaluate(empty_input)

def test_output_consistency(human_centric_prism):
    """Test Human-Centric Prism output consistency across repeated evaluations."""
    input_data = {
        'wellbeing_data': [0.8, 0.9, 0.85],
        'autonomy_score': 0.7,
        'fairness_score': 0.9,
        'privacy_score': 0.75,
        'safety_score': 0.95
    }

    result1 = human_centric_prism.evaluate(input_data)
    result2 = human_centric_prism.evaluate(input_data)

    assert result1 == result2, "Output results should be consistent across repeated evaluations"
