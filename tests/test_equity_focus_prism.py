import pytest
from AEPF_Core.ethical_prisms.equity_focused import EquityFocusedPrism


@pytest.fixture
def equity_focused_prism():
    """Fixture to initialize EquityFocusedPrism."""
    return EquityFocusedPrism()

def test_valid_input(equity_focused_prism):
    """Test EquityFocusedPrism with valid input data."""
    valid_input = {
        "bias_mitigation_score": 0.85,
        "accessibility_score": 0.9,
        "representation_score": 0.75,
        "demographic_equity_score": 0.8,
        "resource_fairness_score": 0.7
    }

    results = equity_focused_prism.evaluate(valid_input)
    assert results["prism"] == "Equity-Focused"
    assert "metrics" in results
    metrics = results["metrics"]

    # Check all expected metrics
    for metric in ["bias_mitigation", "accessibility", "representation", "demographic_equity", "resource_fairness"]:
        assert metric in metrics
        assert "value" in metrics[metric]
        assert "narrative" in metrics[metric]

    # Validate metric values
    assert metrics["bias_mitigation"]["value"] == 0.85
    assert metrics["accessibility"]["value"] == 0.9
    assert metrics["representation"]["value"] == 0.75
    assert metrics["demographic_equity"]["value"] == 0.8
    assert metrics["resource_fairness"]["value"] == 0.7


def test_missing_input(equity_focused_prism):
    """Test EquityFocusedPrism with missing input data."""
    invalid_input = {
        "bias_mitigation_score": 0.85,
        "accessibility_score": 0.9,
        # "representation_score" is missing
        "demographic_equity_score": 0.8,
        "resource_fairness_score": 0.7
    }

    with pytest.raises(ValueError, match="Missing required input data: representation_score"):
        equity_focused_prism.evaluate(invalid_input)


def test_invalid_input_type(equity_focused_prism):
    """Test EquityFocusedPrism with invalid input data type."""
    invalid_input = {
        "bias_mitigation_score": 0.85,
        "accessibility_score": 0.9,
        "representation_score": "invalid",  # Invalid type
        "demographic_equity_score": 0.8,
        "resource_fairness_score": 0.7
    }

    with pytest.raises(ValueError, match="representation_score must be a numeric value between 0 and 1."):
        equity_focused_prism.evaluate(invalid_input)


def test_out_of_bounds_input(equity_focused_prism):
    """Test EquityFocusedPrism with input values out of bounds."""
    invalid_input = {
        "bias_mitigation_score": 1.2,  # Out of bounds
        "accessibility_score": 0.9,
        "representation_score": 0.75,
        "demographic_equity_score": 0.8,
        "resource_fairness_score": 0.7
    }

    with pytest.raises(ValueError, match="bias_mitigation_score must be a numeric value between 0 and 1."):
        equity_focused_prism.evaluate(invalid_input)


def test_empty_input(equity_focused_prism):
    """Test EquityFocusedPrism with empty input."""
    empty_input = {}

    with pytest.raises(ValueError, match="Missing required input data: bias_mitigation_score"):
        equity_focused_prism.evaluate(empty_input)
