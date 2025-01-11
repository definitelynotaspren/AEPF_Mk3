import pytest
from AEPF_Core.ethical_prisms.innovation_focused import InnovationFocusedPrism


@pytest.fixture
def innovation_focused_prism():
    """Fixture to initialize InnovationFocusedPrism."""
    return InnovationFocusedPrism()

def test_valid_input(innovation_focused_prism):
    """Test InnovationFocusedPrism with valid input data."""
    valid_input = {
        "financial_risk": 0.3,
        "reputational_risk": 0.4,
        "technological_risk": 0.2,
        "economic_benefit": 0.8,
        "societal_benefit": 0.9
    }

    results = innovation_focused_prism.evaluate(valid_input)
    assert results["prism"] == "Innovation-Focused"
    assert "metrics" in results
    metrics = results["metrics"]

    # Check all expected metrics
    for metric in ["financial_risk_score", "reputational_risk_score", "technological_risk_score", "economic_benefit_score", "societal_benefit_score"]:
        assert metric in metrics
        assert "value" in metrics[metric]
        assert "narrative" in metrics[metric]

    # Validate metric values
    assert metrics["financial_risk_score"]["value"] == 0.7  # 1 - 0.3
    assert metrics["reputational_risk_score"]["value"] == 0.6  # 1 - 0.4
    assert metrics["technological_risk_score"]["value"] == 0.8  # 1 - 0.2
    assert metrics["economic_benefit_score"]["value"] == 0.8
    assert metrics["societal_benefit_score"]["value"] == 0.9


def test_missing_input(innovation_focused_prism):
    """Test InnovationFocusedPrism with missing input data."""
    invalid_input = {
        "financial_risk": 0.3,
        "reputational_risk": 0.4,
        # "technological_risk" is missing
        "economic_benefit": 0.8,
        "societal_benefit": 0.9
    }

    with pytest.raises(ValueError, match="Missing required input data: technological_risk"):
        innovation_focused_prism.evaluate(invalid_input)


def test_invalid_input_type(innovation_focused_prism):
    """Test InnovationFocusedPrism with invalid input data type."""
    invalid_input = {
        "financial_risk": 0.3,
        "reputational_risk": "invalid",  # Invalid type
        "technological_risk": 0.2,
        "economic_benefit": 0.8,
        "societal_benefit": 0.9
    }

    with pytest.raises(ValueError, match="reputational_risk must be a numeric value between 0 and 1."):
        innovation_focused_prism.evaluate(invalid_input)


def test_out_of_bounds_input(innovation_focused_prism):
    """Test InnovationFocusedPrism with input values out of bounds."""
    invalid_input = {
        "financial_risk": 1.2,  # Out of bounds
        "reputational_risk": 0.4,
        "technological_risk": 0.2,
        "economic_benefit": 0.8,
        "societal_benefit": 0.9
    }

    with pytest.raises(ValueError, match="financial_risk must be a numeric value between 0 and 1."):
        innovation_focused_prism.evaluate(invalid_input)


def test_empty_input(innovation_focused_prism):
    """Test InnovationFocusedPrism with empty input."""
    empty_input = {}

    with pytest.raises(ValueError, match="Missing required input data: financial_risk"):
        innovation_focused_prism.evaluate(empty_input)
