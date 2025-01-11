import pytest
from AEPF_Core.ethical_prisms.ecocentric import EcocentricPrism


@pytest.fixture
def ecocentric_prism():
    """Create an EcocentricPrism instance for testing."""
    return EcocentricPrism()

def test_valid_input(ecocentric_prism):
    """Test EcocentricPrism with valid input data."""
    valid_input = {
        'environmental_impact': 0.9,
        'biodiversity_preservation': 0.8,
        'carbon_neutrality': 0.7,
        'water_conservation': 0.6,
        'renewable_resource_use': 0.9
    }

    results = ecocentric_prism.evaluate(valid_input)

    assert results['prism'] == 'Ecocentric'
    assert 'metrics' in results
    metrics = results['metrics']

    # Check all expected metrics are present
    assert 'environmental_score' in metrics
    assert 'biodiversity_score' in metrics
    assert 'carbon_score' in metrics
    assert 'water_score' in metrics
    assert 'renewable_score' in metrics

    # Verify metric values
    assert metrics['environmental_score']['value'] == 0.9
    assert metrics['biodiversity_score']['value'] == 0.8
    assert metrics['carbon_score']['value'] == 0.7
    assert metrics['water_score']['value'] == 0.6
    assert metrics['renewable_score']['value'] == 0.9


def test_missing_input(ecocentric_prism):
    """Test EcocentricPrism with missing input data."""
    invalid_input = {
        'environmental_impact': 0.9,
        'biodiversity_preservation': 0.8,
        # 'carbon_neutrality' is missing
        'water_conservation': 0.6,
        'renewable_resource_use': 0.9
    }

    with pytest.raises(ValueError, match="Missing required input data: carbon_neutrality"):
        ecocentric_prism.evaluate(invalid_input)


def test_invalid_input_type(ecocentric_prism):
    """Test EcocentricPrism with invalid input data type."""
    invalid_input = {
        'environmental_impact': "invalid",  # Invalid type
        'biodiversity_preservation': 0.8,
        'carbon_neutrality': 0.7,
        'water_conservation': 0.6,
        'renewable_resource_use': 0.9
    }

    with pytest.raises(ValueError, match="environmental_impact must be a numeric value between 0 and 1."):
        ecocentric_prism.evaluate(invalid_input)


def test_out_of_bounds_input(ecocentric_prism):
    """Test EcocentricPrism with out-of-bounds input values."""
    invalid_input = {
        'environmental_impact': 1.2,  # Out of bounds
        'biodiversity_preservation': 0.8,
        'carbon_neutrality': 0.7,
        'water_conservation': 0.6,
        'renewable_resource_use': 0.9
    }

    with pytest.raises(ValueError, match="Value 1.2 for environmental_impact must be between 0 and 1"):
        ecocentric_prism.evaluate(invalid_input)


def test_empty_input(ecocentric_prism):
    """Test EcocentricPrism with empty input."""
    empty_input = {}

    with pytest.raises(ValueError, match="Missing required input data: environmental_impact, biodiversity_preservation, carbon_neutrality, water_conservation, renewable_resource_use"):
        ecocentric_prism.evaluate(empty_input)


def test_low_impact_scenario(ecocentric_prism):
    """Test EcocentricPrism with a low-impact scenario."""
    low_impact_input = {
        'environmental_impact': 0.2,  # Low impact
        'biodiversity_preservation': 0.3,
        'carbon_neutrality': 0.4,
        'water_conservation': 0.5,
        'renewable_resource_use': 0.6
    }

    results = ecocentric_prism.evaluate(low_impact_input)

    metrics = results['metrics']
    assert metrics['environmental_score']['value'] == 0.2
    assert metrics['biodiversity_score']['value'] == 0.3
    assert metrics['carbon_score']['value'] == 0.4
    assert metrics['water_score']['value'] == 0.5
    assert metrics['renewable_score']['value'] == 0.6
