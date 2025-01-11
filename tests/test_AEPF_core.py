import pytest
from AEPF_Core.ethical_governor import EthicalGovernor

@pytest.fixture
def ethical_governor():
    """Fixture to initialize EthicalGovernor."""
    return EthicalGovernor()

def test_end_to_end_valid_input(ethical_governor):
    """Test the full AEPF Core workflow with valid inputs."""
    input_data = {
        "ecocentric": {
            "environmental_impact": 0.8,
            "biodiversity_preservation": 0.7,
            "carbon_neutrality": 0.9,
            "water_conservation": 0.6,
            "renewable_resource_use": 0.85
        },
        "equity_focused": {
            "bias_mitigation_score": 0.75,
            "accessibility_score": 0.8,
            "representation_score": 0.7,
            "demographic_equity_score": 0.9,
            "resource_fairness_score": 0.6
        },
        "human_centric": {
            "wellbeing_data": 0.9,
            "autonomy_score": 0.85,
            "fairness_score": 0.75,
            "privacy_score": 0.8,
            "safety_score": 0.9
        },
        "innovation_focused": {
            "financial_risk": 0.4,
            "reputational_risk": 0.65,
            "technological_risk": 0.6,
            "economic_benefit": 0.8,
            "societal_benefit": 0.85
        },
        "sentient_first": {
            "sentience_recognition": 0.7,
            "sentient_welfare": 0.75,
            "empathy_score": 0.8
        }
    }
    context_parameters = {
        "severity": "critical",
        "impact_scope": "global"
    }

    result = ethical_governor.evaluate(input_data, context_parameters)

    # Verify the structure of the output
    assert "summary" in result
    assert "full_report" in result

    summary = result["summary"]
    full_report = result["full_report"]

    # Validate summary content
    assert "final_score" in summary
    assert "five_star_rating" in summary
    assert "detected_scenario" in summary
    assert 0 <= summary["final_score"] <= 1
    assert 0 <= summary["five_star_rating"] <= 5

    # Validate full report content
    assert "context_analysis" in full_report
    assert "prism_results" in full_report
    assert "weighted_scores" in full_report
    assert "normalized_scores" in full_report

    # Ensure all prisms are evaluated
    for prism in input_data.keys():
        assert prism in full_report["prism_results"]

    # Check the normalized scores sum to 1
    assert abs(sum(full_report["normalized_scores"].values()) - 1) < 0.01

def test_end_to_end_invalid_inputs(ethical_governor):
    """Test the AEPF Core with invalid inputs."""
    input_data = {
        "ecocentric": {
            "environmental_impact": 1.5,  # Invalid: out of range
        },
        "equity_focused": {
            "bias_mitigation_score": "invalid"  # Invalid: non-numeric
        },
        "human_centric": {},
        "innovation_focused": {
            "financial_risk": None  # Invalid: None value
        },
        "sentient_first": {
            "sentience_recognition": -0.5  # Invalid: out of range
        }
    }
    context_parameters = {
        "severity": "low",
        "impact_scope": "individual"
    }

    result = ethical_governor.evaluate(input_data, context_parameters)

    # Verify that errors are handled gracefully
    assert "summary" in result
    assert "full_report" in result

    summary = result["summary"]
    full_report = result["full_report"]

    assert summary["final_score"] == 0  # No valid data to calculate score
    assert summary["five_star_rating"] == 0

    # Ensure each invalid input is logged and handled
    for prism, output in full_report["prism_results"].items():
        assert "error" in output

def test_end_to_end_context_weighting(ethical_governor):
    """Test the effect of context weighting on prism scores."""
    input_data = {
        "ecocentric": {
            "environmental_impact": 0.8,
            "biodiversity_preservation": 0.7,
            "carbon_neutrality": 0.6,
            "water_conservation": 0.75,
            "renewable_resource_use": 0.85
        },
        "equity_focused": {
            "bias_mitigation_score": 0.7,
            "accessibility_score": 0.65,
            "representation_score": 0.75,
            "demographic_equity_score": 0.8,
            "resource_fairness_score": 0.7
        }
    }
    context_parameters = {
        "severity": "high",
        "impact_scope": "regional"
    }

    result = ethical_governor.evaluate(input_data, context_parameters)

    full_report = result["full_report"]

    # Validate context weighting is applied
    assert "context_analysis" in full_report
    context_analysis = full_report["context_analysis"]
    assert "weight_adjustments" in context_analysis

    weight_adjustments = context_analysis["weight_adjustments"]
    weighted_scores = full_report["weighted_scores"]

    for prism, weight in weight_adjustments.items():
        assert prism in weighted_scores
        assert weighted_scores[prism] == weight * full_report["prism_results"][prism]["score"]

