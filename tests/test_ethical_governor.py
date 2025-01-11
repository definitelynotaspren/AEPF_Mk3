import pytest
from AEPF_Core.ethical_governor import EthicalGovernor

@pytest.fixture
def ethical_governor():
    """Fixture to initialize EthicalGovernor."""
    return EthicalGovernor()

def test_evaluate_with_complete_input(ethical_governor):
    input_data = {
        "ecocentric": {
            "environmental_impact": 0.7,
            "biodiversity_preservation": 0.8,
            "carbon_neutrality": 0.9,
            "water_conservation": 0.6,
            "renewable_resource_use": 0.7
        },
        "equity_focused": {
            "bias_mitigation_score": 0.7,
            "accessibility_score": 0.8,
            "representation_score": 0.6,
            "demographic_equity_score": 0.9,
            "resource_fairness_score": 0.7
        },
        "human_centric": {
            "wellbeing_data": 0.9,
            "autonomy_score": 0.8,
            "fairness_score": 0.7,
            "privacy_score": 0.8,
            "safety_score": 0.9
        },
        "innovation_focused": {
            "financial_risk": 0.4,
            "reputational_risk": 0.6,
            "technological_risk": 0.5,
            "economic_benefit": 0.8,
            "societal_benefit": 0.9
        },
        "sentient_first": {
            "sentience_recognition": 0.6,
            "sentient_welfare": 0.7,
            "empathy_score": 0.8
        }
    }
    context_parameters = {"severity": "high", "impact_scope": "global"}
    result = ethical_governor.evaluate(input_data, context_parameters)

    assert "summary" in result
    assert "full_report" in result
    assert 0 <= result["summary"]["final_score"] <= 1
    assert 0 <= result["summary"]["five_star_rating"] <= 5

def test_missing_input(ethical_governor):
    incomplete_data = {"ecocentric": {"environmental_impact": 0.7}}
    context_parameters = {"severity": "medium", "impact_scope": "regional"}
    result = ethical_governor.evaluate(incomplete_data, context_parameters)

    assert result["summary"]["final_score"] < 0.5
