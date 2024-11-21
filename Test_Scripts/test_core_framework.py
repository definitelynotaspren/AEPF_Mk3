import pytest
import warnings
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AEPF_Core.ethical_governor import EthicalGovernor
from AEPF_Core.context_engine import ContextEngine
from AEPF_Core.location_context import LocationContextManager, RegionalContext

class TestAEPFCore:
    @pytest.fixture
    def setup_core_components(self):
        """Setup core components with mocked dependencies"""
        context_engine = Mock(spec=ContextEngine)
        governor = EthicalGovernor(context_engine)
        return {
            'context_engine': context_engine,
            'governor': governor
        }

    def test_end_to_end_decision_flow(self, setup_core_components):
        """Test the complete decision-making flow through the core framework"""
        components = setup_core_components
        governor = components['governor']
        
        # Test scenario with proper criteria structure for all prisms
        scenario_data = {
            'type': 'data_processing',
            'impact_scope': 'individual',
            'severity': 'high',
            # Required criteria for all prisms
            'wellbeing': {
                'safety': 0.8,
                'mental_health': 0.7,
                'physical_health': 0.9
            },
            'fairness': {
                'equal_opportunity': 0.8,
                'non_discrimination': 0.9
            },
            'autonomy': {
                'freedom_of_choice': 0.7,
                'agency': 0.8
            },
            'privacy': {
                'data_protection': 0.9,
                'information_access': 0.8
            },
            'sentient_impact': {
                'direct_impact': 0.8,
                'indirect_impact': 0.7,
                'long_term_impact': 0.6
            },
            'organizational_welfare': {
                'employee_wellbeing': 0.8,
                'stakeholder_benefits': 0.7
            },
            'resource_sustainability': {
                'resource_efficiency': 0.8,
                'long_term_viability': 0.7
            }
        }
        
        result = governor.evaluate_action({'action': 'test'}, scenario_data)
        
        # Verify result structure
        assert isinstance(result.confidence_score, float)
        assert 0 <= result.confidence_score <= 1.0
        assert isinstance(result.prism_scores, dict)
        
        # Verify prism-specific results
        assert 'sentient_first' in result.prism_scores
        prism_result = result.prism_scores['sentient_first']
        assert isinstance(prism_result, dict)
        assert 'score' in prism_result
        assert 'details' in prism_result

    def test_error_handling(self, setup_core_components):
        """Test comprehensive error handling in the core framework"""
        components = setup_core_components
        governor = components['governor']

        # Test with None inputs
        with pytest.raises(ValueError, match="Action and context must not be None"):
            governor.evaluate_action(None, None)

        # Test with invalid action type
        with pytest.raises(TypeError, match="Action must be string or dict"):
            governor.evaluate_action(123, {})

        # Test with invalid context type
        with pytest.raises(TypeError, match="Context must be dict"):
            governor.evaluate_action("test", "invalid")

        # Test with missing required criteria
        invalid_context = {
            'type': 'data_processing',
            'impact_scope': 'individual',
        }
        with pytest.raises(ValueError) as exc_info:
            governor.evaluate_action({'action': 'test'}, invalid_context)
        assert "No valid criteria provided for evaluation" in str(exc_info.value)

        # Test with invalid criteria values
        invalid_values_context = {
            'type': 'data_processing',
            'impact_scope': 'individual',
            'sentient_impact': {
                'direct_impact': 1.5,  # Invalid value > 1
                'indirect_impact': -0.1  # Invalid value < 0
            }
        }
        with pytest.raises(ValueError) as exc_info:
            governor.evaluate_action({'action': 'test'}, invalid_values_context)
        assert "Invalid" in str(exc_info.value)
        assert "sentient_impact" in str(exc_info.value)

    def test_partial_criteria_handling(self, setup_core_components):
        """Test handling of partially valid criteria sets"""
        components = setup_core_components
        governor = components['governor']
        
        # Test with partial but valid criteria
        partial_valid_context = {
            'type': 'data_processing',
            'impact_scope': 'individual',
            'severity': 'high',
            # Include minimum required criteria for at least one prism
            'sentient_impact': {
                'direct_impact': 0.8,
                'indirect_impact': 0.7
            },
            'organizational_welfare': {
                'employee_wellbeing': 0.8,
                'stakeholder_benefits': 0.7
            },
            'resource_sustainability': {
                'resource_efficiency': 0.8,
                'long_term_viability': 0.7
            }
        }
        
        # Should log warnings but not fail for partial criteria
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = governor.evaluate_action({'action': 'test'}, partial_valid_context)
            
            # Verify warnings were logged
            assert len(w) > 0
            assert any("Missing criteria" in str(warning.message) for warning in w)
            
            # Verify result still contains valid data
            assert isinstance(result.confidence_score, float)
            assert len(result.reasoning) > 0

if __name__ == '__main__':
    pytest.main([__file__]) 