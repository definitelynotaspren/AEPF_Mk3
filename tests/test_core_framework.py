import pytest
import warnings
from src.ethical_governor import EthicalGovernor
from collections import defaultdict

class TestEthicalGovernor:
    @pytest.fixture
    def governor(self):
        return EthicalGovernor()

    @pytest.fixture
    def complete_valid_context(self):
        """Fixture providing a complete set of valid criteria for all prisms"""
        return {
            # Environmental prism
            'environmental_impact': 0.75,
            'sustainability': 0.80,
            
            # Social prism
            'social_impact': 0.85,
            'fairness': 0.70,
            'transparency': 0.90,
            
            # Wellbeing prism
            'wellbeing': 0.80,
            'safety': 0.85,
            'privacy': 0.75,
            
            # Governance prism
            'accountability': 0.80
        }

    def test_comprehensive_valid_criteria(self, governor, complete_valid_context):
        """Test evaluation with complete valid criteria across all prisms"""
        result = governor.evaluate_action(complete_valid_context)
        assert result == True

    def test_prism_combinations(self, governor, complete_valid_context):
        """Test various combinations of prism values"""
        test_cases = [
            # All high values
            {
                **complete_valid_context,
                'environmental_impact': 0.9,
                'sustainability': 0.95,
                'social_impact': 0.85,
                'fairness': 0.90,
                'transparency': 0.95
            },
            # All minimum acceptable values
            {
                **complete_valid_context,
                'environmental_impact': 0.5,
                'sustainability': 0.5,
                'social_impact': 0.5,
                'fairness': 0.5,
                'transparency': 0.5
            },
            # Mixed values across prisms
            {
                **complete_valid_context,
                'environmental_impact': 0.9,
                'sustainability': 0.8,
                'social_impact': 0.5,
                'fairness': 0.6,
                'wellbeing': 0.7
            }
        ]

        for case in test_cases:
            result = governor.evaluate_action(case)
            assert result == True

    def test_invalid_prism_combinations(self, governor, complete_valid_context):
        """Test invalid combinations across different prisms"""
        test_cases = [
            # Invalid environmental prism
            {
                **complete_valid_context,
                'environmental_impact': -0.1,
                'sustainability': 1.2
            },
            # Invalid social prism
            {
                **complete_valid_context,
                'social_impact': 1.5,
                'fairness': -0.3,
                'transparency': 2.0
            },
            # Invalid wellbeing prism
            {
                **complete_valid_context,
                'wellbeing': 2.0,
                'safety': -0.5,
                'privacy': 1.1
            }
        ]

        for case in test_cases:
            with pytest.warns(RuntimeWarning) as warning_info:
                result = governor.evaluate_action(case)
                assert result == False

    def test_missing_prism_scenarios(self, governor, complete_valid_context):
        """Test scenarios with missing prisms"""
        # Remove one prism at a time
        for field in complete_valid_context.keys():
            incomplete_set = complete_valid_context.copy()
            del incomplete_set[field]
            
            with pytest.warns(RuntimeWarning) as warning_info:
                result = governor.evaluate_action(incomplete_set)
                assert result == False
                assert f"Missing required fields: {field}" in str(warning_info[0].message)

    def test_threshold_boundaries_comprehensive(self, governor, complete_valid_context):
        """Test threshold boundaries with complete prism set"""
        
        # Test at threshold
        threshold_case = {
            key: 0.5 for key in complete_valid_context.keys()
        }
        result = governor.evaluate_action(threshold_case)
        assert result == True

        # Test below threshold
        below_threshold = {
            key: 0.4 for key in complete_valid_context.keys()
        }
        with pytest.warns(RuntimeWarning):
            result = governor.evaluate_action(below_threshold)
            assert result == False

    def test_test_mode_comprehensive(self, governor, complete_valid_context):
        """Test test_mode with complete prism set"""
        
        invalid_test_case = {
            **complete_valid_context,
            'environmental_impact': 1.5,
            'wellbeing': -0.1,
            'social_impact': 2.0,
            'fairness': -0.5,
            'transparency': 1.2
        }

        # Should not raise warnings in test mode
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = governor.evaluate_action(invalid_test_case, test_mode=True)
            assert result == False

    def test_weighted_prism_combinations(self, governor, complete_valid_context):
        """Test various weighted combinations of prisms"""
        
        # Test case where high-weight prisms are high
        weighted_high = {
            **complete_valid_context,
            'environmental_impact': 0.9,  # Higher weight (0.4)
            'wellbeing': 0.9,            # Higher weight
            'social_impact': 0.4         # Lower weight
        }
        result = governor.evaluate_action(weighted_high)
        assert result == True

        # Test case where high-weight prisms are low
        weighted_low = {
            **complete_valid_context,
            'environmental_impact': 0.3,  # Higher weight (0.4)
            'wellbeing': 0.3,            # Higher weight
            'social_impact': 0.9         # Lower weight
        }
        with pytest.warns(RuntimeWarning):
            result = governor.evaluate_action(weighted_low)
            assert result == False

    @pytest.mark.parametrize("field,value", [
        ('environmental_impact', 0.75),
        ('wellbeing', 0.80),
        ('social_impact', 0.85),
        ('fairness', 0.70),
        ('transparency', 0.90),
        ('accountability', 0.85),
        ('sustainability', 0.80),
        ('privacy', 0.75),
        ('safety', 0.95)
    ])
    def test_individual_prism_validation(self, governor, complete_valid_context, field, value):
        """Test validation of individual prisms"""
        test_case = complete_valid_context.copy()
        test_case[field] = value
        result = governor.evaluate_action(test_case)
        assert result == True

    def test_fill_missing_criteria(self, governor):
        """Test filling missing criteria with default values"""
        
        # Test with partial data
        partial_context = {
            'environmental_impact': 0.8,
            'wellbeing': 0.7
        }
        
        # Test with defaults
        filled_context = governor.fill_missing_criteria(partial_context, use_defaults=True)
        assert len(filled_context) == len(governor.DEFAULT_VALUES)
        assert filled_context['environmental_impact'] == 0.8  # Original value preserved
        assert filled_context['wellbeing'] == 0.7  # Original value preserved
        assert filled_context['social_impact'] == 0.5  # Default value
        assert filled_context['accountability'] == 0.5  # Default value
        
        # Test without defaults (None values)
        filled_context_none = governor.fill_missing_criteria(partial_context, use_defaults=False)
        assert len(filled_context_none) == len(governor.DEFAULT_VALUES)
        assert filled_context_none['environmental_impact'] == 0.8  # Original value preserved
        assert filled_context_none['wellbeing'] == 0.7  # Original value preserved
        assert filled_context_none['social_impact'] is None  # None value
        assert filled_context_none['accountability'] is None  # None value

    def test_fill_missing_criteria_empty_context(self, governor):
        """Test filling missing criteria with empty initial context"""
        
        empty_context = {}
        
        # Test with defaults
        filled_context = governor.fill_missing_criteria(empty_context, use_defaults=True)
        assert len(filled_context) == len(governor.DEFAULT_VALUES)
        for criterion, value in filled_context.items():
            assert value == governor.DEFAULT_VALUES[criterion]
        
        # Test without defaults
        filled_context_none = governor.fill_missing_criteria(empty_context, use_defaults=False)
        assert len(filled_context_none) == len(governor.DEFAULT_VALUES)
        for criterion, value in filled_context_none.items():
            assert value is None

    def test_fill_missing_criteria_preserves_values(self, governor):
        """Test that filling missing criteria preserves existing values"""
        
        # Test with all custom values
        custom_context = {
            criterion: 0.75 for criterion in governor.DEFAULT_VALUES.keys()
        }
        
        filled_context = governor.fill_missing_criteria(custom_context, use_defaults=True)
        assert filled_context == custom_context  # Should not modify existing values
        
        # Test with some custom values outside normal range
        custom_context['environmental_impact'] = 1.5
        filled_context = governor.fill_missing_criteria(custom_context, use_defaults=True)
        assert filled_context['environmental_impact'] == 1.5  # Should preserve even invalid values

    def test_validate_context_structure(self, governor):
        """Test validation of context structure for missing and invalid criteria"""
        
        # Test completely valid context
        valid_context = {
            'environmental_impact': 0.7,
            'sustainability': 0.8,
            'social_impact': 0.6,
            'fairness': 0.7,
            'transparency': 0.8,
            'wellbeing': 0.9,
            'safety': 0.7,
            'privacy': 0.6,
            'accountability': 0.8
        }
        result = governor._validate_context_structure(valid_context)
        assert result.is_valid
        assert len(result.issues) == 0

        # Test missing criteria
        missing_context = {
            'environmental_impact': 0.7,
            # missing sustainability
            'social_impact': 0.6
        }
        result = governor._validate_context_structure(missing_context)
        assert not result.is_valid
        missing_issues = [i for i in result.issues if i.issue_type == 'missing']
        assert len(missing_issues) == 6  # Should identify all missing fields
        assert 'sustainability' in [i.field for i in missing_issues]
        assert all(i.prism is not None for i in missing_issues)

    def test_validate_context_invalid_values(self, governor):
        """Test validation of invalid criteria values"""
        
        # Test various invalid value types
        invalid_context = {
            'environmental_impact': -0.1,        # Below range
            'sustainability': 1.1,               # Above range
            'social_impact': "0.6",             # Wrong type (string)
            'fairness': None,                   # None value
            'transparency': 0.8,                # Valid value
            'wellbeing': float('inf'),          # Invalid float
            'safety': complex(1, 2),            # Invalid complex number
            'privacy': 0.6,                     # Valid value
            'accountability': [0.8]             # Invalid list
        }
        
        result = governor._validate_context_structure(invalid_context)
        assert not result.is_valid
        
        invalid_issues = [i for i in result.issues if i.issue_type == 'invalid']
        assert len(invalid_issues) == 7  # Should identify all invalid fields
        
        # Check specific validation failures
        issue_fields = {i.field: i.details for i in invalid_issues}
        assert "environmental_impact" in issue_fields
        assert "value -0.1 not between 0 and 1" in issue_fields['environmental_impact']
        assert "sustainability" in issue_fields
        assert "value 1.1 not between 0 and 1" in issue_fields['sustainability']

    def test_validate_context_mixed_issues(self, governor):
        """Test validation with both missing and invalid criteria"""
        
        mixed_context = {
            'environmental_impact': 1.5,     # Invalid
            'sustainability': 0.7,           # Valid
            # social_impact missing
            'fairness': -0.1,               # Invalid
            'transparency': 0.8,            # Valid
            # wellbeing missing
            'safety': 2.0,                  # Invalid
            # privacy missing
            'accountability': 0.6           # Valid
        }
        
        result = governor._validate_context_structure(mixed_context)
        assert not result.is_valid
        
        # Check missing issues
        missing_issues = [i for i in result.issues if i.issue_type == 'missing']
        assert len(missing_issues) == 3
        missing_fields = {i.field for i in missing_issues}
        assert 'social_impact' in missing_fields
        assert 'wellbeing' in missing_fields
        assert 'privacy' in missing_fields
        
        # Check invalid issues
        invalid_issues = [i for i in result.issues if i.issue_type == 'invalid']
        assert len(invalid_issues) == 3
        invalid_fields = {i.field for i in invalid_issues}
        assert 'environmental_impact' in invalid_fields
        assert 'fairness' in invalid_fields
        assert 'safety' in invalid_fields

    def test_validate_context_prism_grouping(self, governor):
        """Test validation issues are correctly grouped by prism"""
        
        context = {
            'environmental_impact': 1.5,     # Invalid environmental
            # sustainability missing         # Missing environmental
            'social_impact': -0.1,          # Invalid social
            'fairness': 2.0,                # Invalid social
            # transparency missing          # Missing social
            'wellbeing': 0.7,               # Valid wellbeing
            'safety': 0.8,                  # Valid wellbeing
            'privacy': 0.6,                 # Valid wellbeing
            # accountability missing        # Missing governance
        }
        
        result = governor._validate_context_structure(context)
        assert not result.is_valid
        
        # Group issues by prism
        prism_issues = defaultdict(list)
        for issue in result.issues:
            prism_issues[issue.prism].append(issue)
        
        # Check environmental prism issues
        assert len(prism_issues['environmental']) == 2
        assert any(i.field == 'sustainability' and i.issue_type == 'missing' 
                  for i in prism_issues['environmental'])
        assert any(i.field == 'environmental_impact' and i.issue_type == 'invalid' 
                  for i in prism_issues['environmental'])
        
        # Check social prism issues
        assert len(prism_issues['social']) == 3
        
        # Check governance prism issues
        assert len(prism_issues['governance']) == 1
        assert prism_issues['governance'][0].field == 'accountability'
        
        # Check wellbeing prism has no issues
        assert 'wellbeing' not in prism_issues

    def test_prism_specific_thresholds(self, governor, complete_valid_context):
        """Test threshold boundaries for each prism"""
        
        # Test environmental prism at threshold
        env_threshold = {
            **complete_valid_context,
            'environmental_impact': 0.5,
            'sustainability': 0.5
        }
        result = governor.evaluate_action(env_threshold)
        assert result == True

        # Test social prism below threshold
        social_below = {
            **complete_valid_context,
            'social_impact': 0.3,
            'fairness': 0.3,
            'transparency': 0.3
        }
        with pytest.warns(RuntimeWarning):
            result = governor.evaluate_action(social_below)
            assert result == False

    def test_prism_weights(self, governor, complete_valid_context):
        """Test weight calculations for each prism"""
        
        # High environmental, low others
        env_weighted = {
            **complete_valid_context,
            'environmental_impact': 1.0,  # 0.6 weight in env prism
            'sustainability': 1.0,        # 0.4 weight in env prism
            'social_impact': 0.5,         # Lower weights for other prisms
            'wellbeing': 0.5,
            'accountability': 0.5
        }
        result = governor.evaluate_action(env_weighted)
        assert result == True

        # Test balanced weights
        balanced = {
            **complete_valid_context,
            'environmental_impact': 0.7,
            'sustainability': 0.7,
            'social_impact': 0.7,
            'fairness': 0.7,
            'transparency': 0.7
        }
        result = governor.evaluate_action(balanced)
        assert result == True

    @pytest.mark.parametrize("prism,criteria", [
        ('environmental', ['environmental_impact', 'sustainability']),
        ('social', ['social_impact', 'fairness', 'transparency']),
        ('wellbeing', ['wellbeing', 'safety', 'privacy']),
        ('governance', ['accountability'])
    ])
    def test_prism_validation(self, governor, complete_valid_context, prism, criteria):
        """Test validation for each prism's criteria"""
        
        # Test valid values for each criterion in the prism
        for criterion in criteria:
            test_context = complete_valid_context.copy()
            test_context[criterion] = 0.75
            result = governor.evaluate_action(test_context)
            assert result == True

        # Test invalid values for each criterion
        for criterion in criteria:
            test_context = complete_valid_context.copy()
            test_context[criterion] = 1.5  # Invalid value
            with pytest.warns(RuntimeWarning):
                result = governor.evaluate_action(test_context)
                assert result == False