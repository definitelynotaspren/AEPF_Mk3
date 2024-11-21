import pytest
from AEPF_Core.ethical_prisms.human_centric import HumanCentricPrism, CriterionType
import logging

class TestHumanCentricPrism:
    @pytest.fixture
    def prism(self):
        return HumanCentricPrism()

    @pytest.fixture
    def valid_inputs(self):
        """Fixture providing valid inputs for all criteria"""
        return {
            "wellbeing": {
                "safety": 0.8,
                "mental_health": 0.7,
                "physical_health": 0.9
            },
            "autonomy": {
                "freedom_of_choice": 0.75,
                "agency": 0.85
            },
            "fairness": {
                "equal_opportunity": 0.8,
                "non_discrimination": 0.9
            },
            "privacy": {
                "data_protection": 0.85,
                "information_access": 0.7
            }
        }

    def test_initialization(self, prism):
        """Test proper initialization of the prism"""
        assert prism.name == "human_centric"
        assert sum(prism.CRITERIA_WEIGHTS.values()) == pytest.approx(1.0)
        
        # Test sub-criteria weights sum to 1 for each criterion
        for criterion_type in CriterionType:
            sub_weights = sum(prism.SUB_CRITERIA[criterion_type].values())
            assert sub_weights == pytest.approx(1.0)

    def test_property_access(self, prism):
        """Test access to criteria properties"""
        assert "safety" in prism.wellbeing
        assert "freedom_of_choice" in prism.autonomy
        assert "equal_opportunity" in prism.fairness
        assert "data_protection" in prism.privacy

    def test_calculate_score_valid_input(self, prism, valid_inputs):
        """Test score calculation with valid inputs"""
        result = prism.calculate_score(valid_inputs)
        
        assert "score" in result
        assert "details" in result
        assert 0 <= result["score"] <= 100
        
        # Check details structure
        details = result["details"]
        assert "criteria_scores" in details
        assert "weighted_scores" in details
        assert "sub_criteria_details" in details

    def test_calculate_score_weights(self, prism, valid_inputs):
        """Test that weights are properly applied in score calculation"""
        result = prism.calculate_score(valid_inputs)
        
        # Check that each criterion's weighted score reflects its weight
        for criterion_type in CriterionType:
            criterion_name = criterion_type.value
            weighted_score = result["details"]["weighted_scores"][criterion_name]
            raw_score = result["details"]["criteria_scores"][criterion_name]
            expected_weight = prism.CRITERIA_WEIGHTS[criterion_type]
            
            assert weighted_score == pytest.approx(raw_score * expected_weight)

    def test_invalid_inputs(self, prism, valid_inputs):
        """Test handling of invalid inputs with detailed error message validation"""
        
        test_cases = [
            {
                'criterion_type': CriterionType.WELLBEING,
                'field': 'safety',
                'invalid_value': 1.5,
                'expected_phrase': 'Must be between 0 and 1'
            },
            {
                'criterion_type': CriterionType.AUTONOMY,
                'field': 'freedom_of_choice',
                'invalid_value': -0.1,
                'expected_phrase': 'Must be between 0 and 1'
            },
            {
                'criterion_type': CriterionType.FAIRNESS,
                'field': 'equal_opportunity',
                'invalid_value': "invalid",
                'expected_phrase': 'Must be between 0 and 1'
            },
            {
                'criterion_type': CriterionType.PRIVACY,
                'field': 'data_protection',
                'invalid_value': None,
                'expected_phrase': 'Must be between 0 and 1'
            }
        ]
        
        for test_case in test_cases:
            # Prepare invalid input data
            invalid_inputs = valid_inputs.copy()
            criterion_name = test_case['criterion_type'].value
            invalid_inputs[criterion_name] = invalid_inputs[criterion_name].copy()
            invalid_inputs[criterion_name][test_case['field']] = test_case['invalid_value']
            
            try:
                prism.calculate_score(invalid_inputs)
                pytest.fail(f"Expected ValueError for {test_case['field']} = {test_case['invalid_value']}")
            except ValueError as e:
                error_message = str(e)
                
                # Primary assertion for expected phrase
                if test_case['expected_phrase'] not in error_message:
                    # Log unexpected error message format
                    print(f"\nUnexpected error message format:")
                    print(f"Expected phrase: {test_case['expected_phrase']}")
                    print(f"Actual message: {error_message}")
                    
                    # Fail the test
                    assert test_case['expected_phrase'] in error_message, \
                        f"Error message for {test_case['field']} does not contain expected phrase"
                
                # Additional validation
                assert criterion_name in error_message, \
                    f"Error message should mention criterion '{criterion_name}'"
                assert test_case['field'] in error_message, \
                    f"Error message should mention field '{test_case['field']}'"
                assert str(test_case['invalid_value']) in error_message, \
                    f"Error message should include the invalid value"

    def test_invalid_inputs_edge_cases(self, prism, valid_inputs):
        """Test handling of edge cases for invalid inputs"""
        
        edge_cases = [
            {
                'criterion_type': CriterionType.WELLBEING,
                'field': 'safety',
                'value': float('inf'),
                'expected_phrase': 'Must be between 0 and 1'
            },
            {
                'criterion_type': CriterionType.AUTONOMY,
                'field': 'freedom_of_choice',
                'value': float('nan'),
                'expected_phrase': 'Must be between 0 and 1'
            },
            {
                'criterion_type': CriterionType.FAIRNESS,
                'field': 'equal_opportunity',
                'value': complex(1, 2),
                'expected_phrase': 'Must be between 0 and 1'
            },
            {
                'criterion_type': CriterionType.PRIVACY,
                'field': 'data_protection',
                'value': [0.5],  # List instead of float
                'expected_phrase': 'Must be between 0 and 1'
            }
        ]
        
        for case in edge_cases:
            invalid_inputs = valid_inputs.copy()
            criterion_name = case['criterion_type'].value
            invalid_inputs[criterion_name] = invalid_inputs[criterion_name].copy()
            invalid_inputs[criterion_name][case['field']] = case['value']
            
            with pytest.raises(ValueError) as exc_info:
                prism.calculate_score(invalid_inputs)
                
            error_message = str(exc_info.value)
            
            try:
                assert case['expected_phrase'] in error_message
                assert criterion_name in error_message
                assert case['field'] in error_message
            except AssertionError:
                # Log detailed information about the failure
                print(f"\nFailed validation for edge case:")
                print(f"Criterion: {criterion_name}")
                print(f"Field: {case['field']}")
                print(f"Value: {case['value']}")
                print(f"Expected phrase: {case['expected_phrase']}")
                print(f"Actual error message: {error_message}")
                raise

    def test_missing_criteria(self, prism, valid_inputs):
        """Test handling of missing criteria"""
        incomplete_inputs = valid_inputs.copy()
        del incomplete_inputs["wellbeing"]
        
        with pytest.raises(ValueError) as exc_info:
            prism.calculate_score(incomplete_inputs)
        assert "Missing criterion: wellbeing" in str(exc_info.value)

    def test_missing_sub_criteria(self, prism, valid_inputs):
        """Test handling of missing sub-criteria"""
        incomplete_inputs = valid_inputs.copy()
        del incomplete_inputs["wellbeing"]["safety"]
        
        with pytest.raises(ValueError) as exc_info:
            prism.calculate_score(incomplete_inputs)
        assert "Missing sub-criterion: safety" in str(exc_info.value)

    def test_score_calculation_extremes(self, prism):
        """Test score calculation with extreme values"""
        # All maximum values
        max_inputs = {
            criterion_type.value: {
                sub_name: 1.0
                for sub_name in prism.SUB_CRITERIA[criterion_type].keys()
            }
            for criterion_type in CriterionType
        }
        max_result = prism.calculate_score(max_inputs)
        assert max_result["score"] == pytest.approx(100.0)
        
        # All minimum values
        min_inputs = {
            criterion_type.value: {
                sub_name: 0.0
                for sub_name in prism.SUB_CRITERIA[criterion_type].keys()
            }
            for criterion_type in CriterionType
        }
        min_result = prism.calculate_score(min_inputs)
        assert min_result["score"] == pytest.approx(0.0)

    def test_score_details_structure(self, prism, valid_inputs):
        """Test the structure and content of score details"""
        result = prism.calculate_score(valid_inputs)
        details = result["details"]
        
        # Check structure for each criterion
        for criterion_type in CriterionType:
            criterion_name = criterion_type.value
            
            # Check criterion scores exist
            assert criterion_name in details["criteria_scores"]
            assert criterion_name in details["weighted_scores"]
            
            # Check sub-criteria details
            sub_details = details["sub_criteria_details"][criterion_name]
            for sub_name in prism.SUB_CRITERIA[criterion_type].keys():
                assert sub_name in sub_details
                sub_info = sub_details[sub_name]
                assert "score" in sub_info
                assert "weight" in sub_info
                assert "weighted_score" in sub_info

    def test_weight_validation(self):
        """Test that weights are properly validated during initialization"""
        # This should pass as weights are correct in the class definition
        prism = HumanCentricPrism()
        
        # Verify that main criteria weights sum to 1
        total_weight = sum(prism.CRITERIA_WEIGHTS.values())
        assert total_weight == pytest.approx(1.0)
        
        # Verify that sub-criteria weights sum to 1 for each criterion
        for criterion_type in CriterionType:
            sub_weights = sum(prism.SUB_CRITERIA[criterion_type].values())
            assert sub_weights == pytest.approx(1.0)

    def test_multiple_validation_errors(self, prism):
        """Test handling of multiple validation errors."""
        invalid_inputs = {
            "wellbeing": {
                "safety": 1.5,  # Out of bounds
                "invalid_sub": 0.5,  # Unknown sub-criterion
                # missing mental_health
            },
            "invalid_criterion": {  # Unknown criterion
                "some_value": 0.5
            },
            "autonomy": {
                "freedom_of_choice": "invalid",  # Invalid type
                "agency": -0.1  # Out of bounds
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            prism.calculate_score(invalid_inputs)
        
        error_message = str(exc_info.value)
        
        # Check error message structure
        assert "Multiple validation errors detected:" in error_message
        assert "UNKNOWN errors:" in error_message
        assert "OUT_OF_BOUNDS errors:" in error_message
        assert "MISSING errors:" in error_message
        assert "INVALID_TYPE errors:" in error_message
        
        # Check specific error details
        assert "wellbeing.safety: Value 1.5 must be between 0 and 1" in error_message
        assert "wellbeing.invalid_sub: Unknown sub-criterion" in error_message
        assert "wellbeing.mental_health: Missing required sub-criterion" in error_message
        assert "invalid_criterion: Unknown criterion" in error_message
        assert "autonomy.freedom_of_choice: Value must be numeric" in error_message
        assert "autonomy.agency: Value -0.1 must be between 0 and 1" in error_message

    def test_validation_error_aggregation(self, prism):
        """Test that all validation errors are collected before raising."""
        invalid_inputs = {
            "wellbeing": {
                "safety": 1.5,
                "mental_health": -0.1
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            prism.calculate_score(invalid_inputs)
        
        error_message = str(exc_info.value)
        
        # Both errors should be reported
        assert "safety: Value 1.5 must be between 0 and 1" in error_message
        assert "mental_health: Value -0.1 must be between 0 and 1" in error_message
        assert "physical_health: Missing required sub-criterion" in error_message

    def test_validation_logging(self, prism, caplog):
        """Test that validation errors are properly logged."""
        invalid_inputs = {
            "wellbeing": {
                "safety": 1.5
            }
        }
        
        with pytest.raises(ValueError):
            with caplog.at_level(logging.WARNING):
                prism.calculate_score(invalid_inputs)
        
        # Check log messages
        assert "Validation errors detected:" in caplog.text
        assert "OUT_OF_BOUNDS errors:" in caplog.text
        assert "MISSING errors:" in caplog.text
        assert "wellbeing.safety: Value 1.5 must be between 0 and 1" in caplog.text 