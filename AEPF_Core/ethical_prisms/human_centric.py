from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import logging
from collections import defaultdict
import numpy as np

@dataclass
class ValidationError:
    """Represents a validation error with context."""
    criterion: str
    error_type: str  # 'unknown', 'missing', 'invalid_type', 'out_of_bounds'
    details: str
    sub_criterion: Optional[str] = None

@dataclass
class ValidationResult:
    """Contains validation results and any errors."""
    is_valid: bool
    errors: List[ValidationError]
    details: Dict[str, Any]

@dataclass
class CriterionScore:
    """Represents a score for a specific criterion with its sub-components."""
    raw_score: float
    weighted_score: float
    sub_criteria: Dict[str, float]
    details: Dict[str, Any]

class CriterionType(Enum):
    """Defines the main criteria types for human-centric evaluation."""
    WELLBEING = "wellbeing"
    AUTONOMY = "autonomy"
    FAIRNESS = "fairness"
    PRIVACY = "privacy"

class HumanCentricPrism:
    """
    Evaluates actions based on human-centric ethical principles.
    
    Attributes:
        CRITERIA_WEIGHTS: Weights for each main criterion
        SUB_CRITERIA: Mapping of criteria to their sub-components
    """
    
    CRITERIA_WEIGHTS = {
        CriterionType.WELLBEING: 0.3,
        CriterionType.AUTONOMY: 0.25,
        CriterionType.FAIRNESS: 0.25,
        CriterionType.PRIVACY: 0.2
    }
    
    SUB_CRITERIA = {
        CriterionType.WELLBEING: {
            "safety": 0.35,
            "mental_health": 0.35,
            "physical_health": 0.30
        },
        CriterionType.AUTONOMY: {
            "freedom_of_choice": 0.5,
            "agency": 0.5
        },
        CriterionType.FAIRNESS: {
            "equal_opportunity": 0.5,
            "non_discrimination": 0.5
        },
        CriterionType.PRIVACY: {
            "data_protection": 0.6,
            "information_access": 0.4
        }
    }

    def __init__(self):
        """Initialize the HumanCentricPrism with default settings."""
        self.name = "human_centric"
        self.logger = logging.getLogger(__name__)
        self._validate_weights()

    @property
    def wellbeing(self) -> Dict[str, float]:
        """Get wellbeing sub-criteria and weights."""
        return self.SUB_CRITERIA[CriterionType.WELLBEING]

    @property
    def autonomy(self) -> Dict[str, float]:
        """Get autonomy sub-criteria and weights."""
        return self.SUB_CRITERIA[CriterionType.AUTONOMY]

    @property
    def fairness(self) -> Dict[str, float]:
        """Get fairness sub-criteria and weights."""
        return self.SUB_CRITERIA[CriterionType.FAIRNESS]

    @property
    def privacy(self) -> Dict[str, float]:
        """Get privacy sub-criteria and weights."""
        return self.SUB_CRITERIA[CriterionType.PRIVACY]

    def calculate_score(self, inputs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate ethical score with enhanced validation.
        
        Args:
            inputs: Dictionary of criteria and sub-criteria scores
            
        Returns:
            Dictionary containing score and details
            
        Raises:
            ValueError: If validation fails, with detailed error messages
        """
        validation_result = self._validate_inputs_comprehensive(inputs)
        if not validation_result.is_valid:
            self._log_validation_errors(validation_result.errors)
            raise ValueError(self._format_validation_errors(validation_result.errors))
            
        return self._calculate_score_internal(inputs)

    def _validate_inputs_comprehensive(self, inputs: Dict[str, Dict[str, float]]) -> ValidationResult:
        """
        Comprehensive input validation with standardized error messages.
        """
        errors = []
        details = defaultdict(list)
        
        # Check for unknown top-level criteria
        unknown_criteria = set(inputs.keys()) - {ct.value for ct in CriterionType}
        for criterion in unknown_criteria:
            errors.append(ValidationError(
                criterion=criterion,
                error_type='unknown',
                details=f"Unknown criterion: {criterion}"
            ))
            details['unknown_criteria'].append(criterion)

        # Validate each criterion and its sub-criteria
        for criterion_type in CriterionType:
            criterion_name = criterion_type.value
            
            # Check if criterion exists
            if criterion_name not in inputs:
                errors.append(ValidationError(
                    criterion=criterion_name,
                    error_type='missing',
                    details=f"Missing criterion: {criterion_name}"  # Simplified message
                ))
                continue

            criterion_data = inputs[criterion_name]
            if not isinstance(criterion_data, dict):
                errors.append(ValidationError(
                    criterion=criterion_name,
                    error_type='invalid_type',
                    details=f"Must be a dictionary"  # Simplified message
                ))
                continue

            # Validate sub-criteria
            expected_sub_criteria = set(self.SUB_CRITERIA[criterion_type].keys())
            provided_sub_criteria = set(criterion_data.keys())
            
            # Check for unknown sub-criteria
            for sub in (provided_sub_criteria - expected_sub_criteria):
                errors.append(ValidationError(
                    criterion=criterion_name,
                    sub_criterion=sub,
                    error_type='unknown',
                    details=f"Unknown sub-criterion: {sub}"  # Simplified message
                ))

            # Check for missing sub-criteria
            for sub in (expected_sub_criteria - provided_sub_criteria):
                errors.append(ValidationError(
                    criterion=criterion_name,
                    sub_criterion=sub,
                    error_type='missing',
                    details=f"Missing sub-criterion: {sub}"  # Simplified message
                ))

            # Validate values
            for sub_name in expected_sub_criteria & provided_sub_criteria:
                value = criterion_data[sub_name]
                
                # Type validation
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(ValidationError(
                        criterion=criterion_name,
                        sub_criterion=sub_name,
                        error_type='invalid_type',
                        details=f"Must be numeric"  # Simplified message
                    ))
                    continue

                # Special value checks
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    errors.append(ValidationError(
                        criterion=criterion_name,
                        sub_criterion=sub_name,
                        error_type='invalid_type',
                        details=f"Must be a finite number"
                    ))
                    continue

                # Range validation
                if not 0 <= value <= 1:
                    errors.append(ValidationError(
                        criterion=criterion_name,
                        sub_criterion=sub_name,
                        error_type='out_of_bounds',
                        details=f"Must be between 0 and 1"  # Matches test expectation
                    ))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            details=dict(details)
        )

    def _log_validation_errors(self, errors: List[ValidationError]) -> None:
        """Log validation errors with detailed context for debugging."""
        if not errors:
            return
            
        self.logger.warning("Validation errors detected:")
        
        # Group errors by type and criterion for clearer logging
        for error in errors:
            log_message = []
            if error.sub_criterion:
                log_message.append(
                    f"{error.criterion}.{error.sub_criterion}"
                )
            else:
                log_message.append(error.criterion)
            
            log_message.append(f"[{error.error_type}]")
            log_message.append(error.details)
            
            self.logger.warning(" - " + " ".join(log_message))

    def _format_validation_errors(self, errors: List[ValidationError]) -> str:
        """Format validation errors to match test expectations exactly."""
        if not errors:
            return ""
            
        error_groups = defaultdict(list)
        for error in errors:
            error_groups[error.error_type].append(error)
        
        lines = ["Multiple validation errors detected:"]
        
        # Order error types to match test expectations
        error_type_order = ['unknown', 'missing', 'invalid_type', 'out_of_bounds']
        
        for error_type in error_type_order:
            if error_type in error_groups:
                lines.append(f"\n{error_type.upper()} errors:")
                for error in error_groups[error_type]:
                    if error.sub_criterion:
                        lines.append(
                            f"  - {error.criterion}.{error.sub_criterion}: {error.details}"
                        )
                    else:
                        lines.append(f"  - {error.criterion}: {error.details}")
        
        return "\n".join(lines)

    def _calculate_score_internal(self, inputs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate the ethical score based on provided inputs.
        
        Args:
            inputs: Dictionary containing scores for each criterion and its sub-criteria.
                   Format: {
                       "wellbeing": {"safety": 0.8, "mental_health": 0.7, ...},
                       "autonomy": {"freedom_of_choice": 0.9, "agency": 0.8},
                       ...
                   }
        
        Returns:
            Dictionary containing:
                - score: Final weighted score (0-100)
                - details: Detailed breakdown of calculations
                - criteria_scores: Individual scores for each criterion
        
        Raises:
            ValueError: If inputs are invalid or missing required criteria
        """
        criteria_scores = {}
        total_score = 0.0
        
        for criterion_type in CriterionType:
            criterion_score = self._calculate_criterion_score(
                criterion_type,
                inputs[criterion_type.value]
            )
            criteria_scores[criterion_type.value] = criterion_score
            total_score += criterion_score.weighted_score

        return {
            "score": round(total_score * 100, 2),
            "details": {
                "criteria_scores": {
                    name: score.raw_score 
                    for name, score in criteria_scores.items()
                },
                "weighted_scores": {
                    name: score.weighted_score 
                    for name, score in criteria_scores.items()
                },
                "sub_criteria_details": {
                    name: score.details 
                    for name, score in criteria_scores.items()
                }
            },
            "criteria_scores": criteria_scores
        }

    def _calculate_criterion_score(
        self, 
        criterion_type: CriterionType, 
        inputs: Dict[str, float]
    ) -> CriterionScore:
        """
        Calculate score for a specific criterion.
        
        Args:
            criterion_type: Type of criterion to evaluate
            inputs: Dictionary of sub-criteria scores
            
        Returns:
            CriterionScore containing raw and weighted scores with details
        """
        sub_criteria = self.SUB_CRITERIA[criterion_type]
        raw_score = 0.0
        details = {}
        
        for sub_name, weight in sub_criteria.items():
            if sub_name not in inputs:
                raise ValueError(f"Missing sub-criterion: {sub_name} for {criterion_type.value}")
            
            sub_score = inputs[sub_name]
            weighted_sub_score = sub_score * weight
            raw_score += weighted_sub_score
            
            details[sub_name] = {
                "score": sub_score,
                "weight": weight,
                "weighted_score": weighted_sub_score
            }
        
        criterion_weight = self.CRITERIA_WEIGHTS[criterion_type]
        weighted_score = raw_score * criterion_weight
        
        return CriterionScore(
            raw_score=raw_score,
            weighted_score=weighted_score,
            sub_criteria=inputs,
            details=details
        )

    def _validate_weights(self) -> None:
        """
        Validate that weights are properly configured.
        
        Raises:
            ValueError: If weights don't sum to 1 or are invalid
        """
        # Validate main criteria weights
        total_weight = sum(self.CRITERIA_WEIGHTS.values())
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating-point variance
            raise ValueError(f"Criteria weights must sum to 1.0, got {total_weight}")
        
        # Validate sub-criteria weights
        for criterion_type, sub_criteria in self.SUB_CRITERIA.items():
            sub_total = sum(sub_criteria.values())
            if not 0.99 <= sub_total <= 1.01:
                raise ValueError(
                    f"Sub-criteria weights for {criterion_type.value} "
                    f"must sum to 1.0, got {sub_total}"
                )
