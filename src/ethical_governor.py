import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import defaultdict

@dataclass
class ValidationIssue:
    field: str
    issue_type: str  # 'missing' or 'invalid'
    details: Optional[str] = None
    prism: Optional[str] = None  # Added to track which prism the issue belongs to

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]

class EthicalGovernor:
    REQUIRED_CRITERIA = {
        'environmental': ['environmental_impact', 'sustainability'],
        'social': ['social_impact', 'fairness', 'transparency'],
        'wellbeing': ['wellbeing', 'safety', 'privacy'],
        'governance': ['accountability']
    }

    # Define default values for each criterion
    DEFAULT_VALUES = {
        'environmental_impact': 0.5,
        'sustainability': 0.5,
        'social_impact': 0.5,
        'fairness': 0.5,
        'transparency': 0.5,
        'wellbeing': 0.5,
        'safety': 0.5,
        'privacy': 0.5,
        'accountability': 0.5
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fill_missing_criteria(self, context: Dict, use_defaults: bool = True) -> Dict:
        """
        Fill missing criteria with default or None values.
        
        Args:
            context: Dictionary containing ethical criteria
            use_defaults: If True, use DEFAULT_VALUES; if False, use None
            
        Returns:
            Dict with all required criteria filled
        """
        filled_context = context.copy()
        
        for prism, criteria in self.REQUIRED_CRITERIA.items():
            for criterion in criteria:
                if criterion not in filled_context:
                    if use_defaults:
                        filled_context[criterion] = self.DEFAULT_VALUES[criterion]
                    else:
                        filled_context[criterion] = None
                        
        return filled_context

    def evaluate_action(self, context: Dict, test_mode: bool = False) -> bool:
        """
        Evaluate an action based on ethical criteria.
        
        Args:
            context: Dictionary containing ethical criteria
            test_mode: If True, suppress all warnings and only log to debug level
            
        Returns:
            bool indicating if action meets ethical criteria
        """
        # Fill missing criteria before validation
        complete_context = self.fill_missing_criteria(context)
        validation_result = self._validate_context_structure(complete_context)
        
        if not validation_result.is_valid:
            if test_mode:
                # In test mode, log at debug level instead of warning
                self.logger.debug(self._format_validation_issues(validation_result.issues))
            else:
                self.log_criteria_warnings(validation_result.issues, test_mode)
            return False

        return self._calculate_ethical_score(complete_context, test_mode)

    def log_criteria_warnings(self, issues: List[ValidationIssue], test_mode: bool):
        """
        Consolidate and log all criteria warnings in a structured format.
        
        Args:
            issues: List of validation issues
            test_mode: Whether to suppress warning emission
        """
        if not issues:
            return

        # Group issues by prism and type
        prism_issues = defaultdict(lambda: defaultdict(list))
        for issue in issues:
            prism = self._get_prism_for_criterion(issue.field)
            prism_issues[prism][issue.issue_type].append(issue)

        # Build structured warning message
        warning_lines = ["Ethical criteria validation issues:"]
        
        for prism, type_issues in prism_issues.items():
            prism_lines = []
            
            if type_issues['missing']:
                missing_fields = [i.field for i in type_issues['missing']]
                prism_lines.append(f"Missing: {', '.join(missing_fields)}")
                
            if type_issues['invalid']:
                invalid_details = [
                    f"{i.field} ({i.details})" for i in type_issues['invalid']
                ]
                prism_lines.append(f"Invalid: {', '.join(invalid_details)}")
            
            if prism_lines:
                warning_lines.append(f"\n{prism.title()} Prism:")
                warning_lines.extend([f"  - {line}" for line in prism_lines])

        consolidated_message = "\n".join(warning_lines)
        
        # Always log the structured message
        self.logger.warning(consolidated_message)
        
        # Emit warning if not in test mode
        if not test_mode:
            warnings.warn(consolidated_message, RuntimeWarning)

    def _get_prism_for_criterion(self, criterion: str) -> str:
        """Find which prism a criterion belongs to."""
        for prism, criteria in self.REQUIRED_CRITERIA.items():
            if criterion in criteria:
                return prism
        return "unknown"

    def _validate_context_structure(self, context: Dict) -> ValidationResult:
        """Validate the context structure with enhanced issue tracking."""
        issues = []
        
        for prism, criteria in self.REQUIRED_CRITERIA.items():
            for criterion in criteria:
                if criterion not in context or context[criterion] is None:
                    issues.append(ValidationIssue(
                        field=criterion,
                        issue_type='missing',
                        prism=prism
                    ))
                    continue
                    
                value = context[criterion]
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    issues.append(ValidationIssue(
                        field=criterion,
                        issue_type='invalid',
                        details=f"value {value} not between 0 and 1",
                        prism=prism
                    ))

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues
        )

    def _calculate_ethical_score(self, context: Dict, test_mode: bool) -> bool:
        """Calculate ethical score with test mode awareness."""
        try:
            # Calculate prism-specific scores
            environmental_score = (
                context['environmental_impact'] * 0.6 +
                context['sustainability'] * 0.4
            )
            
            social_score = (
                context['social_impact'] * 0.4 +
                context['fairness'] * 0.3 +
                context['transparency'] * 0.3
            )
            
            wellbeing_score = (
                context['wellbeing'] * 0.4 +
                context['safety'] * 0.4 +
                context['privacy'] * 0.2
            )
            
            governance_score = context['accountability']
            
            # Calculate final weighted score
            final_score = (
                environmental_score * 0.3 +
                social_score * 0.3 +
                wellbeing_score * 0.3 +
                governance_score * 0.1
            )
            
            if final_score < 0.5:
                message = f"Action scored below acceptable threshold: {final_score:.2f}"
                if test_mode:
                    self.logger.debug(message)
                else:
                    self.logger.warning(message)
                    warnings.warn(message, RuntimeWarning)
                return False
                
            return True
            
        except Exception as e:
            message = f"Error calculating ethical score: {str(e)}"
            if test_mode:
                self.logger.debug(message)
            else:
                self.logger.error(message)
                warnings.warn(message, RuntimeWarning)
            return False

    def _format_validation_issues(self, issues: List[ValidationIssue]) -> str:
        """Format validation issues into a readable message without emitting warnings."""
        # Group issues by prism and type
        prism_issues = defaultdict(lambda: defaultdict(list))
        for issue in issues:
            prism = self._get_prism_for_criterion(issue.field)
            prism_issues[prism][issue.issue_type].append(issue)

        # Build structured message
        warning_lines = ["Ethical criteria validation issues:"]
        
        for prism, type_issues in prism_issues.items():
            prism_lines = []
            
            if type_issues['missing']:
                missing_fields = [i.field for i in type_issues['missing']]
                prism_lines.append(f"Missing: {', '.join(missing_fields)}")
                
            if type_issues['invalid']:
                invalid_details = [
                    f"{i.field} ({i.details})" for i in type_issues['invalid']
                ]
                prism_lines.append(f"Invalid: {', '.join(invalid_details)}")
            
            if prism_lines:
                warning_lines.append(f"\n{prism.title()} Prism:")
                warning_lines.extend([f"  - {line}" for line in prism_lines])

        return "\n".join(warning_lines)