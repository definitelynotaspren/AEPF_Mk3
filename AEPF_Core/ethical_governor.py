from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
from AEPF_Core.context_engine import ContextEngine
from AEPF_Core.ethical_prisms.human_centric import HumanCentricPrism
from AEPF_Core.ethical_prisms.sentient_first import SentientFirstPrism
from AEPF_Core.ethical_prisms.ecocentric import EcocentricPrism
from AEPF_Core.ethical_prisms.innovation_focused import InnovationFocusedPrism
from AEPF_Core.ethical_alignment_tracker import EthicalAlignmentTracker
from AEPF_Core.decision_analysis.probability_scorer import ProbabilityScorer, ProbabilityBand, ProbabilityScore
from AEPF_Core.decision_analysis.feedback_loop import FeedbackLoop, FeedbackLoopResult
from AEPF_Core.decision_analysis.pattern_logger import PatternLogger
import warnings

class DecisionCategory(Enum):
    CRITICAL = "critical"
    HIGH_IMPACT = "high_impact"
    MODERATE = "moderate"
    LOW_IMPACT = "low_impact"

class DecisionOutcome(Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"

@dataclass
class EthicalDecision:
    decision_id: str
    category: DecisionCategory
    recommendation: DecisionOutcome
    confidence_score: float
    prism_scores: Dict[str, float]
    context_snapshot: Dict
    reasoning: List[str]
    risk_factors: List[str]
    mitigation_steps: List[str]
    stakeholder_impact: Dict[str, float]
    timestamp: float
    probability_score: Optional[ProbabilityScore] = None

class EthicalGovernor:
    def __init__(self, context_engine=None):
        self.logger = logging.getLogger(__name__)
        self.context_engine = context_engine or ContextEngine()
        self.alignment_tracker = EthicalAlignmentTracker()
        self.pattern_logger = PatternLogger()
        self.probability_scorer = ProbabilityScorer()
        self.feedback_loop = FeedbackLoop()
        
        self.prisms = {
            'human_centric': HumanCentricPrism(),
            'sentient_first': SentientFirstPrism(),
            'ecocentric': EcocentricPrism(),
            'innovation_focused': InnovationFocusedPrism(),
        }

        self.prism_weights = {
            DecisionCategory.CRITICAL: {'human_centric': 0.4, 'sentient_first': 0.3, 'ecocentric': 0.2, 'innovation_focused': 0.1},
            DecisionCategory.HIGH_IMPACT: {'human_centric': 0.35, 'sentient_first': 0.25, 'ecocentric': 0.25, 'innovation_focused': 0.15},
            DecisionCategory.MODERATE: {'human_centric': 0.3, 'sentient_first': 0.2, 'ecocentric': 0.2, 'innovation_focused': 0.3},
            DecisionCategory.LOW_IMPACT: {'human_centric': 0.25, 'sentient_first': 0.25, 'ecocentric': 0.25, 'innovation_focused': 0.25},
        }
        
        self.decisions_history = []

    def analyze(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Combine insights from alignment tracker and feedback loop
        scenario_context = inputs.get('context', {})
        initial_scores = {name: prism.calculate_score(inputs.get(name, {})) for name, prism in self.prisms.items()}
        
        self.alignment_tracker.evaluate_alignment()
        context_result = self.context_engine.apply_adjustments(initial_scores, scenario_context)
        final_weights = self._adjust_weights_based_on_context_and_alignment()
        
        results = {}
        for name, prism in self.prisms.items():
            adjusted_score = context_result['adjusted_scores'][name]
            results[name] = {
                'score': adjusted_score * final_weights[name],
                'details': prism.latest_result.get("details", {})
            }

        return results

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        # Combine alignment insights, prism scores, and narrative summaries
        report = ["Ethical Analysis Report", "-----------------------"]
        for name, result in analysis_results.items():
            report.append(f"Prism: {name} - Score: {result['score']}")
            report.append(f"Details: {result['details']}")
        
        alignment_narrative = self.alignment_tracker.generate_narrative()
        report.append("Alignment Insights:")
        report.append(alignment_narrative)
        
        return "\n".join(report)
    
    def _adjust_weights_based_on_context_and_alignment(self):
        # Logic to combine context and alignment weight adjustments
        return {name: weight for name, weight in self.prism_weights.items()}

    def evaluate_action(self, action: str, context: Dict) -> EthicalDecision:
        """Evaluate decision with comprehensive error handling and warning emission"""
        if action is None or context is None:
            raise ValueError("Action and context must not be None")

        try:
            # Validate basic input structure
            if not isinstance(action, (str, dict)):
                raise TypeError(f"Action must be string or dict, got {type(action)}")
            if not isinstance(context, dict):
                raise TypeError(f"Context must be dict, got {type(context)}")

            # Validate and process context structure
            validation_results = self._validate_context_structure(context)
            
            # Emit warnings for missing or invalid criteria
            if validation_results['missing']:
                warnings.warn(
                    f"Missing criteria: {validation_results['missing']}", 
                    UserWarning
                )
            if validation_results['invalid']:
                warnings.warn(
                    f"Invalid criteria: {validation_results['invalid']}", 
                    UserWarning
                )
            
            # Check if we have any valid criteria to process
            if not validation_results['valid']:
                error_msg = (
                    "No valid criteria provided for evaluation.\n"
                    f"Missing criteria: {validation_results['missing']}\n"
                    f"Invalid criteria: {validation_results['invalid']}"
                )
                raise ValueError(error_msg)

            # Categorize decision
            try:
                category = self._categorize_decision(action, context)
            except Exception as e:
                self.logger.error(f"Error categorizing decision: {str(e)}")
                raise ValueError(f"Failed to categorize decision: {str(e)}")

            # Calculate prism scores with partial validation support
            prism_scores = {}
            prism_errors = {}
            
            for name, prism in self.prisms.items():
                try:
                    # Create prism-specific context with available valid criteria
                    prism_context = {}
                    required_criteria = set(getattr(prism, 'criteria', {}).keys())
                    available_criteria = set(validation_results['valid'].keys())
                    
                    # Check if we have enough criteria for this prism
                    missing_required = required_criteria - available_criteria
                    if missing_required:
                        warnings.warn(
                            f"Missing required criteria for {name} prism: {missing_required}",
                            UserWarning
                        )
                        # Only skip if all required criteria are missing
                        if len(missing_required) == len(required_criteria):
                            prism_scores[name] = {
                                "score": 0, 
                                "details": {"error": "Missing all required criteria"}
                            }
                            continue

                    # Include available valid criteria
                    for criterion in required_criteria & available_criteria:
                        prism_context[criterion] = validation_results['valid'][criterion]
                    
                    # Calculate score with available criteria
                    result = prism.calculate_score(prism_context)
                    if isinstance(result, dict) and 'score' in result:
                        prism_scores[name] = result
                    else:
                        warnings.warn(
                            f"Invalid result format from {name} prism",
                            UserWarning
                        )
                        prism_scores[name] = {
                            "score": 0, 
                            "details": {"error": "Invalid result format"}
                        }
                    
                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Error in {name} prism: {error_msg}")
                    prism_errors[name] = error_msg
                    prism_scores[name] = {
                        "score": 0, 
                        "details": {"error": error_msg}
                    }

            # Calculate confidence using available scores
            try:
                confidence = self._calculate_confidence(prism_scores, category)
            except Exception as e:
                self.logger.error(f"Error calculating confidence: {str(e)}")
                confidence = 0.0

            # Generate decision with available components
            decision = EthicalDecision(
                decision_id=str(len(self.decisions_history) + 1),
                category=category,
                recommendation=self._determine_recommendation(confidence, 
                    self._identify_risks(prism_scores)),
                confidence_score=confidence,
                prism_scores=prism_scores,
                context_snapshot=context,
                reasoning=self._generate_reasoning(prism_scores, confidence),
                risk_factors=self._identify_risks(prism_scores),
                mitigation_steps=self._generate_mitigation_steps(
                    self._identify_risks(prism_scores), 
                    category
                ),
                stakeholder_impact=self._calculate_stakeholder_impact(prism_scores),
                timestamp=time.time(),
            )

            # Add validation warnings to reasoning if any
            if validation_results['missing'] or validation_results['invalid']:
                decision.reasoning.extend([
                    "Warning: Partial evaluation performed due to:",
                    f"- Missing criteria: {validation_results['missing']}",
                    f"- Invalid criteria: {validation_results['invalid']}"
                ])

            self.decisions_history.append(decision)
            return decision

        except Exception as e:
            self.logger.error(f"Critical error in evaluate_action: {str(e)}")
            raise

    def _determine_recommendation(self, confidence: float, risks: List[str]) -> DecisionOutcome:
        """Determine recommendation based on confidence and risks"""
        if confidence < 0.4 or len(risks) > 3:
            return DecisionOutcome.REJECT
        elif confidence < 0.6 or risks:
            return DecisionOutcome.REVIEW
        elif confidence < 0.8:
            return DecisionOutcome.ESCALATE
        return DecisionOutcome.APPROVE

    def _generate_reasoning(self, prism_scores: Dict[str, Any], confidence: float) -> List[str]:
        """Generate reasoning based on prism scores and confidence"""
        reasoning = []
        
        # Add overall confidence assessment
        reasoning.append(f"Overall confidence score: {confidence:.2f}")
        
        # Add insights from each prism
        for prism_name, score_data in prism_scores.items():
            if isinstance(score_data, dict):
                score = score_data.get('score', 0)
                impact_level = score_data.get('impact_level', 'unknown')
                reasoning.append(f"{prism_name}: Score {score:.1f}, Impact Level: {impact_level}")
                
                # Add any specific recommendations from the prism
                if 'recommendations' in score_data:
                    reasoning.extend(f"- {rec}" for rec in score_data['recommendations'])
        
        return reasoning

    def _calculate_stakeholder_impact(self, prism_scores: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on different stakeholders based on prism scores"""
        stakeholder_impact = {}
        
        # Extract relevant scores from prisms
        if 'human_centric' in prism_scores:
            human_score = prism_scores['human_centric'].get('score', 0) / 100.0
            stakeholder_impact['individual'] = human_score
            
        if 'sentient_first' in prism_scores:
            sentient_score = prism_scores['sentient_first'].get('score', 0) / 100.0
            stakeholder_impact['collective'] = sentient_score
            
        if 'ecocentric' in prism_scores:
            eco_score = prism_scores['ecocentric'].get('score', 0) / 100.0
            stakeholder_impact['environment'] = eco_score
        
        return stakeholder_impact

    def _categorize_decision(self, action: Dict, context: Dict) -> DecisionCategory:
        """Categorize the decision based on action and context"""
        severity = context.get('severity', 'low').lower()
        impact_scope = context.get('impact_scope', 'individual').lower()
        
        if severity == 'high' or impact_scope == 'global':
            return DecisionCategory.CRITICAL
        elif severity == 'medium' or impact_scope == 'regional':
            return DecisionCategory.HIGH_IMPACT
        elif severity == 'low' and impact_scope == 'individual':
            return DecisionCategory.LOW_IMPACT
        else:
            return DecisionCategory.MODERATE

    def _calculate_confidence(self, prism_scores: Dict[str, Any], category: DecisionCategory) -> float:
        """Calculate confidence score based on prism scores and decision category"""
        try:
            # Extract numeric scores from prism results
            numeric_scores = {}
            for prism_name, score_data in prism_scores.items():
                if isinstance(score_data, dict) and 'score' in score_data:
                    numeric_scores[prism_name] = score_data['score'] / 100.0  # Convert percentage to decimal
                elif isinstance(score_data, (int, float)):
                    numeric_scores[prism_name] = float(score_data)
                else:
                    self.logger.warning(f"Invalid score format for {prism_name}: {score_data}")
                    continue

            # Get weights for the decision category
            weights = self.prism_weights[category]
            
            # Calculate weighted scores only for prisms with valid numeric scores
            weighted_scores = []
            for prism_name, weight in weights.items():
                if prism_name in numeric_scores:
                    weighted_scores.append(numeric_scores[prism_name] * weight)
                else:
                    self.logger.warning(f"Missing score for prism: {prism_name}")

            if not weighted_scores:
                self.logger.error("No valid scores available for confidence calculation")
                return 0.0

            return sum(weighted_scores)
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _identify_risks(self, prism_scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify potential risks based on prism scores"""
        risks = []
        try:
            for prism_name, score_data in prism_scores.items():
                # Extract numeric score from prism result
                if isinstance(score_data, dict) and 'score' in score_data:
                    score = score_data['score'] / 100.0  # Convert percentage to decimal
                elif isinstance(score_data, (int, float)):
                    score = float(score_data)
                else:
                    self.logger.warning(f"Invalid score format for {prism_name}: {score_data}")
                    continue

                # Check for risks based on score
                if score < 0.6:
                    risks.append(f"Low {prism_name} score: {score:.2f}")
                    
                # Add specific risk checks based on prism details
                if 'details' in score_data:
                    details = score_data['details']
                    if prism_name == 'sentient_first' and details.get('impact_level') == 'critical':
                        risks.append("Critical sentient impact detected")
                    elif prism_name == 'human_centric' and details.get('privacy', {}).get('data_protection', 1.0) < 0.7:
                        risks.append("Insufficient data protection measures")

        except Exception as e:
            self.logger.error(f"Error identifying risks: {str(e)}")
            
        return risks

    def _generate_mitigation_steps(self, risks: List[str], category: DecisionCategory) -> List[str]:
        """Generate mitigation steps based on identified risks"""
        steps = []
        
        try:
            # Map risks to specific mitigation steps
            risk_mitigations = {
                "Low human_centric score": [
                    "Review impact on human wellbeing",
                    "Enhance privacy protections",
                    "Strengthen fairness measures"
                ],
                "Low sentient_first score": [
                    "Assess sentient impact more thoroughly",
                    "Implement additional safeguards",
                    "Consider alternative approaches"
                ],
                "Critical sentient impact": [
                    "URGENT: Immediate review required",
                    "Implement enhanced monitoring",
                    "Develop contingency plans"
                ],
                "Insufficient data protection": [
                    "Strengthen data protection measures",
                    "Review privacy compliance",
                    "Implement additional security controls"
                ]
            }

            # Generate steps for each identified risk
            for risk in risks:
                # Find matching mitigation steps
                for risk_pattern, mitigations in risk_mitigations.items():
                    if risk_pattern in risk:
                        steps.extend(mitigations)
                        break
                else:
                    # Default mitigation if no specific match found
                    steps.append(f"Mitigate: {risk}")

            # Add category-specific steps
            if category in [DecisionCategory.CRITICAL, DecisionCategory.HIGH_IMPACT]:
                steps.append("Implement additional monitoring and oversight")
                steps.append("Schedule regular impact assessments")
                steps.append("Establish feedback mechanisms")

        except Exception as e:
            self.logger.error(f"Error generating mitigation steps: {str(e)}")
            steps.append("ERROR: Failed to generate complete mitigation steps")

        return list(set(steps))  # Remove duplicates

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate inputs for all prisms"""
        missing_criteria = {}
        invalid_values = {}

        for prism_name, prism in self.prisms.items():
            try:
                required_criteria = set(prism.criteria.keys())
                provided_criteria = set(inputs.keys())
                missing = required_criteria - provided_criteria
                
                if missing:
                    missing_criteria[prism_name] = list(missing)

                # Validate values for provided criteria
                for criterion in required_criteria & provided_criteria:
                    value = inputs[criterion]
                    if isinstance(value, dict):
                        # Check nested values
                        invalid = [k for k, v in value.items() 
                                 if not isinstance(v, (int, float)) or not 0 <= float(v) <= 1]
                        if invalid:
                            invalid_values[f"{prism_name}.{criterion}"] = invalid
                    elif not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                        invalid_values[f"{prism_name}.{criterion}"] = value

            except Exception as e:
                raise ValueError(f"Error validating {prism_name} inputs: {str(e)}")

        if missing_criteria:
            raise ValueError(f"Missing required criteria: {missing_criteria}")
        if invalid_values:
            raise ValueError(f"Invalid values found: {invalid_values}")

    def _validate_context_structure(self, context: Dict) -> Dict[str, Dict]:
        """Validate context structure while allowing partial evaluations"""
        validation_results = {
            'valid': {},
            'missing': {},
            'invalid': {},
            'warnings': []
        }

        required_criteria = {
            'human_centric': ['wellbeing', 'fairness', 'autonomy', 'privacy'],
            'sentient_first': ['sentient_impact', 'organizational_welfare', 'resource_sustainability'],
            'ecocentric': ['environmental_impact', 'biodiversity_preservation', 'carbon_neutrality'],
            'innovation_focused': ['risk_assessment', 'benefit_potential', 'feasibility']
        }

        # Process and validate each criterion
        for prism_name, criteria_list in required_criteria.items():
            missing_criteria = []
            for criterion in criteria_list:
                if criterion not in context:
                    missing_criteria.append(criterion)
                    continue

                value = context[criterion]
                if isinstance(value, dict):
                    # Validate nested values
                    valid_nested = {}
                    invalid_nested = {}
                    
                    for k, v in value.items():
                        if isinstance(v, (int, float)) and 0 <= v <= 1:
                            valid_nested[k] = v
                        else:
                            invalid_nested[k] = v
                    
                    if valid_nested:
                        validation_results['valid'][criterion] = valid_nested
                    if invalid_nested:
                        validation_results['invalid'][criterion] = invalid_nested
                        validation_results['warnings'].append(
                            f"Invalid nested values in {criterion}: {invalid_nested}"
                        )
                elif isinstance(value, (int, float)) and 0 <= value <= 1:
                    validation_results['valid'][criterion] = value
                else:
                    validation_results['invalid'][criterion] = value
                    validation_results['warnings'].append(
                        f"Invalid value for {criterion}: {value}"
                    )

            if missing_criteria:
                validation_results['missing'][prism_name] = missing_criteria
                validation_results['warnings'].append(
                    f"Missing criteria for {prism_name}: {missing_criteria}"
                )

        # Log warnings but continue processing
        for warning in validation_results['warnings']:
            self.logger.warning(warning)

        return validation_results

# Note: This combines the best elements while retaining modularity and extensibility.
