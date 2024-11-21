from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class SubCriterionScore:
    raw_score: float
    weight: float
    feedback_history: List[float] = None
    success_count: int = 0
    failure_count: int = 0
    
    def __post_init__(self):
        self.feedback_history = self.feedback_history or []

class CriterionGroup:
    def __init__(self, weight: float, sub_criteria: Dict[str, float]):
        self.weight = weight
        self.sub_criteria = {
            name: SubCriterionScore(0.0, sub_weight)
            for name, sub_weight in sub_criteria.items()
        }
        self.risk_success_ratio = 1.0  # Track success rate of risky decisions
    
    def calculate_group_score(self, inputs: Dict[str, float]) -> float:
        """Calculate weighted score for the criterion group"""
        total_score = 0
        for name, score_obj in self.sub_criteria.items():
            if name in inputs:
                total_score += inputs[name] * score_obj.weight
        return total_score

    def update_success_ratio(self):
        """Update risk-success ratio based on feedback history"""
        total_attempts = sum(score.success_count + score.failure_count 
                           for score in self.sub_criteria.values())
        if total_attempts > 0:
            total_successes = sum(score.success_count 
                                for score in self.sub_criteria.values())
            self.risk_success_ratio = total_successes / total_attempts

class InnovationFocusedPrism:
    def __init__(self):
        self.name = "innovation_focused"
        # Define main criteria groups with sub-criteria
        self.criteria = {
            "risk_assessment": CriterionGroup(0.4, {
                "financial_risk": 0.35,
                "reputational_risk": 0.35,
                "technological_risk": 0.30
            }),
            "benefit_potential": CriterionGroup(0.4, {
                "economic_benefit": 0.35,
                "societal_benefit": 0.35,
                "scientific_advancement": 0.30
            }),
            "feasibility": CriterionGroup(0.2, {
                "implementation_practicality": 0.6,
                "resource_availability": 0.4
            })
        }
        
        # Feedback and adjustment parameters
        self.feedback_adjustment_rate = 0.05
        self.success_threshold = 0.7  # Threshold for considering feedback positive
        self.risk_tolerance_max = 0.5  # Maximum weight for risk assessment
        self.risk_tolerance_min = 0.3  # Minimum weight for risk assessment
        self.feedback_history = []
        
        # Scoring parameters
        self.benefit_risk_threshold = 0.2
        self.high_benefit_multiplier = 1.2
        self.risk_thresholds = {
            "high": 0.8,
            "moderate": 0.6,
            "low": 0.4
        }

    def calculate_score(self, inputs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate innovation score with risk-benefit analysis"""
        if not self._validate_inputs(inputs):
            raise ValueError("Missing required criteria in inputs")

        total_score = 0
        details = {}
        risk_level = "low"

        # Calculate scores for each main criterion
        group_scores = {}
        for criterion, group in self.criteria.items():
            group_score = group.calculate_group_score(inputs[criterion])
            weighted_score = group_score * group.weight
            total_score += weighted_score
            
            group_scores[criterion] = {
                "raw_score": group_score,
                "weight": group.weight,
                "weighted_score": weighted_score,
                "sub_scores": {
                    name: {
                        "raw_score": inputs[criterion].get(name, 0),
                        "weight": score_obj.weight,
                        "weighted_score": inputs[criterion].get(name, 0) * score_obj.weight
                    }
                    for name, score_obj in group.sub_criteria.items()
                }
            }

        # Calculate risk level
        risk_score = group_scores["risk_assessment"]["raw_score"]
        if risk_score >= self.risk_thresholds["high"]:
            risk_level = "high"
        elif risk_score >= self.risk_thresholds["moderate"]:
            risk_level = "moderate"

        # Calculate benefit-risk ratio and apply multiplier if applicable
        benefit_score = group_scores["benefit_potential"]["raw_score"]
        risk_score = group_scores["risk_assessment"]["raw_score"]
        
        benefit_risk_ratio = (benefit_score - risk_score) / max(risk_score, 0.1)
        final_score = total_score * 100

        # Apply multiplier for high benefit-to-risk ratio
        if benefit_risk_ratio > self.benefit_risk_threshold:
            final_score *= self.high_benefit_multiplier
            details["score_adjustments"] = [
                f"Applied {self.high_benefit_multiplier}x multiplier for favorable benefit-risk ratio"
            ]

        # Calculate innovation metrics
        innovation_metrics = self._calculate_innovation_metrics(group_scores)

        return {
            "score": round(final_score, 2),
            "details": group_scores,
            "risk_level": risk_level,
            "benefit_risk_ratio": round(benefit_risk_ratio, 2),
            "innovation_metrics": innovation_metrics,
            "recommendations": self._generate_recommendations(group_scores, risk_level)
        }

    def _validate_inputs(self, inputs: Dict[str, Dict[str, float]]) -> bool:
        """Validate that all required criteria are present"""
        for criterion, group in self.criteria.items():
            if criterion not in inputs:
                return False
            for sub_criterion in group.sub_criteria.keys():
                if sub_criterion not in inputs[criterion]:
                    return False
                if not 0 <= inputs[criterion][sub_criterion] <= 1:
                    raise ValueError(f"{criterion}.{sub_criterion} must be between 0 and 1")
        return True

    def _calculate_innovation_metrics(self, group_scores: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional innovation metrics"""
        risk_score = group_scores["risk_assessment"]["raw_score"]
        benefit_score = group_scores["benefit_potential"]["raw_score"]
        feasibility_score = group_scores["feasibility"]["raw_score"]
        
        return {
            "innovation_potential": round(
                (benefit_score * 0.6 + feasibility_score * 0.4) * 100, 2
            ),
            "risk_exposure": round(risk_score * 100, 2),
            "feasibility_rating": round(feasibility_score * 100, 2),
            "benefit_risk_index": round(
                (benefit_score / max(risk_score, 0.1)) * 100, 2
            )
        }

    def _generate_recommendations(self, group_scores: Dict[str, Any], risk_level: str) -> List[str]:
        """Generate innovation-specific recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == "high":
            recommendations.append("HIGH RISK: Implement comprehensive risk mitigation strategies")
            
            # Check specific risk factors
            risk_scores = group_scores["risk_assessment"]["sub_scores"]
            for risk_type, details in risk_scores.items():
                if details["raw_score"] >= self.risk_thresholds["high"]:
                    recommendations.append(f"Develop specific mitigation plan for {risk_type}")
        
        # Benefit optimization recommendations
        benefit_scores = group_scores["benefit_potential"]["sub_scores"]
        for benefit_type, details in benefit_scores.items():
            if details["raw_score"] < 0.5:
                recommendations.append(f"Explore ways to enhance {benefit_type}")
        
        # Feasibility recommendations
        feasibility_scores = group_scores["feasibility"]["sub_scores"]
        if feasibility_scores["implementation_practicality"]["raw_score"] < 0.6:
            recommendations.append("Review and enhance implementation strategy")
        if feasibility_scores["resource_availability"]["raw_score"] < 0.6:
            recommendations.append("Assess and secure necessary resources")
        
        return recommendations

    def apply_feedback(self, feedback: Dict[str, Any]) -> None:
        """Apply feedback to adjust risk-reward balance"""
        print("\nApplying Risk-Reward Feedback:")
        
        # Track original weights
        original_weights = {
            criterion: group.weight 
            for criterion, group in self.criteria.items()
        }
        
        # Process outcome feedback
        if "outcomes" in feedback:
            self._process_outcome_feedback(feedback["outcomes"])
        
        # Process risk tolerance feedback
        if "risk_tolerance" in feedback:
            self._adjust_risk_tolerance(feedback["risk_tolerance"])
        
        # Update success ratios
        for group in self.criteria.values():
            group.update_success_ratio()
        
        # Adjust weights based on success patterns
        self._adjust_weights_from_patterns()
        
        # Print weight changes
        print("\nWeight Adjustments:")
        for criterion, group in self.criteria.items():
            old_weight = original_weights[criterion]
            new_weight = group.weight
            success_ratio = group.risk_success_ratio
            print(f"{criterion}:")
            print(f"  Weight: {old_weight:.3f} -> {new_weight:.3f}")
            print(f"  Success Ratio: {success_ratio:.2f}")

    def _process_outcome_feedback(self, outcomes: List[Dict[str, Any]]) -> None:
        """Process feedback on decision outcomes"""
        for outcome in outcomes:
            risk_level = outcome.get("risk_level", "low")
            success = outcome.get("success", False)
            affected_criteria = outcome.get("affected_criteria", [])
            
            for criterion in affected_criteria:
                if criterion not in self.criteria:
                    continue
                
                group = self.criteria[criterion]
                for sub_name, sub_score in group.sub_criteria.items():
                    if success:
                        sub_score.success_count += 1
                    else:
                        sub_score.failure_count += 1

    def _adjust_risk_tolerance(self, tolerance_feedback: float) -> None:
        """Adjust risk tolerance based on feedback"""
        risk_group = self.criteria["risk_assessment"]
        benefit_group = self.criteria["benefit_potential"]
        
        # Calculate adjustment
        adjustment = tolerance_feedback * self.feedback_adjustment_rate
        
        # Apply adjustment with bounds
        new_risk_weight = max(
            self.risk_tolerance_min,
            min(self.risk_tolerance_max,
                risk_group.weight + adjustment)
        )
        
        # Adjust benefit weight proportionally
        weight_diff = new_risk_weight - risk_group.weight
        new_benefit_weight = benefit_group.weight - weight_diff
        
        # Update weights
        risk_group.weight = new_risk_weight
        benefit_group.weight = new_benefit_weight
        
        # Normalize all weights
        self._normalize_weights()

    def _adjust_weights_from_patterns(self) -> None:
        """Adjust weights based on success patterns"""
        risk_group = self.criteria["risk_assessment"]
        benefit_group = self.criteria["benefit_potential"]
        
        # If high-risk decisions are consistently successful
        if risk_group.risk_success_ratio > self.success_threshold:
            # Increase risk tolerance slightly
            adjustment = self.feedback_adjustment_rate
            risk_group.weight = max(
                self.risk_tolerance_min,
                min(self.risk_tolerance_max,
                    risk_group.weight - adjustment)  # Reduce risk weight
            )
            benefit_group.weight += adjustment  # Increase benefit weight
        
        # If high-risk decisions are consistently failing
        elif risk_group.risk_success_ratio < (1 - self.success_threshold):
            # Decrease risk tolerance
            adjustment = self.feedback_adjustment_rate
            risk_group.weight = max(
                self.risk_tolerance_min,
                min(self.risk_tolerance_max,
                    risk_group.weight + adjustment)  # Increase risk weight
            )
            benefit_group.weight -= adjustment  # Decrease benefit weight
        
        # Normalize weights
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Ensure weights sum to 1.0"""
        total_weight = sum(group.weight for group in self.criteria.values())
        if total_weight == 0:
            return
        
        for group in self.criteria.values():
            group.weight /= total_weight

    def get_status(self) -> Dict[str, Any]:
        """Return current status of the prism"""
        return {
            "name": self.name,
            "criteria_weights": {
                criterion: {
                    "group_weight": group.weight,
                    "sub_criteria": {
                        name: score_obj.weight
                        for name, score_obj in group.sub_criteria.items()
                    }
                }
                for criterion, group in self.criteria.items()
            },
            "risk_thresholds": self.risk_thresholds,
            "benefit_risk_threshold": self.benefit_risk_threshold
        }

    def train_ml_model(self, historical_data: List[Dict[str, Any]]) -> None:
        """Train ML model on historical innovation outcomes"""
        if not historical_data:
            return

        print("\nTraining ML Model on Innovation Outcomes:")
        
        # Track success rates for different risk levels
        risk_success_data = {
            "high": {"success": 0, "total": 0},
            "moderate": {"success": 0, "total": 0},
            "low": {"success": 0, "total": 0}
        }
        
        # Analyze historical outcomes
        for entry in historical_data:
            risk_level = self._determine_risk_level(entry)
            success = entry.get("success", False)
            
            risk_success_data[risk_level]["total"] += 1
            if success:
                risk_success_data[risk_level]["success"] += 1
        
        # Calculate success rates
        success_rates = {}
        for level, data in risk_success_data.items():
            if data["total"] > 0:
                success_rates[level] = data["success"] / data["total"]
                print(f"{level.title()} Risk Success Rate: {success_rates[level]:.2f}")
        
        # Adjust risk tolerance based on success patterns
        self._adjust_risk_tolerance(success_rates)
        
        # Update ML confidence
        if any(data["total"] > 0 for data in risk_success_data.values()):
            total_decisions = sum(data["total"] for data in risk_success_data.values())
            total_successes = sum(data["success"] for data in risk_success_data.values())
            self.ml_confidence = total_successes / total_decisions
            print(f"ML Confidence: {self.ml_confidence:.2f}")

    def _determine_risk_level(self, data: Dict[str, Any]) -> str:
        """Determine risk level from historical data"""
        risk_score = 0
        risk_inputs = data.get("risk_assessment", {})
        
        if risk_inputs:
            risk_score = sum(risk_inputs.values()) / len(risk_inputs)
        
        if risk_score >= self.risk_thresholds["high"]:
            return "high"
        elif risk_score >= self.risk_thresholds["moderate"]:
            return "moderate"
        return "low"

    def _adjust_risk_tolerance(self, success_rates: Dict[str, float]) -> None:
        """Adjust risk tolerance based on success rates"""
        print("\nAdjusting Risk Tolerance:")
        
        # Base adjustments on high-risk success rate
        high_risk_success = success_rates.get("high", 0)
        
        if high_risk_success > 0.7:  # High success rate with high risk
            print("High success rate with high-risk decisions - increasing risk tolerance")
            self.risk_tolerance_max *= 1.1  # Increase maximum risk tolerance
            self.benefit_risk_threshold *= 0.9  # Lower threshold for benefit-risk ratio
            
        elif high_risk_success < 0.3:  # Low success rate with high risk
            print("Low success rate with high-risk decisions - decreasing risk tolerance")
            self.risk_tolerance_max *= 0.9  # Decrease maximum risk tolerance
            self.benefit_risk_threshold *= 1.1  # Increase threshold for benefit-risk ratio
        
        # Keep thresholds within reasonable bounds
        self.risk_tolerance_max = max(0.4, min(0.6, self.risk_tolerance_max))
        self.benefit_risk_threshold = max(0.1, min(0.3, self.benefit_risk_threshold))
        
        print(f"Updated Risk Tolerance Max: {self.risk_tolerance_max:.2f}")
        print(f"Updated Benefit-Risk Threshold: {self.benefit_risk_threshold:.2f}")

    def generate_summary(self) -> str:
        """Generate a concise summary of the prism's latest analysis"""
        if not hasattr(self, 'latest_result'):
            return "No analysis has been performed yet."
        
        result = self.latest_result
        details = result["details"]
        risk_level = result["risk_level"]
        benefit_ratio = result["benefit_risk_ratio"]
        
        # Build narrative
        summary = [f"Innovation analysis shows {risk_level} risk level"]
        
        # Add benefit-risk details
        if benefit_ratio > self.benefit_risk_threshold:
            summary.append(f"with favorable benefit-risk ratio of {benefit_ratio:.2f}")
        
        # Add specific metrics
        metrics = result["innovation_metrics"]
        summary.append(f"\nKey metrics:")
        summary.append(f"- Innovation Potential: {metrics['innovation_potential']}%")
        summary.append(f"- Risk Exposure: {metrics['risk_exposure']}%")
        summary.append(f"- Feasibility Rating: {metrics['feasibility_rating']}%")
        
        if hasattr(self, 'ml_confidence') and self.ml_confidence > 0:
            summary.append(f"\nML analysis confidence: {self.ml_confidence:.2f}")
        
        return "\n".join(summary)
