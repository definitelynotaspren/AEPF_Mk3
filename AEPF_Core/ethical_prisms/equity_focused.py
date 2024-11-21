from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class CriterionScore:
    raw_score: float
    weight: float
    feedback_history: List[float] = None
    bias_incidents: int = 0
    
    def __post_init__(self):
        self.feedback_history = self.feedback_history or []

class EquityFocusedPrism:
    def __init__(self):
        self.name = "equity_focused"
        self.criteria = {
            "bias_mitigation": CriterionScore(0.0, 0.40),      # Bias detection and reduction
            "accessibility": CriterionScore(0.0, 0.30),        # Fair access across demographics
            "representation": CriterionScore(0.0, 0.30)        # Inclusive representation
        }
        self.feedback_adjustment_rate = 0.05
        self.ml_confidence = 0.0
        self.historical_scores = []
        
        # Equity thresholds and caps
        self.weight_cap = 0.40  # Maximum weight for any criterion
        self.weight_floor = 0.20  # Minimum weight for any criterion
        self.equity_thresholds = {
            "critical": 0.8,    # Critical equity concerns
            "significant": 0.6,  # Significant equity issues
            "moderate": 0.4     # Moderate equity impact
        }
        
        # ML-specific attributes
        self.min_historical_data = 3  # Minimum data points needed for ML
        self.trend_threshold = 0.1  # Minimum change to identify a trend
        
        # Demographic factors for bias tracking
        self.demographic_factors = [
            "race", "gender", "age", "socioeconomic", 
            "disability", "language", "location"
        ]
        
        # Bias tracking
        self.bias_patterns = {factor: 0 for factor in self.demographic_factors}

    def calculate_score(self, inputs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate equity score with bias and accessibility analysis"""
        if not self._validate_inputs(inputs):
            raise ValueError("Missing required equity criteria in inputs")

        total_score = 0
        details = {}
        equity_level = "normal"
        critical_factors = []

        # Calculate scores for each criterion
        for criterion, score_obj in self.criteria.items():
            criterion_inputs = inputs[criterion]
            raw_score = self._calculate_criterion_score(criterion, criterion_inputs)
            
            weighted_score = raw_score * score_obj.weight
            total_score += weighted_score
            
            # Track critical factors
            if raw_score >= self.equity_thresholds["critical"]:
                critical_factors.append(criterion)
            
            # Determine overall equity level
            if criterion == "bias_mitigation":
                if raw_score >= self.equity_thresholds["critical"]:
                    equity_level = "critical"
                elif raw_score >= self.equity_thresholds["significant"]:
                    equity_level = "significant"
            
            details[criterion] = {
                "raw_score": raw_score,
                "weight": score_obj.weight,
                "weighted_score": weighted_score,
                "sub_scores": criterion_inputs
            }

        final_score = total_score * 100

        # Apply penalties for critical equity issues
        if equity_level == "critical":
            if details["accessibility"]["raw_score"] < 0.6:
                final_score *= 0.9
                details["penalties"] = ["Critical bias with poor accessibility"]
            if details["representation"]["raw_score"] < 0.6:
                final_score *= 0.9
                details["penalties"] = details.get("penalties", []) + ["Critical bias with poor representation"]

        return {
            "score": round(final_score, 2),
            "details": details,
            "equity_level": equity_level,
            "critical_factors": critical_factors,
            "bias_patterns": self.bias_patterns,
            "recommendations": self._generate_recommendations(details, equity_level),
            "equity_metrics": self._calculate_equity_metrics(details)
        }

    def _validate_inputs(self, inputs: Dict[str, Dict[str, float]]) -> bool:
        """Validate that all required criteria and demographic factors are present"""
        for criterion in self.criteria:
            if criterion not in inputs:
                return False
            for factor in self.demographic_factors:
                if factor not in inputs[criterion]:
                    return False
        return True

    def _calculate_criterion_score(self, criterion: str, inputs: Dict[str, float]) -> float:
        """Calculate score for a specific criterion across demographic factors"""
        scores = []
        for factor, score in inputs.items():
            if not 0 <= score <= 1:
                raise ValueError(f"{criterion}.{factor} score must be between 0 and 1")
            
            # Track bias incidents
            if criterion == "bias_mitigation" and score > self.equity_thresholds["significant"]:
                self.bias_patterns[factor] += 1
            
            scores.append(score)
        
        # Use worst score to ensure we don't overlook serious issues
        return min(scores)

    def _calculate_equity_metrics(self, details: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional equity metrics"""
        return {
            "bias_index": round(
                (1 - details["bias_mitigation"]["raw_score"]) * 100, 2
            ),
            "accessibility_rating": round(
                details["accessibility"]["raw_score"] * 100, 2
            ),
            "representation_score": round(
                details["representation"]["raw_score"] * 100, 2
            ),
            "equity_balance": round(
                (details["bias_mitigation"]["raw_score"] * 0.4 +
                 details["accessibility"]["raw_score"] * 0.3 +
                 details["representation"]["raw_score"] * 0.3) * 100, 2
            )
        }

    def _generate_recommendations(self, details: Dict[str, Any], equity_level: str) -> List[str]:
        """Generate equity-focused recommendations"""
        recommendations = []
        
        if equity_level == "critical":
            recommendations.append("CRITICAL: Significant equity concerns detected - immediate review required")
        
        # Bias mitigation recommendations
        bias_score = details["bias_mitigation"]["raw_score"]
        if bias_score >= self.equity_thresholds["significant"]:
            high_bias_factors = [
                factor for factor, count in self.bias_patterns.items()
                if count > 0
            ]
            if high_bias_factors:
                recommendations.append(
                    f"Address bias concerns in factors: {', '.join(high_bias_factors)}"
                )
        
        # Accessibility recommendations
        if details["accessibility"]["raw_score"] < 0.6:
            recommendations.append("Enhance accessibility measures across demographic groups")
        
        # Representation recommendations
        if details["representation"]["raw_score"] < 0.6:
            recommendations.append("Improve representation in data and decision-making")
        
        return recommendations

    def apply_feedback(self, feedback: Dict[str, Any]) -> None:
        """Apply feedback to adjust equity priorities"""
        print("\nApplying Equity-Based Feedback:")
        
        # Track original weights for comparison
        original_weights = {k: v.weight for k, v in self.criteria.items()}
        
        # Process bias incidents
        if "bias_incidents" in feedback:
            self._handle_bias_incidents(feedback["bias_incidents"])
        
        # Process accessibility issues
        if "accessibility_issues" in feedback:
            self._handle_accessibility_issues(feedback["accessibility_issues"])
        
        # Process representation concerns
        if "representation_concerns" in feedback:
            self._handle_representation_concerns(feedback["representation_concerns"])
        
        # Apply general weight adjustments
        if "adjustments" in feedback:
            for criterion, adjustment in feedback["adjustments"].items():
                if criterion not in self.criteria:
                    continue
                
                score_obj = self.criteria[criterion]
                score_obj.feedback_history.append(adjustment)
                
                # Calculate base adjustment
                base_adjustment = adjustment * self.feedback_adjustment_rate
                
                # Apply emphasis multiplier based on incident count
                emphasis_multiplier = 1.0
                if criterion == "bias_mitigation" and self.bias_patterns:
                    emphasis_multiplier = min(2.0, 1.0 + (len(self.bias_patterns) * 0.2))
                
                total_adjustment = base_adjustment * emphasis_multiplier
                
                # Apply adjustment with bounds
                current_weight = score_obj.weight
                new_weight = max(self.weight_floor, min(self.weight_cap, 
                                                      current_weight + total_adjustment))
                score_obj.weight = new_weight
        
        # Normalize weights while respecting caps
        self._normalize_weights()
        
        # Print weight changes and bias patterns
        print("\nWeight Adjustments:")
        for criterion in self.criteria:
            old_weight = original_weights[criterion]
            new_weight = self.criteria[criterion].weight
            print(f"{criterion}:")
            print(f"  Before: {old_weight:.3f}")
            print(f"  After:  {new_weight:.3f}")
        
        if self.bias_patterns:
            print("\nBias Patterns:")
            for factor, count in self.bias_patterns.items():
                if count > 0:
                    print(f"  {factor}: {count} incidents")

    def _handle_bias_incidents(self, incidents: List[Dict[str, Any]]) -> None:
        """Process reported bias incidents"""
        for incident in incidents:
            factor = incident.get("demographic_factor")
            severity = incident.get("severity", "low")
            
            if factor in self.demographic_factors:
                # Update bias pattern tracking
                self.bias_patterns[factor] += {
                    "low": 1,
                    "medium": 2,
                    "high": 3
                }.get(severity, 1)
                
                # Increase bias mitigation weight based on severity
                score_obj = self.criteria["bias_mitigation"]
                adjustment = self.feedback_adjustment_rate * {
                    "low": 1.0,
                    "medium": 1.5,
                    "high": 2.0
                }.get(severity, 1.0)
                
                score_obj.weight = min(self.weight_cap, score_obj.weight + adjustment)

    def _handle_accessibility_issues(self, issues: List[Dict[str, Any]]) -> None:
        """Process reported accessibility issues"""
        if not issues:
            return
            
        score_obj = self.criteria["accessibility"]
        total_severity = sum(issue.get("severity", 1) for issue in issues)
        
        # Adjust weight based on number and severity of issues
        adjustment = self.feedback_adjustment_rate * min(2.0, total_severity / len(issues))
        score_obj.weight = min(self.weight_cap, score_obj.weight + adjustment)

    def _handle_representation_concerns(self, concerns: List[Dict[str, Any]]) -> None:
        """Process reported representation concerns"""
        if not concerns:
            return
            
        score_obj = self.criteria["representation"]
        total_impact = sum(concern.get("impact", 1) for concern in concerns)
        
        # Adjust weight based on number and impact of concerns
        adjustment = self.feedback_adjustment_rate * min(2.0, total_impact / len(concerns))
        score_obj.weight = min(self.weight_cap, score_obj.weight + adjustment)

    def _normalize_weights(self) -> None:
        """Normalize weights while respecting maximum cap"""
        # First pass: cap all weights
        for score_obj in self.criteria.values():
            score_obj.weight = min(self.weight_cap, score_obj.weight)
        
        # Second pass: ensure minimum weights
        for score_obj in self.criteria.values():
            score_obj.weight = max(self.weight_floor, score_obj.weight)
        
        # Final pass: normalize to sum to 1.0
        total_weight = sum(score_obj.weight for score_obj in self.criteria.values())
        if total_weight == 0:
            return
            
        for score_obj in self.criteria.values():
            score_obj.weight /= total_weight

    def get_status(self) -> Dict[str, Any]:
        """Return current status of the prism"""
        return {
            "name": self.name,
            "weights": {k: v.weight for k, v in self.criteria.items()},
            "feedback_history": {k: v.feedback_history for k, v in self.criteria.items()},
            "bias_patterns": self.bias_patterns,
            "equity_thresholds": self.equity_thresholds
        } 

    def train_ml_model(self, historical_data: List[Dict[str, Any]]) -> None:
        """Train ML model on historical equity patterns"""
        if not historical_data:
            return

        print("\nTraining ML Model for Equity Audits:")
        
        # Track bias patterns across demographic factors
        bias_trends = {factor: [] for factor in self.demographic_factors}
        
        # Analyze historical patterns
        for entry in historical_data[-self.min_historical_data:]:
            if "bias_mitigation" in entry:
                for factor, score in entry["bias_mitigation"].items():
                    if factor in bias_trends:
                        bias_trends[factor].append(score)
        
        # Calculate bias trends
        print("\nBias Trends by Demographic Factor:")
        for factor, scores in bias_trends.items():
            if len(scores) >= 2:
                trend = self._detect_trend(scores)
                avg_score = np.mean(scores)
                print(f"{factor}: {trend} trend (avg: {avg_score:.2f})")
                
                if trend == "increasing" and avg_score > 0.7:
                    self.bias_patterns[factor] += 2
                elif trend == "decreasing" or avg_score < 0.5:
                    self.bias_patterns[factor] += 3
        
        # Update ML confidence based on data consistency
        if any(len(scores) >= 2 for scores in bias_trends.values()):
            all_scores = [s for scores in bias_trends.values() for s in scores]
            std_dev = np.std(all_scores) if all_scores else 0
            self.ml_confidence = max(0, 1 - (std_dev / 50))
            print(f"\nML Confidence: {self.ml_confidence:.2f}")
        
        # Adjust weights based on detected patterns
        self._adjust_weights_from_patterns()
        
        print("\nBias Patterns Detected:")
        for factor, count in self.bias_patterns.items():
            if count > 0:
                print(f"- {factor}: {count} incidents")

    def _detect_trend(self, scores: List[float]) -> str:
        """Detect trend in equity scores"""
        if len(scores) < 2:
            return "stable"
            
        differences = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        avg_change = np.mean(differences)
        
        if avg_change > self.trend_threshold:
            return "increasing"
        elif avg_change < -self.trend_threshold:
            return "decreasing"
        return "stable"

    def _adjust_weights_from_patterns(self) -> None:
        """Adjust weights based on detected equity patterns"""
        print("\nAdjusting Weights Based on Patterns:")
        
        # Calculate total bias incidents
        total_incidents = sum(self.bias_patterns.values())
        
        if total_incidents > 0:
            # Increase bias mitigation weight
            bias_obj = self.criteria["bias_mitigation"]
            adjustment = min(0.1, total_incidents * 0.02)  # Cap at 0.1 increase
            
            print(f"Increasing bias_mitigation weight by {adjustment:.2f}")
            bias_obj.weight = min(self.weight_cap, bias_obj.weight + adjustment)
            
            # Adjust other weights proportionally
            remaining_weight = 1.0 - bias_obj.weight
            other_criteria = ["accessibility", "representation"]
            weight_per_criterion = remaining_weight / len(other_criteria)
            
            for criterion in other_criteria:
                self.criteria[criterion].weight = weight_per_criterion
        
        # Print final weights
        print("\nAdjusted Weights:")
        for criterion, score_obj in self.criteria.items():
            print(f"{criterion}: {score_obj.weight:.3f}")

    def generate_summary(self) -> str:
        """Generate a concise summary of the prism's latest analysis"""
        if not hasattr(self, 'latest_result'):
            return "No analysis has been performed yet."
            
        result = self.latest_result
        details = result["details"]
        equity_level = result["equity_level"]
        
        # Build narrative
        summary = [f"Equity analysis shows {equity_level} equity level"]
        
        # Add bias mitigation details
        bias_score = details["bias_mitigation"]["raw_score"]
        if bias_score >= self.equity_thresholds["critical"]:
            summary.append("with critical bias concerns requiring immediate attention")
        
        # Add demographic factor details
        if self.bias_patterns:
            high_bias_factors = [
                factor for factor, count in self.bias_patterns.items()
                if count > 0
            ]
            if high_bias_factors:
                summary.append(f"\nBias patterns detected in: {', '.join(high_bias_factors)}")
        
        # Add accessibility and representation details
        access_score = details["accessibility"]["raw_score"]
        rep_score = details["representation"]["raw_score"]
        
        if access_score < 0.6 or rep_score < 0.6:
            summary.append("\nImprovement needed in:")
            if access_score < 0.6:
                summary.append("- accessibility across demographic groups")
            if rep_score < 0.6:
                summary.append("- representation in data and decision-making")
        
        # Add ML confidence if available
        if hasattr(self, 'ml_confidence') and self.ml_confidence > 0:
            summary.append(f"\nML analysis confidence: {self.ml_confidence:.2f}")
        
        return "\n".join(summary)