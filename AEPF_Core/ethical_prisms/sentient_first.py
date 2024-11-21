from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import numpy as np
from collections import defaultdict
import logging

@dataclass
class CriterionScore:
    raw_score: float
    weight: float
    feedback_history: List[float] = None
    conflict_count: int = 0
    historical_scores: List[float] = None
    
    def __post_init__(self):
        self.feedback_history = self.feedback_history or []
        self.historical_scores = self.historical_scores or []

class SentientFirstPrism:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.name = "sentient_first"
        self.criteria = {
            "sentient_impact": CriterionScore(0.0, 0.50),
            "organizational_welfare": CriterionScore(0.0, 0.30),
            "resource_sustainability": CriterionScore(0.0, 0.20)
        }
        self.feedback_adjustment_rate = 0.05
        self.ml_confidence = 0.0
        self.historical_scores = []
        self.impact_thresholds = {
            "critical": 0.8,
            "high": 0.7,
            "moderate": 0.5
        }
        # ML-specific attributes
        self.ml_adjustments = {k: 1.0 for k in self.criteria.keys()}
        self.trend_patterns = defaultdict(int)
        self.min_historical_data = 5
        self.trend_threshold = 0.1  # Minimum change to identify a trend
        self.max_ml_adjustment = 1.3  # Maximum ML weight multiplier
        self.priority_threshold = 0.8  # Add this line
        self.priority_weight_increase = 0.2  # Add this line
        self.conflict_history = []

    def train_ml_model(self, historical_data: List[Dict[str, Any]]) -> None:
        """Train ML model on historical data to detect impact patterns"""
        if len(historical_data) < self.min_historical_data:
            return

        print("\nTraining ML Model on Historical Data:")
        
        # Track historical scores for each criterion
        for criterion in self.criteria:
            scores = [entry.get(criterion, 0) for entry in historical_data]
            self.criteria[criterion].historical_scores.extend(scores)
        
        # Calculate ML confidence based on data consistency
        impact_scores = self.criteria["sentient_impact"].historical_scores[-self.min_historical_data:]
        if len(impact_scores) >= 2:
            std_dev = np.std(impact_scores)
            self.ml_confidence = max(0, 1 - (std_dev / 50))
            print(f"ML Confidence: {self.ml_confidence:.2f}")

        # Analyze impact patterns
        self._analyze_impact_patterns(historical_data)
        
        # Update ML adjustments based on patterns
        self._update_ml_adjustments()
        
        print("\nML Analysis Results:")
        print(f"Detected Patterns: {dict(self.trend_patterns)}")
        print("ML Adjustments:", {k: f"{v:.2f}" for k, v in self.ml_adjustments.items()})

    def _analyze_impact_patterns(self, historical_data: List[Dict[str, Any]]) -> None:
        """Analyze patterns in impact scores and conflicts"""
        recent_data = historical_data[-self.min_historical_data:]
        
        # Detect trends in impact scores
        impact_scores = [d.get("sentient_impact", 0) for d in recent_data]
        trend = self._detect_trend(impact_scores)
        if trend != "stable":
            self.trend_patterns[f"impact_{trend}"] += 1

        # Analyze conflict patterns
        for i in range(len(recent_data)-1):
            current = recent_data[i]
            next_entry = recent_data[i+1]
            
            # Check for recurring conflicts
            if (current.get("sentient_impact", 0) > self.impact_thresholds["high"] and
                current.get("organizational_welfare", 0) < 0.5):
                self.trend_patterns["impact_welfare_conflict"] += 1
            
            # Check for impact escalation
            if (next_entry.get("sentient_impact", 0) > 
                current.get("sentient_impact", 0) + self.trend_threshold):
                self.trend_patterns["escalating_impact"] += 1

    def _detect_trend(self, scores: List[float]) -> str:
        """Detect trend in a series of scores"""
        if len(scores) < 2:
            return "stable"
            
        differences = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        avg_change = np.mean(differences)
        
        if avg_change > self.trend_threshold:
            return "increasing"
        elif avg_change < -self.trend_threshold:
            return "decreasing"
        return "stable"

    def _update_ml_adjustments(self) -> None:
        """Update ML adjustments based on detected patterns"""
        base_adjustment = 0.05
        
        # Reset adjustments
        self.ml_adjustments = {k: 1.0 for k in self.criteria.keys()}
        
        # Apply pattern-based adjustments
        if self.trend_patterns["impact_welfare_conflict"] > 1:
            # Increase sentient impact weight when conflicts are common
            self.ml_adjustments["sentient_impact"] += base_adjustment * 2
            self.ml_adjustments["organizational_welfare"] -= base_adjustment
            
        if self.trend_patterns["escalating_impact"] > 1:
            # Increase sentient impact weight for escalating patterns
            self.ml_adjustments["sentient_impact"] += base_adjustment * 1.5
            
        if self.trend_patterns["impact_decreasing"] > 1:
            # Increase weights to counter declining trends
            self.ml_adjustments["sentient_impact"] += base_adjustment
            
        # Ensure adjustments stay within bounds
        for criterion in self.ml_adjustments:
            self.ml_adjustments[criterion] = max(0.8, min(self.max_ml_adjustment, 
                                                        self.ml_adjustments[criterion]))

    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Preprocess and validate inputs, handling nested structures"""
        processed_inputs = {}
        
        for criterion, value in inputs.items():
            try:
                if isinstance(value, dict):
                    # Extract numeric values from nested dictionary
                    numeric_values = [
                        float(v) for v in value.values() 
                        if isinstance(v, (int, float))
                    ]
                    if not numeric_values:
                        self.logger.warning(f"No valid numeric values found in {criterion}")
                        continue
                    processed_value = sum(numeric_values) / len(numeric_values)
                elif isinstance(value, (int, float)):
                    processed_value = float(value)
                else:
                    self.logger.warning(f"Skipping non-numeric input for {criterion}: {value}")
                    continue

                # Validate range
                if not 0 <= processed_value <= 1:
                    raise ValueError(f"{criterion} value {processed_value} outside valid range [0,1]")
                
                processed_inputs[criterion] = processed_value
                self.logger.debug(f"Processed {criterion}: {processed_value}")
                
            except Exception as e:
                self.logger.error(f"Error processing {criterion}: {str(e)}")
                raise ValueError(f"Invalid input for {criterion}: {str(e)}")
        
        return processed_inputs

    def calculate_score(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate score with ML adjustments and conflict resolution"""
        self.logger.info("Starting score calculation")
        
        # Preprocess and validate inputs
        try:
            processed_inputs = self._preprocess_inputs(inputs)
        except Exception as e:
            self.logger.error(f"Input preprocessing failed: {str(e)}")
            raise

        # Verify all required criteria are present
        missing_criteria = set(self.criteria.keys()) - set(processed_inputs.keys())
        if missing_criteria:
            self.logger.error(f"Missing required criteria: {missing_criteria}")
            raise ValueError(f"Missing required criteria: {missing_criteria}")

        # Calculate scores using processed inputs
        total_score = 0
        details = {}
        impact_level = "normal"

        # Get base weights and apply ML adjustments
        original_weights = {k: v.weight for k, v in self.criteria.items()}
        adjusted_weights = {
            k: w * self.ml_adjustments[k]
            for k, w in original_weights.items()
        }
        
        # Normalize weights
        weight_sum = sum(adjusted_weights.values())
        adjusted_weights = {k: w/weight_sum for k, w in adjusted_weights.items()}

        # Process each criterion
        for criterion, score_obj in self.criteria.items():
            raw_score = processed_inputs[criterion]
            weighted_score = raw_score * adjusted_weights[criterion]
            total_score += weighted_score
            
            details[criterion] = {
                "raw_score": raw_score,
                "original_weight": original_weights[criterion],
                "ml_adjustment": self.ml_adjustments[criterion],
                "adjusted_weight": adjusted_weights[criterion],
                "weighted_score": weighted_score,
                "original_input": inputs[criterion]  # Keep original for reference
            }

        # Check impact level
        sentient_score = processed_inputs["sentient_impact"]
        if sentient_score >= self.priority_threshold:
            impact_level = "critical"
            self.logger.info("Critical impact level detected")
            
            # Apply critical impact adjustments
            adjusted_weights["sentient_impact"] += self.priority_weight_increase
            reduction_factor = (1.0 - adjusted_weights["sentient_impact"]) / (
                sum(w for k, w in adjusted_weights.items() if k != "sentient_impact")
            )
            for criterion in ["organizational_welfare", "resource_sustainability"]:
                adjusted_weights[criterion] *= reduction_factor

        final_score = total_score * 100

        # Apply penalties if needed
        if impact_level == "critical":
            org_welfare = processed_inputs["organizational_welfare"]
            sustainability = processed_inputs["resource_sustainability"]
            
            if org_welfare < 0.4 or sustainability < 0.4:
                final_score *= 0.9
                details["penalties"] = ["Critical impact with poor supporting scores"]
                self.logger.warning("Applied penalty for poor supporting scores")

        result = {
            "score": round(final_score, 2),
            "details": details,
            "impact_level": impact_level,
            "ml_confidence": self.ml_confidence,
            "ml_adjustments": self.ml_adjustments,
            "trend_patterns": dict(self.trend_patterns),
            "recommendations": self._generate_recommendations(details, impact_level)
        }

        self.logger.info(f"Calculation complete. Final score: {result['score']}")
        return result

    def _generate_recommendations(self, details: Dict[str, Any], impact_level: str) -> List[str]:
        """Generate recommendations with focus on harm minimization"""
        recommendations = []
        
        if impact_level == "critical":
            recommendations.append("CRITICAL: High sentient impact detected - immediate review required")
            recommendations.append("Implement enhanced monitoring and harm reduction measures")
        
        sentient_score = details["sentient_impact"]["raw_score"]
        if sentient_score >= self.impact_thresholds["high"]:
            recommendations.append("Establish additional safeguards for sentient welfare")
        
        org_score = details["organizational_welfare"]["raw_score"]
        if org_score < 0.6:
            recommendations.append("Review and enhance organizational welfare measures")
        
        sust_score = details["resource_sustainability"]["raw_score"]
        if sust_score < 0.6:
            recommendations.append("Develop sustainable resource management plan")
            
        return recommendations

    def _get_conflict_resolution_details(self, impact_level: str) -> Dict[str, Any]:
        """Provide details about any conflict resolution actions"""
        if not self.conflict_history:
            return {"status": "No conflicts detected"}
            
        latest_conflict = self.conflict_history[-1]
        return {
            "status": "Conflict resolved",
            "trigger": latest_conflict["trigger"],
            "weight_adjustments": {
                k: {
                    "before": latest_conflict["original_weights"][k],
                    "after": latest_conflict["adjusted_weights"][k]
                }
                for k in self.criteria.keys()
            }
        }

    def apply_feedback(self, feedback: Dict[str, Any]) -> None:
        """Apply feedback with conflict resolution"""
        print("\nApplying Feedback with Conflict Resolution:")
        
        # Track original weights for comparison
        original_weights = {k: v.weight for k, v in self.criteria.items()}
        
        # Handle conflict feedback
        if "conflicts" in feedback:
            for conflict in feedback["conflicts"]:
                if conflict.get("type") == "welfare_vs_impact":
                    self._handle_welfare_impact_conflict(conflict)
        
        # Apply general feedback adjustments
        if "adjustments" in feedback:
            for criterion, adjustment in feedback["adjustments"].items():
                if criterion not in self.criteria:
                    continue
                    
                score_obj = self.criteria[criterion]
                score_obj.feedback_history.append(adjustment)
                
                current_weight = score_obj.weight
                adjustment_factor = adjustment * self.feedback_adjustment_rate
                
                # Apply adjustment with bounds
                new_weight = max(0.1, min(0.6, current_weight + adjustment_factor))
                score_obj.weight = new_weight

        # Normalize weights
        self._normalize_weights()
        
        # Print weight changes
        print("\nWeight Adjustments:")
        for criterion in self.criteria:
            old_weight = original_weights[criterion]
            new_weight = self.criteria[criterion].weight
            print(f"{criterion}:")
            print(f"  Before: {old_weight:.3f}")
            print(f"  After:  {new_weight:.3f}")

    def _handle_welfare_impact_conflict(self, conflict: Dict[str, Any]) -> None:
        """Handle conflicts between organizational welfare and sentient impact"""
        print("\nHandling Welfare vs Impact Conflict:")
        
        impact_score = conflict.get("sentient_impact", 0)
        welfare_score = conflict.get("organizational_welfare", 0)
        severity = conflict.get("severity", "low")
        
        # Update conflict counts
        self.criteria["sentient_impact"].conflict_count += 1
        self.criteria["organizational_welfare"].conflict_count += 1
        
        # Record conflict
        self.conflict_history.append({
            "type": "welfare_vs_impact",
            "impact_score": impact_score,
            "welfare_score": welfare_score,
            "severity": severity
        })
        
        # Check if we need permanent adjustment
        if self.criteria["sentient_impact"].conflict_count >= self.conflict_threshold:
            print("Conflict threshold reached - implementing permanent adjustment")
            self.criteria["sentient_impact"].weight += self.feedback_adjustment_rate
            self.criteria["organizational_welfare"].weight -= self.feedback_adjustment_rate
        else:
            # Temporary adjustment based on severity
            adjustment = self.feedback_adjustment_rate * {
                "low": 0.5,
                "medium": 1.0,
                "high": 1.5
            }.get(severity, 1.0)
            
            if impact_score > welfare_score:
                self.criteria["sentient_impact"].weight += adjustment
                self.criteria["organizational_welfare"].weight -= adjustment
            else:
                self.criteria["organizational_welfare"].weight += adjustment
                self.criteria["sentient_impact"].weight -= adjustment

    def _normalize_weights(self) -> None:
        """Ensure all weights sum to 1"""
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
            "ml_confidence": self.ml_confidence,
            "impact_thresholds": self.impact_thresholds
        }

    def get_conflict_status(self) -> Dict[str, Any]:
        """Get detailed conflict resolution status"""
        return {
            "conflict_counts": {
                k: v.conflict_count for k, v in self.criteria.items()
            },
            "conflict_history": self.conflict_history,
            "current_weights": {
                k: v.weight for k, v in self.criteria.items()
            }
        }

    def generate_summary(self) -> str:
        """Generate a concise summary of the prism's latest analysis"""
        if not hasattr(self, 'latest_result'):
            return "No analysis has been performed yet."
            
        result = self.latest_result
        details = result["details"]
        impact_level = result["impact_level"]
        
        # Build narrative
        summary = [f"Analysis shows {impact_level} sentient impact"]
        
        # Add impact details
        sentient_score = details["sentient_impact"]["raw_score"]
        if sentient_score >= self.impact_thresholds["critical"]:
            summary.append("requiring immediate attention")
        
        # Add balance details
        org_welfare = details["organizational_welfare"]["raw_score"]
        sustainability = details["resource_sustainability"]["raw_score"]
        
        if org_welfare < 0.6 or sustainability < 0.6:
            summary.append("\nThere are concerns with supporting criteria:")
            if org_welfare < 0.6:
                summary.append("- organizational welfare needs improvement")
            if sustainability < 0.6:
                summary.append("- resource sustainability requires attention")
        
        if hasattr(self, 'ml_confidence') and self.ml_confidence > 0:
            summary.append(f"\nML analysis confidence: {self.ml_confidence:.2f}")
        
        if self.conflict_history:
            summary.append("\nConflict resolution has been applied to balance priorities")
        
        return " ".join(summary) + "."
