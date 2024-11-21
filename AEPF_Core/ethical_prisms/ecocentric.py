from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict

@dataclass
class CriterionScore:
    raw_score: float
    weight: float
    feedback_history: List[float] = None
    priority_count: int = 0
    emphasis_level: str = "normal"
    historical_scores: List[float] = None
    
    def __post_init__(self):
        self.feedback_history = self.feedback_history or []
        self.historical_scores = self.historical_scores or []

class EcocentricPrism:
    def __init__(self):
        self.name = "ecocentric"
        self.criteria = {
            "environmental_impact": CriterionScore(0.0, 0.40),
            "biodiversity_preservation": CriterionScore(0.0, 0.30),
            "carbon_neutrality": CriterionScore(0.0, 0.30)
        }
        self.feedback_adjustment_rate = 0.05
        self.ml_confidence = 0.0
        self.historical_scores = []
        
        # ML-specific attributes
        self.ml_adjustments = {k: 1.0 for k in self.criteria.keys()}
        self.trend_patterns = defaultdict(int)
        self.min_historical_data = 5
        self.trend_threshold = 0.1
        self.max_ml_adjustment = 1.3
        
        # Scoring thresholds
        self.priority_threshold = 0.90
        self.priority_weight_increase = 0.10
        self.impact_thresholds = {
            "critical": 0.85,
            "significant": 0.70,
            "moderate": 0.50
        }

    def train_ml_model(self, historical_data: List[Dict[str, Any]]) -> None:
        """Train ML model on historical environmental data"""
        if len(historical_data) < self.min_historical_data:
            return

        print("\nTraining ML Model on Environmental Data:")
        
        # Track historical scores for each criterion
        for criterion in self.criteria:
            scores = [entry.get(criterion, 0) for entry in historical_data]
            self.criteria[criterion].historical_scores.extend(scores)
        
        # Calculate ML confidence based on data consistency
        impact_scores = self.criteria["environmental_impact"].historical_scores[-self.min_historical_data:]
        if len(impact_scores) >= 2:
            std_dev = np.std(impact_scores)
            self.ml_confidence = max(0, 1 - (std_dev / 50))
            print(f"ML Confidence: {self.ml_confidence:.2f}")

        # Analyze environmental patterns
        self._analyze_environmental_patterns(historical_data)
        
        # Update ML adjustments based on patterns
        self._update_ml_adjustments()
        
        print("\nML Analysis Results:")
        print(f"Detected Patterns: {dict(self.trend_patterns)}")
        print("ML Adjustments:", {k: f"{v:.2f}" for k, v in self.ml_adjustments.items()})

    def _analyze_environmental_patterns(self, historical_data: List[Dict[str, Any]]) -> None:
        """Analyze patterns in environmental scores"""
        recent_data = historical_data[-self.min_historical_data:]
        
        # Analyze trends for each criterion
        for criterion in self.criteria:
            scores = [d.get(criterion, 0) for d in recent_data]
            trend = self._detect_trend(scores)
            if trend != "stable":
                self.trend_patterns[f"{criterion}_{trend}"] += 1
        
        # Analyze critical impact patterns
        for data in recent_data:
            # Check for high environmental impact with poor biodiversity
            if (data.get("environmental_impact", 0) > self.impact_thresholds["critical"] and
                data.get("biodiversity_preservation", 0) < 0.6):
                self.trend_patterns["critical_biodiversity_impact"] += 1
            
            # Check for increasing carbon concerns
            if data.get("carbon_neutrality", 0) < 0.5:
                self.trend_patterns["carbon_concern"] += 1

    def _detect_trend(self, scores: List[float]) -> str:
        """Detect trend in environmental scores"""
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
        """Update ML adjustments based on environmental patterns"""
        base_adjustment = 0.05
        
        # Reset adjustments
        self.ml_adjustments = {k: 1.0 for k in self.criteria.keys()}
        
        # Apply pattern-based adjustments
        if self.trend_patterns["critical_biodiversity_impact"] > 1:
            # Increase biodiversity weight when critical impacts are common
            self.ml_adjustments["biodiversity_preservation"] += base_adjustment * 2
            
        if self.trend_patterns["carbon_concern"] > 1:
            # Increase carbon neutrality weight for persistent concerns
            self.ml_adjustments["carbon_neutrality"] += base_adjustment * 1.5
            
        # Adjust for declining environmental scores
        for criterion in self.criteria:
            if self.trend_patterns.get(f"{criterion}_decreasing", 0) > 1:
                self.ml_adjustments[criterion] += base_adjustment
            
        # Ensure adjustments stay within bounds
        for criterion in self.ml_adjustments:
            self.ml_adjustments[criterion] = max(0.8, min(self.max_ml_adjustment, 
                                                        self.ml_adjustments[criterion]))

    def calculate_score(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """Calculate environmental score with ML adjustments"""
        if not all(k in inputs for k in self.criteria.keys()):
            raise ValueError("Missing required environmental criteria in inputs")

        total_score = 0
        details = {}
        impact_level = "normal"
        priority_criteria = []

        # Get base weights and apply ML adjustments
        original_weights = {k: v.weight for k, v in self.criteria.items()}
        adjusted_weights = {
            k: w * self.ml_adjustments[k]
            for k, w in original_weights.items()
        }
        
        # Normalize adjusted weights
        weight_sum = sum(adjusted_weights.values())
        adjusted_weights = {k: w/weight_sum for k, w in adjusted_weights.items()}

        # Calculate scores with adjusted weights
        for criterion, score_obj in self.criteria.items():
            raw_score = inputs[criterion]
            if not 0 <= raw_score <= 1:
                raise ValueError(f"{criterion} score must be between 0 and 1")
            
            # Check for priority status
            if raw_score >= self.priority_threshold:
                priority_criteria.append(criterion)
            
            # Determine impact level
            if criterion == "environmental_impact":
                if raw_score >= self.impact_thresholds["critical"]:
                    impact_level = "critical"
                elif raw_score >= self.impact_thresholds["significant"]:
                    impact_level = "significant"
            
            weighted_score = raw_score * adjusted_weights[criterion]
            total_score += weighted_score
            
            details[criterion] = {
                "raw_score": raw_score,
                "original_weight": original_weights[criterion],
                "ml_adjustment": self.ml_adjustments[criterion],
                "adjusted_weight": adjusted_weights[criterion],
                "weighted_score": weighted_score,
                "priority_status": criterion in priority_criteria
            }

        final_score = total_score * 100

        # Apply impact-based adjustments
        if impact_level == "critical":
            biodiversity_score = inputs["biodiversity_preservation"]
            carbon_score = inputs["carbon_neutrality"]
            
            if biodiversity_score < 0.6 or carbon_score < 0.6:
                final_score *= 0.9
                details["penalties"] = ["Critical impact with poor biodiversity/carbon scores"]

        return {
            "score": round(final_score, 2),
            "details": details,
            "impact_level": impact_level,
            "priority_criteria": priority_criteria,
            "ml_confidence": self.ml_confidence,
            "ml_adjustments": self.ml_adjustments,
            "trend_patterns": dict(self.trend_patterns),
            "recommendations": self._generate_recommendations(details, impact_level),
            "environmental_metrics": self._calculate_environmental_metrics(details)
        }

    def _generate_recommendations(self, details: Dict[str, Any], impact_level: str) -> List[str]:
        """Generate environmental recommendations"""
        recommendations = []
        
        if impact_level == "critical":
            recommendations.append("CRITICAL: Severe environmental impact detected - immediate action required")
        
        for criterion, detail in details.items():
            if detail["priority_status"]:
                recommendations.append(f"High-priority {criterion} - maintain excellent performance")
            elif detail["raw_score"] < 0.6:
                if criterion == "environmental_impact":
                    recommendations.append("Implement stronger environmental protection measures")
                elif criterion == "biodiversity_preservation":
                    recommendations.append("Enhance biodiversity preservation strategies")
                elif criterion == "carbon_neutrality":
                    recommendations.append("Strengthen carbon reduction initiatives")
            
        return recommendations

    def _calculate_environmental_metrics(self, details: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional environmental metrics"""
        return {
            "environmental_risk": round(
                (1 - details["environmental_impact"]["raw_score"]) * 100, 2
            ),
            "biodiversity_health": round(
                details["biodiversity_preservation"]["raw_score"] * 100, 2
            ),
            "carbon_efficiency": round(
                details["carbon_neutrality"]["raw_score"] * 100, 2
            ),
            "sustainability_index": round(
                (details["environmental_impact"]["raw_score"] * 0.4 +
                 details["biodiversity_preservation"]["raw_score"] * 0.3 +
                 details["carbon_neutrality"]["raw_score"] * 0.3) * 100, 2
            )
        }

    def apply_feedback(self, feedback: Dict[str, Any]) -> None:
        """Apply feedback to adjust environmental priorities"""
        print("\nApplying Environmental Feedback:")
        
        # Track original weights for comparison
        original_weights = {k: v.weight for k, v in self.criteria.items()}
        
        # Process emphasis feedback
        if "emphasis" in feedback:
            self._handle_emphasis_feedback(feedback["emphasis"])
        
        # Process weight adjustments
        if "adjustments" in feedback:
            for criterion, adjustment in feedback["adjustments"].items():
                if criterion not in self.criteria:
                    continue
                    
                score_obj = self.criteria[criterion]
                score_obj.feedback_history.append(adjustment)
                
                # Calculate base adjustment
                base_adjustment = adjustment * self.feedback_adjustment_rate
                
                # Apply emphasis multiplier
                emphasis_multiplier = {
                    "normal": 1.0,
                    "important": 1.5,
                    "critical": 2.0
                }.get(score_obj.emphasis_level, 1.0)
                
                total_adjustment = base_adjustment * emphasis_multiplier
                
                # Apply adjustment with bounds
                current_weight = score_obj.weight
                new_weight = max(self.min_weight, 
                               min(self.max_weight, 
                                   current_weight + total_adjustment))
                score_obj.weight = new_weight
        
        # Normalize weights
        self._normalize_weights()
        
        # Print weight changes
        print("\nWeight Adjustments:")
        for criterion in self.criteria:
            old_weight = original_weights[criterion]
            new_weight = self.criteria[criterion].weight
            emphasis = self.criteria[criterion].emphasis_level
            print(f"{criterion}:")
            print(f"  Before: {old_weight:.3f}")
            print(f"  After:  {new_weight:.3f}")
            print(f"  Emphasis Level: {emphasis}")

    def _handle_emphasis_feedback(self, emphasis_feedback: Dict[str, str]) -> None:
        """Process emphasis feedback for environmental criteria"""
        for criterion, emphasis in emphasis_feedback.items():
            if criterion not in self.criteria:
                continue
                
            score_obj = self.criteria[criterion]
            
            if emphasis == "critical":
                score_obj.priority_count += 2
            elif emphasis == "important":
                score_obj.priority_count += 1
            
            # Update emphasis level based on priority count
            if score_obj.priority_count >= self.emphasis_thresholds["critical"]:
                score_obj.emphasis_level = "critical"
            elif score_obj.priority_count >= self.emphasis_thresholds["important"]:
                score_obj.emphasis_level = "important"
            else:
                score_obj.emphasis_level = "normal"

    def _normalize_weights(self) -> None:
        """Normalize weights while respecting emphasis levels"""
        # Calculate total weight
        total_weight = sum(score_obj.weight for score_obj in self.criteria.values())
        
        if total_weight == 0:
            return
            
        # First pass: normalize all weights
        for score_obj in self.criteria.values():
            score_obj.weight /= total_weight
        
        # Second pass: ensure minimum weights
        for score_obj in self.criteria.values():
            if score_obj.weight < self.min_weight:
                score_obj.weight = self.min_weight
        
        # Final pass: normalize again
        total_weight = sum(score_obj.weight for score_obj in self.criteria.values())
        for score_obj in self.criteria.values():
            score_obj.weight /= total_weight

    def get_status(self) -> Dict[str, Any]:
        """Return current status of the prism"""
        return {
            "name": self.name,
            "weights": {k: v.weight for k, v in self.criteria.items()},
            "emphasis_levels": {k: v.emphasis_level for k, v in self.criteria.items()},
            "priority_counts": {k: v.priority_count for k, v in self.criteria.items()},
            "feedback_history": {k: v.feedback_history for k, v in self.criteria.items()},
            "ml_confidence": self.ml_confidence,
            "impact_thresholds": self.impact_thresholds
        }

    def generate_summary(self) -> str:
        """Generate a concise summary of the prism's latest analysis"""
        if not hasattr(self, 'latest_result'):
            return "No analysis has been performed yet."
            
        result = self.latest_result
        details = result["details"]
        impact_level = result["impact_level"]
        
        # Build narrative
        summary = [f"Environmental analysis shows {impact_level} impact level"]
        
        # Add priority criteria details
        if result.get("priority_criteria"):
            summary.append(f"with high priority in: {', '.join(result['priority_criteria'])}")
        
        # Add specific metrics
        metrics = result["environmental_metrics"]
        summary.append(f"\nKey metrics:")
        summary.append(f"- Environmental Risk: {metrics['environmental_risk']}%")
        summary.append(f"- Biodiversity Health: {metrics['biodiversity_health']}%")
        summary.append(f"- Carbon Efficiency: {metrics['carbon_efficiency']}%")
        
        if hasattr(self, 'ml_confidence') and self.ml_confidence > 0:
            summary.append(f"\nML analysis confidence: {self.ml_confidence:.2f}")
        
        if self.trend_patterns:
            summary.append("\nDetected environmental patterns:")
            for pattern, count in self.trend_patterns.items():
                if count > 0:
                    summary.append(f"- {pattern.replace('_', ' ').title()}")
        
        return "\n".join(summary)
