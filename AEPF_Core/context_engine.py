from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class RegionalContext:
    region: str
    factors: Dict[str, float]

class ContextEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_weights = {
            'human_centric': 0.4,
            'community_centric': 0.3,
            'sentient_first': 0.3
        }
        self.scenario_weights = {}

    def get_nearest_context(self, location_data: Dict[str, Any]) -> Optional[RegionalContext]:
        """Get the nearest regional context based on location data"""
        try:
            region = location_data.get('region', '')
            return RegionalContext(
                region=region,
                factors={
                    'privacy_emphasis': location_data.get('cultural_values', {}).get('privacy_emphasis', 0.5),
                    'collectivist_value': location_data.get('cultural_values', {}).get('collectivist_value', 0.5)
                }
            )
        except Exception as e:
            self.logger.error(f"Error getting regional context: {e}")
            return None

    def adjust_weights(self, weights: Dict[str, float], regional_context: Optional[RegionalContext]) -> Dict[str, float]:
        """Adjust weights based on regional context"""
        if not regional_context:
            return weights.copy()

        adjusted_weights = weights.copy()
        factors = regional_context.factors

        if factors.get('privacy_emphasis', 0) > 0.7:
            adjusted_weights['human_centric'] = adjusted_weights.get('human_centric', 0) * 1.2

        if factors.get('collectivist_value', 0) > 0.7:
            adjusted_weights['community_centric'] = adjusted_weights.get('community_centric', 0) * 1.15

        # Normalize weights
        total = sum(adjusted_weights.values())
        return {k: v/total for k, v in adjusted_weights.items()}

    def analyze_context(self, scenario_data: Dict[str, Any], location_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze context including regional factors"""
        regional_context = self.get_nearest_context(location_data) if location_data else None
        
        adjusted_weights = self.adjust_weights(
            self.base_weights,
            regional_context
        )
        
        return {
            'weights': adjusted_weights,
            'regional_factors': regional_context.factors if regional_context else {},
        }

    def apply_adjustments(self, scores: Dict[str, float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply contextual adjustments to scores"""
        adjusted_scores = scores.copy()
        
        # Apply any context-specific adjustments
        for key in adjusted_scores:
            if context.get('high_risk', False):
                adjusted_scores[key] *= 0.9  # Reduce scores in high-risk contexts
                
        return {
            'adjusted_scores': adjusted_scores,
            'context_factors': context
        } 