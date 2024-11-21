from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RegionalContext:
    """Represents the ethical and cultural context of a geographic region"""
    region: str
    factors: Dict[str, float]
    cultural_values: Dict[str, float]
    legal_framework: Dict[str, Any]

class LocationContextManager:
    def __init__(self):
        # Default regional contexts
        self.regional_contexts = {
            'EU': RegionalContext(
                region='EU',
                factors={
                    'privacy_emphasis': 0.9,
                    'collectivist_value': 0.4,
                    'innovation_focus': 0.7
                },
                cultural_values={
                    'individual_rights': 0.8,
                    'data_protection': 0.9,
                    'environmental_concern': 0.8
                },
                legal_framework={
                    'gdpr_compliant': True,
                    'ai_regulations': 'strict',
                    'privacy_laws': 'comprehensive'
                }
            ),
            'Asia': RegionalContext(
                region='Asia',
                factors={
                    'privacy_emphasis': 0.6,
                    'collectivist_value': 0.8,
                    'innovation_focus': 0.8
                },
                cultural_values={
                    'individual_rights': 0.6,
                    'data_protection': 0.7,
                    'environmental_concern': 0.7
                },
                legal_framework={
                    'gdpr_compliant': False,
                    'ai_regulations': 'moderate',
                    'privacy_laws': 'developing'
                }
            )
        }

    def get_context(self, region: str) -> Optional[RegionalContext]:
        """Get the context for a specific region"""
        return self.regional_contexts.get(region)

    def get_nearest_context(self, location_data: Dict[str, Any]) -> Optional[RegionalContext]:
        """Get the nearest matching context based on location data"""
        region = location_data.get('region')
        if region in self.regional_contexts:
            return self.regional_contexts[region]
        
        # If exact match not found, return closest match or default
        # Currently returns None, but could be extended with proximity matching
        return None

    def adjust_weights(self, weights: Dict[str, float], context: Optional[RegionalContext]) -> Dict[str, float]:
        """Adjust weights based on regional context"""
        if not context:
            return weights

        adjusted_weights = weights.copy()
        
        # Apply regional adjustments
        if context.factors.get('privacy_emphasis', 0) > 0.7:
            # Increase human-centric weight in high-privacy regions
            if 'human_centric' in adjusted_weights:
                adjusted_weights['human_centric'] *= 1.2

        if context.factors.get('collectivist_value', 0) > 0.7:
            # Increase community-centric weight in collectivist regions
            if 'community_centric' in adjusted_weights:
                adjusted_weights['community_centric'] *= 1.15
            if 'sentient_first' in adjusted_weights:
                adjusted_weights['sentient_first'] *= 1.1

        # Normalize weights
        total = sum(adjusted_weights.values())
        return {k: v/total for k, v in adjusted_weights.items()}

    def get_legal_requirements(self, region: str) -> Dict[str, Any]:
        """Get legal requirements for a region"""
        context = self.get_context(region)
        if context:
            return context.legal_framework
        return {}

    def get_cultural_values(self, region: str) -> Dict[str, float]:
        """Get cultural values for a region"""
        context = self.get_context(region)
        if context:
            return context.cultural_values
        return {}