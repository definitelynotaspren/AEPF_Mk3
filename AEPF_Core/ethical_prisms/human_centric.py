from typing import Dict, Any, List, Optional
import logging

class HumanCentricPrism:
    def __init__(self):
        """Initialize the prism."""
        self.logger = logging.getLogger(self.__class__.__module__)

    def evaluate(self, metrics: Dict) -> float:
        """Evaluate human-centric ethical considerations."""
        try:
            # Calculate weighted average of human-centric metrics
            weights = {
                'wellbeing_score': 0.25,
                'autonomy_score': 0.15,
                'privacy_score': 0.20,
                'transparency_score': 0.15,
                'accountability_score': 0.10,
                'fairness_score': 0.10,
                'safety_score': 0.05
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    score += metrics[metric] * weight
                    total_weight += weight
            
            final_score = score / total_weight if total_weight > 0 else 0.0
            
            # Add detailed metrics to results
            self.results = {
                'score': final_score,
                'metrics': metrics,
                'weights': weights,
                'components': {
                    'wellbeing': metrics.get('wellbeing_score', 0),
                    'privacy': metrics.get('privacy_score', 0),
                    'safety': metrics.get('safety_score', 0)
                }
            }
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error in human-centric evaluation: {e}")
            return 0.0
