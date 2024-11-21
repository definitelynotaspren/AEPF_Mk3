from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import logging
from enum import Enum

@dataclass
class ScenarioTemplate:
    name: str
    description: str
    weight_adjustments: Dict[str, float]
    detection_thresholds: Dict[str, float]

class ContextEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scenario templates for ethical adjustments
        self.scenario_templates = {
            'high_risk': ScenarioTemplate(
                name="high_risk",
                description="High-risk scenario requiring careful ethical consideration",
                weight_adjustments={
                    'human_centric': 1.2,
                    'sentient_first': 1.1,
                    'ecocentric': 0.8,
                    'innovation_focused': 0.9,
                    'equity_focused': 1.1
                },
                detection_thresholds={
                    'risk_level': 0.8,
                    'impact_severity': 0.7
                }
            ),
            'resource_limited': ScenarioTemplate(
                name="resource_limited",
                description="Scenario with significant resource constraints",
                weight_adjustments={
                    'human_centric': 1.0,
                    'sentient_first': 0.9,
                    'ecocentric': 1.3,
                    'innovation_focused': 0.8,
                    'equity_focused': 1.0
                },
                detection_thresholds={
                    'resource_availability': 0.4,
                    'sustainability_impact': 0.7
                }
            ),
            'human_centered': ScenarioTemplate(
                name="human_centered",
                description="Scenario primarily affecting human wellbeing",
                weight_adjustments={
                    'human_centric': 1.3,
                    'sentient_first': 1.1,
                    'ecocentric': 0.9,
                    'innovation_focused': 1.0,
                    'equity_focused': 1.2
                },
                detection_thresholds={
                    'human_impact': 0.8,
                    'social_significance': 0.7
                }
            )
        }
        
        # Store for context entries and analysis
        self.context_store: Dict[str, Any] = {}
        self.context_history: List[Dict[str, Any]] = []
        self.current_context: Optional[Dict[str, Any]] = None

    def add_context_entry(self, entry: Any) -> None:
        """Add or update a context entry."""
        self.context_store[entry['id']] = entry
        self.context_history.append({'action': 'add', 'entry_id': entry['id']})
        self.logger.debug(f"Added context entry: {entry['id']}")

    def get_context(self, key: str) -> Optional[Any]:
        """Retrieve context by key."""
        return self.context_store.get(key)

    def remove_context(self, key: str) -> bool:
        """Remove a context entry."""
        if key in self.context_store:
            del self.context_store[key]
            self.context_history.append({'action': 'remove', 'key': key})
            self.logger.debug(f"Removed context: {key}")
            return True
        return False

    def clear_context(self) -> None:
        """Clear all context entries."""
        self.context_store.clear()
        self.context_history.append({'action': 'clear'})
        self.logger.debug("Cleared all context")

    def detect_scenario(self, parameters: Dict[str, float]) -> Tuple[Optional[str], Dict[str, float]]:
        """Detect the most relevant scenario based on parameters."""
        scenario_scores = {}
        for scenario_name, template in self.scenario_templates.items():
            score = sum(
                1 for param, threshold in template.detection_thresholds.items()
                if parameters.get(param, 0) >= threshold
            )
            scenario_scores[scenario_name] = score / len(template.detection_thresholds)

        if scenario_scores:
            best_scenario = max(scenario_scores.items(), key=lambda x: x[1])
            if best_scenario[1] >= 0.7:
                detected_scenario = self.scenario_templates[best_scenario[0]]
                return detected_scenario.name, detected_scenario.weight_adjustments
        
        return None, {key: 1.0 for key in self.scenario_templates['high_risk'].weight_adjustments}

    def analyze_context(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context and return adjustments."""
        scenario_name, weight_adjustments = self.detect_scenario(parameters)
        context = {
            'scenario': scenario_name,
            'parameters': parameters,
            'weight_adjustments': weight_adjustments,
            'timestamp': 'current'
        }
        self.current_context = context
        self.context_history.append(context)
        return {
            'detected_scenario': scenario_name,
            'weight_adjustments': weight_adjustments,
            'analysis_confidence': self._calculate_confidence(parameters)
        }

    def _calculate_confidence(self, parameters: Dict[str, float]) -> float:
        """Calculate confidence in the context analysis."""
        expected_params = {param for template in self.scenario_templates.values() for param in template.detection_thresholds}
        coverage = len(set(parameters) & expected_params) / len(expected_params)
        avg_value = np.mean([value for value in parameters.values() if isinstance(value, (int, float))])
        return round((coverage + avg_value) / 2, 2)

    def get_context_history(self) -> List[Dict[str, Any]]:
        """Return the history of context operations."""
        return self.context_history

    def get_current_context(self) -> Optional[Dict[str, Any]]:
        """Return the current context if available."""
        return self.current_context
