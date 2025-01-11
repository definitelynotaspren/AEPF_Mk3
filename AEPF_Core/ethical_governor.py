from typing import Dict, Any
import logging
from AEPF_Core.ethical_prisms.ecocentric import EcocentricPrism
from AEPF_Core.ethical_prisms.equity_focused import EquityFocusedPrism
from AEPF_Core.ethical_prisms.human_centric import HumanCentricPrism
from AEPF_Core.ethical_prisms.innovation_focused import InnovationFocusedPrism
from AEPF_Core.ethical_prisms.sentient_first import SentientFirstPrism
from AEPF_Core.Context_manager import ContextEngine
from datetime import datetime
from pathlib import Path


class EthicalGovernor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prisms = {
            "ecocentric": EcocentricPrism(),
            "equity_focused": EquityFocusedPrism(),
            "human_centric": HumanCentricPrism(),
            "innovation_focused": InnovationFocusedPrism(),
            "sentient_first": SentientFirstPrism()
        }
        self.context_engine = ContextEngine()
        self.base_path = Path(__file__).parent.parent

    def evaluate_prism(self, prism, name, input_data, weights):
        """Helper method to evaluate a single prism with error handling."""
        try:
            if name not in input_data or not input_data[name]:
                self.logger.warning(f"No input data provided for prism {name}")
                return {"metrics": {}, "score": 0.0, "error": "No input data provided"}

            # Get the raw score from the prism
            score = prism.evaluate(input_data.get(name, {}))
            
            # Validate score
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                self.logger.warning(f"Invalid score from prism {name}: {score}")
                return {"metrics": input_data.get(name, {}), "score": 0.0}
            
            return {
                "metrics": input_data.get(name, {}),
                "score": score
            }

        except Exception as e:
            self.logger.error(f"Error evaluating prism {name}: {e}")
            return {"metrics": {}, "score": 0.0, "error": str(e)}

    def _detect_scenario(self, input_data: Dict) -> str:
        """Detect the scenario type based on input data."""
        if 'model_type' in input_data:
            if input_data['model_type'] == 'loan_default':
                return 'loan_default'
        return 'generic'

    def evaluate(self, input_data: Dict, context_parameters: Dict) -> Dict:
        """Evaluate ethical implications of AI model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.base_path / 'reports' / 'AEPF' / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Define scenario-specific weights
        weights = {
            'equity_focused': 0.35,  # Higher weight due to financial impact
            'innovation_focused': 0.15,
            'human_centric': 0.25,   # Important for loan decisions
            'ecocentric': 0.05,      # Less relevant for loan scenario
            'sentient_first': 0.20   # Moderate importance for human welfare
        }
        
        # Evaluate each prism
        prism_results = {}
        for prism_name, prism in self.prisms.items():
            try:
                if prism_name in input_data:
                    metrics = input_data[prism_name]
                    if isinstance(metrics, dict):
                        # Calculate average of all metric scores
                        scores = []
                        for metric_name, value in metrics.items():
                            if metric_name.endswith('_score'):
                                if isinstance(value, (int, float)):
                                    scores.append(value)
                                elif isinstance(value, dict) and 'value' in value:
                                    scores.append(value['value'])
                        
                        score = sum(scores) / len(scores) if scores else 0.0
                    else:
                        score = 0.0
                    
                    prism_results[prism_name] = {
                        'score': score,
                        'metrics': input_data[prism_name]
                    }
                else:
                    self.logger.warning(f"No input data provided for prism {prism_name}")
            except Exception as e:
                self.logger.error(f"Error evaluating prism {prism_name}: {e}")
        
        # Calculate final score using weighted average
        weighted_scores = []
        for prism_name, result in prism_results.items():
            weight = weights.get(prism_name, 0.0)
            if isinstance(result['score'], (int, float)):
                weighted_scores.append(result['score'] * weight)
            else:
                self.logger.warning(f"Invalid score type for prism {prism_name}")
        
        active_weights = sum(weights[k] for k in prism_results.keys() if k in weights)
        total_score = sum(weighted_scores) / active_weights if active_weights > 0 else 0.0
        
        # Generate HTML report
        report_path = report_dir / f'aepf_report_{timestamp}.html'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <html>
                    <head>
                        <title>AEPF Ethical Analysis</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            .metric {{ margin: 20px 0; padding: 10px; background: #f5f5f5; }}
                            .score {{ font-size: 24px; color: #2c3e50; }}
                            .prism {{ margin: 15px 0; padding: 15px; border: 1px solid #ddd; }}
                            .rating {{ font-size: 20px; color: #f1c40f; }}
                        </style>
                    </head>
                    <body>
                        <h1>Ethical Analysis Report</h1>
                        <div class="metric">
                            <h2>Overall Score</h2>
                            <p class="score">{total_score:.1%}</p>
                            <p class="rating">{"⭐" * int(total_score * 5)}</p>
                        </div>
                        <div class="prisms">
                            <h2>Prism Analysis</h2>
                            {''.join(f"""
                                <div class="prism">
                                    <h3>{prism}</h3>
                                    <p>Score: {data['score']:.1%}</p>
                                    <p>Weight: {weights.get(prism, 0):.0%}</p>
                                </div>
                            """ for prism, data in prism_results.items())}
                        </div>
                    </body>
                </html>
                """)
        except Exception as e:
            self.logger.error(f"Error generating AEPF report: {e}")
            raise
        
        return {
            'summary': {
                'final_score': total_score,
                'five_star_rating': total_score * 5,
                'detected_scenario': self._detect_scenario(input_data),
                'valid_prisms': len(prism_results),
                'total_prisms': len(self.prisms)
            },
            'full_report': {
                'context_analysis': context_parameters,
                'weight_adjustments': weights,
                'prism_results': prism_results
            },
            'aepf_report_path': str(report_path)
        }

