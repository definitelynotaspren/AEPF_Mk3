from typing import Dict, Any, List, Optional
import logging

class EcocentricPrism:
    """Evaluates environmental and ecological impact."""
    
    def __init__(self):
        """Initialize the prism."""
        self.logger = logging.getLogger(self.__class__.__module__)
        
        # Define required criteria
        self.required_inputs = [
            'environmental_impact',
            'biodiversity_preservation',
            'carbon_neutrality',
            'water_conservation',
            'renewable_resource_use'
        ]
    
    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the Ecocentric Prism with the provided input data.
        
        Args:
            input_data: Dictionary containing scores for evaluation
            
        Returns:
            Dict containing evaluated metrics and their narratives
            
        Raises:
            ValueError: If required inputs are missing or invalid
        """
        try:
            # Check for missing fields
            missing_fields = [
                field for field in self.required_inputs
                if field not in input_data
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required input data: {', '.join(missing_fields)}"
                )
            
            # Validate numeric values
            for field in self.required_inputs:
                value = input_data[field]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{field} must be a numeric value between 0 and 1."
                    )
                if not 0 <= float(value) <= 1:
                    raise ValueError(
                        f"Value {value} for {field} must be between 0 and 1"
                    )
            
            # Calculate metrics
            metrics = {
                "environmental_score": {
                    "value": input_data["environmental_impact"],
                    "narrative": f"Environmental impact score: {input_data['environmental_impact']}"
                },
                "biodiversity_score": {
                    "value": input_data["biodiversity_preservation"],
                    "narrative": f"Biodiversity preservation score: {input_data['biodiversity_preservation']}"
                },
                "carbon_score": {
                    "value": input_data["carbon_neutrality"],
                    "narrative": f"Carbon neutrality score: {input_data['carbon_neutrality']}"
                },
                "water_score": {
                    "value": input_data["water_conservation"],
                    "narrative": f"Water conservation score: {input_data['water_conservation']}"
                },
                "renewable_score": {
                    "value": input_data["renewable_resource_use"],
                    "narrative": f"Renewable resource utilization: {input_data['renewable_resource_use']}"
                }
            }
            
            # Return results
            return {
                "prism": "Ecocentric",
                "metrics": metrics
            }
            
        except ValueError as ve:
            self.logger.error("Error in EcocentricPrism: %s", str(ve))
            raise
            
        except Exception as e:
            self.logger.error("Unexpected error in EcocentricPrism: %s", str(e))
            return {"error": str(e)}
