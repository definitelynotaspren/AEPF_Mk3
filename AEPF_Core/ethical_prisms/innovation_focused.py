from typing import Dict, Any
import logging

class InnovationFocusedPrism:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Define required fields
            required_fields = [
                "financial_risk",
                "reputational_risk",
                "technological_risk",
                "economic_benefit",
                "societal_benefit"
            ]

            # Check for missing fields
            missing_fields = [field for field in required_fields if field not in input_data]
            if missing_fields:
                raise ValueError(f"Missing required input data: {', '.join(missing_fields)}")

            # Validate input values
            for field in required_fields:
                if not isinstance(input_data[field], (int, float)):
                    raise ValueError(f"{field} must be a numeric value between 0 and 1.")
                if not 0 <= input_data[field] <= 1:
                    raise ValueError(f"{field} must be a numeric value between 0 and 1.")

            # Compute metrics
            results = {
                "metrics": {
                    "financial_risk_score": {
                        "value": 1 - input_data["financial_risk"],
                        "narrative": f"Financial risk score derived from input as {1 - input_data['financial_risk']}."
                    },
                    "reputational_risk_score": {
                        "value": 1 - input_data["reputational_risk"],
                        "narrative": f"Reputational risk score derived from input as {1 - input_data['reputational_risk']}."
                    },
                    "technological_risk_score": {
                        "value": 1 - input_data["technological_risk"],
                        "narrative": f"Technological risk score derived from input as {1 - input_data['technological_risk']}."
                    },
                    "economic_benefit_score": {
                        "value": input_data["economic_benefit"],
                        "narrative": f"Economic benefit score directly input as {input_data['economic_benefit']}."
                    },
                    "societal_benefit_score": {
                        "value": input_data["societal_benefit"],
                        "narrative": f"Societal benefit score directly input as {input_data['societal_benefit']}."
                    },
                },
                "prism": "Innovation-Focused"
            }

            return results

        except ValueError as ve:
            self.logger.error(f"Error in InnovationFocusedPrism: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error in InnovationFocusedPrism: {e}")
            return {"error": str(e)}
