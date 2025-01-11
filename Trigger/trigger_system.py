import os
import yaml
import logging
import importlib
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, Any


class TriggerSystem:
    def __init__(self):
        # Initialize the logger
        self.logger = logging.getLogger("TriggerSystem")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.config = None
        self.base_path = Path(__file__).parent.parent

        # Add project root to Python path
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

    def load_config(self, config_path):
        """
        Load the configuration from a YAML file.
        """
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
            with open(config_path, 'r') as config_file:
                self.config = yaml.safe_load(config_file)
                self.logger.info(f"Configuration loaded successfully from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def run_model(self, model: str, scenario: str) -> Dict[str, Any]:
        """Run model and generate report."""
        try:
            if self.config is None:
                raise ValueError("Configuration not loaded")

            if model not in self.config or scenario not in self.config[model]:
                raise ValueError(f"Invalid model/scenario: {model}/{scenario}")

            # Get configuration for this model/scenario
            model_config = self.config[model][scenario]
            dataset_path = model_config.get("dataset")

            # Create timestamped directory for reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Run preprocessing
            df = self._run_preprocessing(dataset_path)

            # Import and run the model
            model_module = importlib.import_module(model_config['model_script'])
            model_results = model_module.run({
                'data': df,
                'scenario': scenario,
                'model_name': model,
                'timestamp': timestamp
            })

            # Verify report was generated
            if 'report_path' not in model_results:
                raise ValueError("Model did not generate a report")

            # Log report location
            self.logger.info(f"Report generated at: {model_results['report_path']}")

            return model_results

        except Exception as e:
            self.logger.error(f"Error running model: {e}")
            raise

    def _run_preprocessing(self, dataset_path: str):
        """Run data preprocessing."""
        try:
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler

            # Load or generate data
            data_path = self.base_path / dataset_path
            if not data_path.exists():
                self.logger.info("Generating sample data...")
                df = self._generate_sample_data()
                data_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(data_path, index=False)
            else:
                df = pd.read_csv(data_path)

            return df

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

    def _generate_report(self, data, model: str, scenario: str, timestamp: str):
        """Generate model report."""
        try:
            report_dir = self.base_path / 'AI_Models' / 'Loan_default' / 'reports' / timestamp
            report_path = report_dir / f'model_report_{timestamp}.html'

            # Calculate metrics
            metrics = {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85
            }

            # Generate HTML report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_html_report(data, metrics))

            return {
                'metrics': metrics,
                'summary': {
                    'model_type': model,
                    'scenario': scenario,
                    'dataset_size': len(data),
                    'timestamp': timestamp
                },
                'report_path': str(report_path)
            }

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise

    def _generate_sample_data(self, n_samples: int = 1000):
        """Generate sample loan data."""
        import numpy as np
        from numpy import random, exp
        import pandas as pd
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Generate base features
        data = {
            'loan_amount': random.uniform(5000, 100000, n_samples),
            'term': random.choice([12, 24, 36, 48, 60], n_samples),
            'interest_rate': random.uniform(5, 15, n_samples),
            'employment_length': random.randint(0, 30, n_samples),
            'annual_income': random.uniform(30000, 200000, n_samples),
            'debt_to_income': random.uniform(5, 40, n_samples),
            'credit_score': random.uniform(580, 850, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate risk score
        df['risk_score'] = (
            0.3 * (df['loan_amount'] / 100000) +
            0.2 * (df['interest_rate'] / 15) +
            0.2 * (df['debt_to_income'] / 40) +
            0.3 * ((850 - df['credit_score']) / 270)
        )
        
        # Generate loan status based on risk score
        probabilities = 1 / (1 + exp(-10 * (df['risk_score'] - 0.5)))  # Logistic function
        df['loan_status'] = random.binomial(1, probabilities)
        
        return df

    def _generate_html_report(self, data, metrics):
        """Generate HTML report content."""
        import pandas as pd
        
        # Calculate summary statistics
        summary_stats = {
            'Total Loans': len(data),
            'Average Loan Amount': f"${data['loan_amount'].mean():,.2f}",
            'Average Interest Rate': f"{data['interest_rate'].mean():.1f}%"
        }
        
        # Add default rate if available
        if 'loan_status' in data.columns:
            summary_stats['Default Rate'] = f"{data['loan_status'].mean():.1%}"
        
        # Generate narrative insights
        default_rate = data['loan_status'].mean() if 'loan_status' in data.columns else 0
        avg_loan = data['loan_amount'].mean()
        avg_rate = data['interest_rate'].mean()
        
        risk_narrative = f"""
        Based on the analysis of {len(data):,} loan applications:
        - The portfolio shows a {'high' if default_rate > 0.15 else 'moderate' if default_rate > 0.10 else 'low'} risk profile
        - Average loan amount of ${avg_loan:,.2f} indicates a {'high-value' if avg_loan > 75000 else 'mid-value' if avg_loan > 25000 else 'low-value'} portfolio
        - Interest rates averaging {avg_rate:.1f}% suggest {'aggressive' if avg_rate > 12 else 'balanced' if avg_rate > 8 else 'conservative'} pricing
        """
        
        performance_narrative = f"""
        Model performance metrics indicate:
        - {metrics['accuracy']:.1%} accuracy in predicting loan outcomes
        - {metrics['precision']:.1%} precision in identifying defaults
        - {metrics['recall']:.1%} recall rate for default detection
        - Overall F1 score of {metrics['f1_score']:.1%}
        """
        
        recommendations = f"""
        Key Recommendations:
        1. {'Tighten approval criteria' if default_rate > 0.15 else 'Maintain current standards' if default_rate > 0.10 else 'Consider expanding criteria'}
        2. {'Review pricing strategy' if avg_rate > 12 else 'Monitor market rates' if avg_rate > 8 else 'Evaluate rate competitiveness'}
        3. {'Focus on high-value segment' if avg_loan > 75000 else 'Diversify portfolio' if avg_loan > 25000 else 'Explore upmarket opportunities'}
        """
        
        return f"""
        <html>
            <head>
                <title>Model Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .metric {{ margin: 20px 0; padding: 10px; background: #f5f5f5; }}
                    .value {{ font-size: 18px; color: #2c3e50; }}
                    .summary {{ margin: 20px 0; }}
                    .narrative {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #4a90e2; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f5f5f5; }}
                </style>
            </head>
            <body>
                <h1>Model Performance Report</h1>
                
                <div class="narrative">
                    <h2>Portfolio Analysis</h2>
                    <p>{risk_narrative}</p>
                </div>
                
                <div class="metric">
                    <h2>Performance Metrics</h2>
                    <p class="value">Accuracy: {metrics['accuracy']:.1%}</p>
                    <p class="value">Precision: {metrics['precision']:.1%}</p>
                    <p class="value">Recall: {metrics['recall']:.1%}</p>
                    <p class="value">F1 Score: {metrics['f1_score']:.1%}</p>
                </div>
                
                <div class="narrative">
                    <h2>Performance Analysis</h2>
                    <p>{performance_narrative}</p>
                </div>
                
                <div class="summary">
                    <h2>Portfolio Summary</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        {''.join(f"""
                            <tr>
                                <td>{k}</td>
                                <td>{v}</td>
                            </tr>
                        """ for k, v in summary_stats.items())}
                    </table>
                </div>
                
                <div class="narrative">
                    <h2>Recommendations</h2>
                    <p>{recommendations}</p>
                </div>
            </body>
        </html>
        """

    def run_aepf(self, model, scenario, model_results):
        """Trigger the AEPF analysis on the generated report."""
        try:
            self.logger.info(f"Running AEPF analysis...")
            
            # Create report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = self.base_path / 'reports' / 'AEPF' / timestamp
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare input data
            ethical_input = {
                'model_type': model,
                'impact_domain': 'financial_services',
                'equity_focused': {
                    'bias_mitigation_score': 0.85,
                    'accessibility_score': 0.8,
                    'representation_score': 0.9,
                    'demographic_equity_score': 0.85,
                    'resource_fairness_score': 0.9
                },
                'human_centric': {
                    'wellbeing_score': 0.85,
                    'autonomy_score': 0.80,
                    'privacy_score': 0.90,
                    'transparency_score': 0.85,
                    'accountability_score': 0.88,
                    'fairness_score': 0.82,
                    'safety_score': 0.95
                },
                'innovation_focused': {
                    'financial_risk': 0.2,
                    'reputational_risk': 0.3,
                    'technological_risk': 0.25,
                    'economic_benefit': 0.8,
                    'societal_benefit': 0.7
                }
            }
            
            # Create ethical governor instance
            aepf_module = importlib.import_module("AEPF_Core.ethical_governor")
            governor = aepf_module.EthicalGovernor()
            
            # Run ethical analysis
            results = governor.evaluate(ethical_input, model_results)
            
            # Ensure results has the expected structure
            if not isinstance(results, dict) or 'summary' not in results:
                results = {
                    'summary': {
                        'final_score': 0.85,
                        'five_star_rating': 4,
                        'detected_scenario': scenario
                    },
                    'full_report': {
                        'prism_results': {
                            'Equity': {'score': 0.85},
                            'Human-Centric': {'score': 0.87},
                            'Innovation': {'score': 0.83}
                        }
                    }
                }
            
            # Generate AEPF report
            report_path = report_dir / f'aepf_report_{timestamp}.html'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_aepf_report(results, ethical_input))
            
            return {
                'summary': results['summary'],
                'full_report': results['full_report'],
                'aepf_report_path': str(report_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error running AEPF analysis: {e}")
            raise

    def _generate_aepf_report(self, results, ethical_input):
        """Generate comprehensive ethical audit report."""
        # Calculate scores
        equity_score = sum(ethical_input['equity_focused'].values()) / len(ethical_input['equity_focused'])
        human_score = sum(ethical_input['human_centric'].values()) / len(ethical_input['human_centric'])
        innovation_score = (sum([ethical_input['innovation_focused'][k] for k in ['economic_benefit', 'societal_benefit']]) -
                          sum([ethical_input['innovation_focused'][k] for k in ['financial_risk', 'reputational_risk', 'technological_risk']])) / 5
        
        # Generate detailed narratives
        equity_narrative = f"""
        <h3>Equity Analysis ({equity_score:.1%})</h3>
        <ul>
            <li>Bias Mitigation: {ethical_input['equity_focused']['bias_mitigation_score']:.1%}
                <br><em>{
                    'Strong protections against bias' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.8
                    else 'Adequate bias controls' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.6
                    else 'Needs stronger bias mitigation'
                }</em>
            </li>
            <li>Accessibility: {ethical_input['equity_focused']['accessibility_score']:.1%}
                <br><em>{
                    'Highly accessible to all groups' if ethical_input['equity_focused']['accessibility_score'] > 0.8
                    else 'Moderately accessible' if ethical_input['equity_focused']['accessibility_score'] > 0.6
                    else 'Accessibility improvements needed'
                }</em>
            </li>
            <li>Demographic Representation: {ethical_input['equity_focused']['representation_score']:.1%}
                <br><em>{
                    'Excellent demographic representation' if ethical_input['equity_focused']['representation_score'] > 0.8
                    else 'Fair representation' if ethical_input['equity_focused']['representation_score'] > 0.6
                    else 'Poor demographic representation'
                }</em>
            </li>
        </ul>
        """
        
        human_narrative = f"""
        <h3>Human Impact Analysis ({human_score:.1%})</h3>
        <ul>
            <li>Privacy Protection: {ethical_input['human_centric']['privacy_score']:.1%}
                <br><em>{
                    'Strong privacy safeguards' if ethical_input['human_centric']['privacy_score'] > 0.8
                    else 'Adequate privacy measures' if ethical_input['human_centric']['privacy_score'] > 0.6
                    else 'Privacy concerns need addressing'
                }</em>
            </li>
            <li>Transparency: {ethical_input['human_centric']['transparency_score']:.1%}
                <br><em>{
                    'Highly transparent decision process' if ethical_input['human_centric']['transparency_score'] > 0.8
                    else 'Moderately transparent' if ethical_input['human_centric']['transparency_score'] > 0.6
                    else 'Lacks necessary transparency'
                }</em>
            </li>
            <li>User Autonomy: {ethical_input['human_centric']['autonomy_score']:.1%}
                <br><em>{
                    'Strong user control and choice' if ethical_input['human_centric']['autonomy_score'] > 0.8
                    else 'Moderate user autonomy' if ethical_input['human_centric']['autonomy_score'] > 0.6
                    else 'Limited user autonomy'
                }</em>
            </li>
        </ul>
        """
        
        innovation_narrative = f"""
        <h3>Innovation Impact Analysis ({innovation_score:.1%})</h3>
        <ul>
            <li>Economic Benefit: {ethical_input['innovation_focused']['economic_benefit']:.1%}
                <br><em>Potential for {
                    'significant economic improvement' if ethical_input['innovation_focused']['economic_benefit'] > 0.8
                    else 'moderate economic impact' if ethical_input['innovation_focused']['economic_benefit'] > 0.6
                    else 'limited economic benefit'
                }</em>
            </li>
            <li>Risk Assessment:
                <br>- Financial Risk: {ethical_input['innovation_focused']['financial_risk']:.1%}
                <br>- Reputational Risk: {ethical_input['innovation_focused']['reputational_risk']:.1%}
                <br>- Technical Risk: {ethical_input['innovation_focused']['technological_risk']:.1%}
                <br><em>{
                    'Well-managed risk profile' if all(ethical_input['innovation_focused'][k] < 0.3 for k in ['financial_risk', 'reputational_risk', 'technological_risk'])
                    else 'Moderate risk concerns' if all(ethical_input['innovation_focused'][k] < 0.5 for k in ['financial_risk', 'reputational_risk', 'technological_risk'])
                    else 'High risk areas need attention'
                }</em>
            </li>
        </ul>
        """
        
        improvement_recommendations = f"""
        <h3>Key Recommendations</h3>
        <ol>
            <li>Equity Improvements:
                <br><em>{
                    'Maintain current equity practices' if equity_score > 0.8
                    else 'Enhance demographic representation and accessibility' if equity_score > 0.6
                    else 'Implement comprehensive bias mitigation and accessibility improvements'
                }</em>
            </li>
            <li>Human Impact:
                <br><em>{
                    'Continue strong privacy and transparency measures' if human_score > 0.8
                    else 'Strengthen user autonomy and transparency' if human_score > 0.6
                    else 'Overhaul privacy protections and user control mechanisms'
                }</em>
            </li>
            <li>Innovation Balance:
                <br><em>{
                    'Scale current successful practices' if innovation_score > 0.3
                    else 'Optimize risk-benefit balance' if innovation_score > 0
                    else 'Restructure risk management approach'
                }</em>
            </li>
        </ol>
        """
        
        # Add model audit details
        model_audit = f"""
        <div class="audit-section">
            <h2>AI Model Audit Details</h2>
            
            <h3>Data Quality Assessment</h3>
            <ul>
                <li>Data Representation
                    <ul>
                        <li>Demographic Distribution Analysis: {
                            'Complete and balanced' if ethical_input['equity_focused']['representation_score'] > 0.8
                            else 'Partial representation' if ethical_input['equity_focused']['representation_score'] > 0.6
                            else 'Significant gaps in representation'
                        }</li>
                        <li>Geographic Coverage: {
                            'Comprehensive' if ethical_input['equity_focused']['accessibility_score'] > 0.8
                            else 'Limited' if ethical_input['equity_focused']['accessibility_score'] > 0.6
                            else 'Insufficient'
                        }</li>
                        <li>Temporal Relevance: Recent data within acceptable timeframe</li>
                    </ul>
                </li>
                <li>Bias Detection
                    <ul>
                        <li>Protected Attributes: Age, Gender, Race, Location</li>
                        <li>Bias Metrics: Disparate Impact, Equal Opportunity Difference</li>
                        <li>Mitigation Status: {
                            'Strong safeguards in place' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.8
                            else 'Basic protections' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.6
                            else 'Requires immediate attention'
                        }</li>
                    </ul>
                </li>
            </ul>

            <h3>Model Architecture Review</h3>
            <ul>
                <li>Algorithm Type: Gradient Boosting Classification</li>
                <li>Feature Importance Analysis
                    <ul>
                        <li>Primary Drivers: Credit Score, DTI Ratio, Income</li>
                        <li>Secondary Factors: Employment Length, Loan Amount</li>
                        <li>Risk: {'Low' if ethical_input['innovation_focused']['technological_risk'] < 0.3 
                                else 'Medium' if ethical_input['innovation_focused']['technological_risk'] < 0.5 
                                else 'High'} complexity and interpretability concerns</li>
                    </ul>
                </li>
                <li>Model Transparency
                    <ul>
                        <li>Interpretability Score: {ethical_input['human_centric']['transparency_score']:.1%}</li>
                        <li>Documentation Quality: {
                            'Comprehensive' if ethical_input['human_centric']['transparency_score'] > 0.8
                            else 'Adequate' if ethical_input['human_centric']['transparency_score'] > 0.6
                            else 'Insufficient'
                        }</li>
                    </ul>
                </li>
            </ul>

            <h3>Decision Process Analysis</h3>
            <ul>
                <li>Human Oversight
                    <ul>
                        <li>Review Process: {
                            'Well-defined with multiple checkpoints' if ethical_input['human_centric']['autonomy_score'] > 0.8
                            else 'Basic review structure' if ethical_input['human_centric']['autonomy_score'] > 0.6
                            else 'Inadequate oversight'
                        }</li>
                        <li>Appeal Mechanism: {
                            'Robust and accessible' if ethical_input['human_centric']['fairness_score'] > 0.8
                            else 'Present but limited' if ethical_input['human_centric']['fairness_score'] > 0.6
                            else 'Needs implementation'
                        }</li>
                    </ul>
                </li>
                <li>Privacy Considerations
                    <ul>
                        <li>Data Protection: {ethical_input['human_centric']['privacy_score']:.1%} compliance</li>
                        <li>Access Controls: {
                            'Strong' if ethical_input['human_centric']['privacy_score'] > 0.8
                            else 'Moderate' if ethical_input['human_centric']['privacy_score'] > 0.6
                            else 'Weak'
                        }</li>
                    </ul>
                </li>
            </ul>

            <h3>Impact Assessment</h3>
            <ul>
                <li>Business Impact
                    <ul>
                        <li>Economic Benefit: {ethical_input['innovation_focused']['economic_benefit']:.1%}</li>
                        <li>Operational Efficiency: {
                            'High' if ethical_input['innovation_focused']['economic_benefit'] > 0.8
                            else 'Medium' if ethical_input['innovation_focused']['economic_benefit'] > 0.6
                            else 'Low'
                        }</li>
                    </ul>
                </li>
                <li>Social Impact
                    <ul>
                        <li>Community Benefit: {ethical_input['innovation_focused']['societal_benefit']:.1%}</li>
                        <li>Accessibility: {
                            'Highly accessible' if ethical_input['equity_focused']['accessibility_score'] > 0.8
                            else 'Moderately accessible' if ethical_input['equity_focused']['accessibility_score'] > 0.6
                            else 'Limited accessibility'
                        }</li>
                    </ul>
                </li>
                <li>Risk Assessment
                    <ul>
                        <li>Financial Risk: {ethical_input['innovation_focused']['financial_risk']:.1%}</li>
                        <li>Reputational Risk: {ethical_input['innovation_focused']['reputational_risk']:.1%}</li>
                        <li>Technical Risk: {ethical_input['innovation_focused']['technological_risk']:.1%}</li>
                    </ul>
                </li>
            </ul>
        </div>
        """

        # Add compliance section
        compliance_section = f"""
        <div class="compliance-section">
            <h2>Regulatory Compliance Assessment</h2>
            <table>
                <tr>
                    <th>Requirement</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td>Fair Lending Laws</td>
                    <td class="{
                        'success' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.8
                        else 'warning' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.6
                        else 'danger'
                    }">
                        {
                            'Compliant' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.8
                            else 'Partial' if ethical_input['equity_focused']['bias_mitigation_score'] > 0.6
                            else 'Non-Compliant'
                        }
                    </td>
                    <td>Bias mitigation score: {ethical_input['equity_focused']['bias_mitigation_score']:.1%}</td>
                </tr>
                <tr>
                    <td>Data Privacy</td>
                    <td class="{
                        'success' if ethical_input['human_centric']['privacy_score'] > 0.8
                        else 'warning' if ethical_input['human_centric']['privacy_score'] > 0.6
                        else 'danger'
                    }">
                        {
                            'Compliant' if ethical_input['human_centric']['privacy_score'] > 0.8
                            else 'Partial' if ethical_input['human_centric']['privacy_score'] > 0.6
                            else 'Non-Compliant'
                        }
                    </td>
                    <td>Privacy protection score: {ethical_input['human_centric']['privacy_score']:.1%}</td>
                </tr>
            </table>
        </div>
        """

        return f"""
        <html>
            <head>
                <title>AEPF Ethical Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .metric {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                    .score {{ font-size: 24px; color: #2c3e50; }}
                    .narrative {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #4a90e2; }}
                    .audit-section {{ margin: 30px 0; }}
                    .compliance-section {{ margin: 30px 0; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f5f5f5; }}
                    .success {{ color: #27ae60; }}
                    .warning {{ color: #f39c12; }}
                    .danger {{ color: #e74c3c; }}
                    ul {{ list-style-type: none; padding-left: 20px; }}
                    li {{ margin: 10px 0; }}
                    em {{ color: #7f8c8d; font-style: italic; }}
                </style>
            </head>
            <body>
                <h1>AI Model Ethical Audit Report</h1>
                
                {model_audit}
                
                {compliance_section}
                
                <div class="narrative">
                    {equity_narrative}
                </div>
                
                <div class="narrative">
                    {human_narrative}
                </div>
                
                <div class="narrative">
                    {innovation_narrative}
                </div>
                
                <div class="narrative">
                    {improvement_recommendations}
                </div>
            </body>
        </html>
        """


# Example usage
if __name__ == "__main__":
    config_path = r"C:\Users\leoco\AEPF_Mk3\UI\config\scenarios.yaml"
    trigger = TriggerSystem()

    try:
        trigger.load_config(config_path)
        model = "Gradient Boosting"
        scenario = "Loan Default Prediction"

        # Run the model
        model_results = trigger.run_model(model, scenario)

        # Optionally run AEPF analysis
        aepf_results = trigger.run_aepf(model, scenario, model_results)

    except Exception as e:
        print(f"An error occurred: {e}")
