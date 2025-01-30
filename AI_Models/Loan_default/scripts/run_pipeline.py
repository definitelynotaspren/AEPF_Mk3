from pathlib import Path
from generate_report import generate_sample_report

def run_pipeline():
    """Run the Loan Default model pipeline."""
    try:
        print("Starting Loan Default Model Pipeline...")
        
        # Generate and save report
        report = generate_sample_report()
        
        # Verify report was created
        report_path = Path(__file__).parent.parent / 'outputs/reports/model_report.json'
        if report_path.exists():
            print(f"Report created successfully at {report_path}")
        else:
            print(f"Error: Report not found at {report_path}")
        
        print("Pipeline completed successfully!")
        return report
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    run_pipeline() 