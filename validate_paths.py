"""Validate the existence of critical files for loan default analysis."""
from pathlib import Path
import sys

def check_loan_default_paths():
    base_path = Path.cwd()
    print(f"\nChecking paths from: {base_path}")
    print("-" * 50)
    
    # Critical paths for loan default analysis
    critical_paths = {
        'Trigger Script': 'Trigger/trigger_system.py',
        'Config File': 'config/scenarios.yaml',
        'Preprocessing Script': 'AI_Models/Loan_default/scripts/preprocess_data.py',
        'Report Generator': 'AI_Models/Loan_default/scripts/generate_report.py',
        'Model Tuner': 'AI_Models/Loan_default/scripts/tune_model.py'
    }
    
    missing_files = []
    
    for name, path in critical_paths.items():
        full_path = base_path / path
        exists = full_path.exists()
        status = '✓' if exists else '✗'
        print(f"{status} {name}: {path}")
        if exists:
            print(f"  Located at: {full_path}")
        else:
            missing_files.append(path)
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"- {file}")
            # Create parent directories if they don't exist
            parent_dir = (base_path / file).parent
            if not parent_dir.exists():
                print(f"  Creating directory: {parent_dir}")
                parent_dir.mkdir(parents=True, exist_ok=True)
    
    return len(missing_files) == 0

if __name__ == "__main__":
    success = check_loan_default_paths()
    if not success:
        print("\nSome required files are missing. Please create them before running the trigger system.")
        sys.exit(1)
    else:
        print("\nAll required files found!") 