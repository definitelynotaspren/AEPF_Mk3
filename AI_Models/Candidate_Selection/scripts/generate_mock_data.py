import pandas as pd
import numpy as np

def generate_mock_hr_data():
    """Generate mock HR dataset with exactly 200 rows"""
    np.random.seed(42)  # For reproducibility
    
    n_samples = 200  # Fixed size
    
    data = {
        'technical_skills_score': np.random.uniform(60, 100, n_samples),
        'communication_skills_score': np.random.uniform(50, 100, n_samples),
        'leadership_skills_score': np.random.uniform(40, 100, n_samples),
        'cultural_fit_score': np.random.uniform(50, 100, n_samples),
        'hired': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Save the dataset
    output_path = '../models/mock_hr_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated mock dataset with {n_samples} rows")

if __name__ == "__main__":
    generate_mock_hr_data() 