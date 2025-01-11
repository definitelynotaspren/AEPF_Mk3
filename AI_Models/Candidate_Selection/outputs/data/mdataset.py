import pandas as pd
import numpy as np
import random

# Define the number of records
num_records = 1000

# Seed for reproducibility
np.random.seed(42)

# Generate mock data
data = {
    "candidate_id": range(1, num_records + 1),
    "age": np.random.randint(22, 60, num_records),
    "education_level": np.random.choice(
        ["High School", "Bachelor's", "Master's", "PhD"], num_records, p=[0.1, 0.6, 0.25, 0.05]
    ),
    "years_experience": np.random.randint(0, 35, num_records),
    "position_applied": np.random.choice(
        ["Data Scientist", "Software Engineer", "Project Manager", "Analyst"], num_records
    ),
    "technical_skills_score": np.random.randint(50, 101, num_records),
    "communication_skills_score": np.random.randint(50, 101, num_records),
    "leadership_skills_score": np.random.randint(50, 101, num_records),
    "cultural_fit_score": np.random.randint(50, 101, num_records),
    "previous_companies": np.random.randint(0, 10, num_records),
    "certifications": np.random.randint(0, 5, num_records),
    "willing_to_relocate": np.random.choice(["Yes", "No"], num_records, p=[0.7, 0.3]),
    "availability": np.random.choice(["Immediate", "1 Month", "3 Months"], num_records, p=[0.5, 0.3, 0.2]),
    "salary_expectation": np.random.randint(50000, 150001, num_records),
}

# Create a DataFrame
mock_dataset = pd.DataFrame(data)

# Define a function to determine if a candidate is hired based on certain criteria
def determine_hire(row):
    score = (
        row["technical_skills_score"] * 0.4
        + row["communication_skills_score"] * 0.2
        + row["leadership_skills_score"] * 0.2
        + row["cultural_fit_score"] * 0.2
    )
    if (
        score > 75
        and row["salary_expectation"] <= 120000
        and row["availability"] != "3 Months"
        and row["willing_to_relocate"] == "Yes"
    ):
        return "Yes"
    else:
        return "No"

# Apply the function to create the target variable
mock_dataset["hired"] = mock_dataset.apply(determine_hire, axis=1)

# Save to CSV
output_path = r'C:\Users\leoco\AEPF_Mk3\AI_models\candidate_selection\outputs\data\mock_hr_dataset.csv'
mock_dataset.to_csv(output_path, index=False)

print(f"Mock HR dataset created and saved to {output_path}")
