import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

# Load Dataset
dataset_path = r'C:\Users\leoco\AEPF_Mk3\AI_Models\Candidate_Selection\models\mock_hr_data.csv'
data = pd.read_csv(dataset_path)
print(f"Dataset Loaded: {data.info()}")

# Identify target column and features
target_column = "Hired"  # Update if necessary
X = data.drop(columns=[target_column])
y = data[target_column]

# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store for potential inverse transformation

# Encode target variable
y = LabelEncoder().fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("Training the Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
model_dir = Path(r'C:\Users\leoco\AEPF_Mk3\AI_Models\Candidate_Selection\models')
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "random_forest_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")

# Feature Importance (optional)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save feature importance
feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)
print(f"Feature importance saved at: {model_dir / 'feature_importance.csv'}")
