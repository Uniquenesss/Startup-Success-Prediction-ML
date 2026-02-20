# ===============================
# STARTUP SUCCESS PREDICTION
# ===============================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("File is running...")

# Load dataset
df = pd.read_csv("startup data.csv")

print("Dataset Loaded")
print(df.head())

# Convert target column
df['status'] = df['status'].map({'acquired': 1, 'closed': 0})

# Select important features
df = df[[
    'age_first_funding_year',
    'age_last_funding_year',
    'age_first_milestone_year',
    'age_last_milestone_year',
    'relationships',
    'funding_rounds',
    'funding_total_usd',
    'milestones',
    'avg_participants',
    'status'
]]

# Drop missing values
df = df.dropna()

# Split data
X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "random_forest_model.pkl")

print("Model saved successfully!")