"""
Train churn prediction models and save the best one.
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 1. Load the cleaned dataset
data_path = "data/processed/telco_churn_clean.csv"
data = pd.read_csv(data_path)

# 2. Split into features (x) and target (y)
x = data.drop("Churn", axis=1)
y = data["Churn"]

# 3. Split data into test and train
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale numeric features (important for logistic regression)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 5. Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}

# 6. Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(x_train_scaled if "Logistic" in name else x_train, y_train)

    # Predict
    y_pred = model.predict(x_test_scaled if "Logistic" in name else x_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    results[name] = {"model": model, "accuracy": acc, "auc": auc}

# 7. Choose the best model
best_model_name = max(results, key=lambda x: results[x]["auc"])
best_model = results[best_model_name]["model"]

print(f"\nBest model: {best_model_name}")

# 8. Save model and scaler
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, f"../models/{best_model_name.replace(' ', '_').lower()}.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Model and scaler saved successfully!")
