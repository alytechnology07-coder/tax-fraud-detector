"""
Train a BALANCED tax fraud detection model
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 50)
print("TRAINING BALANCED FRAUD DETECTION MODEL")
print("=" * 50)

# Create balanced dataset
np.random.seed(42)
n_normal = 3000  # Normal cases
n_fraud = 300    # Fraud cases (only 10% fraud)

print(f"\nCreating dataset: {n_normal} normal, {n_fraud} fraud cases")

# NORMAL CASES (Legitimate)
normal_data = {
    'transaction_amount': np.random.uniform(50, 2000, n_normal),
    'declared_income': np.random.uniform(40000, 150000, n_normal),
    'claimed_expenses': np.random.uniform(5000, 30000, n_normal),
    'industry_risk_score': np.random.randint(1, 6, n_normal),
    'prior_audit_flag': np.random.choice([0, 1], n_normal, p=[0.9, 0.1]),
    'dependents_claimed': np.random.choice([0, 1, 2, 3], n_normal, p=[0.2, 0.4, 0.3, 0.1]),
    'business_type': np.random.choice([0, 1, 2], n_normal, p=[0.5, 0.3, 0.2]),
    'filing_history_years': np.random.randint(3, 20, n_normal),
}

# FRAUD CASES (Suspicious)
fraud_data = {
    'transaction_amount': np.random.uniform(3000, 15000, n_fraud),
    'declared_income': np.random.uniform(20000, 80000, n_fraud),
    'claimed_expenses': np.random.uniform(30000, 70000, n_fraud),
    'industry_risk_score': np.random.randint(7, 11, n_fraud),
    'prior_audit_flag': np.random.choice([0, 1], n_fraud, p=[0.4, 0.6]),
    'dependents_claimed': np.random.choice([0, 1, 2, 3, 4, 5], n_fraud, p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1]),
    'business_type': np.random.choice([0, 1, 2], n_fraud, p=[0.6, 0.3, 0.1]),
    'filing_history_years': np.random.randint(1, 10, n_fraud),
}

# Create DataFrames
df_normal = pd.DataFrame(normal_data)
df_fraud = pd.DataFrame(fraud_data)

# Add labels
df_normal['is_fraud'] = 0
df_fraud['is_fraud'] = 1

# Combine
df = pd.concat([df_normal, df_fraud], ignore_index=True)

# Calculate income_expense_ratio
df['income_expense_ratio'] = df['claimed_expenses'] / df['declared_income']

print(f"\nDataset created:")
print(f"  Normal cases: {(df['is_fraud']==0).sum()}")
print(f"  Fraud cases: {(df['is_fraud']==1).sum()}")
print(f"  Fraud percentage: {df['is_fraud'].mean()*100:.1f}%")

# Features
feature_columns = [
    'transaction_amount', 'declared_income', 'claimed_expenses',
    'industry_risk_score', 'prior_audit_flag', 'dependents_claimed',
    'business_type', 'filing_history_years', 'income_expense_ratio'
]

X = df[feature_columns]
y = df['is_fraud']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_scaled, y)

# Test predictions on sample cases
print("\n" + "=" * 50)
print("TESTING MODEL ON SAMPLE CASES")
print("=" * 50)

# Test Case 1: Normal taxpayer
test_normal = pd.DataFrame([{
    'transaction_amount': 500,
    'declared_income': 75000,
    'claimed_expenses': 15000,
    'industry_risk_score': 3,
    'prior_audit_flag': 0,
    'dependents_claimed': 2,
    'business_type': 0,
    'filing_history_years': 8,
    'income_expense_ratio': 15000/75000
}])
test_normal_scaled = scaler.transform(test_normal[feature_columns])
pred_normal = model.predict(test_normal_scaled)[0]
prob_normal = model.predict_proba(test_normal_scaled)[0][1]
print(f"\n✅ NORMAL CASE:")
print(f"   Income: $75k, Expenses: $15k, Transaction: $500")
print(f"   Prediction: {'FRAUD' if pred_normal==1 else 'NORMAL'}")
print(f"   Fraud Probability: {prob_normal*100:.1f}%")

# Test Case 2: Fraudulent taxpayer
test_fraud = pd.DataFrame([{
    'transaction_amount': 8000,
    'declared_income': 50000,
    'claimed_expenses': 40000,
    'industry_risk_score': 9,
    'prior_audit_flag': 1,
    'dependents_claimed': 5,
    'business_type': 0,
    'filing_history_years': 3,
    'income_expense_ratio': 40000/50000
}])
test_fraud_scaled = scaler.transform(test_fraud[feature_columns])
pred_fraud = model.predict(test_fraud_scaled)[0]
prob_fraud = model.predict_proba(test_fraud_scaled)[0][1]
print(f"\n⚠️ FRAUD CASE:")
print(f"   Income: $50k, Expenses: $40k, Transaction: $8k")
print(f"   Prediction: {'FRAUD' if pred_fraud==1 else 'NORMAL'}")
print(f"   Fraud Probability: {prob_fraud*100:.1f}%")

# Save model
print("\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("✅ Model saved successfully!")
print("\nNow push to GitHub and redeploy!")
