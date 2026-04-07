"""
AI Tax Fraud Detection - Model Training Script
This script creates a synthetic dataset and trains a machine learning model
to detect potentially fraudulent tax transactions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

print("=" * 50)
print("TAX FRAUD DETECTION SYSTEM - MODEL TRAINING")
print("=" * 50)

# ============================================
# STEP 1: CREATE SYNTHETIC TAX DATA
# ============================================
print("\n[1/6] Creating synthetic tax transaction data...")

np.random.seed(42)  # For reproducible results

n_samples = 10000

# Create realistic tax-related features
data = {
    # Transaction amount (normal: $50-$5000, suspicious: much higher or round numbers)
    'transaction_amount': np.random.uniform(50, 5000, n_samples),
    
    # Income declared (normal: $30k-$200k)
    'declared_income': np.random.uniform(30000, 200000, n_samples),
    
    # Expense claimed (normal: 10-40% of income)
    'claimed_expenses': np.random.uniform(3000, 80000, n_samples),
    
    # Industry risk score (1-10, higher = more fraud-prone industries)
    'industry_risk_score': np.random.randint(1, 11, n_samples),
    
    # Prior audit flag (0 or 1)
    'prior_audit_flag': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    
    # Number of dependents claimed
    'dependents_claimed': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.3, 0.15, 0.1, 0.05]),
    
    # Business type (0=individual, 1=small business, 2=corporate)
    'business_type': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.35, 0.15]),
    
    # Filing history length in years
    'filing_history_years': np.random.randint(1, 25, n_samples),
    
    # Income-expense ratio (suspicious if too high or low)
    'income_expense_ratio': np.random.uniform(0.5, 3.0, n_samples),
}

df = pd.DataFrame(data)

# ============================================
# STEP 2: CREATE FRAUD LABELS BASED ON PATTERNS
# ============================================
print("\n[2/6] Generating fraud labels based on suspicious patterns...")

# Initialize all as legitimate (0)
fraud_flag = np.zeros(n_samples)

# Pattern 1: Claimed expenses > 70% of declared income
mask1 = df['claimed_expenses'] > (df['declared_income'] * 0.7)
fraud_flag[mask1] = 1

# Pattern 2: Very high transaction amounts (> $4000)
mask2 = df['transaction_amount'] > 4000
fraud_flag[mask2] = 1

# Pattern 3: High industry risk score + prior audit
mask3 = (df['industry_risk_score'] > 7) & (df['prior_audit_flag'] == 1)
fraud_flag[mask3] = 1

# Pattern 4: Unusual income-expense ratio (< 0.8 or > 2.5)
mask4 = (df['income_expense_ratio'] < 0.8) | (df['income_expense_ratio'] > 2.5)
fraud_flag[mask4] = 1

# Pattern 5: Too many dependents (> 4) for individual filers
mask5 = (df['dependents_claimed'] > 4) & (df['business_type'] == 0)
fraud_flag[mask5] = 1

df['is_fraud'] = fraud_flag

# Add some random noise (5% of legitimate cases become fraud randomly)
random_fraud = np.random.random(n_samples) < 0.05
df.loc[random_fraud & (df['is_fraud'] == 0), 'is_fraud'] = 1

fraud_count = df['is_fraud'].sum()
fraud_percentage = (fraud_count / n_samples) * 100

print(f"   Dataset created: {n_samples} transactions")
print(f"   Fraud cases: {fraud_count} ({fraud_percentage:.1f}%)")
print(f"   Legitimate cases: {n_samples - fraud_count}")

# ============================================
# STEP 3: PREPARE FEATURES FOR MODEL
# ============================================
print("\n[3/6] Preparing features for machine learning...")

# Define which columns to use for prediction
feature_columns = [
    'transaction_amount',
    'declared_income',
    'claimed_expenses',
    'industry_risk_score',
    'prior_audit_flag',
    'dependents_claimed',
    'business_type',
    'filing_history_years',
    'income_expense_ratio'
]

X = df[feature_columns]
y = df['is_fraud']

# Scale the features (helps the model learn better)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Features prepared: {len(feature_columns)} features")
print(f"   Features: {', '.join(feature_columns)}")

# ============================================
# STEP 4: SPLIT DATA FOR TRAINING AND TESTING
# ============================================
print("\n[4/6] Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ============================================
# STEP 5: TRAIN THE AI MODEL
# ============================================
print("\n[5/6] Training Random Forest AI model...")

model = RandomForestClassifier(
    n_estimators=100,      # Number of decision trees
    max_depth=10,          # Maximum depth of each tree
    random_state=42,
    class_weight='balanced' # Handles imbalanced fraud cases
)

model.fit(X_train, y_train)
print("   Model training complete!")

# ============================================
# STEP 6: EVALUATE MODEL PERFORMANCE
# ============================================
print("\n[6/6] Evaluating model performance...")

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE RESULTS")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.1f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.1f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.1f}%)")
print("=" * 50)

# Feature importance - shows which factors matter most
print("\n📊 TOP 5 FRAUD INDICATORS (by importance):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(5).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")

# ============================================
# SAVE THE MODEL AND SCALER
# ============================================
print("\n💾 Saving model and scaler files...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
joblib.dump(model, 'models/fraud_model.pkl')
# Save the scaler (needed for future predictions)
joblib.dump(scaler, 'models/scaler.pkl')
# Save feature names for reference
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print("   ✅ Model saved as 'models/fraud_model.pkl'")
print("   ✅ Scaler saved as 'models/scaler.pkl'")
print("   ✅ Feature columns saved as 'models/feature_columns.pkl'")

# Save a sample dataset for the web app
df.to_csv('data/sample_transactions.csv', index=False)
print("   ✅ Sample data saved as 'data/sample_transactions.csv'")

print("\n" + "=" * 50)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 50)
print("\nNext step: Run 'streamlit run app.py' to launch the web app")