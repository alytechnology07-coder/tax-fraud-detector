"""
AI Tax Fraud Detection - Streamlit Web Application (UPDATED with better thresholds)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Tax Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD THE TRAINED MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('models/fraud_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("""
        ❌ Model files not found!
        
        Please run `python train_model.py` first to train the model.
        """)
        return None, None, None

def make_prediction(model, scaler, features_df, feature_columns):
    """Make fraud prediction for input features"""
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    
    X = features_df[feature_columns]
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities

def calculate_risk_score(features):
    """Calculate a custom risk score based on rules (more balanced)"""
    score = 0
    
    # Expense ratio risk (expenses/income)
    expense_ratio = features['claimed_expenses'] / max(features['declared_income'], 1)
    if expense_ratio > 0.7:
        score += 40
    elif expense_ratio > 0.5:
        score += 15
    
    # Transaction amount risk
    if features['transaction_amount'] > 10000:
        score += 30
    elif features['transaction_amount'] > 5000:
        score += 15
    elif features['transaction_amount'] > 2000:
        score += 5
    
    # Industry risk
    if features['industry_risk_score'] > 8:
        score += 20
    elif features['industry_risk_score'] > 6:
        score += 10
    
    # Prior audit flag
    if features['prior_audit_flag'] == 1 and features['industry_risk_score'] > 7:
        score += 15
    
    # Dependent anomaly
    if features['dependents_claimed'] > 4 and features['business_type'] == 0:
        score += 15
    
    return min(score, 100) / 100  # Return as probability between 0-1

def create_risk_gauge(risk_score):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        title={'text': "Fraud Risk Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if risk_score > 0.5 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.title("🔍 AI-Powered Tax Fraud Detection System")
    st.markdown("""
    This system uses machine learning to identify potentially fraudulent tax transactions.
    Enter transaction details below to get a fraud risk assessment.
    """)
    
    model, scaler, feature_columns = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    st.sidebar.info(f"Model: Random Forest | Features: {len(feature_columns)}")
    
    # Main input form
    st.header("📝 Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Income Information")
        declared_income = st.number_input(
            "Declared Annual Income ($)",
            min_value=0,
            max_value=1000000,
            value=75000,
            step=1000
        )
        
        claimed_expenses = st.number_input(
            "Claimed Business Expenses ($)",
            min_value=0,
            max_value=500000,
            value=15000,
            step=1000
        )
        
        transaction_amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0,
            max_value=100000,
            value=500,
            step=100
        )
    
    with col2:
        st.subheader("🏢 Business Information")
        business_type = st.selectbox(
            "Business Type",
            options=["Individual", "Small Business", "Corporate"]
        )
        business_type_map = {"Individual": 0, "Small Business": 1, "Corporate": 2}
        
        industry_risk = st.slider(
            "Industry Risk Score (1-10)",
            min_value=1,
            max_value=10,
            value=3,
            help="Higher score = more fraud-prone industry"
        )
        
        prior_audit = st.selectbox("Prior Audit History", options=["No", "Yes"])
        prior_audit_flag = 1 if prior_audit == "Yes" else 0
        
        dependents = st.number_input("Number of Dependents Claimed", min_value=0, max_value=10, value=2)
        filing_history = st.slider("Filing History (years)", min_value=1, max_value=30, value=8)
    
    # Calculate values
    if declared_income > 0:
        expense_ratio = claimed_expenses / declared_income
        expense_percent = expense_ratio * 100
    else:
        expense_ratio = 0
        expense_percent = 0
    
    # Show expense ratio warning
    if expense_percent > 70:
        st.warning(f"⚠️ Warning: Expenses are {expense_percent:.0f}% of income (normal is <70%)")
    
    # Create feature DataFrame
    input_data = pd.DataFrame([{
        'transaction_amount': transaction_amount,
        'declared_income': declared_income,
        'claimed_expenses': claimed_expenses,
        'industry_risk_score': industry_risk,
        'prior_audit_flag': prior_audit_flag,
        'dependents_claimed': dependents,
        'business_type': business_type_map[business_type],
        'filing_history_years': filing_history,
        'income_expense_ratio': expense_ratio
    }])
    
    # Make prediction button
    if st.button("🔍 Assess Fraud Risk", type="primary"):
        with st.spinner("Analyzing transaction..."):
            # Get ML prediction
            ml_prediction, ml_probability = make_prediction(
                model, scaler, input_data, feature_columns
            )
            
            # Calculate custom risk score
            risk_score = calculate_risk_score({
                'declared_income': declared_income,
                'claimed_expenses': claimed_expenses,
                'transaction_amount': transaction_amount,
                'industry_risk_score': industry_risk,
                'prior_audit_flag': prior_audit_flag,
                'dependents_claimed': dependents,
                'business_type': business_type_map[business_type]
            })
            
            # Use the higher of the two scores (or average)
            final_risk = max(ml_probability[0], risk_score)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.plotly_chart(create_risk_gauge(final_risk), use_container_width=True)
            
            with col2:
                if final_risk > 0.5:
                    st.error("### ⚠️ HIGH RISK DETECTED")
                    st.markdown(f"""
                    **Fraud Probability:** {final_risk*100:.1f}%
                    
                    **Recommendation:** This transaction shows patterns consistent with fraudulent activity.
                    """)
                else:
                    st.success("### ✅ LOW RISK")
                    st.markdown(f"""
                    **Fraud Probability:** {final_risk*100:.1f}%
                    
                    **Recommendation:** This transaction appears legitimate.
                    """)
            
            # Show detailed analysis
            st.markdown("---")
            st.subheader("📋 Detailed Analysis")
            
            risk_factors = []
            
            if expense_ratio > 0.7:
                risk_factors.append(f"⚠️ Expenses are {expense_percent:.0f}% of income (threshold: 70%)")
            else:
                risk_factors.append(f"✅ Expenses are {expense_percent:.0f}% of income (normal range)")
            
            if transaction_amount > 5000:
                risk_factors.append(f"⚠️ Transaction amount ${transaction_amount:,} is unusually high")
            elif transaction_amount > 2000:
                risk_factors.append(f"📊 Transaction amount ${transaction_amount:,} is moderately high")
            else:
                risk_factors.append(f"✅ Transaction amount ${transaction_amount:,} is normal")
            
            if industry_risk > 7 and prior_audit_flag == 1:
                risk_factors.append(f"⚠️ High-risk industry (score: {industry_risk}) with prior audit history")
            elif industry_risk > 7:
                risk_factors.append(f"📊 High-risk industry (score: {industry_risk})")
            else:
                risk_factors.append(f"✅ Industry risk score: {industry_risk}/10 (normal range)")
            
            if dependents > 4 and business_type == "Individual":
                risk_factors.append(f"⚠️ {dependents} dependents claimed for individual filer (higher than normal)")
            
            for factor in risk_factors:
                st.write(factor)

if __name__ == "__main__":
    main()