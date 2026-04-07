"""
AI Tax Fraud Detection - Complete Streamlit Web Application
Includes: Single Check, Batch Upload, Data Overview, and About pages
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

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
    """Calculate a custom risk score based on rules"""
    score = 0
    
    expense_ratio = features['claimed_expenses'] / max(features['declared_income'], 1)
    if expense_ratio > 0.7:
        score += 40
    elif expense_ratio > 0.5:
        score += 15
    
    if features['transaction_amount'] > 10000:
        score += 30
    elif features['transaction_amount'] > 5000:
        score += 15
    elif features['transaction_amount'] > 2000:
        score += 5
    
    if features['industry_risk_score'] > 8:
        score += 20
    elif features['industry_risk_score'] > 6:
        score += 10
    
    if features['prior_audit_flag'] == 1 and features['industry_risk_score'] > 7:
        score += 15
    
    if features['dependents_claimed'] > 4 and features['business_type'] == 0:
        score += 15
    
    return min(score, 100) / 100

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

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("🔍 AI-Powered Tax Fraud Detection System")
    st.markdown("""
    This system uses machine learning to identify potentially fraudulent tax transactions.
    Upload your transaction data or enter details manually to get a fraud risk assessment.
    """)
    
    model, scaler, feature_columns = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Choose an option:",
        ["📝 Single Transaction Check", "📁 Batch File Upload", "📈 Data Overview", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Model Information:**
    - Algorithm: Random Forest
    - Features used: {len(feature_columns)}
    - Model status: ✅ Loaded
    """)
    
    # ============================================
    # PAGE 1: SINGLE TRANSACTION CHECK
    # ============================================
    if page == "📝 Single Transaction Check":
        st.header("📝 Check a Single Transaction")
        
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
                value=3
            )
            
            prior_audit = st.selectbox("Prior Audit History", options=["No", "Yes"])
            prior_audit_flag = 1 if prior_audit == "Yes" else 0
            
            dependents = st.number_input("Number of Dependents Claimed", min_value=0, max_value=10, value=2)
            filing_history = st.slider("Filing History (years)", min_value=1, max_value=30, value=8)
        
        if declared_income > 0:
            expense_ratio = claimed_expenses / declared_income
            expense_percent = expense_ratio * 100
        else:
            expense_ratio = 0
            expense_percent = 0
        
        if expense_percent > 70:
            st.warning(f"⚠️ Warning: Expenses are {expense_percent:.0f}% of income (normal is <70%)")
        
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
        
        if st.button("🔍 Assess Fraud Risk", type="primary"):
            with st.spinner("Analyzing transaction..."):
                ml_prediction, ml_probability = make_prediction(
                    model, scaler, input_data, feature_columns
                )
                
                risk_score = calculate_risk_score({
                    'declared_income': declared_income,
                    'claimed_expenses': claimed_expenses,
                    'transaction_amount': transaction_amount,
                    'industry_risk_score': industry_risk,
                    'prior_audit_flag': prior_audit_flag,
                    'dependents_claimed': dependents,
                    'business_type': business_type_map[business_type]
                })
                
                final_risk = max(ml_probability[0], risk_score)
                
                st.markdown("---")
                st.subheader("📊 Assessment Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(create_risk_gauge(final_risk), use_container_width=True)
                
                with col2:
                    if final_risk > 0.5:
                        st.error("### ⚠️ HIGH RISK DETECTED")
                        st.markdown(f"**Fraud Probability:** {final_risk*100:.1f}%")
                    else:
                        st.success("### ✅ LOW RISK")
                        st.markdown(f"**Fraud Probability:** {final_risk*100:.1f}%")
    
    # ============================================
    # PAGE 2: BATCH FILE UPLOAD
    # ============================================
    elif page == "📁 Batch File Upload":
        st.header("📁 Batch Transaction Analysis")
        
        st.markdown("""
        Upload a CSV file containing transaction data with these columns:
        - `transaction_amount`
        - `declared_income`
        - `claimed_expenses`
        - `industry_risk_score`
        - `prior_audit_flag`
        - `dependents_claimed`
        - `business_type`
        - `filing_history_years`
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded! {len(df)} records found.")
                st.dataframe(df.head())
                
                if 'income_expense_ratio' not in df.columns:
                    df['income_expense_ratio'] = df['claimed_expenses'] / df['declared_income']
                    df['income_expense_ratio'] = df['income_expense_ratio'].fillna(0)
                
                if st.button("🔍 Analyze All Transactions", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} transactions..."):
                        predictions, probabilities = make_prediction(
                            model, scaler, df, feature_columns
                        )
                    
                    df['fraud_prediction'] = predictions
                    df['fraud_probability'] = probabilities
                    df['risk_level'] = df['fraud_probability'].apply(
                        lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.3 else 'Low')
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    high_risk = (df['fraud_probability'] > 0.7).sum()
                    medium_risk = ((df['fraud_probability'] > 0.3) & (df['fraud_probability'] <= 0.7)).sum()
                    low_risk = (df['fraud_probability'] <= 0.3).sum()
                    
                    col1.metric("Total Transactions", len(df))
                    col2.metric("⚠️ High Risk", high_risk)
                    col3.metric("📊 Medium Risk", medium_risk)
                    col4.metric("✅ Low Risk", low_risk)
                    
                    risk_counts = df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
                        title="Fraud Risk Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    # ============================================
    # PAGE 3: DATA OVERVIEW
    # ============================================
    elif page == "📈 Data Overview":
        st.header("📈 Dataset & Model Information")
        
        st.markdown("""
        ### 🚩 Fraud Patterns Detected:
        1. **Excessive Expense Claims**: Expenses exceeding 70% of declared income
        2. **Unusual Transaction Amounts**: Very high amounts (>$5,000)
        3. **High-Risk Profiles**: High industry risk combined with prior audit history
        4. **Income-Expense Discrepancies**: Unusual ratios
        5. **Dependent Anomalies**: Too many dependents for individual filers
        """)
        
        try:
            if model:
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance chart available after model training.")
    
    # ============================================
    # PAGE 4: ABOUT
    # ============================================
    else:
        st.header("ℹ️ About This Project")
        st.markdown("""
        ### AI Tax Fraud Detection System
        
        **Technology Stack:**
        - Web Framework: Streamlit
        - Machine Learning: Scikit-learn (Random Forest)
        - Visualization: Plotly
        - Deployment: Streamlit Cloud
        
        ### How It Works:
        1. Enter transaction details or upload a CSV file
        2. The AI model analyzes patterns and risk factors
        3. Get instant fraud risk assessment with visual gauge
        4. Download detailed results for further review
        
        *For educational purposes only.*
        """)

if __name__ == "__main__":
    main()