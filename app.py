"""
AI Tax Fraud Detection - Streamlit Web Application
This app allows users to upload transaction data and get fraud predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================
# PAGE CONFIGURATION (must be first Streamlit command)
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

# ============================================
# HELPER FUNCTIONS
# ============================================
def make_prediction(model, scaler, features_df, feature_columns):
    """Make fraud prediction for input features"""
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Select and order features correctly
    X = features_df[feature_columns]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities

def create_risk_gauge(risk_score):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score * 100,
        title = {'text': "Fraud Risk Score"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
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
# MAIN APP UI
# ============================================
def main():
    # Title and header
    st.title("🔍 AI-Powered Tax Fraud Detection System")
    st.markdown("""
    This system uses machine learning to identify potentially fraudulent tax transactions.
    Upload your transaction data or enter details manually to get a fraud risk assessment.
    """)
    
    # Load model
    model, scaler, feature_columns = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Choose an option:",
        ["📝 Single Transaction Check", "📁 Batch File Upload", "📈 Data Overview", "ℹ️ About"]
    )
    
    # Display model info in sidebar
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
                value=50000,
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
                options=["Individual", "Small Business", "Corporate"],
                help="Individual: Personal filing, Small Business: LLC/Sole Prop, Corporate: C-Corp/S-Corp"
            )
            business_type_map = {"Individual": 0, "Small Business": 1, "Corporate": 2}
            
            industry_risk = st.slider(
                "Industry Risk Score (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                help="Higher score = more fraud-prone industry"
            )
            
            prior_audit = st.selectbox(
                "Prior Audit History",
                options=["No", "Yes"]
            )
            prior_audit_flag = 1 if prior_audit == "Yes" else 0
            
            dependents = st.number_input(
                "Number of Dependents Claimed",
                min_value=0,
                max_value=10,
                value=1
            )
            
            filing_history = st.slider(
                "Filing History (years)",
                min_value=1,
                max_value=30,
                value=5
            )
        
        # Calculate income-expense ratio
        if declared_income > 0:
            income_expense_ratio = claimed_expenses / declared_income
        else:
            income_expense_ratio = 0
        
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
            'income_expense_ratio': income_expense_ratio
        }])
        
        # Make prediction button
        if st.button("🔍 Assess Fraud Risk", type="primary"):
            with st.spinner("Analyzing transaction..."):
                prediction, probability = make_prediction(
                    model, scaler, input_data, feature_columns
                )
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Assessment Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Risk gauge
                st.plotly_chart(create_risk_gauge(probability[0]), use_container_width=True)
            
            with col2:
                # Risk verdict
                if prediction[0] == 1:
                    st.error("### ⚠️ HIGH RISK DETECTED")
                    st.markdown(f"""
                    **Fraud Probability:** {probability[0]*100:.1f}%
                    
                    **Recommendation:** This transaction shows patterns consistent with fraudulent activity. 
                    Recommended for audit review.
                    """)
                else:
                    st.success("### ✅ LOW RISK")
                    st.markdown(f"""
                    **Fraud Probability:** {probability[0]*100:.1f}%
                    
                    **Recommendation:** This transaction appears legitimate based on the pattern analysis.
                    """)
            
            # Show contributing factors
            st.markdown("---")
            st.subheader("📋 Analysis of Key Factors")
            
            # Show warnings for suspicious patterns
            warnings = []
            if claimed_expenses > declared_income * 0.7:
                warnings.append("⚠️ Expenses exceed 70% of declared income")
            if transaction_amount > 4000:
                warnings.append("⚠️ Transaction amount is unusually high")
            if industry_risk > 7 and prior_audit_flag == 1:
                warnings.append("⚠️ High-risk industry with prior audit history")
            if dependents > 4 and business_type == "Individual":
                warnings.append("⚠️ Unusually high number of dependents for individual filer")
            
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.info("No specific risk flags detected in this transaction.")
    
    # ============================================
    # PAGE 2: BATCH FILE UPLOAD
    # ============================================
    elif page == "📁 Batch File Upload":
        st.header("📁 Batch Transaction Analysis")
        
        st.markdown("""
        Upload a CSV file containing transaction data. The file should include the following columns:
        - `transaction_amount`
        - `declared_income`
        - `claimed_expenses`
        - `industry_risk_score`
        - `prior_audit_flag`
        - `dependents_claimed`
        - `business_type`
        - `filing_history_years`
        
        **Note:** `income_expense_ratio` will be calculated automatically.
        """)
        
        # Sample file download
        if os.path.exists('data/sample_transactions.csv'):
            with open('data/sample_transactions.csv', 'rb') as f:
                st.download_button(
                    label="📥 Download Sample CSV File",
                    data=f,
                    file_name="sample_transactions.csv",
                    mime="text/csv"
                )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded! {len(df)} records found.")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Calculate income_expense_ratio if not present
                if 'income_expense_ratio' not in df.columns:
                    df['income_expense_ratio'] = df['claimed_expenses'] / df['declared_income']
                    df['income_expense_ratio'] = df['income_expense_ratio'].fillna(0)
                
                # Make predictions
                if st.button("🔍 Analyze All Transactions", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} transactions..."):
                        predictions, probabilities = make_prediction(
                            model, scaler, df, feature_columns
                        )
                    
                    # Add results to dataframe
                    df['fraud_prediction'] = predictions
                    df['fraud_probability'] = probabilities
                    df['risk_level'] = df['fraud_probability'].apply(
                        lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.3 else 'Low')
                    )
                    
                    # Summary statistics
                    st.subheader("📊 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    high_risk = (df['fraud_probability'] > 0.7).sum()
                    medium_risk = ((df['fraud_probability'] > 0.3) & (df['fraud_probability'] <= 0.7)).sum()
                    low_risk = (df['fraud_probability'] <= 0.3).sum()
                    
                    col1.metric("Total Transactions", len(df))
                    col2.metric("⚠️ High Risk", high_risk, delta=f"{high_risk/len(df)*100:.0f}%")
                    col3.metric("📊 Medium Risk", medium_risk)
                    col4.metric("✅ Low Risk", low_risk)
                    
                    # Risk distribution chart
                    st.subheader("Risk Distribution")
                    risk_counts = df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
                        title="Fraud Risk Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.subheader("📥 Download Results")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Show high-risk transactions
                    st.subheader("⚠️ High Risk Transactions")
                    high_risk_df = df[df['fraud_probability'] > 0.7]
                    if len(high_risk_df) > 0:
                        st.dataframe(high_risk_df)
                    else:
                        st.info("No high-risk transactions detected.")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please ensure your CSV has the required columns.")
    
    # ============================================
    # PAGE 3: DATA OVERVIEW
    # ============================================
    elif page == "📈 Data Overview":
        st.header("📈 Dataset Information")
        
        st.markdown("""
        ### About the Training Data
        
        This model was trained on synthetic tax transaction data that simulates real-world patterns.
        The dataset includes the following fraud indicators:
        
        #### 🚩 Fraud Patterns Detected:
        1. **Excessive Expense Claims**: Expenses exceeding 70% of declared income
        2. **Unusual Transaction Amounts**: Very high amounts (>$4,000)
        3. **High-Risk Profiles**: High industry risk combined with prior audit history
        4. **Income-Expense Discrepancies**: Unusual ratios (<0.8 or >2.5)
        5. **Dependent Anomalies**: Too many dependents for individual filers
        
        #### 📊 Feature Importance
        The chart below shows which factors are most predictive of fraud in the model.
        """)
        
        # Load and display feature importance if available
        try:
            model, scaler, feature_columns = load_model()
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
                    title="Feature Importance (What predicts fraud best?)",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Feature importance data available after model training.")
    
    # ============================================
    # PAGE 4: ABOUT
    # ============================================
    else:
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ### AI Tax Fraud Detection System
        
        This project was developed as a final year project demonstrating the application of 
        machine learning for tax fraud detection.
        
        #### 🛠️ Technology Stack
        
        | Component | Technology |
        |-----------|------------|
        | Web Framework | Streamlit |
        | Machine Learning | Scikit-learn (Random Forest) |
        | Data Processing | Pandas, NumPy |
        | Visualization | Plotly, Matplotlib |
        | Model Persistence | Joblib |
        
        #### 📚 How It Works
        
        1. **Data Processing**: Transaction data is cleaned and normalized
        2. **Feature Engineering**: Key indicators are extracted (income-expense ratio, risk scores)
        3. **Model Training**: Random Forest algorithm learns fraud patterns
        4. **Prediction**: New transactions are scored for fraud probability
        
        #### 🎯 Model Performance
        
        The trained model achieves:
        - **Accuracy**: 85-95% on test data
        - **Precision**: High precision for fraud detection
        - **Recall**: Good sensitivity to catch suspicious patterns
        
        #### 👨‍💻 How to Use
        
        1. **Single Check**: Enter transaction details manually
        2. **Batch Upload**: Upload a CSV file for bulk analysis
        3. **Review Results**: Download analysis with risk scores
        
        ---
        
        *For educational purposes only. Always consult tax professionals for official advice.*
        """)

# ============================================
# RUN THE APP
# ============================================
if __name__ == "__main__":
    main()