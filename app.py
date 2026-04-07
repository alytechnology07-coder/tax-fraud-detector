import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Tax Fraud Detector", page_icon="🔍", layout="wide")

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_risk(income, expenses, transaction, industry_risk, prior_audit, dependents):
    """Calculate fraud risk score based on rules"""
    risk = 0
    
    # Expense ratio check
    if income > 0:
        ratio = (expenses / income) * 100
        if ratio > 70:
            risk += 50
        elif ratio > 50:
            risk += 25
    
    # Transaction check
    if transaction > 5000:
        risk += 30
    elif transaction > 2000:
        risk += 15
    
    # Industry risk check
    if industry_risk >= 8:
        risk += 20
    elif industry_risk >= 6:
        risk += 10
    
    # Prior audit combo
    if prior_audit == "Yes" and industry_risk >= 7:
        risk += 15
    
    # Dependents check
    if dependents > 4:
        risk += 10
    
    return min(risk, 100)

def analyze_single_transaction(income, expenses, transaction, industry_risk, prior_audit, dependents):
    """Analyze single transaction and return detailed results"""
    risk = calculate_risk(income, expenses, transaction, industry_risk, prior_audit, dependents)
    
    details = []
    if income > 0:
        ratio = (expenses / income) * 100
        if ratio > 70:
            details.append(f"⚠️ Expenses are {ratio:.0f}% of income (highly suspicious)")
        elif ratio > 50:
            details.append(f"📊 Expenses are {ratio:.0f}% of income (moderately high)")
        else:
            details.append(f"✅ Expenses are {ratio:.0f}% of income (normal)")
    
    if transaction > 5000:
        details.append(f"⚠️ Transaction amount ${transaction:,} is very high")
    elif transaction > 2000:
        details.append(f"📊 Transaction amount ${transaction:,} is high")
    else:
        details.append(f"✅ Transaction amount ${transaction:,} is normal")
    
    if industry_risk >= 8:
        details.append(f"⚠️ Industry risk score {industry_risk}/10 (very high risk)")
    elif industry_risk >= 6:
        details.append(f"📊 Industry risk score {industry_risk}/10 (moderate risk)")
    else:
        details.append(f"✅ Industry risk score {industry_risk}/10 (low risk)")
    
    if prior_audit == "Yes" and industry_risk >= 7:
        details.append(f"⚠️ Prior audit in high-risk industry")
    
    if dependents > 4:
        details.append(f"⚠️ {dependents} dependents is unusually high")
    
    return risk, details

def create_gauge(risk):
    """Create gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if risk >= 50 else "green"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 100], "color": "lightcoral"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

# ============================================
# MAIN APP
# ============================================

st.title("🔍 Tax Fraud Detection System")
st.markdown("Enter transaction details to check for fraud risk")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose an option:",
    ["📝 Single Transaction Check", "📁 Batch File Upload", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**
- Analyzes expense to income ratio
- Checks transaction amounts
- Evaluates industry risk
- Considers audit history
""")

# ============================================
# PAGE 1: SINGLE TRANSACTION CHECK
# ============================================

if page == "📝 Single Transaction Check":
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Financial Details")
        income = st.number_input("Annual Income ($)", min_value=0, value=75000, step=5000)
        expenses = st.number_input("Claimed Expenses ($)", min_value=0, value=15000, step=1000)
        transaction = st.number_input("Transaction Amount ($)", min_value=0, value=500, step=100)
    
    with col2:
        st.subheader("⚡ Risk Factors")
        industry_risk = st.slider("Industry Risk (1-10)", 1, 10, 3)
        prior_audit = st.selectbox("Prior Audit?", ["No", "Yes"])
        dependents = st.number_input("Number of Dependents", 0, 10, 2)
    
    # Show warning if expenses are high
    if income > 0:
        expense_pct = (expenses / income) * 100
        if expense_pct > 70:
            st.error(f"⚠️ WARNING: Expenses are {expense_pct:.0f}% of income! (Normal is under 50%)")
        elif expense_pct > 50:
            st.warning(f"📊 Note: Expenses are {expense_pct:.0f}% of income")
    
    # Check button
    if st.button("🔍 Check Fraud Risk", type="primary"):
        risk, details = analyze_single_transaction(income, expenses, transaction, industry_risk, prior_audit, dependents)
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_gauge(risk), use_container_width=True)
        
        with col2:
            if risk >= 50:
                st.error(f"## 🔴 HIGH RISK DETECTED")
                st.metric("Risk Score", f"{risk}%")
                st.markdown("### 📋 Recommendation: Audit Required")
            else:
                st.success(f"## 🟢 LOW RISK")
                st.metric("Risk Score", f"{risk}%")
                st.markdown("### 📋 Recommendation: No Action Needed")
        
        st.markdown("---")
        st.subheader("📋 Detailed Analysis")
        for detail in details:
            st.write(detail)

# ============================================
# PAGE 2: BATCH FILE UPLOAD
# ============================================

elif page == "📁 Batch File Upload":
    st.header("📁 Batch Transaction Analysis")
    
    st.markdown("""
    Upload a CSV file with the following columns:
    - `income` (Annual Income in $)
    - `expenses` (Claimed Expenses in $)
    - `transaction` (Transaction Amount in $)
    - `industry_risk` (Risk score 1-10)
    - `prior_audit` ("Yes" or "No")
    - `dependents` (Number of dependents)
    """)
    
    # Download sample template
    sample_df = pd.DataFrame({
        'income': [75000, 50000, 120000],
        'expenses': [15000, 40000, 30000],
        'transaction': [500, 10000, 2000],
        'industry_risk': [3, 9, 5],
        'prior_audit': ['No', 'Yes', 'No'],
        'dependents': [2, 5, 3]
    })
    
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv,
        file_name="sample_transactions.csv",
        mime="text/csv"
    )
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded! {len(df)} records found.")
            st.dataframe(df.head())
            
            if st.button("📊 Analyze All Transactions", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    risk = calculate_risk(
                        row['income'], row['expenses'], row['transaction'],
                        row['industry_risk'], row['prior_audit'], row['dependents']
                    )
                    results.append({
                        'risk_score': risk,
                        'risk_level': 'HIGH' if risk >= 50 else 'LOW'
                    })
                    progress_bar.progress((i + 1) / len(df))
                
                df['risk_score'] = [r['risk_score'] for r in results]
                df['risk_level'] = [r['risk_level'] for r in results]
                
                st.markdown("---")
                st.subheader("📊 Analysis Summary")
                
                col1, col2, col3 = st.columns(3)
                high_risk = (df['risk_score'] >= 50).sum()
                low_risk = (df['risk_score'] < 50).sum()
                
                col1.metric("Total Transactions", len(df))
                col2.metric("🔴 High Risk", high_risk)
                col3.metric("🟢 Low Risk", low_risk)
                
                # Show results
                st.subheader("📋 Detailed Results")
                st.dataframe(df)
                
                # Download results
                csv_results = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_results,
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("Please make sure your CSV has the correct column names")

# ============================================
# PAGE 3: ABOUT
# ============================================

else:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### AI Tax Fraud Detection System
    
    This system helps identify potentially fraudulent tax transactions using intelligent rule-based analysis.
    
    ### 🎯 How It Works
    
    The system analyzes multiple risk factors:
    
    1. **Expense-to-Income Ratio** - Expenses > 70% of income is highly suspicious
    2. **Transaction Amount** - Unusually large transactions trigger alerts
    3. **Industry Risk** - High-risk industries (8-10) get more scrutiny
    4. **Audit History** - Prior audits in high-risk industries raise flags
    5. **Dependents** - Unusually high number of dependents
    
    ### 📊 Risk Scoring
    
    - **0-49%**: LOW RISK - Normal transaction
    - **50-100%**: HIGH RISK - Recommended for audit
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **Data Processing**: Pandas
    - **Deployment**: Streamlit Cloud
    
    ### 📁 Features
    
    - ✅ Single transaction analysis
    - ✅ Batch CSV upload for multiple transactions
    - ✅ Download results as CSV
    - ✅ Interactive risk gauge
    - ✅ Detailed factor analysis
    
    ### 👨‍💻 How to Use
    
    1. **Single Check**: Enter transaction details manually
    2. **Batch Upload**: Upload CSV file for bulk analysis
    3. **Review**: Get instant risk assessment with recommendations
    
    ---
    
    *For educational purposes only. Always consult tax professionals for official advice.*
    """)