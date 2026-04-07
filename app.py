cat > app.py << 'EOF'
"""
AI Tax Fraud Detection - Working Version with Manual Risk Scoring
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Tax Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

def calculate_fraud_risk(income, expenses, transaction, industry_risk, prior_audit, dependents, business_type, filing_years):
    """Calculate fraud risk score based on clear rules"""
    risk_score = 0
    reasons = []
    
    # 1. Expense ratio check (most important)
    if income > 0:
        expense_ratio = expenses / income
        if expense_ratio > 0.7:
            risk_score += 40
            reasons.append(f"⚠️ Expenses are {expense_ratio*100:.0f}% of income (highly suspicious)")
        elif expense_ratio > 0.5:
            risk_score += 20
            reasons.append(f"📊 Expenses are {expense_ratio*100:.0f}% of income (moderately high)")
        elif expense_ratio < 0.1:
            risk_score += 5
            reasons.append(f"✅ Expenses are only {expense_ratio*100:.0f}% of income (very reasonable)")
        else:
            reasons.append(f"✅ Expenses are {expense_ratio*100:.0f}% of income (normal range)")
    
    # 2. Transaction amount check
    if transaction > 10000:
        risk_score += 30
        reasons.append(f"⚠️ Transaction amount ${transaction:,} is extremely high")
    elif transaction > 5000:
        risk_score += 20
        reasons.append(f"📊 Transaction amount ${transaction:,} is high")
    elif transaction > 2000:
        risk_score += 10
        reasons.append(f"💰 Transaction amount ${transaction:,} is moderate")
    else:
        reasons.append(f"✅ Transaction amount ${transaction:,} is normal")
    
    # 3. Industry risk check
    if industry_risk > 8:
        risk_score += 20
        reasons.append(f"⚠️ Industry risk score {industry_risk}/10 (very high risk industry)")
    elif industry_risk > 6:
        risk_score += 10
        reasons.append(f"📊 Industry risk score {industry_risk}/10 (moderately high risk)")
    else:
        reasons.append(f"✅ Industry risk score {industry_risk}/10 (normal)")
    
    # 4. Prior audit + industry risk combination
    if prior_audit == "Yes" and industry_risk > 7:
        risk_score += 15
        reasons.append(f"⚠️ Prior audit in high-risk industry (significant red flag)")
    elif prior_audit == "Yes":
        reasons.append(f"ℹ️ Has prior audit history")
    
    # 5. Dependent check for individuals
    if business_type == "Individual" and dependents > 4:
        risk_score += 10
        reasons.append(f"⚠️ {dependents} dependents claimed (unusually high for individual)")
    
    # 6. Filing history
    if filing_years < 3:
        risk_score += 5
        reasons.append(f"📊 Short filing history ({filing_years} years)")
    
    # Cap at 100
    final_score = min(risk_score, 100)
    return final_score, reasons

def get_risk_level(score):
    """Get risk level based on score"""
    if score >= 50:
        return "HIGH RISK", "🔴", "red"
    else:
        return "LOW RISK", "🟢", "green"

def create_gauge(score):
    """Create risk gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Fraud Risk Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if score >= 50 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 50], 'color': "yellowgreen"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.title("🔍 AI-Powered Tax Fraud Detection System")
    st.markdown("This system analyzes tax transactions to identify potential fraud patterns.")
    
    # Sidebar
    st.sidebar.title("📊 How It Works")
    st.sidebar.info("""
    The system analyzes:
    - Expense to income ratio
    - Transaction amounts
    - Industry risk scores
    - Audit history
    - Dependent claims
    - Filing history
    
    **Score ≥ 50% = HIGH RISK**
    """)
    
    # Main input form
    st.header("📝 Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Income Information")
        income = st.number_input("Declared Annual Income ($)", min_value=0, value=75000, step=5000)
        expenses = st.number_input("Claimed Business Expenses ($)", min_value=0, value=15000, step=1000)
        transaction = st.number_input("Transaction Amount ($)", min_value=0, value=500, step=100)
    
    with col2:
        st.subheader("🏢 Business Information")
        business_type = st.selectbox("Business Type", ["Individual", "Small Business", "Corporate"])
        industry_risk = st.slider("Industry Risk Score (1-10)", 1, 10, 3)
        prior_audit = st.selectbox("Prior Audit History", ["No", "Yes"])
        dependents = st.number_input("Number of Dependents", 0, 10, 2)
        filing_years = st.slider("Years of Filing History", 1, 30, 8)
    
    # Calculate and show expense ratio warning
    if income > 0:
        expense_pct = (expenses / income) * 100
        if expense_pct > 70:
            st.warning(f"⚠️ Warning: Expenses are {expense_pct:.0f}% of income! (Normal is <50%)")
        elif expense_pct > 50:
            st.info(f"📊 Note: Expenses are {expense_pct:.0f}% of income")
    
    # Analyze button
    if st.button("🔍 Assess Fraud Risk", type="primary"):
        # Calculate risk
        risk_score, reasons = calculate_fraud_risk(
            income, expenses, transaction, industry_risk, 
            prior_audit, dependents, business_type, filing_years
        )
        
        risk_level, emoji, color = get_risk_level(risk_score)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Assessment Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(create_gauge(risk_score), use_container_width=True)
        
        with col2:
            if risk_level == "HIGH RISK":
                st.error(f"### {emoji} {risk_level} DETECTED")
                st.markdown(f"**Risk Score:** {risk_score:.0f}%")
                st.markdown("**Recommendation:** This transaction should be reviewed by tax authorities.")
            else:
                st.success(f"### {emoji} {risk_level}")
                st.markdown(f"**Risk Score:** {risk_score:.0f}%")
                st.markdown("**Recommendation:** This transaction appears legitimate.")
        
        # Show detailed analysis
        st.markdown("---")
        st.subheader("📋 Detailed Analysis")
        
        for reason in reasons:
            if "✅" in reason:
                st.success(reason)
            elif "⚠️" in reason:
                st.error(reason)
            else:
                st.info(reason)
        
        # Show summary
        st.markdown("---")
        if risk_score >= 50:
            st.warning("⚠️ **Summary:** Multiple risk factors detected. Further investigation recommended.")
        else:
            st.success("✅ **Summary:** No major risk factors detected. Transaction appears normal.")

if __name__ == "__main__":
    main()
EOF