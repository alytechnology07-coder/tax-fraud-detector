"""
Tax Fraud Detection - Simple Working Version
"""
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Tax Fraud Detector", page_icon="🔍", layout="wide")

def check_fraud_risk(income, expenses, transaction, industry_risk, prior_audit, dependents):
    risk_score = 0
    warnings = []
    
    if income > 0:
        expense_ratio = (expenses / income) * 100
        if expense_ratio > 70:
            risk_score += 50
            warnings.append(f"❌ Expenses are {expense_ratio:.0f}% of income (should be under 70%)")
        elif expense_ratio > 50:
            risk_score += 25
            warnings.append(f"⚠️ Expenses are {expense_ratio:.0f}% of income (slightly high)")
        else:
            warnings.append(f"✅ Expenses are {expense_ratio:.0f}% of income (normal)")
    
    if transaction > 5000:
        risk_score += 30
        warnings.append(f"❌ Transaction amount ${transaction:,} is very high")
    elif transaction > 2000:
        risk_score += 15
        warnings.append(f"⚠️ Transaction amount ${transaction:,} is moderately high")
    else:
        warnings.append(f"✅ Transaction amount ${transaction:,} is normal")
    
    if industry_risk >= 8:
        risk_score += 20
        warnings.append(f"❌ Industry risk score {industry_risk}/10 (very high risk)")
    elif industry_risk >= 6:
        risk_score += 10
        warnings.append(f"⚠️ Industry risk score {industry_risk}/10 (moderate risk)")
    else:
        warnings.append(f"✅ Industry risk score {industry_risk}/10 (low risk)")
    
    if prior_audit == "Yes" and industry_risk >= 7:
        risk_score += 15
        warnings.append(f"❌ Prior audit in high-risk industry")
    
    if dependents > 4:
        risk_score += 10
        warnings.append(f"❌ {dependents} dependents is unusually high")
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 50:
        risk_level = "HIGH RISK"
        color = "red"
        emoji = "🔴"
    else:
        risk_level = "LOW RISK"
        color = "green"
        emoji = "🟢"
    
    return risk_score, risk_level, color, emoji, warnings

def create_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Risk Score"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 100], "color": "lightcoral"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.title("🔍 Tax Fraud Detection System")
    st.markdown("Enter the transaction details below to check for fraud risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Details")
        income = st.number_input("Annual Income ($)", min_value=0, value=75000, step=5000)
        expenses = st.number_input("Claimed Expenses ($)", min_value=0, value=15000, step=1000)
        transaction = st.number_input("Transaction Amount ($)", min_value=0, value=500, step=100)
    
    with col2:
        st.subheader("Risk Factors")
        industry_risk = st.select_slider("Industry Risk (1-10)", options=[1,2,3,4,5,6,7,8,9,10], value=3)
        prior_audit = st.radio("Prior Audit?", ["No", "Yes"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    
    if income > 0:
        expense_pct = (expenses / income) * 100
        if expense_pct > 70:
            st.error(f"⚠️ WARNING: Expenses are {expense_pct:.0f}% of income!")
        elif expense_pct > 50:
            st.warning(f"Note: Expenses are {expense_pct:.0f}% of income")
    
    if st.button("Check Fraud Risk", type="primary"):
        score, level, color, emoji, warnings = check_fraud_risk(
            income, expenses, transaction, industry_risk, prior_audit, dependents
        )
        
        st.markdown("---")
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_gauge(score, color), use_container_width=True)
        
        with col2:
            if level == "HIGH RISK":
                st.error(f"## {emoji} {level}")
                st.markdown(f"**Risk Score: {score}%**")
                st.markdown("### Recommendation: Audit Required")
            else:
                st.success(f"## {emoji} {level}")
                st.markdown(f"**Risk Score: {score}%**")
                st.markdown("### Recommendation: No Action Needed")
        
        st.markdown("---")
        st.subheader("Detailed Analysis")
        for warning in warnings:
            st.write(warning)

if __name__ == "__main__":
    main()
