import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# START SCREEN
# -------------------------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown("""
        <h1 style='text-align: center;'>🏦 AI Risk Control Center</h1>
        <p style='text-align: center;'>Smart Loan Risk Detection System</p>
    """, unsafe_allow_html=True)

    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)

    if st.button("🚀 Start System"):
        st.session_state.started = True
        st.rerun()

    st.stop()

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="AI Risk Control Center", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("Bankingdata.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# -------------------------------
# RISK RULE
# -------------------------------
def risk_rule(row):
    risk = 0
    if row['loan_amount'] > 50000: risk += 2
    if row['credit_score'] < 600: risk += 2
    if row['dti'] > 0.4: risk += 2
    if row['ip_risk_score'] > 70: risk += 2
    if row['has_document_mismatch'] == 1: risk += 2
    if row['income_mismatch_ratio'] > 0.3: risk += 2

    if risk >= 8: return "High"
    elif risk >= 4: return "Medium"
    else: return "Low"

df["risk_level"] = df.apply(risk_rule, axis=1)

# -------------------------------
# ML MODEL
# -------------------------------
le = LabelEncoder()
df_ml = df.copy()

for col in df_ml.select_dtypes(include=['object']).columns:
    if col != "risk_level":
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

df_ml["risk_level"] = le.fit_transform(df_ml["risk_level"])

X = df_ml.drop(["risk_level", "loan_id"], axis=1, errors="ignore")
y = df_ml["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# AGENTS
# -------------------------------
def risk_agent(sample):
    return ["Low","Medium","High"][model.predict(sample)[0]]

def compliance_agent(doc, inc):
    return "Violation" if doc == 1 or inc > 0.5 else "Compliant"

def decision_agent(risk, comp):
    if comp == "Violation" or risk == "High":
        return "Reject"
    elif risk == "Medium":
        return "Review"
    else:
        return "Approve"

def explanation_agent(risk):
    return {
        "High": "High financial risk detected",
        "Medium": "Moderate risk, requires verification",
        "Low": "Low risk profile"
    }[risk]

def solution_agent(risk, comp):
    if comp == "Violation":
        return "❌ Fix document mismatch and verify income"
    elif risk == "High":
        return "⚠️ Reduce loan amount & improve credit score"
    elif risk == "Medium":
        return "🧐 Manual review required"
    else:
        return "✅ Approve loan"

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🏦 Risk Control Panel")

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

filtered_df = df[df["risk_level"].isin(risk_filter)]

page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🤖 AI Decision Center",
    "🧾 Manual Data Entry"
])

# ===============================
# 📊 DASHBOARD
# ===============================
if page == "📊 Dashboard":

    st.title("📊 Enterprise Risk Dashboard")

    # KPI
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total", len(filtered_df))
    col2.metric("High Risk %", f"{(filtered_df['risk_level']=='High').mean()*100:.1f}%")
    col3.metric("Avg Credit", int(filtered_df['credit_score'].mean()))
    col4.metric("Avg Loan", int(filtered_df['loan_amount'].mean()))

    # Charts
    st.subheader("📊 Risk Analysis")

    colA, colB = st.columns(2)

    with colA:
        fig1 = px.pie(filtered_df, names="risk_level")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.scatter(filtered_df, x="loan_amount", y="credit_score", color="risk_level")
        st.plotly_chart(fig2, use_container_width=True)

    # Sandbox
    st.subheader("🧪 Sandbox Simulation")

    loan = st.number_input("Loan Amount",1000,100000,50000)
    credit = st.number_input("Credit Score",300,850,600)
    dti = st.number_input("DTI",0.0,1.0,0.4)
    ip = st.number_input("IP Risk Score",0,100,70)
    doc = st.selectbox("Document Mismatch",[0,1])
    inc = st.number_input("Income Ratio",0.0,1.0,0.3)

    if st.button("Run Simulation"):
        sample = pd.DataFrame([{
            "loan_amount":loan,
            "credit_score":credit,
            "dti":dti,
            "ip_risk_score":ip,
            "has_document_mismatch":doc,
            "income_mismatch_ratio":inc
        }])

        for col in X.columns:
            if col not in sample:
                sample[col] = 0

        sample = sample[X.columns]

        risk = risk_agent(sample)
        comp = compliance_agent(doc, inc)
        decision = decision_agent(risk, comp)

        st.success(f"Risk: {risk}")
        st.info(f"Compliance: {comp}")
        st.warning(f"Decision: {decision}")

# ===============================
# 🤖 AI DECISION CENTER
# ===============================
elif page == "🤖 AI Decision Center":

    st.title("🤖 AI Decision Intelligence Center")
    st.markdown("### 🧾 Enter Customer Details")

    # FORM
    with st.form("ai_form"):

        col1, col2, col3 = st.columns(3)

        loan_amount = col1.number_input("Loan Amount", 1000, 100000, 20000)
        credit_score = col2.number_input("Credit Score", 300, 850, 650)
        dti = col3.number_input("DTI", 0.0, 1.0, 0.3)

        ip_risk = col1.number_input("IP Risk Score", 0, 100, 50)
        doc_mismatch = col2.selectbox("Has Document Mismatch", [0, 1])
        income_ratio = col3.number_input("Income Mismatch Ratio", 0.0, 1.0, 0.1)

        run = st.form_submit_button("🚀 Analyze Customer")

    # -------------------------------
    # EXTRA INTELLIGENCE FUNCTIONS
    # -------------------------------

    def reason_agent(data):
        reasons = []

        if data["credit_score"] < 600:
            reasons.append("Low credit score")

        if data["loan_amount"] > 50000:
            reasons.append("High loan amount")

        if data["dti"] > 0.4:
            reasons.append("High debt-to-income ratio")

        if data["ip_risk_score"] > 70:
            reasons.append("Suspicious IP risk")

        if data["has_document_mismatch"] == 1:
            reasons.append("Document mismatch")

        return reasons

    def risk_score(data):
        score = 0
        score += data["loan_amount"] / 1000
        score += (700 - data["credit_score"]) / 10
        score += data["dti"] * 100
        return round(score, 2)

    def recommendation_agent(risk, comp):
        if comp == "Violation":
            return "Block transaction and request document verification"
        elif risk == "High":
            return "Reject loan and flag for monitoring"
        elif risk == "Medium":
            return "Send to manual review team"
        else:
            return "Approve with standard monitoring"

    # -------------------------------
    # RUN MODEL
    # -------------------------------
    if run:

        sample = pd.DataFrame([{
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'dti': dti,
            'ip_risk_score': ip_risk,
            'has_document_mismatch': doc_mismatch,
            'income_mismatch_ratio': income_ratio
        }])

        for col in X.columns:
            if col not in sample.columns:
                sample[col] = 0

        sample = sample[X.columns]

        # CORE OUTPUTS
        risk = risk_agent(sample)
        comp = compliance_agent(doc_mismatch, income_ratio)
        decision = decision_agent(risk, comp)
        explanation = explanation_agent(risk)
        solution = solution_agent(risk, comp)

        # EXTRA OUTPUTS
        score = risk_score(sample.iloc[0])
        reasons = reason_agent(sample.iloc[0])
        recommendation = recommendation_agent(risk, comp)

        # -------------------------------
        # DISPLAY
        # -------------------------------
        st.markdown("## 📊 AI Analysis Result")

        col1, col2, col3, col4 = st.columns(4)

        col1.success(f"🧠 Risk: {risk}")
        col2.info(f"⚖️ Compliance: {comp}")
        col3.warning(f"🎯 Decision: {decision}")
        col4.metric("📊 Risk Score", score)

        st.markdown("---")

        # WHY
        st.subheader("🔍 Why this decision?")
        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("No major risk factors detected")

        # EXPLANATION
        st.subheader("💡 Explanation")
        st.write(explanation)

        # RECOMMENDATION
        st.subheader("🛠 Recommended Action")
        st.write(recommendation)

        # SOLUTION
        st.subheader("📌 Solution Strategy")
        st.write(solution)

        # ALERT
        st.subheader("🚨 Risk Alert System")

        if risk == "High" and doc_mismatch == 1:
            st.error("🚨 High Probability Fraud Detected!")
        elif risk == "High":
            st.warning("⚠️ High Risk Customer")
        else:
            st.success("✅ Customer appears safe")

# ===============================
# 🧾 MANUAL ENTRY
# ===============================
elif page == "🧾 Manual Data Entry":

    st.title("🧾 Manual Entry")

    if "data" not in st.session_state:
        st.session_state.data=[]

    loan = st.number_input("Loan",1000,100000,20000)
    credit = st.number_input("Credit",300,850,650)
    dti = st.number_input("DTI",0.0,1.0,0.3)
    ip = st.number_input("IP Risk",0,100,50)
    doc = st.selectbox("Doc",[0,1])
    inc = st.number_input("Income Ratio",0.0,1.0,0.1)

    if st.button("Add"):
        st.session_state.data.append({
            "loan_amount":loan,
            "credit_score":credit,
            "dti":dti,
            "ip_risk_score":ip,
            "has_document_mismatch":doc,
            "income_mismatch_ratio":inc
        })

    if st.session_state.data:
        df_manual = pd.DataFrame(st.session_state.data)
        st.dataframe(df_manual)

        if st.button("Predict All"):
            for col in X.columns:
                if col not in df_manual: df_manual[col]=0
            df_manual = df_manual[X.columns]

            preds = model.predict(df_manual)
            df_manual["risk"] = ["Low" if p==0 else "Medium" if p==1 else "High" for p in preds]

            st.dataframe(df_manual)