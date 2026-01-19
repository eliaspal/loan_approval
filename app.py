import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Loan Approval System", page_icon="🏦")

# 2. Model Loading
@st.cache_resource
def load_model():
    try:
        return joblib.load("modelo_svc.pkl")
    except FileNotFoundError:
        st.error("Error: 'modelo_svc.pkl' file not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

# 3. User Interface
st.title("🏦 Loan Approval Prediction System")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Married?", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
    education = st.selectbox("Education Level", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self-Employed", ['No', 'Yes'])

with col2:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (In thousands of dollars)", min_value=0, value=150)
    loan_term = st.selectbox("Loan Term", ['Short Term', 'Medium Term', 'Long Term'])
    property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])
    credit_history = st.selectbox("Credit History", ['Good history', 'Poor history'])

# 4. Input Processing
if st.button("📊 Analyze Application", use_container_width=True):
    if model is not None:
        # Mapping for the model
        ch_map = {'Good history': 1.0, 'Poor history': 0.0}
        lt_map = {'Short Term': 0, 'Medium Term': 1, 'Long Term': 2}
        
        input_df = pd.DataFrame([{
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': lt_map[loan_term],
            'Credit_History': ch_map[credit_history],
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'Property_Area': property_area
        }])

        prediction = model.predict(input_df)
        
        st.markdown("### Analysis Result:")
        if prediction[0] == 1:
            st.success("✅ THE LOAN HAS BEEN APPROVED")
        else:
            st.error("❌ THE LOAN HAS BEEN REJECTED")
    else:
        st.warning("The model is not available.")

