import streamlit as st
import pandas as pd
import pickle

# 1. Configuración de la página (esto profesionaliza el MVP)
st.set_page_config(page_title="Loan Approval System", page_icon="🏦")

# 2. Carga del modelo (Manejo de errores básico)
@st.cache_resource
def load_model():
    try:
        with open("modelo_svc.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'modelo_svc.pkl'. Entrena el modelo primero.")
        return None

model = load_model()

# 3. Interfaz de Usuario (Limpia y directa)
st.title("🏦 Sistema de Aprobación de Préstamos")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Género", ['Male', 'Female'])
    married = st.selectbox("¿Casado?", ['No', 'Yes'])
    dependents = st.selectbox("Personas a cargo", ['0', '1', '2', '3+'])
    education = st.selectbox("Nivel Educativo", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Autónomo", ['No', 'Yes'])

with col2:
    applicant_income = st.number_input("Ingresos Solicitante ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Ingresos Co-solicitante ($)", min_value=0, value=0)
    loan_amount = st.number_input("Monto del Préstamo (k$)", min_value=0, value=150)
    loan_term = st.selectbox("Plazo del Préstamo", ['Corto Plazo', 'Medio Plazo', 'Largo Plazo'])
    property_area = st.selectbox("Zona de Propiedad", ['Urban', 'Rural', 'Semiurban'])
    credit_history = st.selectbox("Historial Crediticio", ['Buen historial', 'Mal historial'])

# 4. Procesamiento de entrada
if st.button("📊 Analizar Solicitud", use_container_width=True):
    if model is not None:
        # Mapeos necesarios para el modelo
        ch_map = {'Buen historial': 1.0, 'Mal historial': 0.0}
        lt_map = {'Corto Plazo': 0, 'Medio Plazo': 1, 'Largo Plazo': 2}
        
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
        
        st.markdown("### Resultado del Análisis:")
        if prediction[0] == 1:
            st.success("✅ EL PRÉSTAMO HA SIDO APROBADO")
        else:
            st.error("❌ EL PRÉSTAMO HA SIDO RECHAZADO")
    else:
        st.warning("El modelo no está disponible.")