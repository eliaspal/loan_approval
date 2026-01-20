import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sistema de Aprobación de Créditos",
    page_icon="🏦",
    layout="centered"
)

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("modelo_final.pkl")
    except FileNotFoundError:
        st.error("⚠️ Error: No se encuentra 'modelo_final.pkl'. Asegúrate de subir el archivo.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error cargando el modelo: {e}")
        return None

model = load_model()

# --- 3. FUNCIONES DE LÓGICA DE NEGOCIO ---

def preprocesar_datos(data):
    """
    Realiza la ingeniería de variables necesaria antes de pasar al modelo.
    Replica lo que hicimos en el notebook (Feature Engineering).
    """
    df = data.copy()
    
    # 1. Calcular Ingreso Total
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # 2. Calcular Ratio Deuda/Ingreso
    # Sumamos 1 al denominador por seguridad matemática
    df['Debt_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] + 1)
    
    return df

def sistema_hibrido(df_processed, modelo):
    """
    Aplica Reglas Duras (Hard Rules) + IA con Umbrales.
    """
    # Extraemos valores para reglas duras
    ratio = df_processed['Debt_Income_Ratio'].values[0]
    income = df_processed['Total_Income'].values[0]
    
    # --- FASE 1: HARD RULES (FILTRO PREVIO) ---
    if ratio > 0.60:
        return "RECHAZADO", "Riesgo Alto: El cliente destina más del 60% de su ingreso a deuda.", "hard_reject"
    
    if income < 2500:
        return "RECHAZADO", "Riesgo Alto: Ingresos insuficientes para la política del banco.", "hard_reject"

    # --- FASE 2: MODELO DE IA ---
    # Obtenemos la probabilidad de la clase 1 (Aprobado)
    try:
        probabilidad = modelo.predict_proba(df_processed)[:, 1][0]
    except Exception as e:
        return "ERROR", f"Fallo en predicción del modelo: {e}", "error"

    # --- FASE 3: ZONA GRIS ---
    limite_inferior = 0.45
    limite_superior = 0.60
    
    if probabilidad >= limite_superior:
        return "APROBADO", f"Score IA sólido ({probabilidad:.2%}). Cliente confiable.", "success"
    
    elif probabilidad < limite_inferior:
        return "RECHAZADO", f"Score IA bajo ({probabilidad:.2%}). Perfil de riesgo detectado.", "error"
    
    else:
        return "REVISIÓN MANUAL", f"Zona Gris ({probabilidad:.2%}). El modelo duda, requiere analista humano.", "warning"

# --- 4. INTERFAZ DE USUARIO ---

st.title("🏦 Sistema Inteligente de Riesgo Crediticio")
st.markdown("Evaluación basada en **Capacidad de Pago** y **Perfil Demográfico** (Sin Historial Crediticio).")
st.markdown("---")

# Formulario de entrada
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Datos Personales")
        gender = st.selectbox("Género", ['Male', 'Female'])
        married = st.selectbox("Estado Civil", ['No', 'Yes'], help="Yes = Casado")
        dependents = st.selectbox("Dependientes", ['0', '1', '2', '3+'])
        education = st.selectbox("Nivel Educativo", ['Graduate', 'Not Graduate'])
        self_employed = st.selectbox("Trabajador Independiente", ['No', 'Yes'])
        property_area = st.selectbox("Zona de Propiedad", ['Urban', 'Rural', 'Semiurban'])

    with col2:
        st.subheader("💰 Datos Financieros")
        applicant_income = st.number_input("Ingreso Solicitante ($)", min_value=0, value=4000, step=100)
        coapplicant_income = st.number_input("Ingreso Co-Solicitante ($)", min_value=0, value=0, step=100)
        loan_amount = st.number_input("Monto del Préstamo (en miles)", min_value=0, value=120)
        
        # Mapeo del Loan Term para que coincida con el entrenamiento (0, 1, 2)
        loan_term_display = st.selectbox("Plazo del Préstamo", ['Corto (< 6 meses)', 'Medio (6-12 meses)', 'Largo (> 12 meses)'])
        
        # Mapeo interno para el modelo
        lt_map = {'Corto (< 6 meses)': 0, 'Medio (6-12 meses)': 1, 'Largo (> 12 meses)': 2}

    submitted = st.form_submit_button("📊 Analizar Riesgo", use_container_width=True)

# --- 5. LÓGICA DE EJECUCIÓN ---

if submitted:
    if model is None:
        st.error("No se puede realizar el análisis porque el modelo no está cargado.")
    else:
        raw_data = pd.DataFrame([{
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': lt_map[loan_term_display], # Pasamos el número (0, 1 o 2)
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'Property_Area': property_area
        }])

        # 2. Ingeniería de Variables (Calculamos Ratio, Total Income, etc.)
        processed_data = preprocesar_datos(raw_data)
        
        # Mostrar métricas financieras calculadas (para transparencia)
        st.markdown("### 🔍 Métricas Financieras Calculadas")
        m1, m2 = st.columns(2)
        m1.metric("Ingreso Total Familiar", f"${processed_data['Total_Income'].values[0]:,.0f}")
        
        ratio = processed_data['Debt_Income_Ratio'].values[0]
        m2.metric("Ratio Deuda/Ingreso", f"{ratio:.1%}", delta_color="inverse" if ratio > 0.6 else "normal")

        # 3. Predicción con Sistema híbrido
        st.markdown("---")
        decision, mensaje, estado = sistema_hibrido(processed_data, model)
        
        if estado == "success":
            st.success(f"✅ {decision}")
            st.info(mensaje)
        elif estado == "error": # Rechazo
            st.error(f"❌ {decision}")
            st.warning(mensaje)
        elif estado == "hard_reject": # Rechazo por Regla Dura
            st.error(f"⛔ {decision}")
            st.warning(mensaje)
        else: # Zona Gris
            st.warning(f"⚠️ {decision}")
            st.info(mensaje)



