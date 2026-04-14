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

def preprocesar_datos(data, plazo_meses):
    """
    Realiza la ingeniería de variables necesaria antes de pasar al modelo.

    Mantiene la fórmula ORIGINAL de `Debt_Income_Ratio` (la que vio el modelo
    durante el entrenamiento). Además calcula un DTI "realista" dependiente
    del plazo, que se usa sólo para la UI y el hard rule.
    """
    df = data.copy()

    # 1. Ingreso mensual total del hogar
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # 2. Ratio Deuda/Ingreso tal como se entrenó el modelo
    #    (LoanAmount en miles / ingreso mensual — métrica del notebook).
    df['Debt_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] + 1)

    # 3. DTI realista para UI / hard rule: cuota mensual / ingreso mensual.
    cuota_mensual = (df['LoanAmount'] * 1000) / plazo_meses
    df['DTI_display'] = cuota_mensual / (df['Total_Income'] + 1)

    return df

def sistema_hibrido(df_processed, modelo):
    """
    Aplica Reglas Duras (Hard Rules) + IA con Umbrales.
    """
    # Extraemos valores para reglas duras
    # Usamos el DTI realista (dependiente del plazo) para el hard rule.
    ratio = df_processed['DTI_display'].values[0]
    income = df_processed['Total_Income'].values[0]

    # --- FASE 1: HARD RULES (FILTRO PREVIO) ---
    if ratio > 0.60:
        return "RECHAZADO", "Riesgo Alto: El cliente destina más del 60% de su ingreso a deuda.", "hard_reject"
    
    if income < 2500:
        return "RECHAZADO", "Riesgo Alto: Ingresos insuficientes para la política del banco.", "hard_reject"

    # --- FASE 2: MODELO DE IA ---
    # Obtenemos la probabilidad de la clase 1 (Aprobado)
    # El pipeline no utiliza DTI_display; lo quitamos para no ensuciar el input.
    X_model = df_processed.drop(columns=['DTI_display'], errors='ignore')
    try:
        probabilidad = modelo.predict_proba(X_model)[:, 1][0]
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
st.markdown("Evaluación de préstamos basada en **Capacidad de Pago** y **Perfil Demográfico** (Sin Historial Crediticio).")
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
        # Caps alineados a los usados en el preprocesado del notebook:
        #   ApplicantIncome -> p90 ≈ 7500
        #   CoapplicantIncome -> p95 ≈ 5700
        #   LoanAmount -> p90 ≈ 225 (en miles)
        # Defaults en torno a la mediana de entrenamiento.
        applicant_income = st.number_input(
            "Ingreso Solicitante mensual ($)",
            min_value=150, max_value=7500, value=3800, step=100
        )
        coapplicant_income = st.number_input(
            "Ingreso Co-Solicitante mensual ($)",
            min_value=0, max_value=5700, value=0, step=100
        )
        loan_amount = st.number_input(
            "Monto del Préstamo (en miles de $)",
            min_value=9, max_value=225, value=128, step=1,
            help="Rango entrenado: 9-225 (miles de $). Mediana 128 = 128.000 $. Ej: 128 equivale a 128.000 $."
        )

        # 3 buckets corto/medio/largo, interpretados como MESES de hipoteca
        # (coherente con el dataset canónico: umbrales 180 y 360 meses = 15 y 30 años).
        loan_term_display = st.selectbox(
            "Plazo del Préstamo",
            ['Corto (≤ 15 años)', 'Medio (15-30 años)', 'Largo (> 30 años)']
        )

        # Mapeo interno para el modelo (bucket 0/1/2 como en el notebook)
        lt_map = {'Corto (≤ 15 años)': 0, 'Medio (15-30 años)': 1, 'Largo (> 30 años)': 2}
        loan_term_bucket = lt_map[loan_term_display]

        # Plazo representativo en meses por bucket, para el cálculo de la cuota
        # mensual del hard rule.
        lt_months_map = {0: 120, 1: 240, 2: 360}  # 10, 20, 30 años
        loan_term_months = lt_months_map[loan_term_bucket]

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
            'Loan_Amount_Term': loan_term_bucket, # Bucket 0/1/2 que espera el modelo
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'Property_Area': property_area
        }])

        # 2. Ingeniería de Variables (Total Income y Ratio dependiente del plazo real)
        processed_data = preprocesar_datos(raw_data, plazo_meses=loan_term_months)
        
        # Mostrar métricas financieras calculadas (para transparencia)
        st.markdown("### 🔍 Métricas Financieras Calculadas")
        m1, m2 = st.columns(2)
        m1.metric("Ingreso Total Familiar", f"${processed_data['Total_Income'].values[0]:,.0f}")
        
        ratio = processed_data['DTI_display'].values[0]
        m2.metric("Ratio Cuota/Ingreso (según plazo)", f"{ratio:.1%}", delta_color="inverse" if ratio > 0.6 else "normal")

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



