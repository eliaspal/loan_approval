# 🏦 Loan Approval Prediction

## 📌 Índice
1. [📂 Descripción del Dataset](#-descripción-del-dataset)
2. [🎯 Objetivo del Proyecto](#-objetivo-del-proyecto)
3. [📊 Análisis Exploratorio](#-análisis-exploratorio)
4. [📈 Modelado y Entrenamiento](#-modelado-y-entrenamiento)
---

## 📂 Descripción del Dataset  
Este dataset contiene **10,000 registros y 14 columnas** con información sobre clientes de una entidad financiera.  
Su propósito es predecir si un cliente **obtendrá aprobación de un préstamo** basado en diferentes factores.

### 🔹 Variables Principales:
- **CreditScore** 📊: Puntaje de crédito del cliente.
- **Geography** 🌍: País donde reside el cliente.
- **Gender** 👥: Género del cliente (`Male`, `Female`).
- **Age** 🎂: Edad del cliente.
- **Balance** 💰: Saldo en cuenta bancaria.
- **NumOfProducts** 🛍️: Número de productos contratados.
- **HasCrCard** 💳: Si posee tarjeta de crédito (`1=Sí`, `0=No`).
- **IsActiveMember** 🔄: Si es un cliente activo.
- **EstimatedSalary** 💵: Salario estimado.
- **Exited** 🎯: Variable objetivo (`1=Cliente se va`, `0=Cliente permanece`).

---

## 🎯 Objetivo del Proyecto  
El problema de clasificación a resolver es predecir si un cliente **será aprobado o rechazado** para un préstamo.  
Esto permitirá a las entidades financieras **mejorar la toma de decisiones** y reducir riesgos de impago.

✔️ **Entender los factores que más influyen en la aprobación.**  
✔️ **Construir un modelo de Machine Learning para predecir el resultado.**  
✔️ **Evaluar el rendimiento del modelo con métricas de clasificación.**  

---

## 📊 Análisis Exploratorio  
Se realizaron los siguientes pasos para entender el dataset:

🔹 **Visualización de datos** con `matplotlib` y `seaborn`.  
🔹 **Distribución de variables** (edades, ingresos, saldo bancario).  
🔹 **Análisis de correlación** entre variables.  
🔹 **Manejo de valores nulos y outliers**.  

---

## 📈 Modelado y Entrenamiento  
Se probaron diferentes algoritmos de **Machine Learning** para predecir la aprobación de préstamos:

🔹 **Regresión Logística**  
🔹 **Random Forest** 🌲  
🔹 **SVC** 🚀  

Antes de entrenar el modelo, realizamos **preprocesamiento** de los datos:

1️⃣ **Manejo de valores nulos**  
2️⃣ **Codificación de variables categóricas**  
3️⃣ **Escalado de variables numéricas**  
