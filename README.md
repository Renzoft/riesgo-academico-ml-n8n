# Sistema Inteligente para la Detección Temprana de Estudiantes con Riesgo de Abandono Académico

Este repositorio contiene el código fuente, modelos y flujos del proyecto final del curso **Software Inteligente (2026-I)** de la Universidad Nacional Mayor de San Marcos (UNMSM).

El objetivo principal del sistema es identificar oportunamente a los estudiantes con mayor probabilidad de abandonar sus estudios o no culminarlos exitosamente. A partir de esta predicción, el sistema automatiza acciones de seguimiento y alertas tempranas para que los tutores o coordinadores académicos puedan intervenir.

## Integrantes del Equipo

- Contreras Quispe Harumi
- Mantilla Flores Shamir
- Morales Mallqui Denilson Teofilo
- Munayco Vivanco Renzo Alexander

## Tecnologías Utilizadas

- **Lenguaje Base:** Python
- **Machine Learning:** TensorFlow / Keras (Red Neuronal Multicapa - MLP)
- **Gestión del Modelo:** MLflow
- **Automatización y Orquestación:** n8n
- **Base de Datos:** SQLite
- **Control de Versiones:** Git / GitHub

## Arquitectura del Sistema

1. **Módulo de Análisis de Datos:** Procesamiento de variables académicas, demográficas y socioeconómicas.
2. **Módulo Inteligente:** Red Neuronal (MLP) que clasifica al estudiante en tres posibles estados: `Dropout` (Riesgo Alto), `Enrolled` (Riesgo Medio) o `Graduate` (Riesgo Bajo).
3. **Módulo de Automatización:** Flujos de n8n que reciben las inferencias y ejecutan acciones programadas (ej. envío de correos, alertas a tutores, registro en base de datos).

## Instrucciones de Configuración y Uso

### 1. Preparación del Entorno
Clona este repositorio y asegúrate de tener Python instalado. Crea un entorno virtual e instala las dependencias:

```bash
python -m venv .venv
source .venv/Scripts/activate  # En Windows Git Bash
pip install pandas numpy scikit-learn tensorflow mlflow joblib
```

### 2. Dataset
Descarga el dataset *"Higher Education Predictors of Student Retention"* de Kaggle, renómbralo a `dataset.csv` y colócalo en la raíz del proyecto.

### 3. Ejecución de la Red Neuronal
Para preparar los datos y entrenar el modelo predictivo, ejecuta:
```bash
python entrenamiento_modelo.py
```
Este script normalizará los datos, entrenará la red neuronal, guardará el modelo en la carpeta `models/` y registrará las métricas.

Puedes visualizar los resultados del entrenamiento iniciando MLflow:
```bash
mlflow ui
```
*(Accede a http://localhost:5000 en tu navegador)*

### 4. Levantamiento del Sistema (API + n8n) con Docker
Para cumplir con la arquitectura técnica, el proyecto está contenerizado usando **Docker Compose**. Esto levantará simultáneamente la API de nuestra Red Neuronal (construida en FastAPI) y el orquestador n8n.

Asegúrate de tener Docker Desktop abierto y ejecuta en tu terminal:
```bash
docker-compose up -d --build
```

Una vez que termine, los servicios estarán disponibles en:
- **API (Swagger UI para probar predicciones):** http://localhost:8000/docs
- **n8n (Panel de orquestación):** http://localhost:5678

---
*Nota: El repositorio será actualizado conforme se agreguen las pruebas (Entregable 6 y 7).*
