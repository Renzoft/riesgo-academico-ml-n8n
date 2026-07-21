# Sistema Inteligente para la Detección Temprana de Estudiantes con Riesgo de Abandono Académico

La organización interna del proyecto se describe en
[`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md). La API vive en el paquete
`api/`; los comandos `preparacion_datos.py`, `entrenamiento_modelo.py` y
`pruebas_sistema.py` se mantienen como fachadas compatibles sin duplicar
lógica.

Este repositorio contiene el código fuente, modelos, pruebas y flujos del proyecto final del curso **Software Inteligente (2026-I)** de la Universidad Nacional Mayor de San Marcos (UNMSM).

El objetivo principal del sistema es identificar oportunamente a los estudiantes con mayor probabilidad de abandonar sus estudios o no culminarlos exitosamente. A partir de esta predicción, el sistema automatiza acciones de seguimiento y alertas tempranas vía correo electrónico para que los tutores o coordinadores académicos puedan intervenir de forma inmediata.

## Integrantes del Equipo

- Contreras Quispe Harumi
- Mantilla Flores Shamir
- Morales Mallqui Denilson Teofilo
- Munayco Vivanco Renzo Alexander

## Tecnologías Utilizadas

- **Lenguaje Base:** Python 3.11+
- **Machine Learning:** TensorFlow / Keras (Red Neuronal Multicapa - MLP), Scikit-Learn
- **Gestión del Modelo:** MLflow (con SQLite)
- **Automatización y Orquestación:** n8n
- **API y Backend:** FastAPI, Uvicorn
- **Simulador de Correos:** MailHog
- **Contenedores:** Docker & Docker Compose
- **Pruebas y Validación:** Pytest, Requests

## Arquitectura del Sistema Completado

1. **Módulo de Preparación de Datos:** Procesamiento de variables académicas, demográficas y socioeconómicas extraídas del dataset original, estandarizando los valores mediante `StandardScaler` y codificando la variable objetivo con `LabelEncoder`.
2. **Módulo Inteligente (MLP):** Red Neuronal Multicapa entrenada y guardada. Clasifica al estudiante en tres estados: `Dropout` (Riesgo Alto), `Enrolled` (Riesgo Medio) o `Graduate` (Riesgo Bajo).
3. **Módulo API REST:** Una interfaz expuesta a través de FastAPI que recibe los datos de un estudiante, realiza la predicción en tiempo real utilizando el modelo de TensorFlow cargado en memoria, y ejecuta acciones locales (como envío de correos vía SMTP).
4. **Módulo de Automatización (n8n):** Un orquestador visual que define las reglas de negocio. Escucha las peticiones entrantes, evalúa el nivel de riesgo predicho por la API y decide la ruta de acción (enviar alerta crítica, programar seguimiento o registrar éxito).
5. **Servidor SMTP (MailHog):** Captura de forma aislada los correos electrónicos de alerta generados por el sistema sin necesidad de usar un servidor de correo real.

---

## Instrucciones de Configuración y Uso

### 1. Preparación Inicial
Clona este repositorio. Es recomendable crear un entorno virtual e instalar las dependencias básicas para la ejecución de scripts locales y pruebas:

```bash
python -m venv .venv
source .venv/Scripts/activate  # En Windows Git Bash
pip install -r requirements.txt
```

### 2. Dataset y Entrenamiento (MLflow)
Descarga el dataset *"Higher Education Predictors of Student Retention"* de Kaggle, renómbralo a `dataset.csv` y colócalo en la raíz del proyecto.

Luego, ejecuta los scripts de preparación y entrenamiento:
```bash
python preparacion_datos.py
python entrenamiento_modelo.py
```
*Este proceso entrenará la red neuronal, registrará el historial de experimentos en una base de datos local de MLflow (`mlflow.db`), y guardará el modelo `.keras` en la carpeta `models/`.*

### 3. Levantamiento de la Arquitectura (Docker)
Todo el sistema está contenerizado. Asegúrate de tener **Docker Desktop** abierto y ejecuta:

```bash
docker-compose up -d --build
```
Una vez finalizado, los servicios estarán en línea:
- **API (FastAPI Swagger UI):** http://localhost:8000/docs
- **Orquestador (n8n):** http://localhost:5678
- **Bandeja de Correos (MailHog):** http://localhost:8025

### 4. Configuración del Orquestador
1. Ingresa a **n8n** (http://localhost:5678).
2. Crea una cuenta local (puedes usar datos ficticios).
3. Ve a "Workflows" > "Add Workflow".
4. En el menú superior derecho (`...`), selecciona **Import from File** y carga el archivo `n8n/workflow_riesgo_academico_completo.json`.
5. En la esquina superior derecha, haz clic en **Publish** (o cambia el switch a *Active*) para habilitar el webhook de producción.

### 5. Pruebas de Sistema y Simulación
Con los contenedores corriendo y el flujo de n8n activo, ejecuta el simulador de pruebas de sistema que buscará casos de prueba (riesgo Alto, Medio y Bajo) y los enviará al flujo:

```bash
python pruebas_sistema.py
```

**Validación Final:**
1. Los resultados detallados en JSON se guardarán en `reports/metrics/resultados_pruebas_sistema.json`.
2. Para el estudiante evaluado con **Riesgo Alto**, ve a la bandeja de **MailHog** (http://localhost:8025) y observarás el correo automático de alerta enviado al tutor encargado.

---
*Todos los entregables del proyecto (1 al 8) han sido completados e integrados satisfactoriamente en esta versión.*
