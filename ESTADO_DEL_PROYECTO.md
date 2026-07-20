# 📊 Estado del Proyecto: Sistema Inteligente de Riesgo Académico

¡Hola equipo! (Harumi, Shamir, Denilson y Renzo). 

Este documento es una bitácora detallada para que todos estemos alineados sobre **qué hemos avanzado hasta el momento**, **cómo funciona lo que se ha construido** y **qué tareas faltan** para completar los entregables finales (5, 6, 7 y 8) del curso de Software Inteligente.

---

## ✅ LO QUE YA HEMOS AVANZADO (Completado)

Hasta el momento, ya tenemos lista la parte más técnica de Inteligencia Artificial y la estructura base del sistema.

### 1. Preparación de Datos (`preparacion_datos.py`)
- **¿Qué hace?** Lee el dataset `dataset.csv` (descargado de Kaggle) que contiene variables socioeconómicas y académicas de estudiantes. Limpia valores nulos, transforma la variable objetivo de texto a números (0=Dropout, 1=Enrolled, 2=Graduate) mediante *LabelEncoder* y normaliza todas las características usando *StandardScaler* para que la Red Neuronal pueda procesarlas.
- **¿Cómo se relaciona con el proyecto?** Cumple con la definición de Entradas de nuestro "Entregable 2".

### 2. Entrenamiento de la Red Neuronal y MLflow (`entrenamiento_modelo.py`)
- **¿Qué hace?** Construye una Red Neuronal Multicapa (MLP) utilizando **TensorFlow y Keras**. Consiste en una capa de entrada de 64 neuronas, una oculta de 32 y una salida de 3 neuronas con activación *Softmax* para clasificación multiclase. 
- **Logro destacado:** El modelo ya fue entrenado con éxito logrando una precisión superior al **77%**.
- **Gestión (MLflow):** El script usa MLflow con una base de datos local SQLite (`mlflow.db`) para guardar el historial de hiperparámetros y la precisión alcanzada, cumpliendo con la exigencia tecnológica de nuestro "Entregable 4".
- **Artefactos generados:** El modelo final, el escalador y el codificador quedaron guardados en la carpeta `models/`.

### 3. Arquitectura para la Automatización (`api.py`, `Dockerfile`, `docker-compose.yml`)
- **¿Qué hace?** Como **n8n** no ejecuta Python directamente de forma nativa/fácil, se construyó una pequeña API utilizando **FastAPI** (`api.py`). Esta API carga el modelo de la carpeta `models/` y abre un puerto (8000) para recibir peticiones web de predicción.
- Se ha configurado Docker (`docker-compose.yml`) para que al ejecutar un solo comando se levanten dos servidores simultáneamente:
  1. El servidor con la API de nuestra Red Neuronal (`api_inteligente`).
  2. El servidor del orquestador **n8n**.
- *Nota: Todo este bloque de código ya está escrito y comiteado en la rama `main` de GitHub, pero aún no se ha ejecutado.*

---

## 🚀 LO QUE FALTA POR HACER (Siguientes Pasos)

A partir de este punto, el equipo debe enfocarse en levantar los servicios y realizar el diseño de los flujos de automatización visual.

### PASO 3: Levantar los contenedores (Inmediato)
Cualquier integrante que haya clonado el repositorio debe tener Docker Desktop abierto y ejecutar en su terminal:
```bash
docker-compose up -d --build
```
Esto encenderá n8n en el puerto `5678` y la API en el puerto `8000`.

### PASO 4: Diseñar el Flujo en n8n (Entregable 5)
Esta es la tarea principal restante. 
1. **Entrar a n8n:** Abren `http://localhost:5678` en su navegador.
2. **Nodo Inicio (Trigger):** Pueden usar un nodo de tipo *Webhook* para simular que un sistema de matrícula envía la información de un estudiante nuevo.
3. **Nodo HTTP Request:** Debe configurarse para enviar un POST a `http://api_inteligente:8000/predict` con los datos del estudiante en formato JSON.
4. **Nodo Switch:** Recibe la respuesta de la API. Debe evaluar la variable `nivel_riesgo` ("Alto", "Medio", "Bajo").
5. **Nodos de Acción (Reglas de negocio):**
   - Si es "Alto" (Dropout): Usar nodo de Gmail/SendGrid para enviar una alerta al correo del Tutor.
   - Si es "Medio" (Enrolled): Insertar un registro en una hoja de Google Sheets / SQLite programando seguimiento.
   - Si es "Bajo" (Graduate): Solo registrar el historial de éxito.

### PASO 5: Pruebas y Validación (Entregables 6 y 7)
- **Extraer Métricas:** Abrir MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) y documentar gráficamente los resultados de precisión (*accuracy*) y pérdida (*loss*).
- **Matriz de Confusión:** Programar un pequeño script extra (o agregarlo al de entrenamiento) para visualizar y poner en el informe cuántos Dropouts predijo correctamente vs cuántos falló.
- **Evidencias n8n:** Hacer capturas de pantalla de ejecuciones exitosas (en verde) del flujo de n8n, mostrando cómo llegan los correos de alerta. 

### PASO 6: Documento Final y Defensa (Entregable 8)
- Pegar las capturas de arquitectura, MLflow y n8n en el documento PDF Final.
- Repartir las partes para la exposición oral. Deben justificar por qué usamos un MLP (clasificación multiclase), por qué normalizamos los datos y cómo el Docker Compose solucionó la comunicación entre la red neuronal y n8n.

---
**¡Mucho éxito equipo! Ya tienen la base sólida terminada, ahora solo queda ensamblar la lógica del negocio visualmente.**
