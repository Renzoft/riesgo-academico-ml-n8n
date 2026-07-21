# Ejecución rápida

## 1. Entorno

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. MLflow y entrenamiento

Terminal 1:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Terminal 2:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python preparacion_datos.py
python entrenamiento_modelo.py
```

Verifica que existan:

- `models/modelo_estudiantes.keras`
- `models/scaler.pkl`
- `models/encoder.pkl`
- `models/feature_names.json`
- `reports/metrics/metricas_modelo.json`
- `reports/figures/matriz_confusion.png`

## 3. MLflow local

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Abrir: http://localhost:5000

## 4. Docker

```powershell
docker compose down
docker compose up -d --build
docker compose ps
```

Servicios:

- API: http://localhost:8000/docs
- n8n: http://localhost:5678
- MailHog: http://localhost:8025
- MLflow: http://localhost:5000 (servidor local iniciado antes)

## 5. n8n

Importa `n8n/workflow_base_riesgo_academico.json` y completa las tres
acciones siguiendo `n8n/CONFIGURACION_FLUJO.md`.

## 6. Pruebas

```powershell
python -m pytest
python pruebas_sistema.py
```

## 7. Evidencias

Captura:

- Métricas de la terminal.
- Curvas de accuracy y loss.
- Matriz de confusión.
- Run de MLflow.
- Workflow completo de n8n.
- Ejecuciones verdes de las tres ramas.
- Correo en MailHog.
- Historial: http://localhost:8000/history
