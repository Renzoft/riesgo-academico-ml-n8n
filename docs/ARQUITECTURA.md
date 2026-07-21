# Arquitectura del proyecto

El proyecto usa capas con dependencias explícitas y conserva los contratos
externos de Docker, n8n, SMTP y MLflow.

## Capas

```text
api/             FastAPI, esquemas y composición HTTP
core/            calidad, preparación y monitoreo de datos
infrastructure/  persistencia de releases y adaptadores externos
ml/pipeline/     preprocesamiento canónico
ml/models/       definición de la red MLP
ml/training/     entrenamiento y registro de experimentos
ml/evaluation/   evaluación, validación cruzada y comparación
tracking/        manifiestos y trazabilidad MLflow
scripts/         comandos operativos y pruebas integrales
```

Los artefactos entrenados continúan en `models/`, los reportes en `reports/`
y el workflow público en `n8n/workflow_riesgo_academico_completo.json`.

## Compatibilidad

- Docker conserva `uvicorn api:app` y el puerto 8000.
- n8n conserva las rutas `/predict` y `/actions/execute`.
- SMTP conserva `SMTP_HOST`, `SMTP_PORT` y `SENDER_EMAIL`.
- MLflow conserva `MLFLOW_TRACKING_URI` y el experimento
  `Sistema_Riesgo_Academico_MLP`.
- Los archivos raíz de preparación, preprocesamiento, entrenamiento y pruebas
  son fachadas; la lógica reside una sola vez dentro de su capa.

## Dirección de dependencias

```text
api -> core, infrastructure
ml/training -> core, ml/pipeline, ml/models, tracking
ml/evaluation -> core, ml/pipeline, ml/models
scripts -> core, infrastructure
```

`core` no debe depender de FastAPI, Docker ni scripts operativos.

## Comandos

```powershell
python entrenamiento_modelo.py
python pruebas_sistema.py
python -m ml.training.entrenamiento_modelo
python -m ml.evaluation.evaluacion_avanzada
python -m ml.evaluation.evaluacion_cv_mlp
python -m ml.evaluation.comparar_modelos
python -m scripts.monitorear_produccion --source production
python -m scripts.gestionar_release --help
```
