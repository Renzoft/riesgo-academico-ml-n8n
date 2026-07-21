# Publicación segura de modelos

El modelo activo continúa siendo la versión legada mientras no exista
`models/active_release.json`. Empaquetar una release no la activa.

## Validar una candidata

```powershell
python -m scripts.gestionar_release validate `
  --candidate-dir models/candidate_v2 `
  --candidate-metrics reports/candidate_v2/metricas_modelo.json
```

La validación comprueba artefactos, clases, dimensiones entre preprocesador y
modelo, y que no retrocedan accuracy, macro F1 ni recall de riesgo alto.

## Empaquetar sin activar

```powershell
python -m scripts.gestionar_release package `
  --candidate-dir models/candidate_v2 `
  --candidate-metrics reports/candidate_v2/metricas_modelo.json `
  --release-id preprocessing-v2-97984f56 `
  --run-id 97984f5628b8463a98b9ced76189e7cf
```

## Activar explícitamente

La activación cambia únicamente un puntero JSON mediante una operación
atómica. Se debe reconstruir o reiniciar la API después de activarla.

```powershell
python -m scripts.gestionar_release activate `
  --manifest models/releases/preprocessing-v2-97984f56/release.json
```

## Rollback

```powershell
python -m scripts.gestionar_release rollback
```

El rollback restaura el puntero anterior guardado en
`models/release_history`. Después se debe reiniciar la API.
