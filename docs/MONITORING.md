# Monitoreo de producción

Cada predicción nueva debe indicar su origen:

```json
{
  "prediction_source": "production"
}
```

Valores permitidos: `production`, `manual` y `system_test`. Si se omite, la
API utiliza `manual` para evitar que pruebas de Thunder/Postman contaminen las
métricas productivas. `pruebas_sistema.py` utiliza `system_test`.

## Consultar el reporte

```text
GET http://localhost:8000/monitoring?source=production
```

O localmente:

```powershell
python -m scripts.monitorear_produccion --source production
```

El monitor espera al menos 30 predicciones antes de emitir alertas. Reporta
distribución de riesgos, confianza, versión del modelo, PSI numérico, deriva
categórica, categorías desconocidas y cobertura de resultados reales.

## Registrar el resultado académico real

```text
POST http://localhost:8000/outcomes
```

```json
{
  "student_id": "EST-001",
  "actual_status": "Graduate"
}
```

Los registros históricos anteriores a esta mejora quedan identificados como
`legacy`; no se mezclan con producción porque no conservan la entrada completa.
