# Contrato de entrada de la API

Las integraciones nuevas deben enviar las 34 características dentro de
`student_data` usando los nombres definidos en
`app/domain/feature_schema.py`. La API
construye internamente el vector en el orden exacto que espera el modelo; el
orden de las propiedades JSON no afecta la predicción.

El formato anterior `features: []` se conserva temporalmente para que los
workflows existentes de n8n sigan funcionando durante la migración. No se
permite enviar `student_data` y `features` en una misma petición.

El esquema completo, los campos obligatorios y sus restricciones se pueden
consultar en `http://localhost:8000/docs`.

## Ejemplo nombrado

```json
{
  "student_id": "EST-001",
  "email_tutor": "tutor@universidad.local",
  "student_data": {
    "marital_status": 1,
    "application_mode": 8,
    "application_order": 5,
    "course": 2,
    "daytime_evening_attendance": 1,
    "previous_qualification": 1,
    "nationality": 1,
    "mothers_qualification": 13,
    "fathers_qualification": 10,
    "mothers_occupation": 6,
    "fathers_occupation": 10,
    "displaced": 1,
    "educational_special_needs": 0,
    "debtor": 0,
    "tuition_fees_up_to_date": 1,
    "gender": 1,
    "scholarship_holder": 0,
    "age_at_enrollment": 20,
    "international": 0,
    "curricular_units_1st_sem_credited": 0,
    "curricular_units_1st_sem_enrolled": 0,
    "curricular_units_1st_sem_evaluations": 0,
    "curricular_units_1st_sem_approved": 0,
    "curricular_units_1st_sem_grade": 0,
    "curricular_units_1st_sem_without_evaluations": 0,
    "curricular_units_2nd_sem_credited": 0,
    "curricular_units_2nd_sem_enrolled": 0,
    "curricular_units_2nd_sem_evaluations": 0,
    "curricular_units_2nd_sem_approved": 0,
    "curricular_units_2nd_sem_grade": 0,
    "curricular_units_2nd_sem_without_evaluations": 0,
    "unemployment_rate": 10.8,
    "inflation_rate": 1.4,
    "gdp": 1.74
  }
}
```
