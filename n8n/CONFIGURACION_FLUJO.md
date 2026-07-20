# Configuración rápida del workflow de n8n

El JSON incluido crea los tres primeros nodos:

1. Webhook.
2. HTTP Request hacia FastAPI.
3. Switch por nivel de riesgo.

Después de importarlo, agrega tres nodos HTTP Request conectados a las
salidas Alto, Medio y Bajo del Switch.

## Configuración común de los tres nodos

- Method: POST
- URL: `http://api_inteligente:8000/actions/execute`
- Send Body: activado
- Body Content Type: RAW
- Content Type: `application/json`

## Rama Alto

Body:

```javascript
={{ JSON.stringify({
  prediction_id: $json.prediction_id,
  student_id: $json.student_id,
  email_tutor: $('Webhook').item.json.body.email_tutor || 'tutor@universidad.local',
  estado_predicho: $json.estado_predicho,
  nivel_riesgo: $json.nivel_riesgo,
  confianza: $json.confianza,
  accion: 'alerta_correo'
}) }}
```

Nombre sugerido: `Enviar alerta local`

## Rama Medio

Body:

```javascript
={{ JSON.stringify({
  prediction_id: $json.prediction_id,
  student_id: $json.student_id,
  email_tutor: $('Webhook').item.json.body.email_tutor || 'tutor@universidad.local',
  estado_predicho: $json.estado_predicho,
  nivel_riesgo: $json.nivel_riesgo,
  confianza: $json.confianza,
  accion: 'programar_seguimiento'
}) }}
```

Nombre sugerido: `Programar seguimiento`

## Rama Bajo

Body:

```javascript
={{ JSON.stringify({
  prediction_id: $json.prediction_id,
  student_id: $json.student_id,
  email_tutor: $('Webhook').item.json.body.email_tutor || 'tutor@universidad.local',
  estado_predicho: $json.estado_predicho,
  nivel_riesgo: $json.nivel_riesgo,
  confianza: $json.confianza,
  accion: 'registrar_exito'
}) }}
```

Nombre sugerido: `Registrar riesgo bajo`

## Respuesta

Agrega un nodo `Respond to Webhook` y conecta a él los tres nodos anteriores.

- Respond With: JSON
- Response Body: `={{ $json }}`

Activa el workflow y exporta la versión terminada como:

`n8n/workflow_riesgo_academico.json`
