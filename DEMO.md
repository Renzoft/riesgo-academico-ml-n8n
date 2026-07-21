# Guía de Demostración del Sistema

Guion y comandos para la demostración en vivo del Sistema Inteligente de Detección
Temprana de Riesgo de Abandono Académico.

Todos los comandos de este documento fueron ejecutados y verificados en la rama
`denilson-rama`.

---

## Estado verificado

| Componente | Estado |
|---|---|
| Contenedores (API, n8n, MailHog, MLflow) | Funcionando |
| Modelo cargado | `standard_scaler_v1`, release `legacy` |
| Trazabilidad a MLflow | `run_id` en `/health`, `/model-info` y cada predicción |
| Flujo n8n (4 ramas + por defecto) | 200 en todas |
| Caso de error | 422 con mensaje descriptivo |
| Correo a MailHog | Entregado |
| MLflow | Cuarto servicio del compose, puerto 5001 |

> **Importante:** el sistema corre con el modelo *legacy* (34 variables,
> `StandardScaler`, accuracy 0.7605). El modelo nuevo con One-Hot Encoding
> (231 variables, accuracy 0.7786) está empaquetado en
> `models/releases/preprocessing-v2-97984f56/` pero **no activado**, porque no
> existe el archivo `models/active_release.json`. Ver la sección
> [Activar el modelo nuevo](#activar-el-modelo-nuevo-opcional).

---

## Requisitos previos

- **Docker Desktop** abierto y corriendo.
- **`dataset.csv`** en la raíz del proyecto (no está en el repositorio).
- **Modelo entrenado** en `models/`: `modelo_estudiantes.keras`, `scaler.pkl`,
  `encoder.pkl`, `feature_names.json`.
- **Workflow importado y publicado** en n8n.

No se necesita Python instalado en el equipo. Todos los scripts se ejecutan
dentro de contenedores, lo que evita problemas de versión (TensorFlow no
soporta Python 3.14).

---

## Preparación (hacer antes de la exposición, no en vivo)

### 1. Levantar el sistema

```bash
cd /Volumes/T7Isao/UNMSM2/ciclo-9/software-inteligente/riesgo-academico-ml-n8n

docker compose up -d --build
```

La primera vez tarda varios minutos porque instala TensorFlow. En arranques
posteriores usar `docker compose up -d` sin `--build`.

### 2. Verificar que todo respondió

```bash
docker compose ps
curl -s http://localhost:8000/health
```

Debe mostrar los tres contenedores arriba y `"model_loaded": true`.

### 3. Comprobar la comunicación entre contenedores

```bash
docker exec n8n_orquestador wget -qO- http://api_inteligente:8000/health
```

Este comando es importante: demuestra que n8n alcanza la API usando el **nombre
del servicio** dentro de la red de Docker. Si falla, el flujo no funcionará.

### 4. Publicar el workflow en n8n

Solo la primera vez. El volumen `n8n_data` lo conserva entre reinicios.

1. Abrir <http://localhost:5678> y crear la cuenta local.
2. **Workflows** → **Add workflow** → menú `...` → **Import from File**.
3. Seleccionar `n8n/workflow_riesgo_academico_completo.json`.
4. Verificar que el nodo *Predecir riesgo* apunte a
   `http://api_inteligente:8000/predict`.
5. Pulsar **Publish** y confirmar.

### 5. MLflow

Ya no requiere ningún comando aparte: es el cuarto servicio del `docker-compose`
y se levanta junto con los demás. Disponible en <http://localhost:5001>.

Se publica en el 5001 porque en macOS el puerto 5000 lo ocupa AirPlay.

### 6. Dejar abiertas estas pestañas

| Pestaña | URL |
|---|---|
| n8n | <http://localhost:5678> |
| MailHog | <http://localhost:8025> |
| API (Swagger) | <http://localhost:8000/docs> |
| MLflow | <http://localhost:5001> |

---

## Guion de la demostración

Duración estimada: 8 a 10 minutos.

### Acto 1 · La arquitectura (1 min)

```bash
docker compose ps
```

**Qué decir:** el sistema son tres servicios independientes levantados con un
solo comando. La API con la red neuronal, el orquestador n8n y un servidor de
correo local. Todo con software libre, sin licencias y sin salir de la máquina.

```bash
docker exec n8n_orquestador wget -qO- http://api_inteligente:8000/health
```

**Qué decir:** n8n llega a la API usando `api_inteligente` como dirección, no
`localhost`. Docker Compose crea un DNS interno donde el nombre del servicio es
el hostname. Esto fue lo que resolvió la comunicación entre el orquestador y el
modelo.

---

### Acto 2 · El módulo inteligente (2 min)

Mostrar MLflow en <http://localhost:5001>.

**Qué mostrar:** el experimento `Sistema_Riesgo_Academico_MLP`, los
hiperparámetros registrados y las métricas de cada corrida.

**Qué decir:** se usó una red neuronal multicapa de 64 y 32 neuronas con
activación ReLU y salida Softmax de 3 clases. MLflow registra cada entrenamiento
con sus parámetros y métricas, lo que permite comparar versiones y mantener
trazabilidad.

Mostrar también `reports/figures/matriz_confusion.png`.

**Qué decir:** la exactitud es 76%. El modelo distingue bien Dropout y Graduate,
y se confunde en Enrolled, que es el estado intermedio y ambiguo. Lo importante
para el propósito del sistema es que la precisión en Dropout es 0.89: cuando
alerta, acierta casi nueve de cada diez veces.

---

### Acto 3 · La API de predicción (1 min)

Abrir <http://localhost:8000/docs>.

```bash
curl -s http://localhost:8000/health
```

**Qué decir:** la API carga el modelo entrenado en memoria al arrancar y lo
expone por HTTP. Existe porque n8n no ejecuta Python de forma nativa; esta es la
pieza que conecta el orquestador con la red neuronal.

Desde Swagger, ejecutar `POST /predict` con un estudiante de ejemplo y mostrar
la respuesta: estado predicho, nivel de riesgo, confianza y las tres
probabilidades.

---

### Acto 4 · La orquestación (3 min) — el núcleo de la demo

Abrir n8n en <http://localhost:5678> y mostrar el lienzo del workflow.

**Qué decir:** aquí están las reglas de negocio. n8n recibe la petición, pide la
predicción a la API, evalúa el nivel de riesgo en el nodo Switch y decide qué
hacer. La API no conoce las reglas y n8n no conoce el modelo: son
responsabilidades separadas. Cambiar una regla no requiere tocar código.

Ejecutar el flujo completo:

```bash
docker run --rm \
  --network riesgo-academico-ml-n8n_default \
  -v "$PWD":/work -w /work \
  -e API_URL=http://api_inteligente:8000 \
  -e N8N_WEBHOOK_URL=http://n8n:5678/webhook/riesgo-academico \
  riesgo-academico-ml-n8n-api_inteligente \
  python scripts/pruebas_sistema.py
```

**Qué decir:** el script recorre el dataset buscando un estudiante de cada nivel
de riesgo y los envía al webhook. Las tres ramas responden 200.

Ir a n8n → pestaña **Executions** y mostrar las tres ejecuciones en verde.

---

### Acto 5 · La acción automática (1 min)

Abrir <http://localhost:8025> y mostrar el correo de alerta.

**Qué decir:** solo el caso de riesgo Alto generó correo al tutor. Los de riesgo
Medio y Bajo quedaron registrados sin alerta, que es exactamente lo que definen
las reglas. MailHog es un servidor de correo local: captura los mensajes sin
enviarlos a internet, lo que permite demostrar el envío sin depender de Gmail ni
de ningún servicio de pago.

---

### Acto 6 · Persistencia y manejo de errores (1 min)

```bash
curl -s "http://localhost:8000/history?limit=5"
```

**Qué decir:** cada ejecución deja dos registros en SQLite, uno de la predicción
y otro de la acción ejecutada, lo que permite reconstruir el recorrido completo
de cualquier caso.

Para el caso de error, desde Swagger enviar un `POST /predict` con solo 3
valores en `features`:

```json
{
  "student_id": "DEMO-ERROR",
  "features": [1.0, 2.0, 3.0]
}
```

**Qué decir:** la API responde 422 indicando que esperaba 34 características y
recibió 3. El sistema rechaza los datos incompletos en lugar de generar una
predicción sin sentido.

---

## Comandos de referencia

### Ciclo de vida

```bash
# Levantar (primera vez o tras cambios en el código)
docker compose up -d --build

# Levantar (uso normal)
docker compose up -d

# Ver estado
docker compose ps

# Ver logs de la API
docker logs api_inteligente --tail 30

# Bajar (conserva datos y el workflow de n8n)
docker compose down
```

### Verificación

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model-info
curl -s "http://localhost:8000/history?limit=10"
curl -s http://localhost:8000/monitoring

docker exec n8n_orquestador wget -qO- http://api_inteligente:8000/health
```

### Pruebas del sistema

```bash
docker run --rm \
  --network riesgo-academico-ml-n8n_default \
  -v "$PWD":/work -w /work \
  -e API_URL=http://api_inteligente:8000 \
  -e N8N_WEBHOOK_URL=http://n8n:5678/webhook/riesgo-academico \
  riesgo-academico-ml-n8n-api_inteligente \
  python scripts/pruebas_sistema.py
```

Los resultados quedan en `reports/metrics/resultados_pruebas_sistema.json`.

### Reentrenar el modelo

```bash
docker run --rm -v "$PWD":/work -w /work \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  riesgo-academico-ml-n8n-api_inteligente \
  python ml/training/entrenamiento_modelo.py
```

Genera los artefactos en `models/`, las métricas en `reports/metrics/` y las
figuras en `reports/figures/`.

### Pruebas unitarias

```bash
docker run --rm -v "$PWD":/work -w /work \
  riesgo-academico-ml-n8n-api_inteligente \
  python -m pytest -q
```

---

## URLs del sistema

| Servicio | URL | Uso |
|---|---|---|
| API (Swagger) | <http://localhost:8000/docs> | Probar endpoints |
| API (estado) | <http://localhost:8000/health> | Verificar el modelo |
| n8n | <http://localhost:5678> | Panel de orquestación |
| MailHog | <http://localhost:8025> | Bandeja de alertas |
| MLflow | <http://localhost:5001> | Experimentos |

---

## Si algo falla

**Los contenedores no levantan.** Verificar que Docker Desktop esté abierto y
que los puertos 8000, 5678, 1025 y 8025 estén libres:

```bash
lsof -i :8000 -i :5678 -i :8025
```

**`model_loaded` es `false`.** Faltan los artefactos en `models/`. Reentrenar con
el comando de la sección anterior.

**El flujo de n8n falla en "Predecir riesgo".** Revisar que la URL del nodo sea
`http://api_inteligente:8000/predict`. Si dice `127.0.0.1` o `localhost`, no
funcionará: dentro del contenedor esas direcciones apuntan al propio n8n.

**El webhook devuelve 404.** El workflow no está publicado. Pulsar **Publish** en
n8n.

**MLflow no abre.** En macOS el puerto 5000 lo usa AirPlay. Por eso se mapea al
5001. Se puede desactivar en Ajustes del Sistema → General → AirDrop y Handoff.

**Empezar de cero.** Elimina el historial de predicciones y el workflow de n8n:

```bash
docker compose down -v
rm -rf app_data
```

---

## Activar el modelo nuevo (opcional)

El modelo con One-Hot Encoding está empaquetado pero no activo. Para activarlo:

```bash
docker run --rm -v "$PWD":/work -w /work \
  riesgo-academico-ml-n8n-api_inteligente \
  python scripts/gestionar_release.py activate \
  --release-id preprocessing-v2-97984f56

docker compose restart api_inteligente
curl -s http://localhost:8000/health
```

Tras activarlo, `/health` debe reportar `"preprocessing": "pipeline_v2"` y
`"active_release": "preprocessing-v2-97984f56"`.

Para volver atrás:

```bash
docker run --rm -v "$PWD":/work -w /work \
  riesgo-academico-ml-n8n-api_inteligente \
  python scripts/gestionar_release.py rollback

docker compose restart api_inteligente
```

> **Advertencia:** este paso no se ha probado en la demo. Activarlo cambia el
> preprocesamiento de 34 a 231 variables. Si se hace, hay que volver a ejecutar
> las pruebas del sistema completas antes de la exposición.

---

## Limitaciones conocidas

Conviene tenerlas presentes por si surgen en la defensa:

- **El workflow de n8n envía el formato posicional** (`features`), que es el
  contrato antiguo. La validación por campos nombrados con rangos
  (`student_data`) existe en la API pero no se usa en el flujo actual.
- **El nodo Switch no tiene salida por defecto.** Si la API devolviera un nivel
  de riesgo inesperado, el ítem se descartaría en silencio y el webhook no
  respondería.
- **Los nodos de n8n no tienen manejo de errores** ni reintentos. Si la API está
  caída, el flujo se interrumpe sin aviso.
- **El endpoint `/health` responde 200 aunque el modelo no cargue.** El
  `HEALTHCHECK` de Docker solo mira el código de estado, por lo que puede
  reportar el contenedor como sano cuando no puede predecir.
- **El endpoint `/monitoring` requiere al menos 30 predicciones** con origen
  `production` para generar el reporte de deriva. Con menos, devuelve
  `insufficient_data`.
