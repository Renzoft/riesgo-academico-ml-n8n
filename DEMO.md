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
| Flujo n8n (4 ramas + salida por defecto) | 200 en todas |
| Caso de error | 422 con mensaje descriptivo |
| Panel del tutor (React) | Bandeja y detalle funcionando |
| Datos de demostración | 61 casos con nombre, carrera y ciclo |
| Correos de alerta en MailHog | 19 |

> **Importante:** el sistema corre con el modelo *legacy* (34 variables,
> `StandardScaler`, accuracy 0.7605). El modelo nuevo con One-Hot Encoding
> (231 variables, accuracy 0.7786) está empaquetado en
> `models/releases/preprocessing-v2-97984f56/` pero **no activado**. Ver
> [Activar el modelo nuevo](#activar-el-modelo-nuevo-opcional).

---

## Arquitectura en ejecución

```
Navegador                          Docker Compose
─────────                          ──────────────
Panel React  ──HTTP :8000──┐
(Vite :5173)               │   ┌──────────────────┐
                           ├──▶│ api_inteligente  │──SMTP :1025──▶ mailhog
Sistema externo            │   │  FastAPI + Keras │                :8025 web
  ──HTTP :5678──▶ n8n ─────┘   │  SQLite          │
                  :5678        └──────────────────┘
                                        ▲
                              models/ (solo lectura)
                                        │
                              mlflow_tracking :5001
```

El panel y n8n consumen la **misma API**. No hay un backend aparte para el panel.

---

## Requisitos previos

- **Docker Desktop** abierto y corriendo.
- **`dataset.csv`** en la raíz del proyecto (no está en el repositorio).
- **Modelo entrenado** en `models/`: `modelo_estudiantes.keras`, `scaler.pkl`,
  `encoder.pkl`, `feature_names.json`.
- **Node 18, 20 o 22** para el panel (no Node 24).
- **Workflow importado y publicado** en n8n.

No se necesita Python instalado. Todos los scripts se ejecutan dentro de
contenedores, lo que evita problemas de versión (TensorFlow no soporta
Python 3.14).

---

## Preparación (antes de la exposición, no en vivo)

### 1. Levantar los servicios

```bash
cd /Volumes/T7Isao/UNMSM2/ciclo-9/software-inteligente/riesgo-academico-ml-n8n

docker compose up -d --build
```

La primera vez tarda varios minutos porque instala TensorFlow. Después basta
`docker compose up -d`.

> Usa siempre `-d`. Sin esa opción los contenedores quedan atados a la terminal
> y se detienen al cerrarla o al pulsar Ctrl+C.

### 2. Verificar que respondió todo

```bash
docker compose ps
curl -s http://localhost:8000/health
```

Debe mostrar cuatro contenedores arriba y `"model_loaded": true`.

### 3. Comprobar la comunicación entre contenedores

```bash
docker exec n8n_orquestador wget -qO- http://api_inteligente:8000/health
```

Demuestra que n8n alcanza la API por el **nombre del servicio**. Si falla, el
flujo no funcionará.

### 4. Publicar el workflow en n8n

Solo la primera vez. El volumen `n8n_data` lo conserva entre reinicios.

1. Abrir <http://localhost:5678> y crear la cuenta local.
2. **Workflows** → **Create workflow** → menú `...` → **Import from File**.
3. Seleccionar `n8n/workflow_riesgo_academico_completo.json`.
4. Verificar que sean **13 nodos** y que *Predecir riesgo* apunte a
   `http://api_inteligente:8000/predict`.
5. Pulsar **Publish**.

> Si al importar aparecen nodos terminados en `1` (`Webhook1`, `Predecir
> riesgo1`), significa que se añadieron sobre los existentes. Borra el workflow
> completo y vuelve a importarlo: esta versión de n8n **agrega** en lugar de
> reemplazar.

### 5. Levantar el panel

```bash
cd frontend
npm install      # solo la primera vez
npm run dev
```

Disponible en <http://localhost:5173>.

### 6. Dejar abiertas estas pestañas

| Pestaña | URL |
|---|---|
| Panel del tutor | <http://localhost:5173> |
| n8n | <http://localhost:5678> |
| MailHog | <http://localhost:8025> |
| API (Swagger) | <http://localhost:8000/docs> |
| MLflow | <http://localhost:5001> |

---

## Guion de la demostración

Duración estimada: 10 a 12 minutos.

### Acto 1 · La arquitectura (1 min)

```bash
docker compose ps
docker exec n8n_orquestador wget -qO- http://api_inteligente:8000/health
```

**Qué decir:** cuatro servicios levantados con un solo comando, todo software
libre y en local. n8n llega a la API usando `api_inteligente` como dirección, no
`localhost`, porque dentro de un contenedor `localhost` es el propio contenedor.
Docker Compose crea un DNS interno donde el nombre del servicio es el hostname.

### Acto 2 · El módulo inteligente (2 min)

Mostrar MLflow en <http://localhost:5001>: el experimento
`Sistema_Riesgo_Academico_MLP` con sus hiperparámetros y métricas.

Mostrar `reports/figures/matriz_confusion.png`.

**Qué decir:** red neuronal multicapa de 64 y 32 neuronas con ReLU y salida
Softmax de 3 clases. Exactitud del 76%. Distingue bien Dropout y Graduate, y se
confunde en Enrolled, que es el estado intermedio. Lo relevante es que la
precisión en Dropout es 0.89: cuando alerta, acierta casi nueve de cada diez
veces.

### Acto 3 · Trazabilidad (1 min)

```bash
curl -s http://localhost:8000/model-info | python3 -m json.tool
```

**Qué decir:** la API informa con qué entrenamiento se generó el modelo que está
sirviendo, y cada predicción guarda ese identificador. Copiar el `mlflow_ui` y
abrirlo: lleva al run exacto en MLflow.

### Acto 4 · La orquestación (3 min) — el núcleo del curso

Abrir n8n y mostrar el lienzo.

**Qué decir:** aquí viven las reglas de negocio. n8n recibe la petición, pide la
predicción a la API, y el nodo Switch decide combinando **dos variables**: el
nivel de riesgo y la confianza. Si el riesgo es alto pero la confianza no llega
a 0.60, el caso se deriva a revisión humana en lugar de alertar al tutor.

**Por qué el umbral solo en la rama Alto:** medido sobre 250 estudiantes, la
confianza mediana es 92% en Alto, 82% en Bajo y 60% en Medio. Un umbral global
habría derivado el 47.5% de los casos Medio a revisión manual sin ninguna acción
que filtrar; en la rama Alto solo afecta al 7.4%. El umbral filtra acciones, no
predicciones.

Ejecutar el flujo:

```bash
docker run --rm \
  --network riesgo-academico-ml-n8n_default \
  -v "$PWD":/work -w /work \
  -e API_URL=http://api_inteligente:8000 \
  -e N8N_WEBHOOK_URL=http://n8n:5678/webhook/riesgo-academico \
  riesgo-academico-ml-n8n-api_inteligente \
  python scripts/pruebas_sistema.py
```

Ir a n8n → **Executions** y mostrar las ejecuciones en verde.

### Acto 5 · El panel del tutor (3 min)

Abrir <http://localhost:5173>.

**Qué decir:** esto es lo que ve un tutor. No es un tablero de indicadores, es
una cola de trabajo. Cada fila muestra **por qué** el caso aparece, para poder
priorizar sin abrir uno por uno.

Filtrar por riesgo alto. Abrir un caso.

**Qué decir sobre el detalle:** el tutor no puede hacer nada con un 99.9%. Sí
puede hacer algo con "matriculó 6 cursos y no aprobó ninguno, y no está al día
con la matrícula". Eso le dice a quién llamar y de qué hablar.

Señalar que la confianza aparece en palabras y no como porcentaje, porque un
tutor no interpreta decimales y sugieren una precisión que el modelo no tiene.

Señalar la nota al pie: **los factores son indicadores de contexto, no una
atribución del modelo**. No se le atribuye a la red un razonamiento que no
realiza.

Pulsar **Marcar como contactado** y volver a la bandeja: el caso desaparece de
los pendientes.

### Acto 6 · La acción automática (1 min)

Abrir <http://localhost:8025>.

**Qué decir:** solo los casos de riesgo alto con confianza suficiente generaron
correo. Los de riesgo medio y bajo quedaron registrados sin alertar, y el caso
de confianza baja se derivó a revisión sin notificar al tutor. MailHog captura
los mensajes localmente, sin depender de Gmail ni de ningún servicio de pago.

### Acto 7 · Persistencia y errores (1 min)

```bash
curl -s "http://localhost:8000/history?limit=5"
```

Y desde Swagger, `POST /predict` con solo 3 valores en `features`:

```json
{ "student_id": "DEMO-ERROR", "features": [1.0, 2.0, 3.0] }
```

**Qué decir:** responde 422 indicando que esperaba 34 características y recibió
3. El sistema rechaza datos incompletos en lugar de inventar una predicción.

---

## Comandos de referencia

### Ciclo de vida

```bash
docker compose up -d --build     # tras cambios en el codigo
docker compose up -d             # uso normal
docker compose ps
docker logs api_inteligente --tail 30
docker compose down              # conserva datos y workflow
```

### Verificación

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model-info
curl -s "http://localhost:8000/history?limit=10"
curl -s http://localhost:8000/monitoring

docker exec n8n_orquestador wget -qO- http://api_inteligente:8000/health
```

### Endpoints del panel

```bash
curl -s "http://localhost:8000/panel/estudiantes?nivel_riesgo=Alto&orden=riesgo"
curl -s http://localhost:8000/panel/resumen
curl -s http://localhost:8000/panel/predicciones/1/explicacion
curl -s -X PATCH http://localhost:8000/panel/casos/1 \
  -H "Content-Type: application/json" \
  -d '{"estado":"contactado","responsable":"Tutor"}'
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

### Reentrenar

```bash
docker run --rm -v "$PWD":/work -w /work \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  riesgo-academico-ml-n8n-api_inteligente \
  python ml/training/entrenamiento_modelo.py
```

### Pruebas unitarias

```bash
docker run --rm -v "$PWD":/work -w /work \
  riesgo-academico-ml-n8n-api_inteligente \
  python -m pytest -q
```

### Regenerar los datos de demostración

Borra el historial y genera 60 casos a través del flujo de n8n, con
estudiantes del directorio para que aparezcan con nombre en el panel.

```bash
docker compose stop api_inteligente
mv app_data app_data.respaldo-$(date +%H%M)
docker compose up -d api_inteligente

docker run --rm --network riesgo-academico-ml-n8n_default \
  -v "$PWD":/work -w /work riesgo-academico-ml-n8n-api_inteligente python -u -c "
import csv, json, urllib.request
from collections import Counter
codigos=[x['codigo'] for x in csv.DictReader(open('data/directorio_estudiantes.csv', encoding='utf-8'))]
filas=list(csv.DictReader(open('dataset.csv', encoding='utf-8-sig')))
acciones=Counter()
for i,codigo in enumerate(codigos):
    fila=filas[(i*7) % len(filas)]
    feats=[float(v) for k,v in fila.items() if k!='Target']
    req=urllib.request.Request('http://n8n:5678/webhook/riesgo-academico',
        data=json.dumps({'student_id':codigo,'email_tutor':'tutor@universidad.local',
                         'prediction_source':'production','features':feats}).encode(),
        headers={'Content-Type':'application/json'})
    r=json.loads(urllib.request.urlopen(req, timeout=30).read())
    acciones[r.get('accion','?')]+=1
for a,t in acciones.most_common(): print(f'  {a:24} {t}')
"
```

> El workflow de n8n debe estar publicado antes de ejecutarlo.

---

## URLs del sistema

| Servicio | URL | Uso |
|---|---|---|
| Panel del tutor | <http://localhost:5173> | Bandeja y detalle de casos |
| API (Swagger) | <http://localhost:8000/docs> | Probar endpoints |
| n8n | <http://localhost:5678> | Panel de orquestación |
| MailHog | <http://localhost:8025> | Bandeja de alertas |
| MLflow | <http://localhost:5001> | Experimentos |

---

## Si algo falla

**Los contenedores no levantan.** Verificar que Docker Desktop esté abierto y
que los puertos estén libres:

```bash
lsof -i :8000 -i :5678 -i :8025 -i :5001
```

**`model_loaded` es `false`.** Faltan los artefactos en `models/`. Reentrenar.

**El flujo de n8n falla en "Predecir riesgo".** Revisar que la URL del nodo sea
`http://api_inteligente:8000/predict`. Si dice `127.0.0.1` o `localhost`, no
funcionará: dentro del contenedor esas direcciones apuntan al propio n8n.

**El webhook devuelve 404.** El workflow no está publicado.

**El panel no carga datos.** Abrir la consola del navegador. Si el error
menciona CORS, revisar que `CORS_ORIGINS` en `docker-compose.yml` incluya
`http://localhost:5173`. Si devuelve 401, la variable `PANEL_API_KEY` está
definida en el backend y falta `VITE_API_KEY` en `frontend/.env`.

**`npm run build` falla con "Cannot find native binding".** La plantilla de Vite
genera versiones que requieren Node 22. El proyecto fija Vite 6 y React 18 a
propósito. Si aparece, borrar y reinstalar:

```bash
cd frontend && rm -rf node_modules package-lock.json && npm install
```

**MLflow no abre.** En macOS el puerto 5000 lo ocupa AirPlay, por eso se mapea
al 5001.

**Empezar de cero.**

```bash
docker compose down -v
rm -rf app_data
```

---

## Activar el modelo nuevo (opcional)

El modelo con One-Hot Encoding está empaquetado pero no activo.

```bash
docker run --rm -v "$PWD":/work -w /work \
  riesgo-academico-ml-n8n-api_inteligente \
  python scripts/gestionar_release.py activate \
  --release-id preprocessing-v2-97984f56

docker compose restart api_inteligente
curl -s http://localhost:8000/health
```

Debe reportar `"preprocessing": "pipeline_v2"` y el `release_id` activo.

Para volver atrás:

```bash
docker run --rm -v "$PWD":/work -w /work \
  riesgo-academico-ml-n8n-api_inteligente \
  python scripts/gestionar_release.py rollback

docker compose restart api_inteligente
```

> **No probado en la demo.** Activarlo cambia el preprocesamiento de 34 a 231
> variables. Si se hace, hay que volver a ejecutar las pruebas completas antes
> de exponer.

---

## Limitaciones conocidas

Conviene tenerlas presentes por si surgen en la defensa.

- **El workflow envía el formato posicional** (`features`), que es el contrato
  antiguo. La validación por campos nombrados con rangos (`student_data`) existe
  en la API pero no se usa en el flujo actual.
- **El endpoint `/health` responde 200 aunque el modelo no cargue.** El
  `HEALTHCHECK` de Docker solo mira el código de estado, por lo que puede
  reportar el contenedor como sano cuando no puede predecir.
- **Los factores del panel son reglas, no atribución del modelo.** Se calculan
  sobre las variables del estudiante, no sobre el cálculo interno de la red. La
  atribución real (SHAP o importancia por permutación) queda pendiente.
- **El directorio de estudiantes es ficticio.** Simula el sistema académico
  institucional. En una implantación real se sustituye la función
  `buscar_por_codigos` de `core/directorio.py` por la llamada al sistema real.
- **El panel no tiene autenticación activa por defecto.** `PANEL_API_KEY` está
  vacía para facilitar el desarrollo. En un despliegue real debe definirse.
- **`/monitoring` requiere 30 predicciones** con origen `production` para
  generar el reporte de deriva.
- **La detección es temprana respecto al abandono, no al ingreso.** El modelo
  usa el rendimiento de los dos primeros semestres, así que no puede predecir
  antes de que el estudiante complete un año académico.
