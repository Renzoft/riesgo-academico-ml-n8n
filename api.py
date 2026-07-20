from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import os

app = FastAPI(
    title="API de Predicción de Riesgo Académico",
    description="API para conectar la red neuronal con n8n",
    version="1.0"
)

# Variables globales para el modelo y transformadores
MODEL_PATH = "models/modelo_estudiantes.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"

model = None
scaler = None
encoder = None

# Definir la estructura de los datos de entrada según el dataset
# Nota: Aquí resumimos a una lista de características por simplicidad de conexión con n8n.
class StudentData(BaseModel):
    features: list[float]

@app.on_event("startup")
async def load_model_and_transformers():
    global model, scaler, encoder
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            encoder = joblib.load(ENCODER_PATH)
            print("[+] Modelo, Scaler y Encoder cargados correctamente en la API.")
        else:
            print("[-] ADVERTENCIA: No se encontraron los archivos del modelo en /models.")
    except Exception as e:
        print(f"[-] Error cargando el modelo: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "message": "API de Predicción de Riesgo Académico Operativa"}

@app.post("/predict")
def predict_risk(data: StudentData):
    global model, scaler, encoder
    
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
    
    try:
        # Convertir a array de numpy y redimensionar
        input_data = np.array(data.features).reshape(1, -1)
        
        # 1. Normalizar los datos tal como se hizo en el entrenamiento
        input_scaled = scaler.transform(input_data)
        
        # 2. Hacer la predicción
        prediction_probs = model.predict(input_scaled)
        
        # 3. Obtener la clase con mayor probabilidad
        predicted_class_index = np.argmax(prediction_probs[0])
        
        # 4. Decodificar el número (0,1,2) a texto (Dropout, Enrolled, Graduate)
        predicted_class_text = encoder.inverse_transform([predicted_class_index])[0]
        
        # 5. Asignar nivel de riesgo según su regla de negocio
        riesgo = "Desconocido"
        if predicted_class_text == "Dropout":
            riesgo = "Alto"
        elif predicted_class_text == "Enrolled":
            riesgo = "Medio"
        elif predicted_class_text == "Graduate":
            riesgo = "Bajo"
            
        return {
            "estado_predicho": predicted_class_text,
            "nivel_riesgo": riesgo,
            "confianza": float(prediction_probs[0][predicted_class_index])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
