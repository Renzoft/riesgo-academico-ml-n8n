import json
import os
import smtplib
import sqlite3
from contextlib import asynccontextmanager
from email.message import EmailMessage
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/modelo_estudiantes.keras")
)
SCALER_PATH = Path(
    os.getenv("SCALER_PATH", "models/scaler.pkl")
)
ENCODER_PATH = Path(
    os.getenv("ENCODER_PATH", "models/encoder.pkl")
)
FEATURE_NAMES_PATH = Path(
    os.getenv("FEATURE_NAMES_PATH", "models/feature_names.json")
)
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", "app_data/riesgo_academico.db")
)

SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "1025"))
SENDER_EMAIL = os.getenv(
    "SENDER_EMAIL",
    "alertas@universidad.local",
)

model = None
scaler = None
encoder = None
feature_names: list[str] = []


class StudentData(BaseModel):
    student_id: str = Field(default="SIN-CODIGO", max_length=100)
    email_tutor: str = Field(
        default="tutor@universidad.local",
        max_length=200,
    )
    features: list[float]


class ActionData(BaseModel):
    prediction_id: int | None = None
    student_id: str = "SIN-CODIGO"
    email_tutor: str = "tutor@universidad.local"
    estado_predicho: str
    nivel_riesgo: str
    confianza: float
    accion: str


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                email_tutor TEXT,
                estado_predicho TEXT NOT NULL,
                nivel_riesgo TEXT NOT NULL,
                confianza REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                student_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_status TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            );
            """
        )
        connection.commit()


def load_artifacts() -> None:
    global model, scaler, encoder, feature_names

    required_paths = [
        MODEL_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        FEATURE_NAMES_PATH,
    ]
    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Faltan artefactos del modelo: "
            + ", ".join(missing_paths)
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_names = json.loads(
        FEATURE_NAMES_PATH.read_text(encoding="utf-8")
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()

    try:
        load_artifacts()
        print("[+] Modelo y transformadores cargados correctamente.")
    except Exception as error:
        print(f"[-] No se pudo cargar el modelo: {error}")

    yield


app = FastAPI(
    title="API de Predicción de Riesgo Académico",
    description="API local para conectar la red neuronal con n8n.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "API de Riesgo Académico operativa",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": all(
            artifact is not None
            for artifact in (model, scaler, encoder)
        ),
        "expected_features": (
            int(scaler.n_features_in_)
            if scaler is not None
            else None
        ),
    }


@app.get("/model-info")
def model_info():
    if scaler is None or encoder is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está disponible.",
        )

    return {
        "expected_features": int(scaler.n_features_in_),
        "feature_names": feature_names,
        "classes": encoder.classes_.tolist(),
    }


@app.post("/predict")
def predict_risk(data: StudentData):
    if model is None or scaler is None or encoder is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está cargado. Ejecuta primero "
                "python entrenamiento_modelo.py."
            ),
        )

    expected_features = int(scaler.n_features_in_)

    if len(data.features) != expected_features:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Se esperaban {expected_features} características, "
                f"pero se recibieron {len(data.features)}."
            ),
        )

    try:
        input_data = np.asarray(
            data.features,
            dtype=np.float64,
        ).reshape(1, -1)

        if not np.isfinite(input_data).all():
            raise ValueError(
                "Las características contienen NaN o infinitos."
            )

        input_scaled = scaler.transform(input_data)
        probabilities = model.predict(
            input_scaled,
            verbose=0,
        )[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = str(
            encoder.inverse_transform([predicted_index])[0]
        )

        risk_mapping = {
            "Dropout": "Alto",
            "Enrolled": "Medio",
            "Graduate": "Bajo",
        }
        risk_level = risk_mapping.get(
            predicted_class,
            "Desconocido",
        )
        confidence = float(probabilities[predicted_index])

        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    student_id,
                    email_tutor,
                    estado_predicho,
                    nivel_riesgo,
                    confianza
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    data.student_id,
                    data.email_tutor,
                    predicted_class,
                    risk_level,
                    confidence,
                ),
            )
            connection.commit()
            prediction_id = int(cursor.lastrowid)

        return {
            "prediction_id": prediction_id,
            "student_id": data.student_id,
            "estado_predicho": predicted_class,
            "nivel_riesgo": risk_level,
            "confianza": confidence,
            "probabilidades": {
                str(class_name): float(probability)
                for class_name, probability in zip(
                    encoder.classes_,
                    probabilities,
                )
            },
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


def send_local_email(action: ActionData) -> str:
    alert_path = Path("app_data/alertas.log")
    alert_path.parent.mkdir(parents=True, exist_ok=True)

    alert_content = "\n".join(
        [
            "========================================",
            "ALERTA DE RIESGO ACADÉMICO ALTO",
            f"Estudiante: {action.student_id}",
            f"Tutor: {action.email_tutor}",
            f"Estado predicho: {action.estado_predicho}",
            f"Nivel de riesgo: {action.nivel_riesgo}",
            f"Confianza: {action.confianza:.2%}",
            "Acción: iniciar seguimiento académico",
            "========================================",
            "",
        ]
    )

    with alert_path.open(
        "a",
        encoding="utf-8",
    ) as alert_file:
        alert_file.write(alert_content)

    return f"Alerta local registrada en {alert_path}"


@app.post("/actions/execute")
def execute_action(action: ActionData):
    allowed_actions = {
        "alerta_correo",
        "programar_seguimiento",
        "registrar_exito",
    }

    if action.accion not in allowed_actions:
        raise HTTPException(
            status_code=422,
            detail=(
                "Acción no válida. Valores permitidos: "
                + ", ".join(sorted(allowed_actions))
            ),
        )

    status = "success"

    try:
        if action.accion == "alerta_correo":
            details = send_local_email(action)
        elif action.accion == "programar_seguimiento":
            details = (
                "Caso registrado para seguimiento académico."
            )
        else:
            details = (
                "Predicción de riesgo bajo registrada sin alerta."
            )
    except Exception as error:
        status = "error"
        details = f"No se pudo ejecutar la acción: {error}"

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO actions (
                prediction_id,
                student_id,
                action_type,
                action_status,
                details
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                action.prediction_id,
                action.student_id,
                action.accion,
                status,
                details,
            ),
        )
        connection.commit()
        action_id = int(cursor.lastrowid)

    if status == "error":
        raise HTTPException(
            status_code=502,
            detail={
                "action_id": action_id,
                "message": details,
            },
        )

    return {
        "action_id": action_id,
        "status": status,
        "accion": action.accion,
        "details": details,
    }


@app.get("/history")
def history(limit: int = 50):
    safe_limit = min(max(limit, 1), 200)

    with get_connection() as connection:
        predictions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        ]

        actions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM actions
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        ]

    return {
        "predictions": predictions,
        "actions": actions,
    }
