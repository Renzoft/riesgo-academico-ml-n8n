import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import mlflow
import mlflow.keras

# Importamos la función de preparación de datos que creamos en el Paso 1
from preparacion_datos import preparar_datos

def entrenar_modelo():
    print("=== PASO 2: ENTRENAMIENTO DE LA RED NEURONAL CON MLFLOW ===")

    # 1. Cargar y preparar datos
    ruta_csv = "dataset.csv"
    X_train, X_test, y_train, y_test, encoder, scaler = preparar_datos(ruta_csv)

    # Guardar scaler y encoder para su posterior uso en la API / n8n
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    print("\n[+] Scaler y Encoder guardados en la carpeta 'models/'")

    # 2. Hiperparámetros del Modelo
    input_dim = X_train.shape[1]
    num_classes = len(encoder.classes_) # 3 clases (Dropout, Enrolled, Graduate)
    epochs = 30
    batch_size = 32
    learning_rate = 0.001

    # 3. Iniciar experimento en MLflow
    # Usamos SQLite como backend para evitar errores de rutas con espacios en Windows
    # y cumplir con la arquitectura definida en su proyecto (Uso de SQLite).
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Sistema_Riesgo_Academico_MLP")

    with mlflow.start_run():
        print("\n[+] Iniciando registro de experimento en MLflow...")

        # Construir la arquitectura de la Red Neuronal (Perceptrón Multicapa - MLP)
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2), # Prevenir sobreajuste (overfitting)
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax') # Capa de salida multiclase
        ])

        # Compilar el modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\n[+] Arquitectura del modelo:")
        model.summary()

        # Registrar hiperparámetros en MLflow
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("architecture", "MLP (64-32-3)")

        # Entrenar el modelo
        print("\n[+] Entrenando el modelo...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Evaluar en el conjunto de prueba
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n[✓] Evaluación final - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Registrar métricas finales en MLflow
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Guardar el modelo en formato .keras y registrarlo en MLflow
        modelo_path = "models/modelo_estudiantes.keras"
        model.save(modelo_path)
        print(f"[+] Modelo guardado en: {modelo_path}")

        mlflow.keras.log_model(model, artifact_path="model")
        print("[✓] Experimento y modelo registrados exitosamente en MLflow.")

if __name__ == "__main__":
    entrenar_modelo()
