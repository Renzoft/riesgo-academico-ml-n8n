import json
import hashlib
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from core.preparacion_datos import preparar_datos
from ml.models.mlp import crear_modelo
from tracking.trazabilidad_mlflow import build_traceability_manifest


RANDOM_STATE = 42
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
CANDIDATE_DIR = MODELS_DIR / "candidate_v2"
CANDIDATE_REPORTS_DIR = REPORTS_DIR / "candidate_v2"
FIGURES_DIR = CANDIDATE_REPORTS_DIR / "figures"
MODEL_PATH = CANDIDATE_DIR / "modelo_estudiantes.keras"
PREPROCESSOR_PATH = CANDIDATE_DIR / "preprocessor.pkl"
ENCODER_PATH = CANDIDATE_DIR / "encoder.pkl"
FEATURE_NAMES_PATH = CANDIDATE_DIR / "feature_names.json"
TRANSFORMED_FEATURE_NAMES_PATH = (
    CANDIDATE_DIR / "transformed_feature_names.json"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)
MLFLOW_EXPERIMENT = "Sistema_Riesgo_Academico_MLP"


def configurar_semillas(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def guardar_curvas(history) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    accuracy_path = FIGURES_DIR / "curva_accuracy.png"
    loss_path = FIGURES_DIR / "curva_loss.png"

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy de entrenamiento y validación")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Pérdida de entrenamiento y validación")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    return accuracy_path, loss_path


def guardar_matriz_confusion(
    target_test,
    target_predicted,
    class_names: list[str],
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "matriz_confusion.png"

    matrix = confusion_matrix(
        target_test,
        target_predicted,
        labels=list(range(len(class_names))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(values_format="d", xticks_rotation=25)
    plt.title("Matriz de confusión - MLP")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def entrenar_modelo() -> None:
    print("=== ENTRENAMIENTO DE LA RED NEURONAL CON MLFLOW ===")

    configurar_semillas()

    (
        features_train,
        features_validation,
        features_test,
        target_train,
        target_validation,
        target_test,
        encoder,
        preprocessor,
        feature_names,
    ) = preparar_datos("dataset.csv")

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    FEATURE_NAMES_PATH.write_text(
        json.dumps(feature_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    transformed_feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )
    TRANSFORMED_FEATURE_NAMES_PATH.write_text(
        json.dumps(
            transformed_feature_names,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    input_dim = features_train.shape[1]
    num_classes = len(encoder.classes_)
    maximum_epochs = 80
    batch_size = 32
    learning_rate = 0.001

    class_indices = np.unique(target_train)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=target_train,
    )
    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(
            class_indices,
            class_weight_values,
        )
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    dataset_hash = hashlib.sha256(
        Path("dataset.csv").read_bytes()
    ).hexdigest()

    with mlflow.start_run(run_name="MLP_64_32_preprocessing_v2") as run:
        # Deja el enlace hacia MLflow junto a los artefactos, para que la API
        # pueda informar con que entrenamiento se genero cada prediccion.
        CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
        (CANDIDATE_DIR / "model_run.json").write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "experiment": MLFLOW_EXPERIMENT,
                    "run_name": "MLP_64_32_preprocessing_v2",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        model = crear_modelo(input_dim, num_classes)

        print("\nArquitectura del modelo:")
        model.summary()

        mlflow.log_params(
            {
                "architecture": "MLP_64_32_3",
                "hidden_layer_1": 64,
                "hidden_layer_2": 32,
                "dropout": 0.20,
                "maximum_epochs": maximum_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": "Adam",
                "input_features": input_dim,
                "original_features": len(feature_names),
                "preprocessing_version": "2",
                "categorical_encoding": "one_hot",
                "numeric_scaling": "standard_scaler",
                "missing_numeric": "median",
                "missing_categorical": "most_frequent",
                "dataset_sha256": dataset_hash,
                "class_weight_balancing": True,
                "random_state": RANDOM_STATE,
            }
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=MODEL_PATH,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        print("\nEntrenando modelo...")
        history = model.fit(
            features_train,
            target_train,
            validation_data=(
                features_validation,
                target_validation,
            ),
            epochs=maximum_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        # Carga el mejor checkpoint antes de evaluar el conjunto final.
        model = tf.keras.models.load_model(MODEL_PATH)

        probabilities = model.predict(features_test, verbose=0)
        target_predicted = np.argmax(probabilities, axis=1)

        test_loss, test_accuracy = model.evaluate(
            features_test,
            target_test,
            verbose=0,
        )

        macro_precision = precision_score(
            target_test,
            target_predicted,
            average="macro",
            zero_division=0,
        )
        macro_recall = recall_score(
            target_test,
            target_predicted,
            average="macro",
            zero_division=0,
        )
        macro_f1 = f1_score(
            target_test,
            target_predicted,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            target_test,
            target_predicted,
            average="weighted",
            zero_division=0,
        )

        class_names = encoder.classes_.tolist()
        report = classification_report(
            target_test,
            target_predicted,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        accuracy_path, loss_path = guardar_curvas(history)
        confusion_path = guardar_matriz_confusion(
            target_test,
            target_predicted,
            class_names,
        )

        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "epochs_executed": len(history.history["loss"]),
            "classes": class_names,
            "classification_report": report,
        }

        metrics_path = CANDIDATE_REPORTS_DIR / "metricas_modelo.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "epochs_executed": len(history.history["loss"]),
            }
        )

        mlflow.log_artifact(str(metrics_path), artifact_path="evaluacion")
        mlflow.log_artifact(str(accuracy_path), artifact_path="figuras")
        mlflow.log_artifact(str(loss_path), artifact_path="figuras")
        mlflow.log_artifact(str(confusion_path), artifact_path="figuras")
        mlflow.log_artifact(
            str(FEATURE_NAMES_PATH),
            artifact_path="preprocesamiento",
        )
        mlflow.log_artifact(
            str(TRANSFORMED_FEATURE_NAMES_PATH),
            artifact_path="preprocesamiento",
        )
        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="preprocesamiento",
        )
        mlflow.log_artifact(
            str(ENCODER_PATH),
            artifact_path="preprocesamiento",
        )
        quality_report_path = Path("reports/data_quality/latest.json")
        if quality_report_path.exists():
            mlflow.log_artifact(
                str(quality_report_path),
                artifact_path="calidad_datos",
            )
        traceability = build_traceability_manifest(
            dataset_path=Path("dataset.csv"),
            schema={
                "original_feature_names": feature_names,
                "transformed_feature_names": transformed_feature_names,
            },
            code_paths=[
                Path("ml/training/entrenamiento_modelo.py"),
                Path("core/preparacion_datos.py"),
                Path("ml/pipeline/preprocesamiento_pipeline.py"),
                Path("api/schemas.py"),
                Path("tracking/trazabilidad_mlflow.py"),
            ],
            artifact_roles={
                "model": str(MODEL_PATH),
                "preprocessor": str(PREPROCESSOR_PATH),
                "target_encoder": str(ENCODER_PATH),
                "original_feature_names": str(FEATURE_NAMES_PATH),
                "transformed_feature_names": str(
                    TRANSFORMED_FEATURE_NAMES_PATH
                ),
                "evaluation": str(metrics_path),
            },
        )
        mlflow.log_dict(
            traceability,
            "traceability/run_manifest.json",
        )
        mlflow.set_tags(
            {
                "model_stage": "candidate",
                "preprocessing_version": "2",
                "dataset_sha256": dataset_hash,
            }
        )

        # Compatibilidad entre versiones de MLflow.
        try:
            mlflow.keras.log_model(model, name="model")
        except TypeError:
            mlflow.keras.log_model(model, artifact_path="model")

        print("\n=== RESULTADO FINAL ===")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Épocas ejecutadas: {len(history.history['loss'])}")
        print(f"Modelo guardado en: {MODEL_PATH}")
        print(f"Métricas guardadas en: {metrics_path}")
        print(f"Matriz de confusión: {confusion_path}")


if __name__ == "__main__":
    entrenar_modelo()
