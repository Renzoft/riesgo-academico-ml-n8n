import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from entrenamiento_modelo import configurar_semillas, crear_modelo
from calidad_datos import prepare_training_dataframe
from mlflow_traceability import build_traceability_manifest
from preprocesamiento_pipeline import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    validate_feature_schema,
)


RANDOM_STATE = 42
DATASET_PATH = Path("dataset.csv")
CANDIDATES_DIR = Path("models/candidates")
REPORTS_DIR = Path("reports/metrics/candidates")
BASELINE_METRICS_PATH = Path("reports/metrics/metricas_modelo.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "Sistema_Riesgo_Academico_MLP"


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for block in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clean_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataframe, _ = prepare_training_dataframe(
        DATASET_PATH,
        report_path=Path("reports/data_quality/latest.json"),
    )
    features = dataframe.drop(columns=["Target"])
    validate_feature_schema(features.columns)
    return features, dataframe["Target"].astype(str)


def train_candidate() -> dict:
    configurar_semillas()
    features, target = load_clean_dataset()

    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(target)
    if set(encoder.classes_) != {"Dropout", "Enrolled", "Graduate"}:
        raise ValueError(f"Clases inesperadas: {encoder.classes_.tolist()}")

    x_train, x_temporary, y_train, y_temporary = train_test_split(
        features,
        encoded_target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=encoded_target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temporary,
        y_temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temporary,
    )

    preprocessor = build_preprocessor()
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_validation_transformed = preprocessor.transform(x_validation)
    x_test_transformed = preprocessor.transform(x_test)

    class_indices = np.unique(y_train)
    weight_values = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=y_train,
    )
    class_weights = {
        int(index): float(weight)
        for index, weight in zip(class_indices, weight_values)
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="MLP_64_32_onehot_candidate") as run:
        run_id = run.info.run_id
        candidate_dir = CANDIDATES_DIR / run_id
        report_dir = REPORTS_DIR / run_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        model_path = candidate_dir / "modelo_estudiantes.keras"
        preprocessor_path = candidate_dir / "preprocessor.pkl"
        encoder_path = candidate_dir / "encoder.pkl"
        transformed_names_path = candidate_dir / "feature_names.json"
        schema_path = candidate_dir / "schema.json"

        model = crear_modelo(
            input_dim=x_train_transformed.shape[1],
            num_classes=len(encoder.classes_),
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            x_train_transformed,
            y_train,
            validation_data=(x_validation_transformed, y_validation),
            epochs=80,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        model = tf.keras.models.load_model(model_path)
        probabilities = model.predict(x_test_transformed, verbose=0)
        predicted = np.argmax(probabilities, axis=1)
        test_loss, test_accuracy = model.evaluate(
            x_test_transformed,
            y_test,
            verbose=0,
        )

        macro_f1 = f1_score(y_test, predicted, average="macro")
        weighted_f1 = f1_score(y_test, predicted, average="weighted")
        report = classification_report(
            y_test,
            predicted,
            target_names=encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "run_id": run_id,
            "status": "candidate",
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "epochs_executed": len(history.history["loss"]),
            "input_features_original": features.shape[1],
            "input_features_transformed": x_train_transformed.shape[1],
            "dataset_sha256": dataset_sha256(DATASET_PATH),
            "classification_report": report,
        }

        baseline = {}
        if BASELINE_METRICS_PATH.exists():
            baseline = json.loads(
                BASELINE_METRICS_PATH.read_text(encoding="utf-8")
            )
            metrics["comparison_with_active"] = {
                "accuracy_delta": (
                    metrics["test_accuracy"] - baseline["test_accuracy"]
                ),
                "macro_f1_delta": (
                    metrics["macro_f1"] - baseline["macro_f1"]
                ),
                "weighted_f1_delta": (
                    metrics["weighted_f1"] - baseline["weighted_f1"]
                ),
            }

        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(encoder, encoder_path)
        transformed_names_path.write_text(
            json.dumps(
                preprocessor.get_feature_names_out().tolist(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        schema_path.write_text(
            json.dumps(
                {
                    "categorical": CATEGORICAL_FEATURES,
                    "binary": BINARY_FEATURES,
                    "numeric": NUMERIC_FEATURES,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        metrics_path = report_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        mlflow.log_params(
            {
                "candidate": True,
                "preprocessing": "impute_onehot_scale",
                "categorical_features": len(CATEGORICAL_FEATURES),
                "binary_features": len(BINARY_FEATURES),
                "numeric_features": len(NUMERIC_FEATURES),
                "input_features_transformed": x_train_transformed.shape[1],
                "architecture": "MLP_64_32_3",
                "class_weight_balancing": True,
                "random_state": RANDOM_STATE,
                "dataset_sha256": metrics["dataset_sha256"],
            }
        )
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "epochs_executed": len(history.history["loss"]),
            }
        )
        mlflow.log_artifacts(str(candidate_dir), artifact_path="bundle")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        quality_report_path = Path("reports/data_quality/latest.json")
        if quality_report_path.exists():
            mlflow.log_artifact(
                str(quality_report_path),
                artifact_path="data_quality",
            )
        traceability = build_traceability_manifest(
            dataset_path=DATASET_PATH,
            schema={
                "categorical": CATEGORICAL_FEATURES,
                "binary": BINARY_FEATURES,
                "numeric": NUMERIC_FEATURES,
            },
            code_paths=[
                Path("entrenamiento_modelo_candidato.py"),
                Path("preprocesamiento_pipeline.py"),
                Path("feature_schema.py"),
            ],
            artifact_roles={
                "model": str(model_path),
                "preprocessor": str(preprocessor_path),
                "target_encoder": str(encoder_path),
                "transformed_feature_names": str(
                    transformed_names_path
                ),
                "input_schema": str(schema_path),
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
                "dataset_sha256": metrics["dataset_sha256"],
            }
        )
        try:
            mlflow.keras.log_model(model, name="model")
        except TypeError:
            mlflow.keras.log_model(model, artifact_path="model")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Candidato guardado en: {candidate_dir}")
    print("El modelo activo no fue reemplazado.")
    return metrics


if __name__ == "__main__":
    train_candidate()
