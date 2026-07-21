import json
import os
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

from calidad_datos import prepare_training_dataframe
from entrenamiento_modelo import configurar_semillas, crear_modelo
from evaluacion_avanzada import dropout_recall, evaluate_predictions
from preprocessing_pipeline import create_preprocessing_pipeline


RANDOM_STATE = 42
RESULTS_PATH = Path("reports/advanced_evaluation/results.json")
CV_PATH = Path("reports/advanced_evaluation/mlp_cross_validation.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "Evaluacion_Avanzada_Riesgo_Academico"


def summarize_folds(folds: list[dict]) -> dict:
    keys = ("accuracy", "macro_f1", "weighted_f1", "dropout_recall")
    return {
        key: {
            "mean": float(np.mean([fold[key] for fold in folds])),
            "std": float(np.std([fold[key] for fold in folds])),
        }
        for key in keys
    }


def main() -> dict:
    dataframe, quality_report = prepare_training_dataframe(Path("dataset.csv"))
    x = dataframe.drop(columns=["Target"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(dataframe["Target"].astype(str))
    classes = encoder.classes_.tolist()
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    folds = []

    for fold_number, (development_indices, evaluation_indices) in enumerate(
        outer_cv.split(x, y), start=1
    ):
        x_development = x.iloc[development_indices]
        y_development = y[development_indices]
        x_evaluation = x.iloc[evaluation_indices]
        y_evaluation = y[evaluation_indices]
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_development,
            y_development,
            test_size=0.10,
            random_state=RANDOM_STATE + fold_number,
            stratify=y_development,
        )

        preprocessor = create_preprocessing_pipeline()
        x_train_processed = preprocessor.fit_transform(x_train)
        x_validation_processed = preprocessor.transform(x_validation)
        x_evaluation_processed = preprocessor.transform(x_evaluation)
        configurar_semillas(RANDOM_STATE + fold_number)
        model = crear_modelo(
            input_dim=x_train_processed.shape[1],
            num_classes=len(classes),
        )
        history = model.fit(
            x_train_processed,
            y_train,
            validation_data=(x_validation_processed, y_validation),
            epochs=60,
            batch_size=32,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        probabilities = model.predict(x_evaluation_processed, verbose=0)
        metrics = evaluate_predictions(y_evaluation, probabilities, classes)
        folds.append(
            {
                "fold": fold_number,
                "epochs": len(history.history["loss"]),
                **{
                    key: metrics[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "dropout_recall",
                    )
                },
            }
        )
        tf.keras.backend.clear_session()

    summary = summarize_folds(folds)
    output = {"folds": folds, "summary": summary}
    CV_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    comparison = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    comparison["models"]["mlp_candidate"]["cross_validation"] = summary
    eligible = {
        name: values
        for name, values in comparison["models"].items()
        if values["cross_validation"] is not None
        and values["cross_validation"]["dropout_recall"]["mean"] >= 0.70
    }
    best_name = max(
        eligible,
        key=lambda name: eligible[name]["cross_validation"]["macro_f1"]["mean"],
    )
    comparison["recommendation"] = {
        "model": best_name,
        "status": "evaluation_complete_not_promoted",
        "reason": (
            "Mayor macro F1 promedio en validación cruzada entre modelos "
            "con recall promedio de Dropout de al menos 0.70."
        ),
    }
    RESULTS_PATH.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="comparison_mlp_5fold"):
        mlflow.set_tags(
            {
                "purpose": "comparison_only",
                "model_stage": "evaluation",
                "cross_validation": "5_fold_nested_early_stopping",
                "dataset_sha256": quality_report["before"]["sha256"],
            }
        )
        mlflow.log_params(
            {
                "outer_folds": 5,
                "inner_validation_fraction": 0.10,
                "maximum_epochs": 60,
                "early_stopping_patience": 8,
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.log_metrics(
            {
                **{f"cv_{key}_mean": value["mean"] for key, value in summary.items()},
                **{f"cv_{key}_std": value["std"] for key, value in summary.items()},
            }
        )
        mlflow.log_artifact(str(CV_PATH), artifact_path="evaluation")
        mlflow.log_artifact(str(RESULTS_PATH), artifact_path="evaluation")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Recomendación final: {comparison['recommendation']}")
    return output


if __name__ == "__main__":
    main()
