import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from core.calidad_datos import prepare_training_dataframe
from ml.pipeline.preprocesamiento_pipeline import create_preprocessing_pipeline


RANDOM_STATE = 42
DATASET_PATH = Path("dataset.csv")
OUTPUT_DIR = Path("reports/advanced_evaluation")
OUTPUT_PATH = OUTPUT_DIR / "results.json"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "Evaluacion_Avanzada_Riesgo_Academico"
MLP_CANDIDATE_DIR = Path(
    "models/releases/preprocessing-v2-97984f56"
)


def multiclass_brier_score(y_true, probabilities, class_count: int) -> float:
    expected = np.eye(class_count)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probabilities - expected) ** 2, axis=1)))


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correctness = (predicted == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        include_upper = upper == 1.0
        mask = (confidence >= lower) & (
            confidence <= upper if include_upper else confidence < upper
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return float(error)


def dropout_recall(y_true, y_pred) -> float:
    return float(recall_score(y_true, y_pred, labels=[0], average="macro"))


def evaluate_predictions(y_true, probabilities, classes: list[str]) -> dict:
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted")),
        "dropout_recall": dropout_recall(y_true, predicted),
        "log_loss": float(log_loss(y_true, probabilities, labels=range(len(classes)))),
        "multiclass_brier": multiclass_brier_score(
            y_true, probabilities, len(classes)
        ),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities
        ),
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
        "classification_report": classification_report(
            y_true,
            predicted,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_figure(name: str, metrics: dict, classes: list[str]) -> Path:
    figure_path = OUTPUT_DIR / f"confusion_{name}.png"
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"]),
        display_labels=classes,
    )
    display.plot(values_format="d", xticks_rotation=25)
    plt.title(f"Matriz de confusión - {name}")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()
    return figure_path


def evaluate_mlp_candidate(x_test, y_test, classes: list[str]) -> dict:
    import tensorflow as tf

    preprocessor = joblib.load(MLP_CANDIDATE_DIR / "preprocessor.pkl")
    model = tf.keras.models.load_model(
        MLP_CANDIDATE_DIR / "modelo_estudiantes.keras"
    )
    probabilities = model.predict(preprocessor.transform(x_test), verbose=0)
    return evaluate_predictions(y_test, probabilities, classes)


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe, quality_report = prepare_training_dataframe(
        DATASET_PATH,
        report_path=Path("reports/data_quality/latest.json"),
    )
    x = dataframe.drop(columns=["Target"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(dataframe["Target"].astype(str))
    classes = encoder.classes_.tolist()

    x_train, x_temporary, y_train, y_temporary = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    _, x_test, _, y_test = train_test_split(
        x_temporary,
        y_temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temporary,
    )

    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "dropout_recall": lambda estimator, features, target: dropout_recall(
            target, estimator.predict(features)
        ),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    results = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessing_pipeline()),
                ("classifier", estimator),
            ]
        )
        cv_results = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        pipeline.fit(x_train, y_train)
        test_metrics = evaluate_predictions(
            y_test, pipeline.predict_proba(x_test), classes
        )
        cv_summary = {
            metric.removeprefix("test_"): {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
            for metric, values in cv_results.items()
            if metric.startswith("test_")
        }
        results[name] = {
            "cross_validation": cv_summary,
            "test": test_metrics,
        }
        figure_path = save_confusion_figure(name, test_metrics, classes)
        with mlflow.start_run(run_name=f"comparison_{name}"):
            mlflow.set_tags(
                {
                    "purpose": "comparison_only",
                    "model_stage": "evaluation",
                    "dataset_sha256": quality_report["before"]["sha256"],
                }
            )
            mlflow.log_params(
                {
                    "estimator": estimator.__class__.__name__,
                    "cv_folds": 5,
                    "random_state": RANDOM_STATE,
                    "training_rows": len(x_train),
                    "test_rows": len(x_test),
                }
            )
            mlflow.log_metrics(
                {
                    **{f"test_{key}": value for key, value in test_metrics.items()
                       if isinstance(value, float)},
                    **{f"cv_{key}_mean": value["mean"]
                       for key, value in cv_summary.items()},
                    **{f"cv_{key}_std": value["std"]
                       for key, value in cv_summary.items()},
                }
            )
            mlflow.log_artifact(str(figure_path), artifact_path="evaluation")

    mlp_metrics = evaluate_mlp_candidate(x_test, y_test, classes)
    results["mlp_candidate"] = {
        "cross_validation": None,
        "test": mlp_metrics,
        "note": (
            "MLP evaluado con el artefacto candidato existente; requiere "
            "validación cruzada propia antes de decidir un reemplazo."
        ),
    }
    save_confusion_figure("mlp_candidate", mlp_metrics, classes)
    with mlflow.start_run(run_name="comparison_mlp_candidate"):
        mlflow.set_tags(
            {
                "purpose": "comparison_only",
                "model_stage": "evaluation",
                "cross_validation": "pending",
                "dataset_sha256": quality_report["before"]["sha256"],
            }
        )
        mlflow.log_metrics(
            {
                f"test_{key}": value
                for key, value in mlp_metrics.items()
                if isinstance(value, float)
            }
        )

    eligible = {
        name: values
        for name, values in results.items()
        if values["cross_validation"] is not None
        and values["cross_validation"]["dropout_recall"]["mean"] >= 0.70
    }
    best_name = max(
        eligible,
        key=lambda name: eligible[name]["cross_validation"]["macro_f1"]["mean"],
    )
    output = {
        "protocol": {
            "training_rows": len(x_train),
            "test_rows": len(x_test),
            "cv_folds": 5,
            "stratified": True,
            "test_set_used_for_selection": False,
            "selection_rule": (
                "Mayor macro F1 promedio en CV con recall promedio de "
                "Dropout de al menos 0.70."
            ),
        },
        "classes": classes,
        "models": results,
        "recommendation": {
            "tabular_challenger": best_name,
            "status": "requires_mlp_cross_validation",
            "reason": (
                "Mejor modelo tabular según validación cruzada. No debe "
                "reemplazar al MLP hasta ejecutar CV comparable del MLP."
            ),
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
