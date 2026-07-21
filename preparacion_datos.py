from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from calidad_datos import prepare_training_dataframe
from preprocessing_pipeline import (
    create_preprocessing_pipeline,
    validate_feature_groups,
)


RANDOM_STATE = 42
TARGET_COLUMN = "Target"


def preparar_datos(ruta_dataset: str):
    """Valida, limpia, divide y transforma los datos sin fuga de información."""
    dataset_path = Path(ruta_dataset)
    print(f"Cargando dataset desde: {dataset_path}")

    dataframe, quality_report = prepare_training_dataframe(
        dataset_path,
        report_path=Path("reports/data_quality/latest.json"),
    )
    findings = quality_report["findings"]
    print("\n--- Calidad del dataset ---")
    print(f"Dimensiones originales: {quality_report['before']['rows']} filas")
    print(f"Duplicados eliminados: {findings['exact_duplicates_removed']}")
    print(f"Filas sin Target eliminadas: {findings['missing_target_rows_removed']}")
    print(
        "Nulos predictivos conservados para imputación: "
        f"{findings['predictor_missing_values_retained_for_imputation']}"
    )
    print(f"Perfiles contradictorios: {findings['contradictory_feature_rows']}")
    print("Reporte: reports/data_quality/latest.json")

    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN].astype(str)
    feature_names = list(features.columns)
    validate_feature_groups(feature_names)

    print("\nDistribución de la variable objetivo:")
    print(target.value_counts())

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)
    print("\nMapeo real de clases:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {index}")

    expected_classes = {"Dropout", "Enrolled", "Graduate"}
    if set(label_encoder.classes_) != expected_classes:
        raise ValueError(
            "Las clases no coinciden con las esperadas. "
            f"Encontradas: {sorted(label_encoder.classes_)}"
        )

    features_train, features_temporary, target_train, target_temporary = (
        train_test_split(
            features,
            target_encoded,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=target_encoded,
        )
    )
    features_validation, features_test, target_validation, target_test = (
        train_test_split(
            features_temporary,
            target_temporary,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=target_temporary,
        )
    )

    # El preprocesador se ajusta solo con entrenamiento. Las imputaciones,
    # categorías y estadísticas de escalamiento no ven validación ni prueba.
    preprocessor = create_preprocessing_pipeline()
    features_train_processed = preprocessor.fit_transform(features_train)
    features_validation_processed = preprocessor.transform(features_validation)
    features_test_processed = preprocessor.transform(features_test)

    print("\nDivisión final:")
    print(f"Entrenamiento: {len(features_train_processed)} muestras")
    print(f"Validación: {len(features_validation_processed)} muestras")
    print(f"Prueba final: {len(features_test_processed)} muestras")
    print(f"Variables transformadas: {features_train_processed.shape[1]}")

    return (
        features_train_processed,
        features_validation_processed,
        features_test_processed,
        target_train,
        target_validation,
        target_test,
        label_encoder,
        preprocessor,
        feature_names,
    )


if __name__ == "__main__":
    preparar_datos("dataset.csv")
