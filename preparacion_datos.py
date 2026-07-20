from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Target"


def preparar_datos(ruta_dataset: str):
    """
    Carga el dataset, valida sus columnas y crea conjuntos separados de
    entrenamiento (70 %), validación (15 %) y prueba final (15 %).

    El escalador se ajusta únicamente con los datos de entrenamiento para
    evitar fuga de información.
    """
    dataset_path = Path(ruta_dataset)

    print(f"Cargando dataset desde: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {dataset_path}. "
            "Coloca dataset.csv en la raíz del proyecto."
        )

    dataframe = pd.read_csv(dataset_path)

    print("\n--- Información inicial del dataset ---")
    print(f"Dimensiones: {dataframe.shape}")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"El dataset debe contener la columna '{TARGET_COLUMN}'."
        )

    # Elimina columnas exportadas accidentalmente como índices.
    unnamed_columns = [
        column
        for column in dataframe.columns
        if column.lower().startswith("unnamed")
    ]
    if unnamed_columns:
        dataframe = dataframe.drop(columns=unnamed_columns)

    duplicate_rows = int(dataframe.duplicated().sum())
    null_values = int(dataframe.isna().sum().sum())

    print(f"Filas duplicadas: {duplicate_rows}")
    print(f"Valores nulos: {null_values}")

    if duplicate_rows:
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    if null_values:
        # Para este dataset normalmente no hay nulos. Si aparecen, se
        # eliminan para impedir que TensorFlow reciba valores NaN.
        dataframe = dataframe.dropna().reset_index(drop=True)

    print("\nDistribución de la variable objetivo:")
    print(dataframe[TARGET_COLUMN].value_counts())

    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN].astype(str)

    if not all(pd.api.types.is_numeric_dtype(features[column])
               for column in features.columns):
        non_numeric = [
            column
            for column in features.columns
            if not pd.api.types.is_numeric_dtype(features[column])
        ]
        raise ValueError(
            "Todas las variables predictoras deben ser numéricas. "
            f"Columnas no numéricas: {non_numeric}"
        )

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    print("\nMapeo real de clases:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {index}")

    expected_classes = {"Dropout", "Enrolled", "Graduate"}
    observed_classes = set(label_encoder.classes_)

    if observed_classes != expected_classes:
        raise ValueError(
            "Las clases no coinciden con las esperadas. "
            f"Encontradas: {sorted(observed_classes)}"
        )

    # 70 % entrenamiento y 30 % temporal.
    (
        features_train,
        features_temporary,
        target_train,
        target_temporary,
    ) = train_test_split(
        features,
        target_encoded,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target_encoded,
    )

    # Divide el 30 % temporal en 15 % validación y 15 % prueba.
    (
        features_validation,
        features_test,
        target_validation,
        target_test,
    ) = train_test_split(
        features_temporary,
        target_temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=target_temporary,
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_validation_scaled = scaler.transform(features_validation)
    features_test_scaled = scaler.transform(features_test)

    print("\nDivisión final:")
    print(f"Entrenamiento: {len(features_train_scaled)} muestras")
    print(f"Validación: {len(features_validation_scaled)} muestras")
    print(f"Prueba final: {len(features_test_scaled)} muestras")
    print(f"Variables predictoras: {features_train_scaled.shape[1]}")

    return (
        features_train_scaled,
        features_validation_scaled,
        features_test_scaled,
        target_train,
        target_validation,
        target_test,
        label_encoder,
        scaler,
        list(features.columns),
    )


if __name__ == "__main__":
    preparar_datos("dataset.csv")
