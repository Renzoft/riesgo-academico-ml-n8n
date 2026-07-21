from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_schema import (
    BINARY_API_FIELDS,
    CATEGORICAL_API_FIELDS,
    FEATURE_FIELD_MAP,
    NUMERIC_API_FIELDS,
)

CATEGORICAL_FEATURES = [
    FEATURE_FIELD_MAP[name] for name in CATEGORICAL_API_FIELDS
]

BINARY_FEATURES = [
    FEATURE_FIELD_MAP[name] for name in BINARY_API_FIELDS
]

NUMERIC_FEATURES = [
    FEATURE_FIELD_MAP[name] for name in NUMERIC_API_FIELDS
]

EXPECTED_FEATURES = (
    CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES
)


def validate_feature_schema(columns) -> None:
    observed = list(columns)
    missing = sorted(set(EXPECTED_FEATURES) - set(observed))
    additional = sorted(set(observed) - set(EXPECTED_FEATURES))

    if missing or additional:
        raise ValueError(
            "El dataset no coincide con el esquema de entrenamiento. "
            f"Faltantes: {missing}. Adicionales: {additional}."
        )


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
