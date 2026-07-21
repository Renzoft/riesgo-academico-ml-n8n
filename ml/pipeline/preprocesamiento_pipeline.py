from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from api.schemas import (
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

ALL_MODEL_FEATURES = (
    CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES
)


def create_preprocessing_pipeline() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def validate_feature_groups(feature_names: list[str]) -> None:
    configured = set(ALL_MODEL_FEATURES)
    observed = set(feature_names)
    duplicated = [
        name for name in configured
        if ALL_MODEL_FEATURES.count(name) > 1
    ]
    unclassified = sorted(observed - configured)
    absent = sorted(configured - observed)

    if duplicated or unclassified or absent:
        raise ValueError(
            "La clasificacion de variables no coincide con el dataset. "
            f"Duplicadas: {sorted(set(duplicated))}. "
            f"Sin clasificar: {unclassified}. "
            f"No presentes: {absent}."
        )
