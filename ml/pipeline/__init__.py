"""Pipeline canonico de preprocesamiento."""

from ml.pipeline.preprocesamiento_pipeline import (
    ALL_MODEL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_preprocessing_pipeline,
    validate_feature_groups,
)

__all__ = [
    "ALL_MODEL_FEATURES",
    "BINARY_FEATURES",
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "create_preprocessing_pipeline",
    "validate_feature_groups",
]
