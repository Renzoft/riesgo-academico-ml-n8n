import hashlib
import json
from pathlib import Path

import pandas as pd

from preprocessing_pipeline import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    validate_feature_groups,
)


TARGET_COLUMN = "Target"
EXPECTED_TARGETS = {"Dropout", "Enrolled", "Graduate"}
GRADE_FEATURES = {
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
}
NON_NEGATIVE_FEATURES = {
    "Application order",
    *[name for name in NUMERIC_FEATURES if name.startswith("Curricular units")],
}


def dataframe_sha256(dataframe: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def class_distribution(dataframe: pd.DataFrame) -> dict[str, int]:
    if TARGET_COLUMN not in dataframe:
        return {}
    return {
        str(name): int(count)
        for name, count in dataframe[TARGET_COLUMN].value_counts(dropna=False).items()
    }


def invalid_values(features: pd.DataFrame) -> dict[str, list]:
    issues = {}
    for column in BINARY_FEATURES:
        observed = set(features[column].dropna().unique().tolist())
        invalid = sorted(observed - {0, 1})
        if invalid:
            issues[column] = invalid
    for column in CATEGORICAL_FEATURES:
        mask = features[column].notna() & (features[column] < 1)
        invalid = sorted(features.loc[mask, column].unique().tolist())
        if invalid:
            issues[column] = invalid
    for column in NON_NEGATIVE_FEATURES:
        mask = features[column].notna() & (features[column] < 0)
        invalid = sorted(features.loc[mask, column].unique().tolist())
        if invalid:
            issues[column] = invalid
    for column in GRADE_FEATURES:
        mask = features[column].notna() & ~features[column].between(0, 20)
        invalid = sorted(features.loc[mask, column].unique().tolist())
        if invalid:
            issues[column] = invalid
    age = "Age at enrollment"
    mask = features[age].notna() & ~features[age].between(15, 100)
    invalid_age = sorted(features.loc[mask, age].unique().tolist())
    if invalid_age:
        issues[age] = invalid_age
    return issues


def prepare_training_dataframe(
    dataset_path: Path,
    report_path: Path | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {dataset_path}")
    original = pd.read_csv(dataset_path)
    if TARGET_COLUMN not in original.columns:
        raise ValueError(f"El dataset debe contener '{TARGET_COLUMN}'.")

    unnamed = [
        column for column in original.columns
        if column.lower().startswith("unnamed")
    ]
    working = original.drop(columns=unnamed).copy()
    features_before = working.drop(columns=[TARGET_COLUMN])
    validate_feature_groups(list(features_before.columns))
    non_numeric = [
        column for column in features_before
        if not pd.api.types.is_numeric_dtype(features_before[column])
    ]
    exact_duplicate_mask = working.duplicated(keep="first")
    feature_duplicate_mask = working.duplicated(
        subset=list(features_before.columns), keep=False
    )
    contradictory_indices = []
    if feature_duplicate_mask.any():
        repeated = working.loc[feature_duplicate_mask]
        contradictory = repeated.groupby(
            list(features_before.columns), dropna=False
        )[TARGET_COLUMN].transform("nunique") > 1
        contradictory_indices = repeated.index[contradictory].tolist()

    missing_target_mask = working[TARGET_COLUMN].isna()
    cleaned = working.loc[~missing_target_mask & ~exact_duplicate_mask].copy()
    cleaned = cleaned.reset_index(drop=True)
    features_after = cleaned.drop(columns=[TARGET_COLUMN])
    invalid = invalid_values(features_after) if not non_numeric else {}
    observed_targets = set(cleaned[TARGET_COLUMN].astype(str).unique())
    invalid_targets = sorted(observed_targets - EXPECTED_TARGETS)

    report = {
        "source": str(dataset_path),
        "policy": {
            "exact_duplicates": "remove_to_avoid_split_leakage",
            "missing_target": "remove_because_supervised_training_requires_label",
            "missing_predictors": "retain_for_train_fitted_pipeline_imputation",
            "contradictory_profiles": "report_without_silent_removal",
            "invalid_values": "stop_training_in_strict_mode",
        },
        "before": {
            "rows": int(len(original)),
            "columns": int(len(original.columns)),
            "sha256": dataframe_sha256(original),
            "class_distribution": class_distribution(original),
            "missing_by_column": {
                name: int(count)
                for name, count in original.isna().sum().items() if count
            },
        },
        "findings": {
            "unnamed_columns_removed": unnamed,
            "exact_duplicates_removed": int(exact_duplicate_mask.sum()),
            "missing_target_rows_removed": int(missing_target_mask.sum()),
            "predictor_rows_with_missing_values": int(
                features_before.isna().any(axis=1).sum()
            ),
            "predictor_missing_values_retained_for_imputation": int(
                features_after.isna().sum().sum()
            ),
            "duplicated_feature_rows": int(feature_duplicate_mask.sum()),
            "contradictory_feature_rows": len(contradictory_indices),
            "contradictory_row_indices_sample": contradictory_indices[:50],
            "non_numeric_features": non_numeric,
            "invalid_feature_values": invalid,
            "invalid_targets": invalid_targets,
        },
        "after": {
            "rows": int(len(cleaned)),
            "columns": int(len(cleaned.columns)),
            "sha256": dataframe_sha256(cleaned),
            "class_distribution": class_distribution(cleaned),
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    errors = {
        "non_numeric_features": non_numeric,
        "invalid_feature_values": invalid,
        "invalid_targets": invalid_targets,
    }
    if strict and any(errors.values()):
        raise ValueError(
            "El dataset contiene valores inválidos. "
            + json.dumps(errors, ensure_ascii=False)
        )
    return cleaned, report
