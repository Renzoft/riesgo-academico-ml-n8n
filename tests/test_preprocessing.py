import pandas as pd
import pytest

from preprocesamiento_pipeline import (
    EXPECTED_FEATURES,
    build_preprocessor,
    validate_feature_schema,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for variant in (0, 1):
        row = {}
        for name in EXPECTED_FEATURES:
            row[name] = variant
        row["Marital status"] = variant + 1
        row["Application mode"] = variant + 1
        row["Course"] = variant + 1
        row["Previous qualification"] = variant + 1
        row["Nacionality"] = variant + 1
        row["Mother's qualification"] = variant + 1
        row["Father's qualification"] = variant + 1
        row["Mother's occupation"] = variant + 1
        row["Father's occupation"] = variant + 1
        row["Age at enrollment"] = 18 + variant
        rows.append(row)
    return pd.DataFrame(rows)


def test_schema_rejects_missing_columns():
    columns = EXPECTED_FEATURES[:-1]

    with pytest.raises(ValueError, match="Faltantes"):
        validate_feature_schema(columns)


def test_pipeline_separates_and_transforms_feature_types():
    frame = sample_frame()
    validate_feature_schema(frame.columns)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > len(EXPECTED_FEATURES)
    assert not pd.isna(transformed).any()


def test_pipeline_accepts_unknown_category_after_fit():
    frame = sample_frame()
    preprocessor = build_preprocessor().fit(frame)
    unknown = frame.iloc[[0]].copy()
    unknown["Course"] = 999

    transformed = preprocessor.transform(unknown)

    assert transformed.shape[1] == len(
        preprocessor.get_feature_names_out()
    )
