from fastapi.testclient import TestClient
import pytest

import api
from api.schemas import FEATURE_FIELD_MAP, StudentFeatures
from ml.pipeline.preprocesamiento_pipeline import (
    ALL_MODEL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_preprocessing_pipeline,
    validate_feature_groups,
)


client = TestClient(api.app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_invalid_feature_count():
    response = client.post(
        "/predict",
        json={
            "student_id": "TEST-ERROR",
            "email_tutor": "tutor@universidad.local",
            "features": [1.0, 2.0, 3.0],
        },
    )

    # Si el modelo aún no se cargó devuelve 503; con el modelo devuelve 422.
    assert response.status_code in {422, 503}


def valid_named_features():
    return {
        "marital_status": 1,
        "application_mode": 8,
        "application_order": 5,
        "course": 2,
        "daytime_evening_attendance": 1,
        "previous_qualification": 1,
        "nationality": 1,
        "mothers_qualification": 13,
        "fathers_qualification": 10,
        "mothers_occupation": 6,
        "fathers_occupation": 10,
        "displaced": 1,
        "educational_special_needs": 0,
        "debtor": 0,
        "tuition_fees_up_to_date": 1,
        "gender": 1,
        "scholarship_holder": 0,
        "age_at_enrollment": 20,
        "international": 0,
        "curricular_units_1st_sem_credited": 0,
        "curricular_units_1st_sem_enrolled": 0,
        "curricular_units_1st_sem_evaluations": 0,
        "curricular_units_1st_sem_approved": 0,
        "curricular_units_1st_sem_grade": 0,
        "curricular_units_1st_sem_without_evaluations": 0,
        "curricular_units_2nd_sem_credited": 0,
        "curricular_units_2nd_sem_enrolled": 0,
        "curricular_units_2nd_sem_evaluations": 0,
        "curricular_units_2nd_sem_approved": 0,
        "curricular_units_2nd_sem_grade": 0,
        "curricular_units_2nd_sem_without_evaluations": 0,
        "unemployment_rate": 10.8,
        "inflation_rate": 1.4,
        "gdp": 1.74,
    }


def test_named_features_follow_model_order():
    named = StudentFeatures(**valid_named_features())
    model_order = list(FEATURE_FIELD_MAP.values())

    vector = named.to_model_vector(model_order)

    assert len(vector) == 34
    assert vector[0] == 1
    assert vector[17] == 20
    assert vector[-1] == 1.74


def test_named_features_are_independent_of_json_order():
    values = valid_named_features()
    reversed_values = dict(reversed(list(values.items())))

    normal = StudentFeatures(**values)
    reversed_input = StudentFeatures(**reversed_values)

    assert normal.to_model_vector(list(FEATURE_FIELD_MAP.values())) == (
        reversed_input.to_model_vector(list(FEATURE_FIELD_MAP.values()))
    )


def test_named_features_reject_unknown_fields():
    values = valid_named_features()
    values["unknown_feature"] = 123

    with pytest.raises(ValueError):
        StudentFeatures(**values)


def test_named_features_validate_binary_fields():
    values = valid_named_features()
    values["debtor"] = 2

    with pytest.raises(ValueError):
        StudentFeatures(**values)


def test_all_features_have_exactly_one_preprocessing_group():
    grouped = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

    assert len(grouped) == 34
    assert len(set(grouped)) == 34
    assert set(grouped) == set(FEATURE_FIELD_MAP.values())
    validate_feature_groups(list(FEATURE_FIELD_MAP.values()))


def test_preprocessor_generates_dense_numeric_output():
    import pandas as pd

    named = StudentFeatures(**valid_named_features())
    vector = named.to_model_vector(list(FEATURE_FIELD_MAP.values()))
    frame = pd.DataFrame(
        [vector, vector],
        columns=list(FEATURE_FIELD_MAP.values()),
    )
    preprocessor = create_preprocessing_pipeline()

    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= len(ALL_MODEL_FEATURES)
    assert transformed.dtype.kind == "f"
