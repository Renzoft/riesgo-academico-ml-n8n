import json
import sqlite3

import pandas as pd

from core.monitoring import (
    categorical_drift,
    ensure_monitoring_schema,
    generate_monitoring_report,
    population_stability_index,
)


def test_identical_numeric_distributions_have_zero_psi():
    values = [1, 2, 3, 4, 5] * 10

    assert population_stability_index(values, values) == 0


def test_unknown_category_is_reported():
    result = categorical_drift([1, 1, 2], [1, 99])

    assert result["unseen_rate"] == 0.5
    assert result["total_variation"] > 0


def test_report_separates_prediction_sources(tmp_path):
    reference = pd.read_csv("dataset.csv")
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            student_id TEXT,
            estado_predicho TEXT,
            nivel_riesgo TEXT,
            confianza REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    ensure_monitoring_schema(connection)
    row = reference.drop(columns=["Target"]).iloc[0].to_dict()
    connection.execute(
        """
        INSERT INTO predictions (
            student_id, estado_predicho, nivel_riesgo, confianza,
            prediction_source, input_features, model_release
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("PROD-1", "Dropout", "Alto", 0.8, "production", json.dumps(row), "v2"),
    )
    connection.execute(
        """
        INSERT INTO predictions (
            student_id, estado_predicho, nivel_riesgo, confianza,
            prediction_source, input_features, model_release
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("TEST-1", "Graduate", "Bajo", 0.9, "system_test", json.dumps(row), "v2"),
    )
    connection.commit()

    report = generate_monitoring_report(
        connection, reference_path, source="production", minimum_samples=1
    )

    assert report["sample_count"] == 1
    assert report["risk_distribution"] == {"Alto": 1.0}
    assert report["status"] == "ready"
