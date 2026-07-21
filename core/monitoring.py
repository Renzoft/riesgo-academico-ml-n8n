import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ml.pipeline.preprocesamiento_pipeline import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


RISK_MAPPING = {"Dropout": "Alto", "Enrolled": "Medio", "Graduate": "Bajo"}


def ensure_monitoring_schema(connection: sqlite3.Connection) -> None:
    prediction_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
    }
    if not prediction_columns:
        raise ValueError("La base no contiene la tabla predictions.")
    migrations = {
        "prediction_source": (
            "ALTER TABLE predictions ADD COLUMN prediction_source "
            "TEXT NOT NULL DEFAULT 'legacy'"
        ),
        "input_features": "ALTER TABLE predictions ADD COLUMN input_features TEXT",
        "model_release": (
            "ALTER TABLE predictions ADD COLUMN model_release "
            "TEXT NOT NULL DEFAULT 'legacy'"
        ),
    }
    for column, statement in migrations.items():
        if column not in prediction_columns:
            connection.execute(statement)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            actual_status TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()


def population_stability_index(reference, observed, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    reference = reference[np.isfinite(reference)]
    observed = observed[np.isfinite(observed)]
    if not len(reference) or not len(observed):
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(observed, bins=edges)[0] / len(observed)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_drift(reference, observed) -> dict:
    reference = pd.Series(reference).dropna()
    observed = pd.Series(observed).dropna()
    if observed.empty:
        return {"total_variation": 0.0, "unseen_rate": 0.0}
    reference_distribution = reference.value_counts(normalize=True)
    observed_distribution = observed.value_counts(normalize=True)
    categories = reference_distribution.index.union(observed_distribution.index)
    total_variation = 0.5 * sum(
        abs(
            float(reference_distribution.get(category, 0.0))
            - float(observed_distribution.get(category, 0.0))
        )
        for category in categories
    )
    unseen_rate = float((~observed.isin(set(reference))).mean())
    return {
        "total_variation": float(total_variation),
        "unseen_rate": unseen_rate,
    }


def normalized_distribution(values) -> dict[str, float]:
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    return {
        name: count / total for name, count in sorted(counts.items())
    } if total else {}


def distribution_total_variation(first: dict, second: dict) -> float:
    categories = set(first) | set(second)
    return float(
        0.5 * sum(abs(first.get(name, 0.0) - second.get(name, 0.0))
                  for name in categories)
    )


def generate_monitoring_report(
    connection: sqlite3.Connection,
    reference_path: Path,
    source: str = "production",
    minimum_samples: int = 30,
) -> dict:
    ensure_monitoring_schema(connection)
    rows = connection.execute(
        """
        SELECT id, student_id, estado_predicho, nivel_riesgo, confianza,
               input_features, model_release, created_at
        FROM predictions
        WHERE prediction_source = ?
        ORDER BY id
        """,
        (source,),
    ).fetchall()
    records = [dict(row) for row in rows]
    inputs = []
    valid_records = []
    for record in records:
        try:
            parsed = json.loads(record.get("input_features") or "null")
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            inputs.append(parsed)
            valid_records.append(record)

    reference = pd.read_csv(reference_path)
    observed = pd.DataFrame(inputs)
    drift = {}
    alerts = []
    if not observed.empty:
        for feature in NUMERIC_FEATURES:
            psi = population_stability_index(reference[feature], observed[feature])
            drift[feature] = {"type": "numeric", "psi": psi}
            if psi >= 0.20:
                alerts.append(f"Deriva numérica en {feature}: PSI={psi:.3f}")
        for feature in CATEGORICAL_FEATURES + BINARY_FEATURES:
            values = categorical_drift(reference[feature], observed[feature])
            drift[feature] = {"type": "categorical", **values}
            if values["total_variation"] >= 0.20:
                alerts.append(
                    f"Deriva categórica en {feature}: "
                    f"TV={values['total_variation']:.3f}"
                )
            if values["unseen_rate"] > 0:
                alerts.append(
                    f"Categoría desconocida en {feature}: "
                    f"{values['unseen_rate']:.1%}"
                )

    outcomes = {
        row["student_id"]: row["actual_status"]
        for row in connection.execute(
            """
            SELECT student_id, actual_status
            FROM outcomes
            ORDER BY id
            """
        ).fetchall()
    }
    latest_predictions = {}
    for record in records:
        latest_predictions[record["student_id"]] = record
    matched = [
        (record, outcomes[student_id])
        for student_id, record in latest_predictions.items()
        if student_id in outcomes
    ]
    correct = sum(
        record["estado_predicho"] == actual for record, actual in matched
    )
    sample_count = len(records)
    risk_distribution = normalized_distribution(
        record["nivel_riesgo"] for record in records
    )
    reference_risk_distribution = normalized_distribution(
        RISK_MAPPING[value] for value in reference["Target"]
    )
    risk_distribution_drift = distribution_total_variation(
        risk_distribution, reference_risk_distribution
    ) if records else None
    if risk_distribution_drift is not None and risk_distribution_drift >= 0.20:
        alerts.append(
            "Cambio en distribución de riesgos: "
            f"TV={risk_distribution_drift:.3f}"
        )
    ready = sample_count >= minimum_samples
    return {
        "source": source,
        "sample_count": sample_count,
        "inputs_available": len(valid_records),
        "status": "ready" if ready else "insufficient_data",
        "minimum_samples": minimum_samples,
        "risk_distribution": risk_distribution,
        "reference_risk_distribution": reference_risk_distribution,
        "risk_distribution_total_variation": risk_distribution_drift,
        "confidence": {
            "mean": (
                float(np.mean([record["confianza"] for record in records]))
                if records else None
            ),
            "below_60_percent": sum(
                record["confianza"] < 0.60 for record in records
            ),
        },
        "model_releases": normalized_distribution(
            record["model_release"] for record in records
        ),
        "drift": drift,
        "alerts": alerts if ready else [],
        "alerts_suppressed_until_minimum_samples": (
            len(alerts) if not ready else 0
        ),
        "outcomes": {
            "registered": len(outcomes),
            "matched_students": len(matched),
            "coverage": (
                len(matched) / len(latest_predictions)
                if latest_predictions else 0.0
            ),
            "accuracy": correct / len(matched) if matched else None,
        },
    }
