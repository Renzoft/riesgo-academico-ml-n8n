import json
import os
from pathlib import Path

import pandas as pd
import requests


API_URL = os.getenv("API_URL", "http://localhost:8000")
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "http://localhost:5678/webhook/riesgo-academico",
)
DATASET_PATH = Path("dataset.csv")
OUTPUT_PATH = Path(
    "reports/metrics/resultados_pruebas_sistema.json"
)


def send_request(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=60)

    try:
        body = response.json()
    except ValueError:
        body = {"raw_response": response.text}

    return {
        "status_code": response.status_code,
        "response": body,
    }


def prepare_dataframe() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Coloca dataset.csv en la raíz del proyecto."
        )

    dataframe = pd.read_csv(DATASET_PATH)

    unnamed_columns = [
        column
        for column in dataframe.columns
        if column.lower().startswith("unnamed")
    ]
    if unnamed_columns:
        dataframe = dataframe.drop(columns=unnamed_columns)

    if "Target" not in dataframe.columns:
        raise ValueError(
            "El dataset no contiene la columna Target."
        )

    return dataframe


def find_payloads_by_predicted_risk(
    dataframe: pd.DataFrame,
) -> dict[str, dict]:
    """
    Busca registros que el modelo prediga como Alto, Medio y Bajo.
    Así se garantiza que las tres ramas de n8n sean ejecutadas.
    """
    found: dict[str, dict] = {}

    for row_index, row in dataframe.head(1000).iterrows():
        features = (
            row.drop(labels=["Target"])
            .astype(float)
            .tolist()
        )

        payload = {
            "student_id": f"PRUEBA-{row_index}",
            "email_tutor": "tutor@universidad.local",
            "features": features,
        }

        prediction_result = send_request(
            f"{API_URL}/predict",
            payload,
        )

        if prediction_result["status_code"] != 200:
            continue

        predicted_risk = prediction_result[
            "response"
        ].get("nivel_riesgo")

        if (
            predicted_risk in {"Alto", "Medio", "Bajo"}
            and predicted_risk not in found
        ):
            found[predicted_risk] = payload
            print(
                f"Encontrado caso para riesgo {predicted_risk}: "
                f"{payload['student_id']}"
            )

        if len(found) == 3:
            break

    return found


def main() -> None:
    dataframe = prepare_dataframe()
    payloads = find_payloads_by_predicted_risk(dataframe)

    missing_risks = (
        {"Alto", "Medio", "Bajo"} - set(payloads)
    )
    if missing_risks:
        print(
            "Advertencia: no se encontraron casos para: "
            + ", ".join(sorted(missing_risks))
        )

    results = {}

    for risk_level, payload in payloads.items():
        results[f"flujo_n8n_riesgo_{risk_level.lower()}"] = (
            send_request(
                N8N_WEBHOOK_URL,
                payload,
            )
        )

    invalid_payload = {
        "student_id": "PRUEBA-ERROR",
        "email_tutor": "tutor@universidad.local",
        "features": [1.0, 2.0, 3.0],
    }

    results["caso_erroneo"] = send_request(
        f"{API_URL}/predict",
        invalid_payload,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
