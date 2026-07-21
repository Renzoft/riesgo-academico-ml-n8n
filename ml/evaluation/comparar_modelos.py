import hashlib
import json
from pathlib import Path


ACTIVE_PATH = Path("reports/metrics/metricas_modelo.json")
CANDIDATE_PATHS = [
    Path(
        "models/releases/preprocessing-v2-97984f56/metrics.json"
    ),
]
OUTPUT_PATH = Path("reports/metrics/comparacion_modelos.json")


def load_comparison_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    report = metrics["classification_report"]
    return {
        "source": str(path),
        "run_id": metrics.get("run_id"),
        "accuracy": float(metrics["test_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "dropout_recall": float(report["Dropout"]["recall"]),
        "enrolled_f1": float(report["Enrolled"]["f1-score"]),
        "graduate_f1": float(report["Graduate"]["f1-score"]),
        "dataset_sha256": metrics.get("dataset_sha256"),
    }


def candidate_signature(candidate: dict) -> str:
    comparable = {
        key: candidate[key]
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "dropout_recall",
            "enrolled_f1",
            "graduate_f1",
        )
    }
    return hashlib.sha256(
        json.dumps(comparable, sort_keys=True).encode("utf-8")
    ).hexdigest()


def compare_models() -> dict:
    active = load_comparison_metrics(ACTIVE_PATH)
    candidates = [
        load_comparison_metrics(path)
        for path in CANDIDATE_PATHS
        if path.exists()
    ]

    grouped = {}
    for candidate in candidates:
        grouped.setdefault(candidate_signature(candidate), []).append(candidate)

    unique_candidates = [group[0] for group in grouped.values()]
    best = max(unique_candidates, key=lambda item: item["macro_f1"])
    deltas = {
        metric: best[metric] - active[metric]
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "dropout_recall",
            "enrolled_f1",
            "graduate_f1",
        )
    }
    eligible = (
        deltas["accuracy"] >= 0
        and deltas["macro_f1"] >= 0
        and deltas["dropout_recall"] >= 0
    )
    result = {
        "active": active,
        "best_candidate": best,
        "deltas": deltas,
        "promotion_eligible": eligible,
        "decision": (
            "Candidato apto para validación funcional; no promovido."
            if eligible
            else "Mantener modelo activo."
        ),
        "equivalent_candidate_groups": list(grouped.values()),
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    compare_models()
