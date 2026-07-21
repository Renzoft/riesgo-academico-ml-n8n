import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf

from infrastructure.model_release import (
    load_active_release,
    resolve_release_artifacts,
)


REQUIRED_CANDIDATE_FILES = {
    "model": "modelo_estudiantes.keras",
    "transformer": "preprocessor.pkl",
    "encoder": "encoder.pkl",
    "feature_names": "feature_names.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def promotion_criteria(candidate: dict, active: dict) -> dict:
    checks = {
        "accuracy_not_lower": candidate["test_accuracy"] >= active["test_accuracy"],
        "macro_f1_not_lower": candidate["macro_f1"] >= active["macro_f1"],
        "high_risk_recall_not_lower": (
            candidate["classification_report"]["Dropout"]["recall"]
            >= active["classification_report"]["Dropout"]["recall"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def validate_candidate(
    candidate_dir: Path,
    candidate_metrics_path: Path,
    active_metrics_path: Path,
    dataset_path: Path,
) -> dict:
    paths = {
        name: candidate_dir / filename
        for name, filename in REQUIRED_CANDIDATE_FILES.items()
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan artefactos candidatos: " + ", ".join(missing))

    feature_names = json.loads(paths["feature_names"].read_text(encoding="utf-8"))
    dataframe = pd.read_csv(dataset_path)
    missing_columns = sorted(set(feature_names) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"El dataset no contiene: {missing_columns}")

    transformer = joblib.load(paths["transformer"])
    encoder = joblib.load(paths["encoder"])
    model = tf.keras.models.load_model(paths["model"])
    transformed = transformer.transform(dataframe[feature_names].head(5))
    model_input = int(model.input_shape[-1])
    if transformed.shape[1] != model_input:
        raise ValueError(
            "Dimensión incompatible: el preprocesador produce "
            f"{transformed.shape[1]} y el modelo espera {model_input}."
        )
    if set(encoder.classes_) != {"Dropout", "Enrolled", "Graduate"}:
        raise ValueError(f"Clases incompatibles: {encoder.classes_.tolist()}")

    criteria = promotion_criteria(
        load_raw_metrics(candidate_metrics_path),
        load_raw_metrics(active_metrics_path),
    )
    return {
        "compatible": True,
        "promotion_criteria": criteria,
        "original_features": len(feature_names),
        "transformed_features": transformed.shape[1],
        "classes": encoder.classes_.tolist(),
        "artifact_sha256": {
            name: file_sha256(path) for name, path in paths.items()
        },
    }


def package_release(
    release_id: str,
    run_id: str,
    candidate_dir: Path,
    candidate_metrics_path: Path,
    active_metrics_path: Path,
    dataset_path: Path,
    releases_dir: Path,
) -> Path:
    validation = validate_candidate(
        candidate_dir, candidate_metrics_path, active_metrics_path, dataset_path
    )
    if not validation["promotion_criteria"]["passed"]:
        raise ValueError("El candidato no cumple los criterios de promoción.")

    releases_dir.mkdir(parents=True, exist_ok=True)
    final_dir = releases_dir / release_id
    if final_dir.exists():
        raise FileExistsError(f"La release ya existe: {final_dir}")

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{release_id}-", dir=releases_dir))
    try:
        artifacts = {}
        for name, filename in REQUIRED_CANDIDATE_FILES.items():
            shutil.copy2(candidate_dir / filename, temporary_dir / filename)
            artifacts[name] = filename
        shutil.copy2(candidate_metrics_path, temporary_dir / "metrics.json")
        artifacts["metrics"] = "metrics.json"
        manifest = {
            "release_id": release_id,
            "source_run_id": run_id,
            "status": "packaged",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "transformer_type": "column_transformer",
            "artifacts": artifacts,
            "validation": validation,
        }
        (temporary_dir / "release.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary_dir, final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return final_dir / "release.json"


def activate_release(manifest_path: Path, pointer_path: Path, history_dir: Path) -> None:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    paths = resolve_release_artifacts(manifest)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Release incompleta: " + ", ".join(missing))

    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    if pointer_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.copy2(pointer_path, history_dir / f"{timestamp}.json")

    pointer = {
        "release_id": manifest["release_id"],
        "manifest_path": os.path.relpath(
            manifest_path, start=pointer_path.parent.resolve()
        ),
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_pointer = pointer_path.with_suffix(".json.tmp")
    temporary_pointer.write_text(
        json.dumps(pointer, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_pointer, pointer_path)


def rollback_release(pointer_path: Path, history_dir: Path) -> str:
    history = sorted(history_dir.glob("*.json"))
    if not history:
        raise FileNotFoundError("No existe una release anterior para restaurar.")
    previous_pointer = history[-1]
    current_content = pointer_path.read_text(encoding="utf-8")
    previous_content = previous_pointer.read_text(encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    current_backup = history_dir / f"{timestamp}-replaced.json"
    current_backup.write_text(current_content, encoding="utf-8")
    temporary_pointer = pointer_path.with_suffix(".json.tmp")
    temporary_pointer.write_text(previous_content, encoding="utf-8")
    os.replace(temporary_pointer, pointer_path)
    previous_pointer.unlink()
    return json.loads(previous_content)["release_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    package_parser = subparsers.add_parser("package")
    activate_parser = subparsers.add_parser("activate")
    rollback_parser = subparsers.add_parser("rollback")

    for current in (validate_parser, package_parser):
        current.add_argument("--candidate-dir", type=Path, required=True)
        current.add_argument("--candidate-metrics", type=Path, required=True)
        current.add_argument(
            "--active-metrics", type=Path,
            default=Path("reports/metrics/metricas_modelo.json"),
        )
        current.add_argument("--dataset", type=Path, default=Path("dataset.csv"))

    package_parser.add_argument("--release-id", required=True)
    package_parser.add_argument("--run-id", required=True)
    package_parser.add_argument(
        "--releases-dir", type=Path, default=Path("models/releases")
    )
    activate_parser.add_argument("--manifest", type=Path, required=True)
    activate_parser.add_argument(
        "--pointer", type=Path, default=Path("models/active_release.json")
    )
    activate_parser.add_argument(
        "--history-dir", type=Path, default=Path("models/release_history")
    )
    rollback_parser.add_argument(
        "--pointer", type=Path, default=Path("models/active_release.json")
    )
    rollback_parser.add_argument(
        "--history-dir", type=Path, default=Path("models/release_history")
    )
    args = parser.parse_args()

    if args.command == "validate":
        result = validate_candidate(
            args.candidate_dir, args.candidate_metrics,
            args.active_metrics, args.dataset,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "package":
        manifest = package_release(
            args.release_id, args.run_id, args.candidate_dir,
            args.candidate_metrics, args.active_metrics, args.dataset,
            args.releases_dir,
        )
        print(f"Release empaquetada sin activar: {manifest}")
    elif args.command == "activate":
        activate_release(args.manifest, args.pointer, args.history_dir)
        active = load_active_release(args.pointer)
        print(f"Release activa: {active['release_id']}")
    else:
        release_id = rollback_release(args.pointer, args.history_dir)
        print(f"Rollback completado. Release activa: {release_id}")


if __name__ == "__main__":
    main()
