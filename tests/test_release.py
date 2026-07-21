import json
from pathlib import Path

from scripts.gestionar_release import (
    activate_release,
    promotion_criteria,
    rollback_release,
)
from infrastructure.model_release import load_active_release


def metrics(accuracy=0.8, macro_f1=0.7, dropout_recall=0.75):
    return {
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": {
            "Dropout": {"recall": dropout_recall},
        },
    }


def create_release(root: Path, release_id: str) -> Path:
    release_dir = root / release_id
    release_dir.mkdir(parents=True)
    artifacts = {
        "model": "model.keras",
        "transformer": "preprocessor.pkl",
        "encoder": "encoder.pkl",
        "feature_names": "feature_names.json",
    }
    for filename in artifacts.values():
        (release_dir / filename).write_text("test", encoding="utf-8")
    manifest = {
        "release_id": release_id,
        "artifacts": artifacts,
    }
    manifest_path = release_dir / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_promotion_criteria_protect_high_risk_recall():
    active = metrics(dropout_recall=0.75)
    candidate = metrics(accuracy=0.82, macro_f1=0.72, dropout_recall=0.70)

    result = promotion_criteria(candidate, active)

    assert result["passed"] is False
    assert result["checks"]["high_risk_recall_not_lower"] is False


def test_activation_and_rollback_use_atomic_pointer(tmp_path):
    releases = tmp_path / "releases"
    pointer = tmp_path / "active_release.json"
    history = tmp_path / "history"
    first = create_release(releases, "release-1")
    second = create_release(releases, "release-2")

    activate_release(first, pointer, history)
    activate_release(second, pointer, history)
    assert load_active_release(pointer)["release_id"] == "release-2"

    restored = rollback_release(pointer, history)

    assert restored == "release-1"
    assert load_active_release(pointer)["release_id"] == "release-1"
