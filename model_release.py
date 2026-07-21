import json
import os
from pathlib import Path


ACTIVE_RELEASE_PATH = Path(
    os.getenv("ACTIVE_RELEASE_PATH", "models/active_release.json")
)


def load_active_release(pointer_path: Path = ACTIVE_RELEASE_PATH) -> dict | None:
    if not pointer_path.exists():
        return None

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = Path(pointer["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = pointer_path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def resolve_release_artifacts(manifest: dict) -> dict[str, Path]:
    release_dir = Path(manifest["_manifest_path"]).parent
    return {
        name: (release_dir / relative_path).resolve()
        for name, relative_path in manifest["artifacts"].items()
    }
