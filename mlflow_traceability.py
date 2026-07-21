import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict:
    workspace = Path.cwd().resolve()
    base_command = [
        "git",
        "-c",
        f"safe.directory={workspace.as_posix()}",
    ]
    try:
        commit = subprocess.run(
            base_command + ["rev-parse", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            base_command + ["status", "--porcelain"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty_worktree": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_worktree": None}


def dataset_profile(dataset_path: Path) -> dict:
    dataframe = pd.read_csv(dataset_path)
    target_distribution = {}
    if "Target" in dataframe.columns:
        target_distribution = {
            str(name): int(count)
            for name, count in dataframe["Target"].value_counts().items()
        }

    return {
        "path": str(dataset_path),
        "sha256": sha256_file(dataset_path),
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_values": int(dataframe.isna().sum().sum()),
        "target_distribution": target_distribution,
        "column_dtypes": {
            name: str(dtype) for name, dtype in dataframe.dtypes.items()
        },
    }


def build_traceability_manifest(
    dataset_path: Path,
    schema: dict,
    code_paths: list[Path],
    artifact_roles: dict,
) -> dict:
    code_versions = {
        str(path): sha256_file(path)
        for path in code_paths
        if path.exists()
    }
    requirements_path = Path("requirements.txt")

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_profile(dataset_path),
        "schema": schema,
        "split_policy": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
            "stratified": True,
            "random_state": 42,
        },
        "code": {
            "git": git_metadata(),
            "files_sha256": code_versions,
            "requirements_sha256": (
                sha256_file(requirements_path)
                if requirements_path.exists()
                else None
            ),
        },
        "artifact_roles": artifact_roles,
    }


def write_manifest(manifest: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
