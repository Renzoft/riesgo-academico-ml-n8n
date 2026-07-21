from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.calidad_datos import prepare_training_dataframe


def base_frame() -> pd.DataFrame:
    return pd.read_csv("dataset.csv").head(3).copy()


def write_dataset(tmp_path: Path, dataframe: pd.DataFrame) -> Path:
    path = tmp_path / "dataset.csv"
    dataframe.to_csv(path, index=False)
    return path


def test_predictor_null_is_retained_for_pipeline_imputation(tmp_path):
    dataframe = base_frame()
    dataframe.loc[0, "GDP"] = np.nan
    path = write_dataset(tmp_path, dataframe)

    cleaned, report = prepare_training_dataframe(path)

    assert len(cleaned) == 3
    assert pd.isna(cleaned.loc[0, "GDP"])
    assert report["findings"][
        "predictor_missing_values_retained_for_imputation"
    ] == 1


def test_exact_duplicate_is_removed_and_reported(tmp_path):
    dataframe = base_frame()
    dataframe = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)
    path = write_dataset(tmp_path, dataframe)

    cleaned, report = prepare_training_dataframe(path)

    assert len(cleaned) == 3
    assert report["findings"]["exact_duplicates_removed"] == 1


def test_contradictory_labels_are_detected_without_silent_deletion(tmp_path):
    dataframe = base_frame()
    contradiction = dataframe.iloc[[0]].copy()
    contradiction["Target"] = "Graduate"
    dataframe = pd.concat([dataframe, contradiction], ignore_index=True)
    path = write_dataset(tmp_path, dataframe)

    cleaned, report = prepare_training_dataframe(path)

    assert len(cleaned) == 4
    assert report["findings"]["contradictory_feature_rows"] == 2


def test_missing_target_is_removed(tmp_path):
    dataframe = base_frame()
    dataframe.loc[0, "Target"] = np.nan
    path = write_dataset(tmp_path, dataframe)

    cleaned, report = prepare_training_dataframe(path)

    assert len(cleaned) == 2
    assert report["findings"]["missing_target_rows_removed"] == 1


def test_invalid_binary_value_stops_training_and_writes_report(tmp_path):
    dataframe = base_frame()
    dataframe.loc[0, "Debtor"] = 7
    path = write_dataset(tmp_path, dataframe)
    report_path = tmp_path / "quality.json"

    with pytest.raises(ValueError, match="valores inválidos"):
        prepare_training_dataframe(path, report_path=report_path)

    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert '"Debtor"' in report
