import numpy as np

from evaluacion_avanzada import (
    expected_calibration_error,
    multiclass_brier_score,
)
from evaluacion_cv_mlp import summarize_folds


def test_perfect_probabilities_have_zero_calibration_errors():
    target = np.array([0, 1, 2])
    probabilities = np.eye(3)

    assert multiclass_brier_score(target, probabilities, 3) == 0
    assert expected_calibration_error(target, probabilities) == 0


def test_brier_score_penalizes_wrong_confident_predictions():
    target = np.array([0])
    correct = np.array([[0.9, 0.05, 0.05]])
    wrong = np.array([[0.05, 0.9, 0.05]])

    assert multiclass_brier_score(target, wrong, 3) > (
        multiclass_brier_score(target, correct, 3)
    )


def test_cross_validation_summary_contains_mean_and_deviation():
    folds = [
        {"accuracy": 0.7, "macro_f1": 0.6, "weighted_f1": 0.7,
         "dropout_recall": 0.8},
        {"accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.9,
         "dropout_recall": 0.6},
    ]

    summary = summarize_folds(folds)

    assert summary["accuracy"]["mean"] == 0.8
    assert summary["dropout_recall"]["mean"] == 0.7
    assert summary["macro_f1"]["std"] > 0
