# test_train.py
from train import train_and_save_best


def test_best_model_quality():
    best_r2 = train_and_save_best()
    # quality gate: require at least 0.4 R²
    assert best_r2 > 0.4, f"Validation R² too low: {best_r2}"
