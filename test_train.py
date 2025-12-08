from train import train_and_save_best


def test_best_model_quality():
    best_r2 = train_and_save_best()
    # require at least 0.4 R² (for diabetes dataset, this is reasonable)
    assert best_r2 > 0.4, f"Validation R² too low: {best_r2}"
