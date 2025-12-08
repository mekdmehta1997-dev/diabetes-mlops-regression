import os
import json

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn


EXPERIMENT_NAME = "diabetes_ridge_multimodel"


def train_candidates():
    """
    Train multiple Ridge regression models with different alphas,
    log each run to MLflow, and return the best model info.
    """
    data = load_diabetes()
    X = data.data
    y = data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    alphas = [0.1, 1.0, 10.0, 100.0]

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_r2 = -1.0
    best_model = None
    best_params = None

    for alpha in alphas:
        with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            r2 = r2_score(y_val, preds)

            # Log to MLflow
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("model_type", "Ridge")
            mlflow.log_metric("val_r2", float(r2))
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"alpha={alpha}, val_r2={r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_params = {"alpha": alpha}

    return best_model, best_r2, best_params


def train_and_save_best():
    """
    Train multiple candidates, pick best, and save to artifacts/best_model.pkl.
    Also save metrics to artifacts/metrics.json.
    """
    os.makedirs("artifacts", exist_ok=True)

    best_model, best_r2, best_params = train_candidates()

    if best_model is None:
        raise RuntimeError("No best model found!")

    # Save best model
    model_path = os.path.join("artifacts", "best_model.pkl")
    joblib.dump(best_model, model_path)

    # Save metrics
    metrics = {
        "best_val_r2": float(best_r2),
        "best_params": best_params,
    }
    metrics_path = os.path.join("artifacts", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Best model saved to:", model_path)
    print("Metrics:", metrics)
    return best_r2


if __name__ == "__main__":
    train_and_save_best()
