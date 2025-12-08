from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

MODEL_PATH = os.path.join("artifacts", "best_model.pkl")

app = Flask(__name__)

# Prometheus metrics
PRED_TOTAL = Counter(
    "regression_prediction_total", "Total number of regression predictions"
)
PRED_ERRORS = Counter(
    "regression_prediction_errors_total", "Total number of regression prediction errors"
)
PRED_LATENCY = Histogram(
    "regression_prediction_latency_seconds",
    "Time spent doing regression prediction in seconds",
)
PRED_ABS_ERROR = Summary(
    "regression_prediction_abs_error",
    "Absolute error of predictions (when true value is provided)",
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found at artifacts/best_model.pkl")
    return joblib.load(MODEL_PATH)


model = None
try:
    model = load_model()
    print("Model loaded at startup.")
except Exception as e:
    print("Model NOT loaded at startup:", e)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expect JSON:
    {
      "features": [f1, f2, ..., f10],
      "true_value": <optional numeric, for error tracking>
    }
    """
    global model
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            PRED_ERRORS.inc()
            return jsonify({"error": f"Unable to load model: {e}"}), 500

    data = request.get_json() or {}
    features = data.get("features")
    true_value = data.get("true_value", None)

    if not features or len(features) != 10:
        PRED_ERRORS.inc()
        return (
            jsonify(
                {
                    "error": "Please send JSON with 'features': [10 numeric values for diabetes dataset]"
                }
            ),
            400,
        )

    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
    except Exception as e:
        PRED_ERRORS.inc()
        return jsonify({"error": f"Invalid features: {e}"}), 400

    PRED_TOTAL.inc()
    with PRED_LATENCY.time():
        pred = float(model.predict(arr)[0])

    # If true_value is provided, track absolute error
    if true_value is not None:
        try:
            tv = float(true_value)
            abs_err = abs(pred - tv)
            PRED_ABS_ERROR.observe(abs_err)
        except Exception:
            # ignore if true_value is not numeric
            pass

    return jsonify({"prediction": pred})


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
