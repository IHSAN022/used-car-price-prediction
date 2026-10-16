"""app.py - REST API prediksi harga mobil bekas."""
import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

import predictor

app = Flask(__name__)
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
PRED_LOG = LOG_DIR / "predictions_log.csv"
FEEDBACK_LOG = LOG_DIR / "feedback_log.csv"
PRED_COLS = ["request_id", "timestamp", "status", "latency_ms", "year", "mileage",
             "model", "trim", "review", "review_count", "predicted_price"]
FB_COLS = ["request_id", "timestamp", "actual_price"]


def _append(path, cols, row):
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in cols})


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@app.get("/health")
def health():
    return jsonify(status="ok", model=predictor.META["model_name"],
                   version=predictor.META["model_version"])


@app.post("/predict")
def predict():
    start = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    rid = str(uuid.uuid4())
    log = {"request_id": rid, "timestamp": _now(), **{k: payload.get(k, "") for k in
           ("year", "mileage", "model", "trim", "review", "review_count")}}
    try:
        result = predictor.predict(payload)
        status, code = "ok", 200
        log["predicted_price"] = result["predicted_price"]
        body = {"request_id": rid, **result}
    except predictor.ValidationError as e:
        status, code, body = "invalid_input", 400, {"error": str(e)}
    except Exception as e:  # kesalahan tak terduga
        status, code, body = "server_error", 500, {"error": "internal error"}
    log["status"] = status
    log["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    _append(PRED_LOG, PRED_COLS, log)
    return jsonify(body), code


@app.post("/feedback")
def feedback():
    """Terima harga jual sebenarnya agar performa model bisa dipantau."""
    data = request.get_json(silent=True) or {}
    try:
        actual = float(data["actual_price"])
        rid = str(data["request_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="request_id dan actual_price (angka) wajib diisi"), 400
    _append(FEEDBACK_LOG, FB_COLS, {"request_id": rid, "timestamp": _now(), "actual_price": actual})
    return jsonify(status="saved"), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
