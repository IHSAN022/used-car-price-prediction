"""predictor.py - logika inferensi model harga mobil bekas."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL = joblib.load(BASE_DIR / "used_car_model.joblib")
META = json.loads((BASE_DIR / "model_metadata.json").read_text())
FEATURES = META["num_features"] + META["cat_features"]
_GROUPS = {g.lower(): g for g in META["known_model_groups"]}


class ValidationError(ValueError):
    """Input pengguna tidak valid."""


def _to_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} harus berupa angka")


def build_features(payload: dict) -> pd.DataFrame:
    """Ubah input mentah menjadi fitur yang dipakai saat training."""
    missing = [k for k in ("year", "mileage", "model") if payload.get(k) in (None, "")]
    if missing:
        raise ValidationError(f"Field wajib belum diisi: {missing}")

    year = int(_to_float(payload["year"], "year"))
    mileage = _to_float(payload["mileage"], "mileage")
    lo, hi = META["input_limits"]["year"]
    if not lo <= year <= hi:
        raise ValidationError(f"year harus antara {lo} dan {hi}")
    if not 0 <= mileage <= META["input_limits"]["mileage"][1] * 1.5:
        raise ValidationError("mileage di luar rentang yang wajar")

    review = payload.get("review")
    review = np.nan if review in (None, "") else _to_float(review, "review")
    if not np.isnan(review) and not 0 <= review <= 5:
        raise ValidationError("review harus antara 0 dan 5")
    review_count = payload.get("review_count")
    review_count = np.nan if review_count in (None, "") else _to_float(review_count, "review_count")

    age = META["current_year"] - year
    model = str(payload["model"]).strip()
    trim = str(payload.get("trim") or "").strip() or "Base"

    row = {
        "mileage": mileage,
        "age": age,
        "review": review,
        "review_count": review_count,
        "mileage_per_year": mileage / age,
        "review_missing": int(np.isnan(review)),
        "model_grp": _GROUPS.get(model.lower(), "Other"),
        "trim": trim,
    }
    return pd.DataFrame([row])[FEATURES]


def predict_features(features: pd.DataFrame) -> np.ndarray:
    """Prediksi harga (USD) dari DataFrame fitur yang sudah siap."""
    return np.expm1(MODEL.predict(features))


def predict(payload: dict) -> dict:
    price = float(predict_features(build_features(payload))[0])
    mae = META["test_metrics"]["MAE"]
    return {
        "predicted_price": round(price, 2),
        "currency": "USD",
        "estimated_range": [round(max(price - mae, 0), 2), round(price + mae, 2)],
        "model_version": META["model_version"],
    }
