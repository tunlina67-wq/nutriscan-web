"""Step 5 (Machine learning): Random Forest that predicts the Nutri-Score grade (A–E).

Trained by  train_classifier.py  on Open Food Facts data and saved to
models/grade_rf.joblib. If the file is missing, the app uses the expert rules only.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "grade_rf.joblib"
FEATURES = ["energy_kcal", "total_fat_g", "sat_fat_g", "carbs_g", "sugar_g",
            "protein_g", "sodium_g", "is_beverage"]
_bundle = None
_load_error = None


def _load():
    global _bundle, _load_error
    if _bundle is not None or _load_error is not None:
        return _bundle
    if not MODEL_PATH.exists():
        _load_error = "no model file"
        return None
    try:
        import joblib
        _bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        _load_error = str(e)
    return _bundle


def available() -> bool:
    return _load() is not None


def features_per_100(nutrients: dict, serving_size, serving_unit):
    """Convert per-serving label values to the per-100 g/ml values the model was trained on."""
    if not serving_size:
        return None
    f = 100.0 / float(serving_size)
    get = lambda k: float(nutrients.get(k) or 0) * f
    return [get("energy_kcal"), get("total_fat_g"), get("sat_fat_g"), get("carbs_g"),
            get("sugar_g"), get("protein_g"), get("sodium_mg") / 1000.0,
            1.0 if serving_unit == "ml" else 0.0]


def predict(nutrients: dict, serving_size, serving_unit):
    b = _load()
    if b is None:
        return None
    x = features_per_100(nutrients, serving_size, serving_unit)
    if x is None:
        return {"grade": None, "note": "Serving size needed for the ML model"}
    model = b["model"]
    proba = model.predict_proba([x])[0]
    classes = [str(c).upper() for c in model.classes_]
    best = int(proba.argmax())
    return {"grade": classes[best], "confidence": round(float(proba[best]), 3),
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
            "test_accuracy": b.get("test_accuracy")}
