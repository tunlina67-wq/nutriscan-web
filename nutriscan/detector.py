"""Step 1 (Computer Vision): find the nutrition table with our trained YOLOv8 model.

Put the model trained in the Colab notebook at  models/best.pt.
If the model or the `ultralytics` library is missing, the app still works:
it simply sends the whole photo to OCR instead of the cropped table.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
_model = None
_load_error = None


def _load():
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return _model
    if not MODEL_PATH.exists():
        _load_error = f"no model at {MODEL_PATH.name}"
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    except Exception as e:  # library not installed or bad file
        _load_error = str(e)
    return _model


def available() -> bool:
    return _load() is not None


def detect_and_crop(img, conf: float = 0.25, pad: float = 0.03):
    """Return (image_to_read, info). `img` is a BGR numpy array."""
    model = _load()
    if model is None:
        return img, {"used": False, "found": False, "confidence": None,
                     "note": "YOLO model not loaded, using full photo"}

    r = model.predict(img, conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return img, {"used": True, "found": False, "confidence": None,
                     "note": "No nutrition table detected, using full photo"}

    i = int(r.boxes.conf.argmax())
    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
    h, w = img.shape[:2]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return img[y1:y2, x1:x2], {"used": True, "found": True,
                               "confidence": round(float(r.boxes.conf[i]), 3),
                               "box": [x1, y1, x2, y2]}
